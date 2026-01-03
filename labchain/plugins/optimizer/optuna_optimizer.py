import optuna

from typing import Any, Callable, Dict, Sequence, Union, cast
from labchain import F1
from labchain.container import Container
from labchain.base import BaseMetric, BasePlugin, XYData

from rich import print

from labchain.base import BaseFilter
from labchain.base.base_optimizer import BaseOptimizer


@Container.bind()
class OptunaOptimizer(BaseOptimizer):
    """
    Optuna-based optimizer for hyperparameter tuning.

    This class implements hyperparameter optimization using the Optuna framework.
    It allows for efficient searching of hyperparameter spaces for machine learning models.

    Key Features:
        - Supports various types of hyperparameters (categorical, integer, float)
        - Allows for customizable optimization direction (minimize or maximize)
        - Can resume previous studies or start new ones
        - Integrates with the Framework3 pipeline system

    Usage:
        The OptunaOptimizer can be used to optimize hyperparameters of a machine learning pipeline:

        ```python
        from framework3.plugins.optimizer import OptunaOptimizer
        from framework3.base import XYData

        # Assuming you have a pipeline and data
        pipeline = ...
        x_data = XYData(...)
        y_data = XYData(...)

        optimizer = OptunaOptimizer(direction="minimize", n_trials=100)
        optimizer.optimize(pipeline)
        optimizer.fit(x_data, y_data)

        best_pipeline = optimizer.pipeline
        ```

    Attributes:
        direction (str): The direction of optimization ("minimize" or "maximize").
        n_trials (int): The number of trials for the optimization process.
        load_if_exists (bool): Whether to load an existing study if one exists.
        reset_study (bool): Whether to reset the study before optimization.
        pipeline (BaseFilter | None): The pipeline to be optimized.
        study_name (str | None): The name of the Optuna study.
        storage (str | None): The storage URL for the Optuna study.

    Methods:
        optimize(pipeline: BaseFilter): Set up the optimization process for a given pipeline.
        fit(x: XYData, y: XYData | None): Perform the optimization and fit the best pipeline.
        predict(x: XYData) -> XYData: Make predictions using the optimized pipeline.
        evaluate(x_data: XYData, y_true: XYData | None, y_pred: XYData) -> Dict[str, Any]:
            Evaluate the optimized pipeline.
    """

    def __init__(
        self,
        direction: str,
        n_trials: int = 2,
        load_if_exists: bool = False,
        reset_study: bool = False,
        pipeline: BaseFilter | None = None,
        study_name: str | None = None,
        storage: str | None = None,
        scorer: BaseMetric = F1(),
    ):
        """
        Initialize the OptunaOptimizer.

        Args:
            direction (str): The direction of optimization ("minimize" or "maximize").
            n_trials (int): The number of trials for the optimization process.
            load_if_exists (bool): Whether to load an existing study if one exists.
            reset_study (bool): Whether to reset the study before optimization.
            pipeline (BaseFilter | None): The pipeline to be optimized.
            study_name (str | None): The name of the Optuna study.
            storage (str | None): The storage URL for the Optuna study.
        """
        super().__init__(direction=direction, study_name=study_name, storage=storage)
        self.direction = direction
        self.study_name = study_name
        self.storage = storage
        self.pipeline = pipeline
        self.n_trials = n_trials
        self.load_if_exists = load_if_exists
        self.reset_study = reset_study
        self.scorer = scorer

    def optimize(self, pipeline: BaseFilter):
        """
        Set up the optimization process for a given pipeline.

        This method prepares the Optuna study for optimization.

        Args:
            pipeline (BaseFilter): The pipeline to be optimized.
        """
        self.pipeline = pipeline
        self.pipeline._verbose = False

        if (
            self.reset_study
            and self.study_name is not None
            and self.storage is not None
        ):
            studies = optuna.get_all_study_summaries(storage=self.storage)

            if self.study_name in [study.study_name for study in studies]:
                optuna.delete_study(study_name=self.study_name, storage=self.storage)

        self._study = optuna.create_study(
            study_name=self.study_name,
            direction=self.direction,
            storage=self.storage,
            load_if_exists=self.load_if_exists,
        )

    def get_grid(self, aux: Dict[str, Any], f: Callable):
        """
        Recursively process the grid configuration of a pipeline or filter.

        This method traverses the configuration dictionary and applies the provided
        callable to each grid parameter.

        Args:
            aux (Dict[str, Any]): The configuration dictionary to process.
            f (Callable): A function to apply to each grid parameter.

        Note:
            This method modifies the input dictionary in-place.
        """
        match aux["params"]:
            case {"filters": filters, **r}:
                for filter_config in filters:
                    self.get_grid(filter_config, f)
            case {"pipeline": pipeline, **r}:  # noqa: F841
                self.get_grid(pipeline, f)
            case {"filter": cached_filter, **r}:  # noqa: F841
                self.get_grid(cached_filter, f)
            case p_params:
                if "_grid" in aux:
                    for param, value in aux["_grid"].items():
                        value = f(param, value)
                        p_params.update({param: value})

    def build_pipeline(
        self, dumped_pipeline: Dict[str, Any], f: Callable
    ) -> BaseFilter:
        """
        Build a pipeline from a dumped configuration.

        This method processes the dumped pipeline configuration, applies the provided
        callable to the grid parameters, and constructs a new BaseFilter object.

        Args:
            dumped_pipeline (Dict[str, Any]): The dumped pipeline configuration.
            f (Callable): A function to apply to each grid parameter.

        Returns:
            BaseFilter: The constructed pipeline.

        Note:
            This method uses the Container.ppif for dependency injection when building
            the pipeline components.
        """
        self.get_grid(dumped_pipeline, f)

        pipeline: BaseFilter = cast(
            BaseFilter, BasePlugin.build_from_dump(dumped_pipeline, Container.ppif)
        )
        return pipeline

    def fit(self, x: XYData, y: XYData | None = None):
        """
        Perform the optimization and fit the best pipeline.

        This method runs the Optuna optimization process and fits the best found pipeline.

        Args:
            x (XYData): The input features.
            y (XYData | None): The target values (if applicable).

        Raises:
            ValueError: If the pipeline is not defined before fitting.
        """
        self._print_acction("Fitting with OptunaOptimizer...")
        if self._verbose:
            print(self.pipeline)

        if self.pipeline is not None:
            dumped_pipeline = self.pipeline.item_dump(include=["_grid"])
            print(dumped_pipeline)

            def objective(trial) -> Union[float, Sequence[float]]:
                def matcher(k, v):
                    match v:
                        case list():
                            return trial.suggest_categorical(k, v)
                        case dict():
                            if type(v["low"]) is int and type(v["high"]) is int:
                                return trial.suggest_int(k, v["low"], v["high"])
                            elif type(v["low"]) is float and type(v["high"]) is float:
                                return trial.suggest_float(k, v["low"], v["high"])
                            else:
                                raise ValueError(
                                    f"Inconsistent types in tuple: {k}: {v}"
                                )
                        case (min_v, max_v):
                            if type(min_v) is int and type(max_v) is int:
                                return trial.suggest_int(k, min_v, max_v)
                            elif type(min_v) is float and type(max_v) is float:
                                return trial.suggest_float(k, min_v, max_v)
                            else:
                                raise ValueError(
                                    f"Inconsistent types in tuple: {k}: {v}"
                                )
                        case _:
                            raise ValueError(f"Unsupported type in grid: {k}: {v}")

                pipeline: BaseFilter = self.build_pipeline(dumped_pipeline, matcher)
                pipeline.verbose(False)

                match pipeline.fit(x, y):
                    case None:
                        metrics = pipeline.evaluate(x, y, pipeline.predict(x))
                        return float(metrics.get(self.scorer.__class__.__name__, 0.0))
                    case float() as loss:
                        return loss
                    case dict() as losses:
                        return float(losses.get(self.scorer.__class__.__name__, 0.0))
                    case _:
                        raise ValueError("Unsupported type in pipeline.fit")

            self._study.optimize(
                objective, n_trials=self.n_trials, show_progress_bar=True
            )

            best_params = self._study.best_params
            if best_params:
                print(f"Best params: {best_params}")
                pipeline = self.build_pipeline(
                    dumped_pipeline, lambda k, _: best_params[k]
                ).unwrap()
                pipeline.fit(x, y)
                self.pipeline = pipeline
            else:
                self.pipeline.unwrap().fit(x, y)
        else:
            raise ValueError("Pipeline must be defined before fitting")

    def predict(self, x: XYData) -> XYData:
        """
        Make predictions using the optimized pipeline.

        Args:
            x (XYData): The input features.

        Returns:
            XYData: The predictions.

        Raises:
            ValueError: If the pipeline is not fitted before predicting.
        """
        self._print_acction("Predicting with OptunaOptimizer...")
        if self._verbose:
            print(self.pipeline)

        if self.pipeline is not None:
            return self.pipeline.predict(x)
        else:
            raise ValueError("Pipeline must be fitted before predicting")

    def evaluate(
        self, x_data: XYData, y_true: XYData | None, y_pred: XYData
    ) -> Dict[str, Any]:
        """
        Evaluate the optimized pipeline.

        Args:
            x_data (XYData): The input features.
            y_true (XYData | None): The true target values (if applicable).
            y_pred (XYData): The predicted values.

        Returns:
            Dict[str, Any]: A dictionary containing evaluation metrics.

        Raises:
            ValueError: If the pipeline is not fitted before evaluating.
        """
        if self.pipeline is not None:
            return self.pipeline.evaluate(x_data, y_true, y_pred)
        else:
            raise ValueError("Pipeline must be fitted before evaluating")

    def start(
        self, x: XYData, y: XYData | None, X_: XYData | None
    ) -> XYData | None: ...

    def finish(self) -> None: ...
