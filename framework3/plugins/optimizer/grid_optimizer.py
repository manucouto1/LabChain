from framework3.container import Container
from framework3.base import BaseFilter, BasePlugin, XYData, BaseMetric
from framework3.base.base_optimizer import BaseOptimizer
from typing import Any, Dict, List, Optional, cast
import itertools
import pandas as pd


class GridOptimizer(BaseOptimizer):
    """
    Grid search optimizer for hyperparameter tuning.

    This class implements a grid search for hyperparameter optimization.
    It exhaustively searches through a specified parameter grid to find the best combination of hyperparameters.

    Key Features:
        - Supports various types of hyperparameters (categorical, numerical)
        - Performs an exhaustive search over the specified parameter grid
        - Evaluates each parameter combination on the entire dataset
        - Integrates with the Framework3 pipeline system

    Usage:
        The GridOptimizer can be used to optimize hyperparameters of a machine learning pipeline:

        ```python
        from framework3.plugins.optimizer import GridOptimizer
        from framework3.base import XYData

        # Assuming you have a pipeline and data
        pipeline = ...
        x_data = XYData(...)
        y_data = XYData(...)

        optimizer = GridOptimizer(scoring=some_metric)
        optimizer.optimize(pipeline)
        optimizer.fit(x_data, y_data)

        best_pipeline = optimizer.pipeline
        ```

    Attributes:
        scoring (BaseMetric): The scoring metric to use for evaluation.
        pipeline (BaseFilter | None): The pipeline to be optimized.
        best_params (Dict[str, Any]): The best parameters found during the search.
        best_score (float): The best score achieved during the search.
        _grid (Dict[str, Any]): The parameter grid for the search.
        _results (pd.DataFrame | None): DataFrame containing all evaluation results.

    Methods:
        optimize(pipeline: BaseFilter): Set up the optimization process for a given pipeline.
        fit(x: XYData, y: Optional[XYData]) -> None | float: Perform the grid search and fit the best pipeline.
        predict(x: XYData) -> XYData: Make predictions using the best pipeline found.
        evaluate(x_data: XYData, y_true: XYData | None, y_pred: XYData) -> Dict[str, Any]:
            Evaluate the optimized pipeline.
    """

    def __init__(
        self,
        scorer: BaseMetric,
        pipeline: BaseFilter | None = None,
    ):
        super().__init__(pipeline=pipeline, scorer=scorer)
        self.scorer: BaseMetric = scorer
        self.pipeline: Optional[BaseFilter] = None

        self.best_params: Dict[str, Any] = {}
        self.best_score: float = float("-inf")
        self._grid: Dict[str, Any] = {}
        self._results: pd.DataFrame | None = None

    def get_grid(self, aux: Dict[str, Any], grid: Dict[str, Any]):
        """
        Recursively process the grid configuration of a pipeline or filter.

        This method traverses the configuration dictionary and extracts the grid parameters.

        Args:
            aux (Dict[str, Any]): The configuration dictionary to process.
            grid (Dict[str, Any]): The grid to populate with hyperparameters.

        Note:
            This method modifies the input dictionary in-place.
        """
        match aux["params"]:
            case {"filters": filters, **r}:
                for filter_config in filters:
                    self.get_grid(filter_config, grid)
            case {"pipeline": pipeline, **r}:  # noqa: F841
                self.get_grid(pipeline, grid)
            case {"filter": cached_filter, **r}:  # noqa: F841
                self.get_grid(cached_filter, grid)
            case _:
                if "_grid" in aux:
                    grid[aux["clazz"]] = aux["_grid"]

    def get_from_grid(self, aux: Dict[str, Any], config: Dict[str, Any]):
        """
        Recursively process the grid configuration of a pipeline or filter.

        This method traverses the configuration dictionary and applies the grid parameters.

        Args:
            aux (Dict[str, Any]): The configuration dictionary to process.
            config (Dict[str, Any]): The configuration to apply.

        Note:
            This method modifies the input dictionary in-place.
        """
        match aux["params"]:
            case {"filters": filters, **r}:
                for filter_config in filters:
                    self.get_from_grid(filter_config, config)
            case {"pipeline": pipeline, **r}:  # noqa: F841
                self.get_from_grid(pipeline, config)
            case {"filter": cached_filter, **r}:  # noqa: F841
                self.get_from_grid(cached_filter, config)
            case p_params:
                if "_grid" in aux:
                    for param, value in aux["_grid"].items():
                        p_params.update({param: config[aux["clazz"]][param]})

    def nested_product(self, d: Dict[str, Any]) -> List[Dict[str, Any]]:
        def recurse(current) -> List[Dict[str, Any]]:
            if isinstance(current, dict):
                keys, values = zip(*[(k, recurse(v)) for k, v in current.items()])
                return [dict(zip(keys, v)) for v in itertools.product(*values)]
            elif isinstance(current, list):
                return current
            else:
                return [current]

        return recurse(d)

    def optimize(self, pipeline: BaseFilter):
        """
        Set up the optimization process for a given pipeline.

        This method prepares the pipeline for grid search optimization.

        Args:
            pipeline (BaseFilter): The pipeline to be optimized.
        """
        self.pipeline = pipeline
        self.pipeline.verbose(False)

    def fit(self, x: XYData, y: Optional[XYData]):
        """
        Perform the grid search and fit the best pipeline.

        This method runs the grid search optimization process and fits the best found pipeline.

        Args:
            x (XYData): The input features.
            y (Optional[XYData]): The target values (if applicable).

        Returns:
            None | float: None if the pipeline is fitted successfully, or the best score if available.

        Raises:
            ValueError: If the pipeline is not defined before fitting.
        """
        if self.pipeline is None:
            raise ValueError("Pipeline must be set before fitting")

        dumped_pipeline = self.pipeline.item_dump(include=["_grid"])

        self.get_grid(dumped_pipeline, self._grid)

        print(self._grid)

        results = []
        combinations = self.nested_product(self._grid)

        for param_dict in combinations:
            self.get_from_grid(dumped_pipeline, param_dict)

            pipeline: BaseFilter = cast(
                BaseFilter, BasePlugin.build_from_dump(dumped_pipeline, Container.pif)
            )

            pipeline.verbose(False)

            match pipeline.fit(x, y):
                case None:
                    losses = pipeline.evaluate(x, y, pipeline.predict(x))
                    score = losses.pop(self.scorer.__class__.__name__, 0.0)
                    score_data = {"score": score, **losses}
                case float() as score:
                    score_data = {"score": score}
                    pass
                case dict() as losses:
                    score = losses.pop(self.scorer.__class__.__name__, 0.0)
                    score_data = {"score": score, **losses}
                case _:
                    raise ValueError("Unexpected return type from pipeline.fit()")

            results.append({**param_dict, **score_data})

        # Create DataFrame with all combinations and scores
        self._results = pd.DataFrame(results)
        self._results = self._results.sort_values(
            "score", ascending=not self.scorer.higher_better
        )

        self.best_params = self._results.iloc[0].drop("score").to_dict()

        self.get_from_grid(dumped_pipeline, self.best_params)

        self.pipeline = cast(
            BaseFilter, BasePlugin.build_from_dump(dumped_pipeline, Container.pif)
        )

        return self.pipeline.unwrap().fit(x, y)

    def predict(self, x: XYData) -> XYData:
        """
        Make predictions using the best pipeline found.

        Args:
            x (XYData): The input features.

        Returns:
            XYData: The predictions.

        Raises:
            ValueError: If the pipeline is not fitted before predicting.
        """
        if self.pipeline is not None:
            return self.pipeline.predict(x)
        else:
            raise ValueError("Pipeline must be fitted before predicting")

    def start(self, x: XYData, y: XYData | None, X_: XYData | None) -> XYData | None:
        """
        Start the pipeline execution.

        Args:
            x (XYData): Input data for fitting.
            y (XYData | None): Target data for fitting.
            X_ (XYData | None): Data for prediction (if different from x).

        Returns:
            XYData | None: Prediction results if X_ is provided, else None.

        Raises:
            ValueError: If the pipeline has not been fitted.
        """
        if self.pipeline is not None:
            return self.pipeline.start(x, y, X_)
        else:
            raise ValueError("Pipeline must be fitted before starting")

    def evaluate(
        self, x_data: XYData, y_true: XYData | None, y_pred: XYData
    ) -> Dict[str, Any]:
        """
        Evaluate the optimized pipeline.

        Args:
            x_data (XYData): Input data.
            y_true (XYData | None): True target data.
            y_pred (XYData): Predicted target data.

        Returns:
            Dict[str, Any]: A dictionary containing the evaluation results.
        """
        return (
            self.pipeline.evaluate(x_data, y_true, y_pred)
            if self.pipeline is not None
            else {}
        )
