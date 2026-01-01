from typing import Any, Callable, Dict, Tuple, Optional

from sklearn.model_selection import GridSearchCV
from labchain.base import BaseFilter, XYData
from labchain.base.base_optimizer import BaseOptimizer
from labchain.container.container import Container
from sklearn.pipeline import Pipeline

from labchain.utils.skestimator import SkWrapper
from rich import print
import pandas as pd

__all__ = ["SklearnOptimizer"]


@Container.bind()
class SklearnOptimizer(BaseOptimizer):
    """
    Sklearn-based optimizer for hyperparameter tuning using GridSearchCV.

    This class implements hyperparameter optimization using scikit-learn's GridSearchCV.
    It allows for efficient searching of hyperparameter spaces for machine learning models
    within the Framework3 pipeline system.

    Key Features:
        - Supports various types of hyperparameters (categorical, numerical)
        - Integrates with scikit-learn's GridSearchCV for exhaustive search
        - Allows for customizable scoring metrics
        - Integrates with the Framework3 pipeline system

    Usage:
        The SklearnOptimizer can be used to optimize hyperparameters of a machine learning pipeline:

        ```python
        from framework3.plugins.optimizer import SklearnOptimizer
        from framework3.base import XYData

        # Assuming you have a pipeline and data
        pipeline = ...
        x_data = XYData(...)
        y_data = XYData(...)

        optimizer = SklearnOptimizer(scoring='accuracy', cv=5)
        optimizer.optimize(pipeline)
        optimizer.fit(x_data, y_data)

        best_pipeline = optimizer.pipeline
        ```

    Attributes:
        scoring (str | Callable | Tuple | Dict): The scoring metric for GridSearchCV.
        pipeline (BaseFilter | None): The pipeline to be optimized.
        cv (int): The number of cross-validation folds.
        _grid (Dict): The parameter grid for GridSearchCV.
        _filters (List[Tuple[str, SkWrapper]]): The list of pipeline steps.
        _pipeline (Pipeline): The scikit-learn Pipeline object.
        _clf (GridSearchCV): The GridSearchCV object.

    Methods:
        optimize(pipeline: BaseFilter): Set up the optimization process for a given pipeline.
        fit(x: XYData, y: Optional[XYData]) -> None | float: Fit the GridSearchCV object to the given data.
        predict(x: XYData) -> XYData: Make predictions using the best estimator found by GridSearchCV.
        evaluate(x_data: XYData, y_true: XYData | None, y_pred: XYData) -> Dict[str, Any]:
            Evaluate the optimized pipeline.
    """

    def __init__(
        self,
        scoring: str | Callable | Tuple | Dict,
        pipeline: BaseFilter | None = None,
        cv: int = 2,
        n_jobs: int | None = None,
    ):
        """
        Initialize the SklearnOptimizer.

        Args:
            scoring (str | Callable | Tuple | Dict): Strategy to evaluate the performance of the cross-validated model.
            pipeline (BaseFilter | None): The pipeline to be optimized. Defaults to None.
            cv (int): Determines the cross-validation splitting strategy. Defaults to 2.
        """

        super().__init__(
            scoring=scoring,
            cv=cv,
            pipeline=pipeline,
        )
        self.pipeline = pipeline
        self.n_jobs = n_jobs
        self._grid = {}

    def get_grid(self, aux: Dict[str, Any]) -> None:
        """
        Recursively process the grid configuration of a pipeline or filter.

        This method traverses the configuration dictionary and builds the parameter grid
        for GridSearchCV.

        Args:
            aux (Dict[str, Any]): The configuration dictionary to process.

        Note:
            This method modifies the _grid attribute in-place.
        """
        match aux["params"]:
            case {"filters": filters, **r}:
                for filter_config in filters:
                    self.get_grid(filter_config)
            case {"pipeline": pipeline, **r}:  # noqa: F841
                self.get_grid(pipeline)
            case _:
                if "_grid" in aux:
                    for param, value in aux["_grid"].items():
                        if type(value) is list:
                            self._grid[f'{aux["clazz"]}__{param}'] = value
                        else:
                            self._grid[f'{aux["clazz"]}__{param}'] = [value]

    def optimize(self, pipeline: BaseFilter):
        """
        Set up the optimization process for a given pipeline.

        This method prepares the GridSearchCV object for optimization.

        Args:
            pipeline (BaseFilter): The pipeline to be optimized.
        """
        self.pipeline = pipeline
        self.pipeline.verbose(False)
        self._filters = list(
            map(lambda x: (x.__name__, SkWrapper(x)), self.pipeline.get_types())
        )

        dumped_pipeline = self.pipeline.item_dump(include=["_grid"])
        self.get_grid(dumped_pipeline)

        self._pipeline = Pipeline(self._filters)

        self._clf: GridSearchCV = GridSearchCV(
            estimator=self._pipeline,
            param_grid=self._grid,
            scoring=self.scoring,
            cv=self.cv,
            n_jobs=self.n_jobs,
            verbose=10,
        )

    def start(
        self, x: XYData, y: Optional[XYData], X_: Optional[XYData]
    ) -> Optional[XYData]:
        """
        Start the pipeline execution.

        This method fits the optimizer and makes predictions if X_ is provided.

        Args:
            x (XYData): Input data for fitting.
            y (Optional[XYData]): Target data for fitting.
            X_ (Optional[XYData]): Data for prediction (if different from x).

        Returns:
            Optional[XYData]: Prediction results if X_ is provided, else None.

        Raises:
            Exception: If an error occurs during pipeline execution.
        """
        try:
            self.fit(x, y)
            if X_ is not None:
                return self.predict(X_)
            else:
                return self.predict(x)
        except Exception as e:
            print(f"Error during pipeline execution: {e}")
            raise e

    def fit(self, x: XYData, y: Optional[XYData]) -> None | float:
        """
        Fit the GridSearchCV object to the given data.

        This method performs the grid search and prints the results.

        Args:
            x (XYData): The input features.
            y (Optional[XYData]): The target values.

        Returns:
            None | float: The best score achieved during the grid search.
        """
        self._clf.fit(x.value, y.value if y is not None else None)
        results = self._clf.cv_results_
        results_df = (
            pd.DataFrame(results)
            .iloc[:, 4:]
            .sort_values("mean_test_score", ascending=False)
        )
        print(results_df)
        return self._clf.best_score_  # type: ignore

    def predict(self, x: XYData) -> XYData:
        """
        Make predictions using the best estimator found by GridSearchCV.

        Args:
            x (XYData): The input features.

        Returns:
            XYData: The predicted values wrapped in an XYData object.
        """
        return XYData.mock(self._clf.predict(x.value))  # type: ignore

    def evaluate(
        self, x_data: XYData, y_true: XYData | None, y_pred: XYData
    ) -> Dict[str, Any]:
        """
        Evaluate the optimized pipeline.

        This method applies each metric in the pipeline to the predicted and true values,
        and includes the best score from GridSearchCV.

        Args:
            x_data (XYData): Input data.
            y_true (XYData | None): True target data.
            y_pred (XYData): Predicted target data.

        Returns:
            Dict[str, Any]: A dictionary containing the evaluation results for each metric
                            and the best score from GridSearchCV.

        Example:
            ```python
            >>> evaluation = optimizer.evaluate(x_test, y_test, predictions)
            >>> print(evaluation)
            {'F1Score': 0.85, 'best_score': 0.87}
            ```
        """
        if self.pipeline is None:
            raise Exception("No pipeline set for evaluation.")

        results = self.pipeline.evaluate(x_data, y_true, y_pred)
        results["best_score"] = self._clf.best_score_  # type: ignore
        return results
