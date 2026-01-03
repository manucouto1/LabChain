from typing import Dict, List, Optional
from labchain.base.exceptions import NotTrainableFilterError

from pydantic import ConfigDict
from labchain.base import XYData
from labchain.base import BaseFilter, BaseMetric
from labchain.base import SequentialPipeline
from labchain.container import Container

from rich import print as rprint

__all__ = ["F3Pipeline"]


@Container.bind()
class F3Pipeline(SequentialPipeline):
    """
    A flexible sequential pipeline implementation for machine learning workflows.

    F3Pipeline allows chaining multiple filters together and applying metrics
    for evaluation. It supports fitting, predicting, and evaluating data through the pipeline.

    Key Features:
        - Sequential application of multiple filters
        - Support for various metrics for evaluation
        - Configurable data storage and logging options

    Usage:
        ```python
        from framework3.plugins.pipelines.sequential import F3Pipeline
        from framework3.plugins.filters.transformation import PCAPlugin
        from framework3.plugins.filters.classification import ClassifierSVMPlugin
        from framework3.plugins.metrics.classification import F1, Precision, Recall
        from framework3.base import XYData
        import numpy as np

        # Create a pipeline with PCA and SVM
        pipeline = F3Pipeline(
            filters=[
                PCAPlugin(n_components=2),
                ClassifierSVMPlugin(kernel='rbf', C=1.0)
            ],
            metrics=[F1(), Precision(), Recall()]
        )

        # Prepare some dummy data
        X = XYData(value=np.random.rand(100, 10))
        y = XYData(value=np.random.randint(0, 2, 100))

        # Fit the pipeline
        pipeline.fit(X, y)

        # Make predictions
        y_pred = pipeline.predict(X)

        # Evaluate the pipeline
        results = pipeline.evaluate(X, y, y_pred)
        print(results)
        ```

    Attributes:
        filters (List[BaseFilter]): List of filters to be applied in the pipeline.
        metrics (List[BaseMetric]): List of metrics for evaluation.
        overwrite (bool): Whether to overwrite existing data in storage.
        store (bool): Whether to store intermediate results.
        log (bool): Whether to log pipeline operations.

    Methods:
        fit(x: XYData, y: Optional[XYData]) -> None | float: Fit the pipeline to the input data.
        predict(x: XYData) -> XYData: Make predictions using the fitted pipeline.
        evaluate(x_data: XYData, y_true: XYData | None, y_pred: XYData) -> Dict[str, float]:
            Evaluate the pipeline using specified metrics.
        start(x: XYData, y: Optional[XYData], X_: Optional[XYData]) -> Optional[XYData]:
            Start the pipeline execution by fitting and optionally predicting.
    """

    model_config = ConfigDict(extra="allow")

    def __init__(
        self,
        filters: List[BaseFilter],
        metrics: List[BaseMetric] = [],
        overwrite: bool = False,
        store: bool = False,
        log: bool = False,
    ) -> None:
        """
        Initialize the F3Pipeline.

        Args:
            filters (List[BaseFilter]): List of filters to be applied in the pipeline.
            metrics (List[BaseMetric], optional): List of metrics for evaluation. Defaults to [].
            overwrite (bool, optional): Whether to overwrite existing data. Defaults to False.
            store (bool, optional): Whether to store intermediate results. Defaults to False.
            log (bool, optional): Whether to log pipeline operations. Defaults to False.
        """
        super().__init__(
            filters=filters, metrics=metrics, overwrite=overwrite, store=store, log=log
        )
        self.filters: List[BaseFilter] = filters
        self.metrics: List[BaseMetric] = metrics
        self.overwrite = overwrite
        self.store = store
        self.log = log
        # self._filters: List[BaseFilter] = []

    def start(
        self, x: XYData, y: Optional[XYData], X_: Optional[XYData]
    ) -> Optional[XYData]:
        """
        Start the pipeline execution by fitting the model and making predictions.

        This method initiates the pipeline process by fitting the model to the input data
        and then making predictions.

        Args:
            x (XYData): The input data for fitting and prediction.
            y (Optional[XYData]): The target data for fitting, if available.
            X_ (Optional[XYData]): Additional input data for prediction, if different from x.

        Returns:
            Optional[XYData]: The predictions made by the pipeline, or None if an error occurs.

        Raises:
            Exception: If an error occurs during the fitting or prediction process.
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

    def fit(self, x: XYData, y: Optional[XYData]) -> None | float | dict:
        """
        Fit the pipeline to the input data.

        This method applies each filter in the pipeline sequentially to the input data,
        fitting each filter that requires training.

        Args:
            x (XYData): The input data to fit the pipeline on.
            y (Optional[XYData]): The target data, if available.

        Returns:
            None | float: The loss value from the last fitted filter, if any.

        Note:
            Filters that raise NotTrainableFilterError will be initialized instead of fitted.
        """
        self._print_acction("Fitting pipeline")
        loss = None
        for filter in self.filters:
            if self._verbose:
                rprint(filter)
            try:
                loss = filter.fit(x, y)
            except NotTrainableFilterError:
                if self._verbose:
                    rprint("Skipping not trainable filter:", filter.__class__.__name__)

            x = filter.predict(x)

        return loss

    def predict(self, x: XYData) -> XYData:
        """
        Make predictions using the fitted pipeline.

        This method applies each filter in the pipeline sequentially to the input data
        to generate predictions.

        Args:
            x (XYData): The input data to make predictions on.

        Returns:
            XYData: The predictions made by the pipeline.
        """

        self._print_acction("Predicting pipeline")

        for filter_ in self.filters:
            if self._verbose:
                rprint(filter_)
            x = filter_.predict(x)

        return x

    def evaluate(
        self, x_data: XYData, y_true: XYData | None, y_pred: XYData
    ) -> Dict[str, float]:
        """
        Evaluate the pipeline using the specified metrics.

        This method applies each metric in the pipeline to the predicted and true values,
        returning a dictionary of evaluation results.

        Args:
            x_data (XYData): The input data used for evaluation.
            y_true (XYData | None): The true target values, if available.
            y_pred (XYData): The predicted values.

        Returns:
            Dict[str, float]: A dictionary containing the evaluation results for each metric.

        Example:
            ```python
            >>> results = pipeline.evaluate(x_test, y_test, y_pred)
            >>> print(results)
            {'F1': 0.85, 'Precision': 0.80, 'Recall': 0.90}
            ```
        """

        self._print_acction("Evaluating pipeline...")

        evaluations = {}
        for metric in self.metrics:
            evaluations[metric.__class__.__name__] = metric.evaluate(
                x_data, y_true, y_pred
            )
        return evaluations

    def inner(self) -> List[BaseFilter]:
        """
        Get the list of filters in the pipeline.

        Returns:
            List[BaseFilter]: The list of filters in the pipeline.
        """
        return self.filters
