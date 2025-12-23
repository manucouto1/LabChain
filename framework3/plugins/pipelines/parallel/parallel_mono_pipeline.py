from copy import deepcopy
from typing import List, Dict, Any, Optional, Sequence
import numpy as np
from framework3.base import BaseMetric, XYData
from framework3.base import BaseFilter
from framework3.base import ParallelPipeline
from framework3.base.exceptions import NotTrainableFilterError
from framework3.container.container import Container

__all__ = ["MonoPipeline"]


@Container.bind()
class MonoPipeline(ParallelPipeline):
    """
    A pipeline that combines multiple filters in parallel and constructs new features from their outputs.

    This pipeline allows for simultaneous execution of multiple filters on the same input data,
    and then combines their outputs to create new features. It's particularly useful for
    feature engineering and ensemble methods.

    Key Features:
        - Parallel execution of multiple filters
        - Combination of filter outputs for feature construction
        - Support for evaluation metrics

    Usage:
        ```python
        from framework3.plugins.pipelines.parallel import MonoPipeline
        from framework3.plugins.filters.transformation import PCAPlugin
        from framework3.plugins.filters.classification import KnnFilter
        from framework3.plugins.metrics import F1Score
        from framework3.base import XYData

        pipeline = MonoPipeline(
            filters=[
                PCAPlugin(n_components=2),
                KnnFilter(n_neighbors=3)
            ],
            metrics=[F1Score()]
        )

        x_train = XYData(...)
        y_train = XYData(...)
        x_test = XYData(...)
        y_test = XYData(...)

        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        evaluation = pipeline.evaluate(x_test, y_test, predictions)
        print(evaluation)
        ```

    Attributes:
        filters (Sequence[BaseFilter]): A sequence of filters to be applied in parallel.

    Methods:
        fit(x: XYData, y: Optional[XYData], evaluator: BaseMetric | None = None) -> Optional[float]:
            Fit all filters in parallel.
        predict(x: XYData) -> XYData: Make predictions using all filters in parallel.
        evaluate(x_data: XYData, y_true: XYData | None, y_pred: XYData) -> Dict[str, Any]:
            Evaluate the pipeline using provided metrics.
        combine_features(pipeline_outputs: list[XYData]) -> XYData:
            Combine features from all filter outputs.
        start(x: XYData, y: XYData | None, X_: XYData | None) -> XYData | None:
            Start the pipeline execution.
    """

    def __init__(self, filters: Sequence[BaseFilter]):
        """
        Initialize the MonoPipeline.

        Args:
            filters (Sequence[BaseFilter]): A sequence of filters to be applied in parallel.
        """
        super().__init__(filters=filters)
        self.filters = filters

    def fit(
        self, x: XYData, y: Optional[XYData], evaluator: BaseMetric | None = None
    ) -> Optional[float]:
        """
        Fit all filters in the pipeline to the input data in parallel.

        This method applies the fit operation to each filter in the pipeline
        using the provided input data.

        Args:
            x (XYData): The input data to fit the filters on.
            y (XYData | None): The target data, if available.
            evaluator (BaseMetric | None, optional): An evaluator metric, if needed. Defaults to None.

        Returns:
            Optional[float]: The mean of the losses returned by the filters, if any.

        Note:
            Filters that raise NotTrainableFilterError will be initialized instead of fitted.
        """
        losses = []
        for f in self.filters:
            try:
                losses.append(f.fit(deepcopy(x), y))
            except NotTrainableFilterError:
                print("Skipping not trainable filter:", f.__class__.__name__)

        # filtre los valores None

        match list(filter(lambda x: x is not None, losses)):
            case []:
                return None
            case lss:
                return float(np.mean(lss))

    def predict(self, x: XYData) -> XYData:
        """
        Make predictions using all filters in parallel and combine their outputs.

        This method applies the predict operation to each filter in the pipeline
        and then combines the outputs using the combine_features method.

        Args:
            x (XYData): The input data to make predictions on.

        Returns:
            XYData: The combined predictions from all filters.
        """
        outputs: List[XYData] = [filter.predict(deepcopy(x)) for filter in self.filters]
        return self.combine_features(outputs)

    def evaluate(
        self, x_data: XYData, y_true: XYData | None, y_pred: XYData
    ) -> Dict[str, Any]:
        """
        Evaluate the pipeline using the provided metrics.

        This method applies each metric in the pipeline to the predicted and true values,
        returning a dictionary of evaluation results.

        Args:
            x_data (XYData): Input data used for evaluation.
            y_true (XYData | None): True target data, if available.
            y_pred (XYData): Predicted target data.

        Returns:
            Dict[str, Any]: A dictionary containing the evaluation results for each metric.

        Example:
            ```python
            >>> evaluation = pipeline.evaluate(x_test, y_test, predictions)
            >>> print(evaluation)
            {'F1Score': 0.85, 'Accuracy': 0.90}
            ```
        """
        results = {}
        for metric in self.metrics:
            results[metric.__class__.__name__] = metric.evaluate(x_data, y_true, y_pred)
        return results

    @staticmethod
    def combine_features(pipeline_outputs: list[XYData]) -> XYData:
        """
        Combine features from all filter outputs.

        This method concatenates the features from all filter outputs along the last axis.

        Args:
            pipeline_outputs (List[XYData]): List of outputs from each filter.

        Returns:
            XYData: Combined output with concatenated features.

        Note:
            This method assumes that all filter outputs can be concatenated along the last axis.
            Ensure that your filters produce compatible outputs.
        """
        return XYData.concat(
            [XYData.ensure_dim(output.value) for output in pipeline_outputs], axis=-1
        )

    def start(self, x: XYData, y: XYData | None, X_: XYData | None) -> XYData | None:
        """
        Start the pipeline execution by fitting the model and making predictions.

        This method initiates the pipeline process by fitting the model to the input data
        and then making predictions.

        Args:
            x (XYData): The input data for fitting and prediction.
            y (XYData | None): The target data for fitting, if available.
            X_ (XYData | None): Additional input data for prediction, if different from x.

        Returns:
            XYData | None: The predictions made by the pipeline, or None if an error occurs.

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
