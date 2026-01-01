from copy import deepcopy
from typing import Any, Dict, Optional, Sequence
from labchain.base import XYData, BaseFilter
from labchain.base import ParallelPipeline
from labchain.base.exceptions import NotTrainableFilterError
from labchain.container import Container
from labchain.utils.pyspark import PySparkMapReduce
from labchain.base.base_types import VData

import numpy as np

__all__ = ["HPCPipeline"]


@Container.bind()
class HPCPipeline(ParallelPipeline):
    """
    High Performance Computing Pipeline using MapReduce for parallel feature extraction.

    This pipeline applies a sequence of filters to the input data using a MapReduce approach,
    enabling parallel processing and potentially improved performance on large datasets.

    Key Features:
        - Parallel processing of filters using MapReduce
        - Scalable to large datasets
        - Configurable number of partitions for optimization

    Usage:
        ```python
        from framework3.plugins.pipelines.parallel import HPCPipeline
        from framework3.base import XYData

        filters = [Filter1(), Filter2(), Filter3()]
        pipeline = HPCPipeline(filters, app_name="MyApp", master="local[*]", numSlices=8)

        x_data = XYData(...)
        y_data = XYData(...)

        pipeline.fit(x_data, y_data)
        predictions = pipeline.predict(x_data)
        ```

    Attributes:
        filters (Sequence[BaseFilter]): A sequence of filters to be applied to the input data.
        numSlices (int): The number of partitions to use in the MapReduce process.
        app_name (str): The name of the Spark application.
        master (str): The Spark master URL.

    Methods:
        fit(x: XYData, y: Optional[XYData]): Fit the filters in parallel using MapReduce.
        predict(x: XYData) -> XYData: Make predictions using the fitted filters in parallel.
        evaluate(x_data: XYData, y_true: XYData | None, y_pred: XYData) -> Dict[str, Any]:
            Evaluate the pipeline using provided metrics.
    """

    def __init__(
        self,
        filters: Sequence[BaseFilter],
        app_name: str,
        master: str = "local",
        numSlices: int = 4,
    ):
        """
        Initialize the HPCPipeline.

        Args:
            filters (Sequence[BaseFilter]): A sequence of filters to be applied to the input data.
            app_name (str): The name of the Spark application.
            master (str, optional): The Spark master URL. Defaults to "local".
            numSlices (int, optional): The number of partitions to use in the MapReduce process. Defaults to 4.
        """
        super().__init__(filters=filters)
        self.filters = filters
        self.numSlices = numSlices
        self.app_name = app_name
        self.master = master

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
            return self.predict(x)
        except Exception as e:
            # Handle the exception appropriately
            raise e

    def fit(self, x: XYData, y: Optional[XYData]):
        """
        Fit the filters in the pipeline to the input data using MapReduce.

        This method applies the fit operation to all filters in parallel using MapReduce,
        allowing for efficient processing of large datasets.

        Args:
            x (XYData): The input data to fit the filters on.
            y (XYData | None, optional): The target data, if available.

        Note:
            This method updates the filters in place with their trained versions.
        """

        def fit_function(filter):
            try:
                filter.fit(deepcopy(x), y)
            except NotTrainableFilterError:
                print("Skipping not trainable filter:", filter.__class__.__name__)
            return filter

        spark = PySparkMapReduce(self.app_name, self.master)
        # Apply fit in parallel to the filters
        rdd = spark.map(self.filters, fit_function, numSlices=self.numSlices)
        # Update the filters with the trained versions
        self.filters = rdd.collect()
        spark.stop()

    def predict(self, x: XYData) -> XYData:
        """
        Make predictions using the fitted filters in parallel.

        This method applies the predict operation to all filters in parallel using MapReduce,
        then combines the results into a single prediction.

        Args:
            x (XYData): The input data to make predictions on.

        Returns:
            XYData: The combined predictions from all filters.

        Note:
            The predictions from each filter are stacked horizontally to form the final output.
        """

        def predict_function(filter: BaseFilter) -> VData:
            result: XYData = filter.predict(x)
            m_hash, _ = filter._get_model_key(x._hash)
            return XYData.ensure_dim(result.value)

        # Apply predict in parallel to the filters
        spark = PySparkMapReduce(self.app_name, self.master)
        spark.map(self.filters, predict_function, numSlices=self.numSlices)
        aux = spark.reduce(lambda x, y: np.hstack([x, y]))
        spark.stop()
        # Reduce the results
        return XYData.mock(aux)

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

    def log_metrics(self):
        """
        Log metrics for the pipeline.

        This method can be implemented to log any relevant metrics during the pipeline's execution.
        """
        # Implement metric logging if necessary
        pass

    def finish(self):
        """
        Finish the pipeline's execution.

        This method is called to perform any necessary cleanup or finalization steps.
        """
