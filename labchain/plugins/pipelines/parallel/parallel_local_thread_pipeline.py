from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from typing import Any, Dict, Optional, Sequence, Tuple
from tqdm import tqdm
import numpy as np

from labchain.base import XYData, BaseFilter, ParallelPipeline
from labchain.base.exceptions import NotTrainableFilterError
from labchain.container import Container
from labchain.base.base_types import VData

__all__ = ["LocalThreadPipeline"]


@Container.bind()
class LocalThreadPipeline(ParallelPipeline):
    """
    A pipeline that runs filters in parallel using local multithreading.

    This pipeline is a lightweight version of HPCPipeline, using ThreadPoolExecutor
    for parallel processing instead of a distributed Spark cluster. It allows for
    efficient execution of multiple filters on a single machine with multiple cores.

    Key Features:
        - Parallel execution of filters using local threads
        - Progress bars for fit and predict operations
        - Configurable number of threads for optimization

    Usage:
        ```python
        from framework3.plugins.pipelines.parallel import LocalThreadPipeline
        from framework3.base import XYData

        filters = [Filter1(), Filter2(), Filter3()]
        pipeline = LocalThreadPipeline(filters, num_threads=4)

        x_data = XYData(...)
        y_data = XYData(...)

        pipeline.fit(x_data, y_data)
        predictions = pipeline.predict(x_data)
        ```

    Attributes:
        filters (Sequence[BaseFilter]): A sequence of filters to be applied to the input data.
        num_threads (int): The number of threads to use for parallel processing.

    Methods:
        fit(x: XYData, y: Optional[XYData]) -> Optional[float]: Fit the filters in parallel.
        predict(x: XYData) -> XYData: Make predictions using the fitted filters in parallel.
        evaluate(x_data: XYData, y_true: XYData | None, y_pred: XYData) -> Dict[str, Any]:
            Evaluate the pipeline using provided metrics.
    """

    def __init__(self, filters: Sequence[BaseFilter], num_threads: int = 4):
        """
        Initialize the LocalThreadPipeline.

        Args:
            filters (Sequence[BaseFilter]): A sequence of filters to be applied to the input data.
            num_threads (int, optional): The number of threads to use for parallel processing. Defaults to 4.
        """
        super().__init__(filters=filters)
        self.filters = filters
        self.num_threads = num_threads

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
            raise e

    def fit(self, x: XYData, y: Optional[XYData]) -> Optional[float]:
        """
        Fit the filters in the pipeline to the input data using parallel threads.

        This method applies the fit operation to all filters in parallel using ThreadPoolExecutor,
        allowing for efficient processing on multi-core machines.

        Args:
            x (XYData): The input data to fit the filters on.
            y (XYData | None): The target data, if available.

        Returns:
            Optional[float]: The mean of the losses returned by the filters, if any.

        Note:
            This method updates the filters in place with their trained versions.
        """

        def fit_function(filt: BaseFilter) -> Tuple[BaseFilter, Optional[float | dict]]:
            loss = None
            try:
                loss = filt.fit(deepcopy(x), y)
            except NotTrainableFilterError:
                print("Skipping not trainable filter:", filt.__class__.__name__)
            return filt, loss

        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            self.filters, losses = list(
                zip(
                    *tqdm(
                        executor.map(fit_function, self.filters),
                        total=len(self.filters),
                        desc="Parallel Fitting",
                    )
                )
            )
            match list(filter(lambda x: x is not None, losses)):
                case []:
                    return None
                case lss:
                    return float(np.mean(lss))

    def predict(self, x: XYData) -> XYData:
        """
        Make predictions using the fitted filters in parallel.

        This method applies the predict operation to all filters in parallel using ThreadPoolExecutor,
        then combines the results into a single prediction.

        Args:
            x (XYData): The input data to make predictions on.

        Returns:
            XYData: The combined predictions from all filters.

        Note:
            The predictions from each filter are concatenated to form the final output.
        """

        def predict_function(filt: BaseFilter) -> VData:
            result: XYData = filt.predict(x)
            return XYData.ensure_dim(result.value)

        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            results = list(
                tqdm(
                    executor.map(predict_function, self.filters),
                    total=len(self.filters),
                    desc="Predicting",
                )
            )

        combined = XYData.concat(results)
        return combined

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
