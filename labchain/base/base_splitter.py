from __future__ import annotations  # noqa: D100
from abc import abstractmethod
from typing import Any, Dict, Optional

from labchain.base.base_clases import BaseFilter, XYData
from labchain.base.base_optimizer import BaseOptimizer

__all__ = ["BaseSplitter"]


class BaseSplitter(BaseFilter):
    """
    Base class for splitter components in the framework.

    This abstract class extends BaseFilter and defines the interface for splitter operations.
    It provides a structure for implementing data splitting strategies for pipelines.

    Key Features:
        - Abstract methods for starting splitting process, logging metrics, and splitting pipelines
        - Support for verbose output control
        - Integration with optimizer components

    Usage:
        To create a new splitter type, inherit from this class and implement
        the required methods. For example:

        ```python
        class MyCustomSplitter(BaseSplitter):
            def __init__(self, splitting_params):
                super().__init__()
                self.splitting_params = splitting_params

            def start(self, x: XYData, y: Optional[XYData], X_: Optional[XYData]) -> Optional[XYData]:
                # Implement splitting start logic
                pass

            def split(self, pipeline: BaseFilter) -> None:
                # Implement pipeline splitting logic
                pass

            def log_metrics(self) -> None:
                # Implement metric logging logic
                pass

            def evaluate(self, x_data: XYData, y_true: XYData | None, y_pred: XYData) -> Dict[str, Any]:
                # Implement evaluation logic
                pass
        ```

    Attributes:
        pipeline (BaseFilter): The pipeline to be split.

    Methods:
        start(x: XYData, y: Optional[XYData], X_: Optional[XYData]) -> Optional[XYData]:
            Abstract method to start the splitting process.

        log_metrics() -> None:
            Abstract method to log the metrics of the pipeline.

        split(pipeline: BaseFilter) -> None:
            Abstract method to split the given pipeline.

        verbose(value: bool) -> None:
            Sets the verbosity level for the splitter and its pipeline.

        evaluate(x_data: XYData, y_true: XYData | None, y_pred: XYData) -> Dict[str, Any]:
            Abstract method to evaluate the metric based on the provided data.

        optimizer(optimizer: BaseOptimizer) -> BaseOptimizer:
            Applies an optimizer to the splitter.

        unwrap() -> BaseFilter:
            Returns the underlying pipeline.

    Note:
        This is an abstract base class. Concrete implementations should override
        the abstract methods to provide specific splitting functionality.
    """

    @abstractmethod
    def start(
        self, x: XYData, y: Optional[XYData], X_: Optional[XYData]
    ) -> Optional[XYData]:
        """
        Start the splitting process.

        This abstract method should be implemented by subclasses to define
        the specific logic for initiating the splitting process.

        Args:
            x (XYData): The primary input data.
            y (Optional[XYData]): Optional target data.
            X_ (Optional[XYData]): Optional additional input data.

        Returns:
            Optional[XYData]: The processed data, if any.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        ...

    @abstractmethod
    def split(self, pipeline: BaseFilter) -> None:
        """
        Split the given pipeline.

        This abstract method should be implemented by subclasses to define
        the specific logic for splitting a pipeline.

        Args:
            pipeline (BaseFilter): The pipeline to be split.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        ...

    def verbose(self, value: bool) -> None:
        """
        Set the verbosity of the splitter and its pipeline.

        This method controls the verbosity of both the splitter itself and its associated pipeline.

        Args:
            value (bool): If True, enables verbose output; if False, disables it.
        """
        self._verbose = value
        self.pipeline.verbose(value)

    @abstractmethod
    def evaluate(
        self, x_data: XYData, y_true: XYData | None, y_pred: XYData
    ) -> Dict[str, Any]:
        """
        Evaluate the metric based on the provided data.

        This abstract method should be implemented by subclasses to calculate the specific metric.

        Args:
            x_data (XYData): The input data used for the prediction.
            y_true (XYData | None): The ground truth or actual values.
            y_pred (XYData): The predicted values.

        Returns:
            Dict[str, Any]: A dictionary containing the calculated metric values.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        ...

    def _pre_fit_wrapp(
        self, x: XYData, y: Optional[XYData] = None
    ) -> float | None | dict:
        """
        Wrapper method for pre-fitting.

        This method calls the original fit method of the pipeline.

        Args:
            x (XYData): The input data for fitting.
            y (Optional[XYData]): The target data for fitting, if applicable.

        Returns:
            float | None: The result of the original fit method.
        """
        return self._original_fit(x, y)

    def _pre_predict_wrapp(self, x: XYData) -> XYData:
        """
        Wrapper method for pre-prediction.

        This method calls the original predict method of the pipeline.

        Args:
            x (XYData): The input data for prediction.

        Returns:
            XYData: The result of the original predict method.
        """
        return self._original_predict(x)

    def optimizer(self, optimizer: BaseOptimizer) -> BaseOptimizer:
        """
        Apply an optimizer to the splitter.

        This method allows an optimizer to be applied to the entire splitter.

        Args:
            optimizer (BaseOptimizer): The optimizer to apply to the splitter.

        Returns:
            BaseOptimizer: The optimizer after optimization.
        """
        optimizer.optimize(self)
        return optimizer

    def unwrap(self) -> BaseFilter:
        """
        Unwrap the splitter to get the underlying pipeline.

        This method returns the pipeline that the splitter is operating on.

        Returns:
            BaseFilter: The underlying pipeline.
        """
        return self.pipeline
