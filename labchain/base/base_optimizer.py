from __future__ import annotations  # noqa: D100
from abc import abstractmethod
from typing import Optional

from labchain.base.base_clases import BaseFilter, XYData

__all__ = ["BaseOptimizer"]


class BaseOptimizer(BaseFilter):
    """
    Base class for optimizer components in the framework.

    This abstract class extends BaseFilter and defines the interface for optimizer operations.
    It provides a structure for implementing optimization strategies for pipelines.

    Key Features:
        - Abstract methods for starting optimization process and optimizing pipelines
        - Support for verbose output control

    Usage:
        To create a new optimizer type, inherit from this class and implement
        the required methods. For example:

        ```python
        class MyCustomOptimizer(BaseOptimizer):
            def __init__(self, optimization_params):
                super().__init__()
                self.optimization_params = optimization_params

            def start(self, x: XYData, y: Optional[XYData], X_: Optional[XYData]) -> Optional[XYData]:
                # Implement optimization start logic
                pass

            def optimize(self, pipeline: BaseFilter) -> None:
                # Implement pipeline optimization logic
                pass
        ```

    Attributes:
        pipeline (BaseFilter): The pipeline to be optimized.

    Methods:
        start(x: XYData, y: Optional[XYData], X_: Optional[XYData]) -> Optional[XYData]:
            Abstract method to start the optimization process.

        optimize(pipeline: BaseFilter) -> None:
            Abstract method to optimize the given pipeline.

        verbose(value: bool) -> None:
            Sets the verbosity level for the optimizer and its pipeline.

    Note:
        This is an abstract base class. Concrete implementations should override
        the start and optimize methods to provide specific optimization functionality.
    """

    @abstractmethod
    def start(
        self, x: XYData, y: Optional[XYData], X_: Optional[XYData]
    ) -> Optional[XYData]:
        """
        Start the optimization process.

        This abstract method should be implemented by subclasses to define
        the specific logic for initiating the optimization process.

        Args:
            x (XYData): The primary input data.
            y (Optional[XYData]): Optional target data.
            X_ (Optional[XYData]): Optional additional input data.

        Returns:
            Optional[XYData]: The processed data, if any.

        Raises:
            NotImplementedError: If the subclass does not implement this method.

        Example:
            ```python
            class MyOptimizer(BaseOptimizer):
                def start(self, x: XYData, y: Optional[XYData], X_: Optional[XYData]) -> Optional[XYData]:
                    # Optimization start logic here
                    return processed_data
            ```
        """
        ...

    @abstractmethod
    def optimize(self, pipeline: BaseFilter) -> None:
        """
        Optimize the given pipeline.

        This abstract method should be implemented by subclasses to define
        the specific logic for optimizing a pipeline.

        Args:
            pipeline (BaseFilter): The pipeline to be optimized.

        Returns:
            None

        Raises:
            NotImplementedError: If the subclass does not implement this method.

        Example:
            ```python
            class MyOptimizer(BaseOptimizer):
                def optimize(self, pipeline: BaseFilter) -> None:
                    # Pipeline optimization logic here
                    pass
            ```
        """
        ...

    def verbose(self, value: bool) -> None:
        """
        Set the verbosity of the optimizer and its pipeline.

        This method controls the verbosity of both the optimizer itself and its associated pipeline.

        Args:
            value (bool): If True, enables verbose output; if False, disables it.

        Returns:
            None

        Example:
            ```python
            optimizer = MyOptimizer()
            optimizer.verbose(True)  # Enable verbose output
            ```
        """
        self._verbose = value
        self.pipeline.verbose(value)
