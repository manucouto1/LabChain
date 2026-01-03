from abc import ABC, abstractmethod
from typing import Any, Callable


class MapReduceStrategy(ABC):
    """
    Abstract base class for implementing Map-Reduce strategies.

    This class defines the interface for Map-Reduce operations, providing a structure
    for implementing various distributed computing strategies.

    Key Features:
        - Abstract methods for map and reduce operations
        - Support for custom map and reduce functions
        - Method to stop the Map-Reduce process

    Usage:
        To create a new Map-Reduce strategy, inherit from this class and implement
        all abstract methods. For example:

        ```python
        class SimpleMapReduce(MapReduceStrategy):
            def __init__(self):
                self.intermediate = []
                self.result = None

            def map(self, data: list, map_function: Callable) -> None:
                self.intermediate = [map_function(item) for item in data]

            def reduce(self, reduce_function: Callable) -> Any:
                self.result = reduce_function(self.intermediate)
                return self.result

            def stop(self) -> None:
                self.intermediate = []
                self.result = None

        # Usage
        mr = SimpleMapReduce()
        data = [1, 2, 3, 4, 5]
        mr.map(data, lambda x: x * 2)
        result = mr.reduce(sum)
        print(result)  # Output: 30
        mr.stop()
        ```

    Methods:
        map(data: Any, map_function: Callable) -> Any:
            Abstract method to perform the map operation.
        reduce(reduce_function: Callable) -> Any:
            Abstract method to perform the reduce operation.
        stop() -> None:
            Abstract method to stop the Map-Reduce process.

    Note:
        This is an abstract base class. Concrete implementations should override
        all abstract methods to provide specific Map-Reduce functionality.
    """

    @abstractmethod
    def map(self, data: Any, map_function: Callable) -> Any:
        """
        Perform the map operation on the input data.

        This method should be implemented to apply the map_function to each element
        of the input data, potentially in a distributed manner.

        Args:
            data (Any): The input data to be processed.
            map_function (Callable): The function to be applied to each data element.

        Returns:
            Any: The result of the map operation, which could be a collection of
                 intermediate key-value pairs or any other suitable format.

        Raises:
            NotImplementedError: If the subclass does not implement this method.

        Example:
            ```python
            def map(self, data: list, map_function: Callable) -> list:
                return [map_function(item) for item in data]
            ```
        """
        pass

    @abstractmethod
    def reduce(self, reduce_function: Callable) -> Any:
        """
        Perform the reduce operation on the mapped data.

        This method should be implemented to apply the reduce_function to the
        intermediate results produced by the map operation.

        Args:
            reduce_function (Callable): The function to be used for reducing the mapped data.

        Returns:
            Any: The final result of the Map-Reduce operation.

        Raises:
            NotImplementedError: If the subclass does not implement this method.

        Example:
            ```python
            def reduce(self, reduce_function: Callable) -> Any:
                return reduce_function(self.intermediate_results)
            ```
        """
        pass

    @abstractmethod
    def stop(self) -> None:
        """
        Stop the Map-Reduce process and perform any necessary cleanup.

        This method should be implemented to halt the Map-Reduce operation,
        release resources, and reset the state if needed.

        Raises:
            NotImplementedError: If the subclass does not implement this method.

        Example:
            ```python
            def stop(self) -> None:
                self.intermediate_results = []
                self.final_result = None
                # Additional cleanup code...
            ```
        """
        ...
