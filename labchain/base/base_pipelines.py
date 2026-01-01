from __future__ import annotations  # noqa: D100
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Type
from labchain.base.base_clases import BaseFilter
from labchain.base.base_optimizer import BaseOptimizer
from labchain.base.base_splitter import BaseSplitter
from labchain.base.base_types import XYData


__all__ = ["BasePipeline", "SequentialPipeline", "ParallelPipeline"]


class BasePipeline(BaseFilter):
    """
    Base class for pipeline structures in the framework.

    This abstract class extends BaseFilter and defines the interface for pipeline operations.
    It provides a structure for implementing complex data flows and combinations of filters.

    Key Features:
        - Abstract methods for starting pipeline processing and evaluation
        - Support for verbose output control
        - Methods for initializing filters, getting filter types, and applying optimizers and splitters
        - Access to inner filters of the pipeline

    Usage:
        To create a new pipeline type, inherit from this class and implement
        the required methods. For example:

        ```python
        class MyCustomPipeline(BasePipeline):
            def __init__(self, filters: List[BaseFilter]):
                super().__init__(filters=filters)

            def start(self, x: XYData, y: Optional[XYData], X_: Optional[XYData]) -> Optional[XYData]:
                # Implement pipeline start logic
                pass

            def evaluate(self, x_data: XYData, y_true: XYData | None, y_pred: XYData) -> Dict[str, Any]:
                # Implement evaluation logic
                pass
        ```

    Attributes:
        filters (List[BaseFilter]): List of filters in the pipeline.

    Methods:
        start(x: XYData, y: Optional[XYData], X_: Optional[XYData]) -> Optional[XYData]:
            Abstract method to start the pipeline processing.

        evaluate(x_data: XYData, y_true: XYData | None, y_pred: XYData) -> Dict[str, Any]:
            Abstract method to evaluate the pipeline's performance.

        verbose(value: bool) -> None:
            Sets the verbosity level for the pipeline and its filters.

        init(*args, **kwargs) -> None:
            Initializes the pipeline and its filters.

        get_types() -> List[Type[BaseFilter]]:
            Returns the types of filters in the pipeline.

        optimizer(optimizer: BaseOptimizer) -> BaseOptimizer:
            Applies an optimizer to the pipeline.

        splitter(splitter: BaseSplitter) -> BaseSplitter:
            Applies a splitter to the pipeline.

        inner() -> BaseFilter | List[BaseFilter] | None:
            Returns the inner filters of the pipeline.

    Note:
        This is an abstract base class. Concrete implementations should override
        the start and evaluate methods to provide specific pipeline functionality.
    """

    @abstractmethod
    def start(
        self, x: XYData, y: Optional[XYData], X_: Optional[XYData]
    ) -> Optional[XYData]:
        """
        Start the pipeline processing.

        This abstract method should be implemented by subclasses to define
        the specific logic for initiating the pipeline's data processing.

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
    def evaluate(
        self, x_data: XYData, y_true: XYData | None, y_pred: XYData
    ) -> Dict[str, Any]:
        """
        Evaluate the pipeline's performance.

        This abstract method should be implemented by subclasses to define
        the specific logic for evaluating the pipeline's output.

        Args:
            x_data (XYData): The input data used for prediction.
            y_true (XYData | None): The ground truth or actual values.
            y_pred (XYData): The predicted values.

        Returns:
            Dict[str, Any]: A dictionary containing evaluation metrics.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        ...

    def verbose(self, value: bool) -> None:
        """
        Set the verbosity of the pipeline and its filters.

        This method controls the verbosity of both the pipeline itself and all its constituent filters.

        Args:
            value (bool): If True, enables verbose output; if False, disables it.

        Returns:
            None
        """
        self._verbose = value
        for filter in self.filters:
            filter.verbose(value)

    # def init(self, *args: List[Any], **kwargs: Dict[str, Any]):
    #     """
    #     Initialize the pipeline and its filters.

    #     This method initializes both the pipeline itself and all its constituent filters.

    #     Args:
    #         *args (List[Any]): Variable length argument list.
    #         **kwargs (Dict[str,Any]): Arbitrary keyword arguments.

    #     """
    #     super().init(*args, **kwargs)
    #     for filter in self.filters:
    #         filter.init(*args, **kwargs)

    def get_types(self) -> List[Type[BaseFilter]]:
        """
        Get the types of filters in the pipeline.

        This method returns a list of the types of all filters contained in the pipeline.

        Returns:
            List[Type[BaseFilter]]: A list of filter types in the pipeline.
        """
        return list(map(lambda obj: type(obj), self.filters))

    def optimizer(self, optimizer: BaseOptimizer) -> BaseOptimizer:
        """
        Apply an optimizer to the pipeline.

        This method allows an optimizer to be applied to the entire pipeline.

        Args:
            optimizer (BaseOptimizer): The optimizer to apply to the pipeline.

        Returns:
            BaseOptimizer: The optimizer after optimization.
        """
        optimizer.optimize(self)
        return optimizer

    def splitter(self, splitter: BaseSplitter) -> BaseSplitter:
        """
        Apply a splitter to the pipeline.

        This method allows a splitter to be applied to the entire pipeline.

        Args:
            splitter (BaseSplitter): The splitter to apply to the pipeline.

        Returns:
            BaseSplitter: The splitter after splitting.
        """
        splitter.split(self)
        return splitter

    def inner(self) -> BaseFilter | List[BaseFilter] | None:
        """
        Get the inner filters of the pipeline.

        This method returns the list of filters contained within the pipeline.

        Returns:
            BaseFilter | List[BaseFilter] | None: The inner filters of the pipeline.
        """
        return self.filters


class SequentialPipeline(BasePipeline):
    """
    A pipeline that processes filters sequentially.

    This class implements a pipeline where each filter is applied in sequence,
    with the output of one filter becoming the input of the next.

    Key Features:
        - Sequential processing of filters
        - Implements start method for initiating the pipeline
        - Supports both fit and predict operations

    Usage:
        ```python
        from framework3.base import SequentialPipeline, XYData
        from framework3.plugins.filters import StandardScaler, PCA, LogisticRegression

        pipeline = SequentialPipeline([
            StandardScaler(),
            PCA(n_components=5),
            LogisticRegression()
        ])

        X_train = XYData.mock(np.random.rand(100, 10))
        y_train = XYData.mock(np.random.randint(0, 2, 100))

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_train)
        ```

    Methods:
        start(x: XYData, y: Optional[XYData], X_: Optional[XYData]) -> Optional[XYData]:
            Starts the sequential processing of filters in the pipeline.
        _pre_fit(x: XYData, y: Optional[XYData]) -> Tuple[str, str, str]:
            Prepares the pipeline for fitting by initializing model attributes and pre-fitting filters.
        _pre_predict(x: XYData) -> XYData:
            Prepares the pipeline for prediction by applying pre-predict operations on all filters.

    Note:
        This class extends BasePipeline and provides a concrete implementation
        for sequential processing of filters.
    """

    def _pre_fit(self, x: XYData, y: Optional[XYData]):
        """
        Prepare the pipeline for fitting.

        This method initializes model attributes (hash, path, and string representation)
        and performs pre-fit operations on all filters in the pipeline.

        Args:
            x (XYData): The input data for fitting.
            y (Optional[XYData]): The target data for fitting, if applicable.

        Returns:
            Tuple[str, str, str]: A tuple containing the model hash, path, and string representation.

        Note:
            This method is called internally before the actual fit operation and
            should not be called directly by users.
        """
        m_hash, m_str = self._get_model_key(
            data_hash=f'{x._hash}, {y._hash if y is not None else ""}'
        )
        m_path = f"{self._get_model_name()}/{m_hash}"

        self._m_hash = m_hash
        self._m_path = m_path
        self._m_str = m_str

        new_x = x

        for filter in self.filters:
            if filter._original_fit.__func__ is not BaseFilter.fit:
                filter._pre_fit(new_x, y)
                new_x = filter._pre_predict(new_x)

        return m_hash, m_path, m_str

    def _pre_predict(self, x: XYData):
        """
        Prepare the pipeline for prediction.

        This method checks if the pipeline has been properly fitted and then
        applies the pre-predict operation on all filters in the pipeline sequentially.

        Args:
            x (XYData): The input data for prediction.

        Returns:
            XYData: The transformed input data after applying all filters' pre-predict operations.

        Raises:
            ValueError: If the pipeline model has not been trained or loaded.

        Note:
            This method is called internally before the actual predict operation and
            should not be called directly by users.
        """
        if not self._m_hash or not self._m_path or not self._m_str:
            raise ValueError("Cached filter model not trained or loaded")

        aux_x = x
        for filter in self.filters:
            aux_x = filter._pre_predict(aux_x)
        return aux_x


class ParallelPipeline(BasePipeline):
    """
    A pipeline that processes filters in parallel.

    This class implements a pipeline where filters can be applied concurrently,
    potentially improving performance for certain types of operations.

    Note:
        The implementation details for this class are not provided in the given code snippet.
        It is expected that concrete implementations will define the specific behavior
        for parallel processing of filters.
    """

    ...
