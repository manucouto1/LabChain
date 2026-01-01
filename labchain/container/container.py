from __future__ import annotations

from typing import Any, Callable, Type, Optional, TypeVar

from labchain.base import BaseDatasetManager, BaseFactory
from labchain.base import BaseFilter, BaseMetric, BasePlugin
from labchain.base import BaseStorage
from labchain.base import BasePipeline
from labchain.container.overload import fundispatch


F = TypeVar("F", bound=type)

__all__ = ["Container"]


class Container:
    """
    A container class for managing various components of the framework.

    This class provides a centralized location for storing and managing different types of
    objects such as filters, pipelines, metrics, storage, and plugins. It uses factories
    to create and store these objects.

    Key Features:
        - Centralized management of framework components
        - Factory-based creation and storage of objects
        - Static binding method for easy registration of components
        - Support for multiple component types (filters, pipelines, metrics, storage, plugins)

    Usage:
        To use the Container, you can register components and then retrieve them as needed:

        ```python
        from framework3.container import Container
        from framework3.base import BaseFilter, BasePipeline

        @Container.bind()
        class MyFilter(BaseFilter):
            def fit(self, x, y):
                pass
            def predict(self, x):
                return x

        @Container.bind()
        class MyPipeline(BasePipeline):
            def fit(self, x, y):
                pass
            def predict(self, x):
                return x
            def init(self):
                pass
            def start(self, x, y, X_):
                return None
            def log_metrics(self):
                pass
            def finish(self):
                pass
            def evaluate(self, x_data, y_true, y_pred):
                return {}

        # Retrieving and using registered components
        filter_instance = Container.ff["MyFilter"]()
        pipeline_instance = Container.pf["MyPipeline"]()

        result = pipeline_instance.run(filter_instance.process("hello"))
        print(result)
        ```

    Attributes:
        storage (BaseStorage): An instance of BaseStorage for handling storage operations.
        ds (BaseDatasetManager): An instance of BaseDatasetManager for managing datasets.
        ff (BaseFactory[BaseFilter]): Factory for creating and storing BaseFilter objects.
        pf (BaseFactory[BasePipeline]): Factory for creating and storing BasePipeline objects.
        mf (BaseFactory[BaseMetric]): Factory for creating and storing BaseMetric objects.
        sf (BaseFactory[BaseStorage]): Factory for creating and storing BaseStorage objects.
        pif (BaseFactory[BasePlugin]): Factory for creating and storing BasePlugin objects.

    Methods:
        bind(manager: Optional[Any] = dict, wrapper: Optional[Any] = dict) -> Callable:
            A decorator for binding various components to the Container.

    Note:
        The Container class is designed to be used as a singleton, with all its methods
        and attributes being class-level (static) to ensure a single point of access
        for all components across the framework.
    """

    storage: BaseStorage
    ds: BaseDatasetManager
    ff: BaseFactory[BaseFilter] = BaseFactory[BaseFilter]()
    pf: BaseFactory[BasePipeline] = BaseFactory[BasePipeline]()
    mf: BaseFactory[BaseMetric] = BaseFactory[BaseMetric]()
    sf: BaseFactory[BaseStorage] = BaseFactory[BaseStorage]()
    pif: BaseFactory[BasePlugin] = BaseFactory[BasePlugin]()

    @staticmethod
    def bind(manager: Optional[Any] = dict, wrapper: Optional[Any] = dict) -> Callable:
        """
        A decorator for binding various components to the Container.

        This method uses function dispatching to register different types of components
        (filters, pipelines, metrics, storage) with their respective factories in the Container.

        Args:
            manager (Optional[Any]): An optional manager for the binding process. Defaults to dict.
            wrapper (Optional[Any]): An optional wrapper for the binding process. Defaults to dict.

        Returns:
            Callable: A decorator function that registers the decorated class with the appropriate factory.

        Raises:
            NotImplementedError: If no decorator is registered for the given function.

        Example:
            ```python
            @Container.bind()
            class MyCustomFilter(BaseFilter):
                def fit(self, x, y):
                    # Implementation
                    pass

                def predict(self, x):
                    # Implementation
                    return x

            # The MyCustomFilter class is now registered and can be accessed via Container.ff["MyCustomFilter"]
            ```

        Note:
            This method uses the @fundispatch decorator to provide different implementations
            based on the type of the decorated class. It automatically registers the class
            with the appropriate factory based on its base class (BaseFilter, BasePipeline, etc.).
        """

        @fundispatch  # type: ignore
        def inner(func: Any):
            """
            Default inner function for the bind decorator.

            This function is called when no specific registration is found for the decorated class.

            Args:
                func (Any): The class being decorated.

            Raises:
                NotImplementedError: Always raised to indicate that no suitable decorator was found.
            """
            raise NotImplementedError(f"No decorator registered for {func.__name__}")

        @inner.register(BaseFilter)  # type: ignore
        def _(func: Type[BaseFilter]) -> Type[BaseFilter]:
            """
            Register a BaseFilter class with the Container.

            This function is called when the decorated class is a subclass of BaseFilter.

            Args:
                func (Type[BaseFilter]): The BaseFilter subclass being decorated.

            Returns:
                Type[BaseFilter]: The decorated class, now registered with the Container.
            """
            Container.ff[func.__name__] = func
            Container.pif[func.__name__] = func
            return func

        @inner.register(BasePipeline)  # type: ignore
        def _(func: Type[BasePipeline]) -> Type[BasePipeline]:
            """
            Register a BasePipeline class with the Container.

            This function is called when the decorated class is a subclass of BasePipeline.

            Args:
                func (Type[BasePipeline]): The BasePipeline subclass being decorated.

            Returns:
                Type[BasePipeline]: The decorated class, now registered with the Container.
            """
            Container.pf[func.__name__] = func
            Container.pif[func.__name__] = func
            return func

        @inner.register(BaseMetric)  # type: ignore
        def _(func: Type[BaseMetric]) -> Type[BaseMetric]:
            """
            Register a BaseMetric class with the Container.

            This function is called when the decorated class is a subclass of BaseMetric.

            Args:
                func (Type[BaseMetric]): The BaseMetric subclass being decorated.

            Returns:
                Type[BaseMetric]: The decorated class, now registered with the Container.
            """
            Container.mf[func.__name__] = func
            Container.pif[func.__name__] = func
            return func

        @inner.register(BaseStorage)  # type: ignore
        def _(func: Type[BaseStorage]) -> Type[BaseStorage]:
            """
            Register a BaseStorage class with the Container.

            This function is called when the decorated class is a subclass of BaseStorage.

            Args:
                func (Type[BaseStorage]): The BaseStorage subclass being decorated.

            Returns:
                Type[BaseStorage]: The decorated class, now registered with the Container.
            """
            Container.sf[func.__name__] = func
            Container.pif[func.__name__] = func
            return func

        @inner.register(BasePlugin)  # type: ignore
        def _(func: Type[BasePlugin]) -> Type[BasePlugin]:
            """
            Register a BasePlugin class with the Container.

            This function is called when the decorated class is a subclass of BasePlugin.

            Args:
                func (Type[BasePlugin]): The BasePlugin subclass being decorated.

            Returns:
                Type[BasePlugin]: The decorated class, now registered with the Container.
            """
            Container.pif[func.__name__] = func
            return func

        return inner
