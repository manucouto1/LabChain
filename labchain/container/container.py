# labchain/container/container.py

from __future__ import annotations

from typing import Any, Callable, Type, Optional, TypeVar

from labchain.plugins.storage.local_storage import LocalStorage
from labchain.base import BaseFactory
from labchain.base import BaseFilter, BaseMetric, BasePlugin
from labchain.base import BaseStorage
from labchain.base import BasePipeline
from labchain.container.overload import fundispatch
from labchain.container.persistent.pet_class_manager import PetClassManager
from labchain.container.persistent.pet_factory import PetFactory


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
        - Optional persistent storage with version control (via persist=True)

    Usage:
        To use the Container, you can register components and then retrieve them as needed:

        ```python
        from labchain.container import Container
        from labchain.base import BaseFilter, BasePipeline

        @Container.bind()
        class MyFilter(BaseFilter):
            def fit(self, x, y):
                pass
            def predict(self, x):
                return x

        # With persistence enabled
        @Container.bind(persist=True)
        class MyPersistentFilter(BaseFilter):
            def predict(self, x):
                return x

        # Push persistent classes to storage
        Container.ppif.push_all()

        # Retrieving and using registered components
        filter_instance = Container.ff["MyFilter"]()
        persistent_filter = Container.ppif["MyPersistentFilter"]()
        ```

    Attributes:
        storage (BaseStorage): An instance of BaseStorage for handling storage operations.
        ds (BaseDatasetManager): An instance of BaseDatasetManager for managing datasets.
        ff (BaseFactory[BaseFilter]): Factory for creating and storing BaseFilter objects.
        pf (BaseFactory[BasePipeline]): Factory for creating and storing BasePipeline objects.
        mf (BaseFactory[BaseMetric]): Factory for creating and storing BaseMetric objects.
        sf (BaseFactory[BaseStorage]): Factory for creating and storing BaseStorage objects.
        pif (BaseFactory[BasePlugin]): Factory for creating and storing BasePlugin objects.
        pcm (PetClassManager): Manager for persistent class storage operations.
        ppif (PetFactory[BasePlugin]): Persistent factory with version control for all plugins.

    Methods:
        bind(manager: Optional[Any] = dict, wrapper: Optional[Any] = dict, persist: bool = False) -> Callable:
            A decorator for binding various components to the Container.

    Note:
        The Container class is designed to be used as a singleton, with all its methods
        and attributes being class-level (static) to ensure a single point of access
        for all components across the framework.
    """

    storage: BaseStorage = LocalStorage()
    ff: BaseFactory[BaseFilter] = BaseFactory[BaseFilter]()
    pf: BaseFactory[BasePipeline] = BaseFactory[BasePipeline]()
    mf: BaseFactory[BaseMetric] = BaseFactory[BaseMetric]()
    sf: BaseFactory[BaseStorage] = BaseFactory[BaseStorage]()
    pif: BaseFactory[BasePlugin] = BaseFactory[BasePlugin]()

    pcm: PetClassManager = PetClassManager(storage)
    ppif: PetFactory[BasePlugin] = PetFactory[BasePlugin](pcm, pif)

    @staticmethod
    def bind(
        manager: Optional[Any] = dict,
        wrapper: Optional[Any] = dict,
        persist: bool = False,
        auto_push: bool = False,
    ) -> Callable:
        """
        A decorator for binding various components to the Container.

        This method uses function dispatching to register different types of components
        (filters, pipelines, metrics, storage) with their respective factories in the Container.

        Args:
            manager (Optional[Any]): An optional manager for the binding process. Defaults to dict.
            wrapper (Optional[Any]): An optional wrapper for the binding process. Defaults to dict.
            persist (bool): If True, register class in persistent factory (ppif) for version control
                           and storage synchronization. Defaults to False.
            auto_push (bool): If True and persist=True, automatically push class to storage after
                             binding. Defaults to False.

        Returns:
            Callable: A decorator function that registers the decorated class with the appropriate factory.

        Raises:
            NotImplementedError: If no decorator is registered for the given function.
            RuntimeError: If persist=True but Container.storage is not configured.

        Example:
        ```python
            # Standard (non-persistent) binding
            @Container.bind()
            class MyCustomFilter(BaseFilter):
                def predict(self, x):
                    return x

            # Persistent binding with version control
            @Container.bind(persist=True)
            class MyPersistentFilter(BaseFilter):
                def predict(self, x):
                    return x

            # Persistent with automatic push to storage
            @Container.bind(persist=True, auto_push=True)
            class MyAutoPushFilter(BaseFilter):
                def predict(self, x):
                    return x

            # Access classes
            filter1 = Container.ff["MyCustomFilter"]()
            filter2 = Container.ppif["MyPersistentFilter"]()  # Also in ppif for persistence
        ```

        Note:
            This method uses the @fundispatch decorator to provide different implementations
            based on the type of the decorated class. It automatically registers the class
            with the appropriate factory based on its base class (BaseFilter, BasePipeline, etc.).

            When persist=True:
            - Class is registered in both standard factory AND ppif (persistent factory)
            - Class hash is automatically computed for version tracking
            - Class can be pushed to storage for remote access
            - Class can be reconstructed from storage on other machines
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

            if persist:
                cls_hash = Container.pcm.get_class_hash(func)
                func._hash = cls_hash
                Container.ppif[func.__name__] = func
                if auto_push:
                    Container.pcm.push(func)

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

            if persist:
                cls_hash = Container.pcm.get_class_hash(func)
                func._hash = cls_hash
                Container.ppif[func.__name__] = func
                if auto_push:
                    Container.pcm.push(func)

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

            if persist:
                cls_hash = Container.pcm.get_class_hash(func)
                func._hash = cls_hash
                Container.ppif[func.__name__] = func
                if auto_push:
                    Container.pcm.push(func)

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

            if persist:
                cls_hash = Container.pcm.get_class_hash(func)
                func._hash = cls_hash
                Container.ppif[func.__name__] = func
                if auto_push:
                    Container.pcm.push(func)

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

            if persist:
                cls_hash = Container.pcm.get_class_hash(func)
                func._hash = cls_hash
                Container.ppif[func.__name__] = func
                if auto_push:
                    Container.pcm.push(func)

            return func

        return inner
