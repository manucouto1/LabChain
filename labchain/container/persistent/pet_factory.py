# labchain/container/persistent/pet_factory.py

from typing import Dict, Type, cast
from labchain import BasePlugin
from labchain.base import BaseFactory, TypePlugable
from labchain.container.persistent.pet_class_manager import PetClassManager
from rich import print as rprint
import os


class PetFactory(BaseFactory[TypePlugable]):
    """
    Persistent factory with version control and lazy loading.

    This factory extends BaseFactory to provide automatic class persistence
    and retrieval from storage. It maintains local version tracking and
    can automatically fetch classes from storage when they're not in memory.

    Key Features:

        - Automatic hash computation on registration
        - Lazy loading from storage when class not in memory
        - Version-specific class retrieval for pipeline reconstruction
        - Bulk push operation for all registered classes
        - Integration with PetClassManager for storage operations

    Usage:
        ```python
        from labchain.storage import S3Storage
        from labchain.container.persistent import PetClassManager, PetFactory

        storage = S3Storage(bucket="my-models")
        manager = PetClassManager(storage)
        factory = PetFactory(manager)

        # Register classes
        factory["MyFilter"] = MyFilter
        factory["AnotherFilter"] = AnotherFilter

        # Push to storage
        factory.push_all()

        # Later, on another machine
        # Class auto-loads if not in memory
        cls = factory["MyFilter"]

        # Get specific version
        old_version = factory.get_version("MyFilter", "abc123...")
        ```

    Attributes:
        manager (PetClassManager): Manager for storage operations.
        version_control (Dict[str, str]): Maps class names to their current hashes.

    Methods:
        __setitem__(name: str, value: Type[TypePlugable]) -> None:
            Register class and compute its hash.

        __getitem__(name: str) -> Type[TypePlugable]:
            Get class, with automatic storage fallback.

        get(name: str, default: Optional[Type[TypePlugable]] = None) -> Optional[Type[TypePlugable]]:
            Get class with default fallback.

        get_version(name: str, code_hash: str) -> Type[TypePlugable]:
            Get specific version by hash.

        push_all() -> None:
            Push all registered classes to storage.

    Note:
        This factory is designed to work seamlessly with PetContainer
        and BasePlugin.build_from_dump() for full pipeline portability.
    """

    def __init__(self, class_manager: PetClassManager, fallback_factory: BaseFactory):
        """
        Initialize the PetFactory with a class manager.

        Args:
            class_manager (PetClassManager): Manager for storage operations.

        Example:
        ```python
            manager = PetClassManager(storage)
            factory = PetFactory(manager)
        ```
        """
        super().__init__()
        os.environ["TYPEGUARD_DISABLE"] = "1"
        self._manager = class_manager
        self._fallback_factory = fallback_factory
        self._version_control: Dict[str, str] = {}

    def __setitem__(self, name: str, value: Type[TypePlugable]) -> None:
        """
        Register class locally and compute its content hash.

        When a class is registered, this method:

            1. Computes the SHA-256 hash of the class source code
            2. Stores the hash in version_control for tracking
            3. Registers the class in the parent factory

        Args:
            name (str): Name to register the class under.
            value (Type[TypePlugable]): The class to register.

        Returns:
            None

        Example:
        ```python
            factory["MyFilter"] = MyFilter
            # Hash is automatically computed and tracked
        ```

        Note:
            The hash is computed immediately but the class is not pushed
            to storage until push_all() is called.
        """
        # code_hash = self._manager.get_class_hash(value)
        class_hash = cast(BasePlugin, value)._hash
        if class_hash is not None:
            self._version_control[name] = class_hash
            super().__setitem__(name, value)
        else:
            raise ValueError("Class must be a BasePlugin and should have a hash")

    def __getitem__(
        self, name: str, default: Type[TypePlugable] | None = None
    ) -> Type[TypePlugable]:
        """
        Get class with automatic lazy loading from storage.

        This method implements the following fallback chain:

            1. Try to get from local memory (parent factory)
            2. If not found, check storage for 'latest' version
            3. If found in storage, pull and register it
            4. If not found anywhere, raise AttributeError

        Args:
            name (str): Name of the class to retrieve.

        Returns:
            Type[TypePlugable]: The requested class.

        Raises:
            AttributeError: If class not found locally or in storage.

        Example:
        ```python
            # Class in memory - instant return
            cls = factory["MyFilter"]

            # Class not in memory - auto-pulls from storage
            cls = factory["RemoteFilter"]
        ```

        Note:
            Lazy loading is transparent - users don't need to explicitly
            call pull() before accessing classes.
        """
        try:
            return super().__getitem__(name)
        except AttributeError:
            # Not in memory, try to fetch 'latest' from storage
            remote_meta = self._manager._get_remote_latest_meta(name)

            if remote_meta:
                code_hash = remote_meta["hash"]
                rprint(
                    f"[yellow]Class {name} missing. Pulling latest: {code_hash[:8]}...[/yellow]"
                )

                # Pull and register dynamically
                recovered_cls: Type[TypePlugable] = self._manager.pull(name, code_hash)
                self._version_control[name] = code_hash
                super().__setitem__(name, recovered_cls)
                return recovered_cls

            if name in self._fallback_factory:
                return self._fallback_factory[name]

            raise AttributeError(
                f"Class '{name}' not found locally and no remote version exists."
            )

    def get(
        self, name: str, default: Type[TypePlugable] | None = None
    ) -> Type[TypePlugable] | None:
        """
        Get class with optional default, attempting lazy load from storage.

        This method overrides the parent's get() to add lazy loading capability.
        It provides a safe way to retrieve classes with a fallback value.

        Args:
            name (str): Name of the class to retrieve.
            default (Optional[Type[TypePlugable]]): Default value if not found.

        Returns:
            Optional[Type[TypePlugable]]: The class or default value.

        Example:
        ```python
            # With default
            cls = factory.get("MaybeExists", DefaultFilter)

            # Without default (returns None if not found)
            cls = factory.get("MaybeExists")
            if cls is None:
                print("Class not found")
        ```

        Note:
            Unlike __getitem__, this method returns the default value
            (or None) instead of raising an exception when the class
            is not found.
        """
        try:
            return self.__getitem__(name)
        except AttributeError:
            return default

    def get_version(self, name: str, code_hash: str) -> Type[TypePlugable]:
        """
        Get a specific version of a class by its hash.

        This method is essential for reconstructing old pipelines via
        BasePlugin.build_from_dump(). It allows loading exact versions
        of classes as they existed when a pipeline was saved.

        Args:
            name (str): Name of the class.
            code_hash (str): SHA-256 hash of the specific version.

        Returns:
            Type[TypePlugable]: The requested version of the class.

        Example:
        ```python
            # Save pipeline with current versions
            config = pipeline.item_dump()  # Includes version_hash

            # Later, reconstruct with exact versions
            old_filter = factory.get_version(
                "MyFilter",
                config["params"]["filter"]["version_hash"]
            )
        ```

        Note:
            This method does NOT register the fetched version in the
            factory's main registry, avoiding conflicts with the current
            'latest' version. It simply returns the requested class.
        """
        if name in self._foundry and self._version_control.get(name) == code_hash:
            return self._foundry[name]

        rprint(
            f"[cyan]Reconstructing specific version {code_hash[:8]}... for {name}[/cyan]"
        )
        recovered_cls: Type[TypePlugable] = self._manager.pull(name, code_hash)

        return recovered_cls

    def push_all(self) -> None:
        """
        Push all locally registered classes to storage.

        This method iterates through all classes in the factory and
        pushes each one to storage, updating their 'latest' pointers.
        This is the primary way to synchronize local classes with storage.

        Returns:
            None

        Example:
        ```python
            # Register multiple classes
            factory["Filter1"] = Filter1
            factory["Filter2"] = Filter2
            factory["Filter3"] = Filter3

            # Push all at once
            factory.push_all()
        ```

        Note:
            This operation is idempotent - pushing the same class
            multiple times without changes will not create duplicate
            storage entries (same hash = same storage path).
        """
        for name, cls in self._foundry.items():
            rprint(f"[blue]Publishing {name}...[/blue]")
            self._manager.push(cls)
