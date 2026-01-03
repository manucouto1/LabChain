# labchain/container/persistent/pet_class_manager.py

import hashlib
import inspect
import sys
import types
import cloudpickle
from typing import Type, Optional, Dict, cast

from labchain import BasePlugin
from labchain.base import BaseStorage, TypePlugable
from rich import print as rprint
import json


class PetClassManager:
    """
    Manager for persistent class storage with version tracking.

    This class handles the serialization, storage, and retrieval of plugin classes
    using cloudpickle and a hash-based versioning system. Each class version is
    stored immutably by its content hash, with a 'latest' pointer for convenience.

    Key Features:

        - Hash-based version tracking using SHA-256 of source code
        - Immutable storage of class versions
        - 'latest' pointer management for development workflow
        - Status checking (synced, untracked, out_of_sync)
        - Push/pull operations for class synchronization

    Storage Structure:
    ```
        plugins/
        ├── ClassName/
        │   ├── abc123...pkl  # Immutable class binary by hash
        │   ├── def456...pkl  # Another version
        │   └── latest.json   # Pointer to current version
        └── AnotherClass/
            └── ...
    ```

    Usage:
    ```python
        from labchain.storage import LocalStorage
        from labchain.container.persistent import PetClassManager

        storage = LocalStorage("./storage")
        manager = PetClassManager(storage)

        # Check status
        status = manager.check_status(MyFilter)

        # Push class
        manager.push(MyFilter)

        # Pull latest
        cls = manager.pull("MyFilter")

        # Pull specific version
        cls = manager.pull("MyFilter", code_hash="abc123...")
    ```

    Attributes:
        storage (BaseStorage): Storage backend for class persistence.

    Methods:
        _get_remote_latest_meta(class_name: str) -> Optional[Dict[str, str]]:
            Retrieve the 'latest.json' manifest from storage.

        check_status(class_obj: Type[TypePlugable]) -> str:
            Compare local vs remote versions.

        get_class_hash(class_obj: Type[TypePlugable]) -> str:
            Generate SHA-256 hash of class source code.

        persist_class(class_obj: Type[TypePlugable]) -> str:
            Serialize and upload class if it doesn't exist.

        push(class_obj: Type[TypePlugable]) -> None:
            Push local version and mark as 'latest' in storage.

        pull(class_name: str, code_hash: Optional[str] = None) -> Type[TypePlugable]:
            Fetch a specific version or 'latest' from storage.

        recover_class(class_name: str, code_hash: str) -> Type[TypePlugable]:
            Download and reconstruct class from its hash path.

    Note:
        This class is designed to work with the PetFactory and PetContainer
        to provide seamless class persistence across different environments.
    """

    def __init__(self, storage: BaseStorage):
        """
        Initialize the PetClassManager with a storage backend.

        Args:
            storage (BaseStorage): Storage backend to use for class persistence.

        Example:
        ```python
            from labchain.storage import S3Storage

            storage = S3Storage(bucket="my-ml-models")
            manager = PetClassManager(storage)
        ```
        """
        self.storage = storage

    def _get_remote_latest_meta(self, class_name: str) -> Optional[Dict[str, str]]:
        """
        Retrieve the 'latest.json' manifest from storage for a specific class.

        This internal method fetches the metadata that points to the current
        'latest' version of a class in storage.

        Args:
            class_name (str): Name of the class to look up.

        Returns:
            Optional[Dict[str, str]]: Dictionary with 'hash' and 'class_name' keys,
                                      or None if not found or error occurs.

        Note:
            This method handles different return types from storage.download_file
            (bytes, str, or dict) and normalizes them to a dict.
        """
        # Path WITHOUT "plugins/" prefix since context="plugins" adds it
        path = f"{class_name}/latest.json"

        if self.storage.check_if_exists(path, context="plugins"):
            try:
                data = self.storage.download_file(path, context="plugins")

                if isinstance(data, bytes):
                    return json.loads(data.decode("utf-8"))
                elif isinstance(data, str):
                    return json.loads(data)
                return data

            except Exception as e:
                rprint(
                    f"[red]Error reading remote metadata for {class_name}: {e}[/red]"
                )
                return None

        return None

    def check_status(self, class_obj: Type[TypePlugable]) -> str:
        """
        Compare local class version vs remote version.

        This method computes the hash of the local class and compares it
        with the hash stored in the remote 'latest' pointer.

        Args:
            class_obj (Type[TypePlugable]): The class to check.

        Returns:

            str: Status string - one of:

                - 'synced': Local and remote hashes match
                - 'out_of_sync': Local and remote hashes differ
                - 'untracked': No remote version exists

        Example:
        ```python
            status = manager.check_status(MyFilter)
            if status == 'out_of_sync':
                manager.push(MyFilter)
        ```
        """
        local_hash = self.get_class_hash(class_obj)
        remote_meta = self._get_remote_latest_meta(class_obj.__name__)

        if not remote_meta:
            return "untracked"
        if local_hash == remote_meta["hash"]:
            return "synced"

        return "out_of_sync"

    def get_class_hash(self, class_obj: Type[TypePlugable]) -> str:
        """
        Generate SHA-256 hash based on class source code.

        This method extracts the source code of the class using inspect.getsource
        and computes a SHA-256 hash. For built-in or dynamically created classes
        without accessible source, it falls back to hashing the module and qualified name.

        Args:
            class_obj (Type[TypePlugable]): The class to hash.

        Returns:
            str: Hex digest of the SHA-256 hash (64 characters).

        Example:
        ```python
            hash1 = manager.get_class_hash(MyFilter)
            # Modify MyFilter source code
            hash2 = manager.get_class_hash(MyFilter)
            assert hash1 != hash2  # Hashes differ after modification
        ```

        Note:
            The hash is deterministic - the same source code will always
            produce the same hash, enabling reliable version tracking.
        """
        try:
            h = hashlib.sha256()

            h.update(class_obj.__module__.encode())
            h.update(class_obj.__qualname__.encode())

            # 2. Base classes (order matters)
            for base in class_obj.__bases__:
                h.update(base.__module__.encode())
                h.update(base.__qualname__.encode())

            # 3. Methods defined in this class only
            for name, obj in sorted(class_obj.__dict__.items()):
                if isinstance(obj, types.FunctionType):
                    code = obj.__code__

                    h.update(name.encode())

                    # Core bytecode
                    h.update(code.co_code)

                    # Constants and names affect semantics
                    h.update(repr(code.co_consts).encode())
                    h.update(repr(code.co_names).encode())

                    # Signature (API-level change)
                    sig = inspect.signature(obj)
                    h.update(str(sig).encode())
            return h.hexdigest()

        except (TypeError, OSError):
            identifier = f"{class_obj.__module__}.{class_obj.__qualname__}"
            return hashlib.sha256(identifier.encode("utf-8")).hexdigest()

    def persist_class(self, class_obj: Type[TypePlugable]) -> str:
        """
        Serialize the class and upload to storage if it doesn't exist.

        This method computes the class hash, serializes it with cloudpickle,
        and uploads it to storage only if a class with that hash doesn't
        already exist (avoiding redundant uploads).

        Args:
            class_obj (Type[TypePlugable]): The class to persist.

        Returns:
            str: Hash of the persisted class.

        Example:
        ```python
            hash_value = manager.persist_class(MyFilter)
            print(f"Persisted MyFilter with hash: {hash_value}")
        ```

        Note:
            This method does not update the 'latest' pointer. Use push()
            for the full push workflow including pointer update.
        """

        code_hash = cast(BasePlugin, class_obj)._hash
        if code_hash is not None:
            path = f"{class_obj.__name__}/{code_hash}.pkl"

            if not self.storage.check_if_exists(path, context="plugins"):
                binary = cloudpickle.dumps(class_obj)
                self.storage.upload_file(binary, file_name=path, context="plugins")
            return code_hash
        else:
            raise ValueError("Class must have a hash attribute.")

    def push(self, class_obj: Type[TypePlugable]) -> None:
        """
        Push local version and mark it as 'latest' in storage.

        This method performs a complete push workflow:

        1. Computes the class hash
        2. Serializes and uploads the class binary (if not already present)
        3. Updates the 'latest.json' pointer to reference this version

        Args:
            class_obj (Type[TypePlugable]): The class to push.

        Returns:
            None

        Example:
        ```python
            # After modifying MyFilter locally
            manager.push(MyFilter)
            # Now remote 'latest' points to the new version
        ```

        Note:
            Pushing creates immutable snapshots. Old versions remain accessible
            by their hash, enabling rollback and version-specific reconstruction.
        """
        code_hash = cast(BasePlugin, class_obj)._hash
        class_name = class_obj.__name__

        # Path WITHOUT "plugins/" prefix since context="plugins" adds it
        path = f"{class_name}/{code_hash}.pkl"
        if not self.storage.check_if_exists(path, context="plugins"):
            self.storage.upload_file(class_obj, file_name=path, context="plugins")

        # 2. Update 'latest' development pointer
        manifest = {"hash": code_hash, "class_name": class_name}
        self.storage.upload_file(
            manifest,
            file_name=f"{class_name}/latest.json",
            context="plugins",
        )

    def pull(
        self, class_name: str, code_hash: Optional[str] = None
    ) -> Type[TypePlugable]:  # type: ignore
        """
        Fetch a specific version or 'latest' from storage.

        This method retrieves a class from storage. If no specific hash is
        provided, it follows the 'latest' pointer to get the current version.

        Args:
            class_name (str): Name of the class to fetch.
            code_hash (Optional[str]): Specific hash to fetch. If None, fetches 'latest'.

        Returns:
            Type[TypePlugable]: The reconstructed class object.

        Raises:
            ValueError: If no remote versions exist for the class.

        Example:
        ```python
            # Pull latest version
            MyFilter = manager.pull("MyFilter")

            # Pull specific version
            MyFilterV1 = manager.pull("MyFilter", code_hash="abc123...")
        ```

        Note:
            The returned class is fully functional and can be instantiated
            immediately. All methods and attributes are preserved.
        """
        target_hash = code_hash

        # If no specific hash requested, fetch latest pointer
        if not target_hash:
            remote_meta = self._get_remote_latest_meta(class_name)
            if not remote_meta:
                raise ValueError(f"No remote versions found for {class_name}")
            target_hash = remote_meta["hash"]

        # Recover the class using the final hash
        return self.recover_class(class_name, target_hash)

    @staticmethod
    def _rehydrate_class_globals(clz: Type) -> None:
        """
        Reinject module globals into all methods of a deserialized class.
        This is required for cloudpickle-loaded classes that reference
        external symbols (e.g. XYData, labchain, etc).
        """
        module_name = getattr(clz, "__module__", None)

        if not module_name:
            return

        module = sys.modules.get(module_name)
        if not module:
            return

        module_globals = module.__dict__

        for attr in clz.__dict__.values():
            if isinstance(attr, (types.FunctionType, types.MethodType)):
                attr.__globals__.update(module_globals)  # type: ignore

    def recover_class(self, class_name: str, code_hash: str) -> Type[TypePlugable]:  # type: ignore
        """
        Download and reconstruct class from its hash path.

        This method fetches the serialized class binary from storage using
        the class name and hash, then deserializes it with cloudpickle.

        Args:
            class_name (str): Name of the class.
            code_hash (str): Hash of the version to recover.

        Returns:
            Type[TypePlugable]: The reconstructed class object.

        Example:
        ```python
            # Recover a specific version directly
            MyFilterV1 = manager.recover_class("MyFilter", "abc123...")

            instance = MyFilterV1(param=42)
        ```

        Note:
            This is a lower-level method typically called by pull().
            Prefer using pull() for most use cases.
        """
        # Path WITHOUT "plugins/" prefix since context="plugins" adds it

        path = f"{class_name}/{code_hash}.pkl"
        class_obj = self.storage.download_file(path, context="plugins")
        if hasattr(class_obj.__init__, "__typeguard_original_function__"):
            # Restauramos la función original sin chequeo
            class_obj.__init__ = class_obj.__init__.__typeguard_original_function__

        PetClassManager._rehydrate_class_globals(class_obj)

        return class_obj
