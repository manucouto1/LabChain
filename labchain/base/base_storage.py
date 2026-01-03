from __future__ import annotations

from abc import abstractmethod
import time
from typing import Dict, List, Any, Type

from labchain.base import BasePlugin

__all__ = ["BaseStorage", "BaseSingleton"]


class BaseSingleton:
    """
    A base class for implementing the Singleton pattern.

    This class ensures that only one instance of each derived class is created.

    Key Features:
        - Implements the Singleton design pattern
        - Allows derived classes to have only one instance

    Usage:
        To create a Singleton class, inherit from BaseSingleton:

        ```python
        class MySingleton(BaseSingleton):
            def __init__(self):
                self.value = 0

            def increment(self):
                self.value += 1

        # Usage
        instance1 = MySingleton()
        instance2 = MySingleton()
        assert instance1 is instance2  # True
        ```

    Attributes:
        _instances (Dict[Type[BaseSingleton], Any]): A class-level dictionary to store instances.

    Methods:
        __new__(cls: Type[BaseSingleton], *args: Any, **kwargs: Any) -> BaseStorage:
            Creates a new instance or returns the existing one.

    Note:
        This class should be used as a base class for any class that needs to implement
        the Singleton pattern.
    """

    _instances: Dict[Type[BaseSingleton], Any] = {}
    _verbose: bool = True

    def __new__(cls: Type[BaseSingleton], *args: Any, **kwargs: Any) -> BaseStorage:
        """
        Create a new instance of the class if it doesn't exist, otherwise return the existing instance.

        This method implements the core logic of the Singleton pattern.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            BaseStorage: The single instance of the class.

        Note:
            This method is called before __init__ when creating a new instance of the class.
        """
        if cls not in cls._instances:
            cls._instances[cls] = super().__new__(cls)  # type: ignore
        return cls._instances[cls]


class BaseStorage(BasePlugin, BaseSingleton):
    """
    An abstract base class for storage operations.

    This class defines the interface for storage-related operations and inherits
    from BasePlugin for plugin functionality and BaseSingleton for single instance behavior.

    Key Features:
        - Abstract methods for common storage operations
        - Singleton behavior ensures only one instance per storage type
        - Inherits plugin functionality from BasePlugin

    Usage:
        To create a new storage type, inherit from BaseStorage and implement all abstract methods:

        ```python
        class MyCustomStorage(BaseStorage):
            def __init__(self, root_path: str):
                self.root_path = root_path

            def get_root_path(self) -> str:
                return self.root_path

            def upload_file(self, file, file_name: str, context: str, direct_stream: bool = False) -> str | None:
                # Implement file upload logic
                ...

            # Implement other abstract methods
            ...

        # Usage
        storage = MyCustomStorage("/path/to/storage")
        storage.upload_file(file_object, "example.txt", "documents")
        ```

    Methods:
        get_root_path() -> str:
            Abstract method to get the root path of the storage.
        upload_file(file: object, file_name: str, context: str, direct_stream: bool = False) -> str | None:
            Abstract method to upload a file to the storage.
        download_file(hashcode: str, context: str) -> Any:
            Abstract method to download a file from the storage.
        list_stored_files(context: str) -> List[Any]:
            Abstract method to list all files stored in a specific context.
        get_file_by_hashcode(hashcode: str, context: str) -> Any:
            Abstract method to retrieve a file by its hashcode.
        check_if_exists(hashcode: str, context: str) -> bool:
            Abstract method to check if a file exists in the storage.
        delete_file(hashcode: str, context: str):
            Abstract method to delete a file from the storage.

    Note:
        This is an abstract base class. Concrete implementations should override
        all abstract methods to provide specific storage functionality.
    """

    @abstractmethod
    def get_root_path(self) -> str:
        """
        Get the root path of the storage.

        This method should be implemented to return the base directory or path
        where the storage system keeps its files.

        Returns:
            str: The root path of the storage.

        Raises:
            NotImplementedError: If the subclass does not implement this method.

        Example:
            ```python
            def get_root_path(self) -> str:
                return "/var/data/storage"
            ```
        """
        ...

    @abstractmethod
    def upload_file(
        self, file: object, file_name: str, context: str, direct_stream: bool = False
    ) -> str | None:
        """
        Upload a file to the storage.

        This method should be implemented to handle file uploads to the storage system.

        Args:
            file (object): The file object to upload.
            file_name (str): The name of the file.
            context (str): The context or directory for the file.
            direct_stream (bool, optional): Whether to use direct streaming. Defaults to False.

        Returns:
            str | None: The identifier of the uploaded file, or None if upload failed.

        Raises:
            NotImplementedError: If the subclass does not implement this method.

        Example:
            ```python
            def upload_file(self, file: object, file_name: str, context: str, direct_stream: bool = False) -> str | None:
                path = os.path.join(self.get_root_path(), context, file_name)
                with open(path, 'wb') as f:
                    f.write(file.read())
                return file_name
            ```
        """
        ...

    @abstractmethod
    def download_file(self, hashcode: str, context: str) -> Any:
        """
        Download a file from the storage.

        This method should be implemented to retrieve files from the storage system.

        Args:
            hashcode (str): The identifier of the file to download.
            context (str): The context or directory of the file.

        Returns:
            Any: The downloaded file object.

        Raises:
            NotImplementedError: If the subclass does not implement this method.

        Example:
            ```python
            def download_file(self, hashcode: str, context: str) -> Any:
                path = os.path.join(self.get_root_path(), context, hashcode)
                with open(path, 'rb') as f:
                    return f.read()
            ```
        """
        ...

    @abstractmethod
    def list_stored_files(self, context: str) -> List[Any]:
        """
        List all files stored in a specific context.

        This method should be implemented to return a list of files in a given context.

        Args:
            context (str): The context or directory to list files from.

        Returns:
            List[Any]: A list of file objects or file information.

        Raises:
            NotImplementedError: If the subclass does not implement this method.

        Example:
            ```python
            def list_stored_files(self, context: str) -> List[Any]:
                path = os.path.join(self.get_root_path(), context)
                return os.listdir(path)
            ```
        """
        ...

    @abstractmethod
    def get_file_by_hashcode(self, hashcode: str, context: str) -> Any:
        """
        Retrieve a file by its hashcode.

        This method should be implemented to fetch a specific file using its identifier.

        Args:
            hashcode (str): The identifier of the file.
            context (str): The context or directory of the file.

        Returns:
            Any: The file object or file information.

        Raises:
            NotImplementedError: If the subclass does not implement this method.

        Example:
            ```python
            def get_file_by_hashcode(self, hashcode: str, context: str) -> Any:
                return self.download_file(hashcode, context)
            ```
        """
        ...

    @abstractmethod
    def check_if_exists(self, hashcode: str, context: str) -> bool:
        """
        Check if a file exists in the storage.

        This method should be implemented to verify the existence of a file.

        Args:
            hashcode (str): The identifier of the file.
            context (str): The context or directory of the file.

        Returns:
            bool: True if the file exists, False otherwise.

        Raises:
            NotImplementedError: If the subclass does not implement this method.

        Example:
            ```python
            def check_if_exists(self, hashcode: str, context: str) -> bool:
                path = os.path.join(self.get_root_path(), context, hashcode)
                return os.path.exists(path)
            ```
        """
        ...

    @abstractmethod
    def delete_file(self, hashcode: str, context: str):
        """
        Delete a file from the storage.

        This method should be implemented to remove a file from the storage system.

        Args:
            hashcode (str): The identifier of the file to delete.
            context (str): The context or directory of the file.

        Raises:
            NotImplementedError: If the subclass does not implement this method.

        Example:
            ```python
            def delete_file(self, hashcode: str, context: str):
                path = os.path.join(self.get_root_path(), context, hashcode)
                if os.path.exists(path):
                    os.remove(path)
                else:
                    raise FileNotFoundError(f"File {hashcode} not found in {context}")
            ```
        """
        ...


class BaseLockingStorage(BaseStorage):
    """Base class for storage backends with distributed locking support.

    This abstract class defines the interface for storage backends that support
    both file operations and distributed locking mechanisms. The locking system
    ensures safe concurrent access to cached artifacts across multiple processes
    or machines.

    Storage backends must implement atomic lock acquisition to prevent race
    conditions when multiple processes try to generate the same cached artifact
    simultaneously.

    Examples:
        Basic storage implementation pattern:

        ```python
        class MyStorage(BaseStorage):
            def exists(self, path: str) -> bool:
                # Check if file exists in your storage
                return my_storage_client.file_exists(path)

            def try_acquire_lock(self, lock_name: str, ttl: int = 3600) -> bool:
                # Atomic lock acquisition
                return my_storage_client.create_if_not_exists(
                    f"locks/{lock_name}.lock"
                )
        ```

    Note:
        All lock operations must be atomic to guarantee correctness in
        distributed environments. Non-atomic implementations will lead to
        race conditions.
    """

    @abstractmethod
    def try_acquire_lock(
        self, lock_name: str, ttl: int = 3600, heartbeat_interval: int | None = None
    ) -> bool:
        """Try to acquire a distributed lock atomically.

        This operation must be atomic to prevent race conditions. Only one
        process across all machines should be able to acquire a lock with
        the same name at any given time.

        The lock includes a TTL (time-to-live) for automatic recovery from
        crashed processes. If a process crashes while holding a lock, other
        processes can steal it after the TTL expires.

        Args:
            lock_name: Unique identifier for the lock. Should be descriptive,
                e.g., "model_abc123" or "data_xyz789".
            ttl: Time-to-live in seconds. After this time, the lock is
                considered stale and can be stolen by other processes.
                Default is 3600 (1 hour).
            heartbeat_interval: Optional interval in seconds for heartbeat updates.
                If provided, enables crash detection. Should be ttl/10 or smaller.
                Default None (no heartbeat).


        Returns:
            True if the lock was successfully acquired, False if another
            process holds the lock.

        Examples:
            ```python
            # Try to acquire lock for training a model
            if storage.try_acquire_lock("model_abc123", ttl=7200):
                try:
                    # Train model - you have exclusive access
                    train_model()
                    save_model()
                finally:
                    storage.release_lock("model_abc123")
            else:
                # Another process is training, wait for it
                storage.wait_for_unlock("model_abc123")
                load_cached_model()
            ```

        Note:
            Always release locks in a try-finally block to prevent deadlocks.
        """
        pass

    @abstractmethod
    def release_lock(self, lock_name: str) -> None:
        """Release a previously acquired lock.

        Makes the lock available for other processes. Safe to call even if
        the lock doesn't exist or was already released.

        Args:
            lock_name: Identifier of the lock to release.

        Examples:
            ```python
            storage.acquire_lock("model_abc123")
            try:
                train_model()
            finally:
                storage.release_lock("model_abc123")
            ```
        """
        pass

    def wait_for_unlock(
        self,
        lock_name: str,
        timeout: int = 7200,
        initial_poll_interval: float = 0.5,
        max_poll_interval: float = 10.0,
        backoff_factor: float = 1.5,
    ) -> bool:
        """Wait for a lock to be released with exponential backoff.

                Uses exponential backoff to reduce polling frequency over time,
                minimizing resource usage while still being responsive.

                Args:
                    lock_name: Identifier of the lock to wait for.
                    timeout: Maximum time to wait in seconds. Default 7200 (2 hours).
                    initial_poll_interval: Initial time between checks in seconds. Default 0.5.
                    max_poll_interval: Maximum time between checks in seconds. Default 10.0.
                    backoff_factor: Multiplier for poll interval after each check. Default 1.5.

                Returns:
                    True if the lock was released within the timeout, False if timeout
                    was reached.

                Examples:
        ```python
                    # Quick response initially, then less frequent checks
                    if storage.wait_for_unlock("model_abc123", timeout=3600):
                        load_model()
                    else:
                        raise TimeoutError("Training took too long")
        ```

                Note:
                    Polling pattern with default settings (backoff=1.5):

                    | Check | Interval | Cumulative Time |
                    |-------|----------|----------------|
                    | 1     | 0.5s     | 0.5s          |
                    | 2     | 0.75s    | 1.25s         |
                    | 3     | 1.13s    | 2.38s         |
                    | 4     | 1.69s    | 4.07s         |
                    | 5     | 2.53s    | 6.60s         |
                    | ...   | ...      | ...           |
                    | 15    | 10.0s    | ~60s          |
                    | ...   | 10.0s    | ...           |

                    For a 30-minute wait:
                    - Fixed 0.5s polling: 3,600 checks
                    - Exponential backoff: ~150 checks (24x reduction!)

        """
        start_time = time.time()
        poll_interval = initial_poll_interval
        check_count = 0

        while time.time() - start_time < timeout:
            if not self._is_locked(lock_name):
                if self._verbose:
                    print(
                        f"\t✓ Lock released after {check_count} checks "
                        f"({time.time() - start_time:.1f}s)"
                    )
                return True

            time.sleep(poll_interval)
            check_count += 1

            # Exponential backoff
            poll_interval = min(poll_interval * backoff_factor, max_poll_interval)

        if self._verbose:
            print(f"\t⏱️ Timeout after {check_count} checks " f"({timeout}s elapsed)")

        return False

    @abstractmethod
    def _is_locked(self, lock_name: str) -> bool:
        """Check if a lock exists and is not stale.

        Internal method used by wait_for_unlock. Must consider TTL when
        checking lock validity.

        Args:
            lock_name: Identifier of the lock to check.

        Returns:
            True if lock exists and is fresh (not expired), False otherwise.
        """
        pass

    def _get_lock_path(self, lock_name: str) -> str:
        """Convert lock name to storage path.

        Provides a consistent naming convention for lock files across all
        storage backends.

        Args:
            lock_name: Lock identifier.

        Returns:
            Storage path for the lock file, e.g., "locks/model_abc123.lock".
        """
        return f"locks/{lock_name}.lock"
