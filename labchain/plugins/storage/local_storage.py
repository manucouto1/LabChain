from typing import Any, List
from labchain.base import BaseStorage
import pickle

import os
from pathlib import Path


__all__ = ["LocalStorage"]


class LocalStorage(BaseStorage):
    """
    A local file system storage implementation for storing and retrieving files.

    This class provides methods to interact with the local file system, allowing
    storage operations such as uploading, downloading, and deleting files.

    Key Features:
        - Simple interface for file operations
        - Support for creating nested directory structures
        - File existence checking
        - Listing stored files in a given context

    Usage:
        ```python
        from framework3.plugins.storage import LocalStorage

        # Initialize local storage
        storage = LocalStorage(storage_path='my_cache')

        # Upload a file
        storage.upload_file("Hello, World!", "greeting.txt", "/tmp")

        # Download and read a file
        content = storage.download_file("greeting.txt", "/tmp")
        print(content)  # Output: Hello, World!

        # Check if a file exists
        exists = storage.check_if_exists("greeting.txt", "/tmp")
        print(exists)  # Output: True

        # List files in a directory
        files = storage.list_stored_files("/tmp")
        print(files)  # Output: ['greeting.txt']

        # Delete a file
        storage.delete_file("greeting.txt", "/tmp")
        ```

    Attributes:
        storage_path (str): The full path to the storage directory.

    Methods:
        get_root_path() -> str: Get the root path of the storage.
        upload_file(file, file_name: str, context: str, direct_stream: bool = False) -> str | None:
            Upload a file to the specified context.
        list_stored_files(context: str) -> List[str]: List all files in the specified context.
        get_file_by_hashcode(hashcode: str, context: str) -> Any: Get a file by its hashcode.
        check_if_exists(hashcode: str, context: str) -> bool: Check if a file exists in the specified context.
        download_file(hashcode: str, context: str) -> Any: Download and load a file from the specified context.
        delete_file(hashcode: str, context: str) -> None: Delete a file from the specified context.
    """

    def __init__(self, storage_path: str = "cache/"):
        """
        Initialize the LocalStorage.

        Args:
            storage_path (str, optional): The base path for storage. Defaults to 'cache'.
        """
        super().__init__()
        self.storage_path = (
            storage_path
            if storage_path.endswith("/") or storage_path == ""
            else f"{storage_path}/"
        )

    def get_root_path(self) -> str:
        """
        Get the root path of the storage.

        Returns:
            str: The full path to the storage directory.
        """
        return self.storage_path

    def upload_file(
        self, file, file_name: str, context: str, direct_stream: bool = False
    ) -> str | None:
        """
        Upload a file to the specified context.

        Args:
            file (Any): The file content to be uploaded.
            file_name (str): The name of the file.
            context (str): The directory path where the file will be saved.
            direct_stream (bool, optional): Not used in this implementation. Defaults to False.

        Returns:
            str | None: The file name if successful, None otherwise.
        """
        try:
            prefix = f"{context}/" if context and not context.endswith("/") else context
            Path(prefix).mkdir(parents=True, exist_ok=True)
            if self._verbose:
                print(f"\t * Saving in local path: {prefix}{file_name}")
            pickle.dump(file, open(f"{prefix}{file_name}", "wb"))
            if self._verbose:
                print("\t * Saved !")
            return file_name
        except Exception as ex:
            print(ex)
        return None

    def list_stored_files(self, context: str) -> List[str]:
        """
        List all files in the specified context.

        Args:
            context (str): The directory path to list files from.

        Returns:
            List[str]: A list of file names in the specified context.
        """
        return os.listdir(context)

    def get_file_by_hashcode(self, hashcode: str, context: str) -> Any:
        """
        Get a file by its hashcode (filename in this implementation).

        Args:
            hashcode (str): The hashcode (filename) of the file.
            context (str): The directory path where the file is located.

        Returns:
            Any: A file object if found.

        Raises:
            FileNotFoundError: If the file is not found in the specified context.
        """
        prefix = f"{context}/" if context and not context.endswith("/") else context
        if hashcode in os.listdir(prefix):
            return open(f"{prefix}{hashcode}", "rb")
        else:
            raise FileNotFoundError(f"Couldn't find file {hashcode} in path {context}")

    def check_if_exists(self, hashcode: str, context: str) -> bool:
        """
        Check if a file exists in the specified context.

        Args:
            hashcode (str): The hashcode (filename) of the file.
            context (str): The directory path where to check for the file.

        Returns:
            bool: True if the file exists, False otherwise.
        """
        try:
            for file_n in os.listdir(context):
                if file_n == hashcode:
                    return True
            return False
        except FileNotFoundError:
            return False

    def download_file(self, hashcode: str, context: str) -> Any:
        """
        Download and load a file from the specified context.

        Args:
            hashcode (str): The hashcode (filename) of the file to download.
            context (str): The directory path where the file is located.

        Returns:
            Any: The content of the file, unpickled if it was pickled.
        """
        stream = self.get_file_by_hashcode(hashcode, context)
        if self._verbose:
            print(f"\t * Downloading: {stream}")
        loaded = pickle.load(stream)
        return pickle.loads(loaded) if isinstance(loaded, bytes) else loaded

    def delete_file(self, hashcode: str, context: str):
        """
        Delete a file from the specified context.

        Args:
            hashcode (str): The hashcode (filename) of the file to delete.
            context (str): The directory path where the file is located.

        Raises:
            FileExistsError: If the file does not exist in the specified context.
        """
        prefix = f"{context}/" if context and not context.endswith("/") else context
        if os.path.exists(f"{prefix}{hashcode}"):
            os.remove(f"{prefix}{hashcode}")
        else:
            raise FileExistsError("No existe en la carpeta")


# class LockingLocalStorage(BaseLockingStorage):
#     """Local filesystem storage backend with atomic file-based locking.

#     Implements storage and locking using the local filesystem. Lock atomicity
#     is guaranteed through the O_CREAT | O_EXCL flags in os.open(), which
#     provide atomic file creation at the kernel level.

#     This backend is ideal for:
#     - Single-machine parallel processing
#     - Development and testing
#     - CI/CD pipelines
#     - Local experimentation

#     The locking mechanism is safe for multiple processes on the same machine
#     accessing the same filesystem (including NFS if properly configured).

#     Attributes:
#         base_path: Root directory for all storage operations.
#         locks_dir: Directory where lock files are stored.

#     Examples:
#         Basic usage:

#         ```python
#         # Initialize storage
#         storage = LocalStorage("./cache")

#         # Store and retrieve files
#         storage.upload_file("/tmp/model.pkl", "models/abc123/model")
#         storage.download_file("models/abc123/model", "/tmp/loaded.pkl")

#         # Distributed locking
#         if storage.try_acquire_lock("model_abc123"):
#             try:
#                 # Your exclusive work here
#                 train_and_save_model()
#             finally:
#                 storage.release_lock("model_abc123")
#         ```

#         Parallel processing example:

#         ```python
#         from multiprocessing import Pool

#         def train_model(model_id):
#             storage = LocalStorage("./cache")
#             lock_name = f"model_{model_id}"

#             if storage.try_acquire_lock(lock_name):
#                 try:
#                     print(f"Training {model_id}")
#                     # Train model
#                 finally:
#                     storage.release_lock(lock_name)
#             else:
#                 print(f"Waiting for {model_id}")
#                 storage.wait_for_unlock(lock_name)
#                 # Load cached model

#         # Run in parallel - only one process trains each model
#         with Pool(4) as p:
#             p.map(train_model, ["model_1", "model_2", "model_3"])
#         ```

#     Note:
#         Lock files contain metadata (hostname, PID, timestamp) for debugging
#         and stale lock detection.
#     """

#     def __init__(self, base_path: str):
#         """Initialize local storage backend.

#         Creates the base directory and locks subdirectory if they don't exist.

#         Args:
#             base_path: Root directory for storage. Can be relative or absolute.
#                 Will be created if it doesn't exist.

#         Examples:
#             ```python
#             # Relative path
#             storage = LocalStorage("./cache")

#             # Absolute path
#             storage = LocalStorage("/var/ml-cache")

#             # User home directory
#             storage = LocalStorage("~/ml-experiments/cache")
#             ```
#         """
#         self.storage_path = Path(base_path).expanduser().resolve()
#         self.locks_dir = self.storage_path / "locks"
#         self.locks_dir.mkdir(parents=True, exist_ok=True)
