from typing import Any, List
from labchain.base import BaseStorage
import cloudpickle as pickle

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
            file_name (str): The name of the file (can include subdirectories).
            context (str): The directory path where the file will be saved.
            direct_stream (bool, optional): Not used in this implementation. Defaults to False.

        Returns:
            str | None: The file name if successful, None otherwise.
        """
        try:
            prefix = (
                f"{self.storage_path}{context}/"
                if context and not context.endswith("/")
                else f"{self.storage_path}{context}"
            )
            full_path = Path(prefix) / file_name

            # Create parent directory for the file (including any subdirectories in file_name)
            full_path.parent.mkdir(parents=True, exist_ok=True)

            if self._verbose:
                print(f"\t * Saving in local path: {full_path}")

            pickle.dump(file, open(full_path, "wb"))

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
        return os.listdir(f"{self.storage_path}{context}")

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
        prefix = (
            f"{self.storage_path}{context}/"
            if context and not context.endswith("/")
            else f"{self.storage_path}{context}"
        )
        full_path = Path(prefix) / hashcode

        if full_path.exists():
            return open(full_path, "rb")
        else:
            raise FileNotFoundError(f"Couldn't find file {hashcode} in path {prefix}")

    def check_if_exists(self, hashcode: str, context: str) -> bool:
        """
        Check if a file exists in the specified context.

        Args:
            hashcode (str): The hashcode (filename) of the file.
            context (str): The directory path where to check for the file.

        Returns:
            bool: True if the file exists, False otherwise.
        """
        prefix = (
            f"{self.storage_path}{context}/"
            if context and not context.endswith("/")
            else f"{self.storage_path}{context}"
        )
        full_path = Path(prefix) / hashcode
        return full_path.exists()

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
        prefix = (
            f"{self.storage_path}{context}/"
            if context and not context.endswith("/")
            else f"{self.storage_path}{context}"
        )
        if os.path.exists(f"{prefix}{hashcode}"):
            os.remove(f"{prefix}{hashcode}")
        else:
            raise FileExistsError("No existe en la carpeta")
