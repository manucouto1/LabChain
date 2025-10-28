import boto3
import pickle
import io
import sys
from typing import Any, List
from botocore.exceptions import ClientError
from framework3.base import BaseStorage
from framework3.container import Container

__all__ = ["S3Storage"]


@Container.bind()
class S3Storage(BaseStorage):
    """
    A storage implementation for Amazon S3 to store and retrieve files.

    This class provides methods to interact with Amazon S3, allowing storage operations
    such as uploading, downloading, and deleting files in an S3 bucket.

    Key Features:
        - Simple interface for S3 file operations
        - Support for file existence checking
        - Listing stored files in the bucket
        - Direct streaming support for large files

    Usage:
        ```python
        from framework3.plugins.storage import S3Storage

        # Initialize S3 storage
        storage = S3Storage(bucket='my-bucket', region_name='us-west-2',
                            access_key_id='YOUR_ACCESS_KEY', access_key='YOUR_SECRET_KEY')

        # Upload a file
        storage.upload_file("Hello, World!", "greeting.txt", "my-folder")

        # Download and read a file
        content = storage.download_file("greeting.txt", "my-folder")
        print(content)  # Output: Hello, World!

        # Check if a file exists
        exists = storage.check_if_exists("greeting.txt", "my-folder")
        print(exists)  # Output: True

        # List files in the bucket
        files = storage.list_stored_files("")
        print(files)  # Output: ['my-folder/greeting.txt']

        # Delete a file
        storage.delete_file("greeting.txt", "my-folder")
        ```

    Attributes:
        _client (boto3.client): The boto3 S3 client.
        bucket (str): The name of the S3 bucket.

    Methods:
        get_root_path() -> str: Get the root path (bucket name) of the storage.
        upload_file(file: object, file_name: str, context: str, direct_stream: bool = False) -> str:
            Upload a file to the specified context in S3.
        list_stored_files(context: str) -> List[str]: List all files in the S3 bucket.
        get_file_by_hashcode(hashcode: str, context: str) -> bytes: Get a file by its hashcode (key in S3).
        check_if_exists(hashcode: str, context: str) -> bool: Check if a file exists in S3.
        download_file(hashcode: str, context: str) -> Any: Download a file from S3.
        delete_file(hashcode: str, context: str) -> None: Delete a file from S3.
    """

    def __init__(
        self,
        bucket: str,
        region_name: str,
        access_key_id: str,
        access_key: str,
        endpoint_url: str | None = None,
        storage_path: str = "",
    ):
        """
        Initialize the S3Storage.

        Args:
            bucket (str): The name of the S3 bucket.
            region_name (str): The AWS region name.
            access_key_id (str): The AWS access key ID.
            access_key (str): The AWS secret access key.
            endpoint_url (str | None, optional): The endpoint URL for the S3 service. Defaults to None.
        """
        super().__init__()
        self._client = boto3.client(
            service_name="s3",
            region_name=region_name,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=access_key,
            endpoint_url=endpoint_url,
            use_ssl=True,
        )
        self.bucket = bucket
        self.storage_path = (
            storage_path
            if storage_path.endswith("/") or storage_path == ""
            else f"{storage_path}/"
        )

    def get_root_path(self) -> str:
        """
        Get the root path (bucket name) of the storage.

        Returns:
            str: The name of the S3 bucket.
        """
        return self.storage_path

    def upload_file(
        self, file: object, file_name: str, context: str, direct_stream: bool = False
    ) -> str:
        """
        Upload a file to the specified context in S3.

        Args:
            file (object): The file content to be uploaded.
            file_name (str): The name of the file.
            context (str): The directory path where the file will be saved.
            direct_stream (bool, optional): If True, assumes file is already a BytesIO object. Defaults to False.

        Returns:
            str: The file name if successful.
        """
        prefix = f"{context}/" if context and not context.endswith("/") else context

        if isinstance(file, (bytes, bytearray)):
            stream = io.BytesIO(file)
        elif isinstance(file, io.BytesIO):
            stream = file
        else:
            stream = io.BytesIO(pickle.dumps(file))

        if self._verbose:
            print("- Binary prepared!")
            print("- Stream ready!")
            print(f" \t * Object size {sys.getsizeof(stream) * 1e-9} GBs ")

        self._client.put_object(
            Body=stream, Bucket=self.bucket, Key=f"{prefix}{file_name}"
        )

        if self._verbose:
            print("Upload Complete!")

        return file_name

    def list_stored_files(self, context: str) -> List[str]:
        """
        List all files in a specific folder (context) in the S3 bucket.

        Args:
            context (str): The folder path within the bucket to list files from.

        Returns:
            List[str]: A list of object keys in the specified folder.
        """
        # Ensure the context ends with a trailing slash if it's not empty
        prefix = f"{context}/" if context and not context.endswith("/") else context

        paginator = self._client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=self.bucket, Prefix=prefix)

        file_list = []
        for page in pages:
            if "Contents" in page:
                for obj in page["Contents"]:
                    # Remove the prefix from the key to get the relative path
                    relative_path = obj["Key"]
                    if relative_path:  # Ignore the folder itself
                        file_list.append(relative_path)

        return file_list

    def get_file_by_hashcode(self, hashcode: str, context: str) -> bytes:
        """
        Get a file by its hashcode (key in S3).

        Args:
            hashcode (str): The hashcode (key) of the file.
            context (str): Not used in this implementation.

        Returns:
            bytes: The content of the file.
        """
        prefix = f"{context}/" if context and not context.endswith("/") else context
        obj = self._client.get_object(Bucket=self.bucket, Key=f"{prefix}{hashcode}")
        return obj["Body"].read()

    def check_if_exists(self, hashcode: str, context: str) -> bool:
        """
        Check if a file exists in S3.

        Args:
            hashcode (str): The name of the file.
            context (str): The directory path where the file is located.

        Returns:
            bool: True if the file exists, False otherwise.
        """
        try:
            prefix = f"{context}/" if context and not context.endswith("/") else context
            self._client.head_object(Bucket=self.bucket, Key=f"{prefix}{hashcode}")
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            else:
                print(f"An error ocurred > {e}")
                return False
        return True

    def download_file(self, hashcode: str, context: str) -> Any:
        """
        Download a file from S3.

        Args:
            hashcode (str): The name of the file.
            context (str): The directory path where the file is located.

        Returns:
            Any: The deserialized content of the file.
        """
        prefix = f"{context}/" if context and not context.endswith("/") else context
        obj = self._client.get_object(Bucket=self.bucket, Key=f"{prefix}{hashcode}")
        return pickle.loads(obj["Body"].read())

    def delete_file(self, hashcode: str, context: str) -> None:
        """
        Delete a file from S3.

        Args:
            hashcode (str): The name of the file.
            context (str): The directory path where the file is located.

        Raises:
            Exception: If the file couldn't be deleted.
            FileExistsError: If the file doesn't exist in the bucket.
        """

        if self.check_if_exists(hashcode, context):
            prefix = f"{context}/" if context and not context.endswith("/") else context
            self._client.delete_object(Bucket=self.bucket, Key=f"{prefix}{hashcode}")
            if self.check_if_exists(hashcode, context):
                raise Exception("Couldn't delete file")
            else:
                print("Deleted!")
        else:
            raise FileExistsError("No existe en el bucket")
