from typing import List
from labchain.base import BaseDatasetManager, VData, XYData
from labchain.container.container import Container


@Container.bind()
class DatasetManager(BaseDatasetManager):
    """
    A manager for handling datasets within the framework3 ecosystem.

    This class provides functionality for listing, saving, updating, loading, and deleting datasets.
    It integrates with the framework's storage system to manage dataset files.

    Key Features:
        - List available datasets
        - Save new datasets
        - Update existing datasets
        - Load datasets for use in models
        - Delete datasets from storage

    Usage:
        The DatasetManager can be used to manage datasets in your project:

        ```python
        from framework3.plugins.ingestion.dataset_manager import DatasetManager
        from framework3.base import XYData
        import numpy as np

        # Create a DatasetManager instance
        dataset_manager = DatasetManager()

        # List available datasets
        available_datasets = dataset_manager.list()
        print("Available datasets:", available_datasets)

        # Save a new dataset
        new_data = XYData.mock(np.array([[1, 2], [3, 4]]))
        dataset_manager.save("new_dataset", new_data)

        # Load a dataset
        loaded_data = dataset_manager.load("new_dataset")
        print("Loaded data:", loaded_data.value)

        # Update a dataset
        updated_data = XYData.mock(np.array([[5, 6], [7, 8]]))
        dataset_manager.update("new_dataset", updated_data)

        # Delete a dataset
        dataset_manager.delete("new_dataset")
        ```

    Methods:
        list() -> List[str]:
            List all available datasets.
        save(name: str, data: VData) -> None:
            Save a new dataset.
        update(name: str, data: VData) -> None:
            Update an existing dataset.
        load(name: str) -> XYData:
            Load a dataset from storage.
        delete(name: str) -> None:
            Delete a dataset from storage.

    Note:
        This class relies on the Container's storage system for file operations.
        Ensure that the storage system is properly configured in your project.
    """

    def list(self) -> List[str]:
        """
        List all available datasets.

        This method retrieves a list of all dataset names currently stored in the system.

        Returns:
            List[str]: A list of dataset names.

        Note:
            This method uses the Container's storage system to list files in the datasets directory.
        """
        return Container.storage.list_stored_files(
            f"{Container.storage.get_root_path()}/datasets"
        )

    def save(self, name: str, data: VData) -> None:
        """
        Save a new dataset.

        This method saves a new dataset to the storage system. If a dataset with the given name
        already exists, it raises a ValueError.

        Args:
            name (str): Unique name for the dataset.
            data (VData): The data to be saved.

        Raises:
            ValueError: If a dataset with the given name already exists.

        Note:
            This method uses the Container's storage system to check for existing files
            and to upload the new dataset.
        """
        if Container.storage.check_if_exists(
            name, f"{Container.storage.get_root_path()}/datasets"
        ):  # type: ignore
            raise ValueError(f"Dataset '{name}' already exists.")
        Container.storage.upload_file(
            data, name, f"{Container.storage.get_root_path()}/datasets"
        )

    def update(self, name: str, data: VData) -> None:
        """
        Update an existing dataset.

        This method updates an existing dataset in the storage system. If a dataset with the given name
        doesn't exist, it raises a ValueError.

        Args:
            name (str): Unique name for the dataset.
            data (VData): The updated data to be saved.

        Raises:
            ValueError: If a dataset with the given name doesn't exist.

        Note:
            This method uses the Container's storage system to check for existing files
            and to upload the updated dataset.
        """
        if not Container.storage.check_if_exists(
            name, f"{Container.storage.get_root_path()}/datasets"
        ):  # type: ignore
            raise ValueError(f"Dataset '{name}' does not exist.")
        Container.storage.upload_file(
            data, name, f"{Container.storage.get_root_path()}/datasets"
        )

    def load(self, name: str) -> XYData:
        """
        Load a dataset from storage.

        This method retrieves a dataset from the storage system. If the dataset doesn't exist,
        it raises a ValueError.

        Args:
            name (str): Name of the dataset to load.

        Returns:
            XYData: The loaded dataset.

        Raises:
            ValueError: If the dataset does not exist.

        Note:
            This method uses the Container's storage system to check for existing files
            and to download the dataset. The actual data is loaded lazily when accessed.
        """
        if not Container.storage.check_if_exists(
            name, f"{Container.storage.get_root_path()}/datasets"
        ):
            raise ValueError(f"Dataset '{name}' does not exist.")
        return XYData(
            _hash=name,
            _path="datasets",
            _value=lambda: Container.storage.download_file(
                name, f"{Container.storage.get_root_path()}/datasets"
            ),
        )

    def delete(self, name: str) -> None:
        """
        Delete a dataset from storage.

        This method removes a dataset from the storage system. If the dataset doesn't exist,
        it raises a ValueError.

        Args:
            name (str): Name of the dataset to delete.

        Raises:
            ValueError: If the dataset does not exist.

        Note:
            This method uses the Container's storage system to check for existing files
            and to delete the dataset.
        """
        if not Container.storage.check_if_exists(
            name, f"{Container.storage.get_root_path()}/datasets"
        ):
            raise ValueError(f"Dataset '{name}' does not exist.")
        Container.storage.delete_file(
            name, f"{Container.storage.get_root_path()}/datasets"
        )
