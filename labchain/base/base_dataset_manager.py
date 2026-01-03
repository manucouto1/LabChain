from abc import ABC, abstractmethod
from typing import List
from labchain.base import BasePlugin, VData, XYData


class BaseDatasetManager(BasePlugin, ABC):
    @abstractmethod
    def list(self) -> List[str]:
        """
        List all available datasets.

        Returns:
            List[str]: A list of dataset names.
        """
        ...

    @abstractmethod
    def save(self, name: str, data: VData) -> None:
        """
        Save a dataset.

        Args:
            name (str): Unique name for the dataset.
            data (XYData): The data to be saved.

        Raises:
            ValueError: If a dataset with the given name already exists.
        """
        ...

    @abstractmethod
    def update(self, name: str, data: VData) -> None:
        """
        Update a dataset.

        Args:
            name (str): Unique name for the dataset.
            data (XYData): The data to be saved.

        Raises:
            ValueError: If a dataset with the given name doesn't exists.
        """
        ...

    @abstractmethod
    def load(self, name: str) -> XYData:
        """
        Load a dataset.

        Args:
            name (str): Name of the dataset to load.

        Returns:
            XYData: The loaded dataset.

        Raises:
            ValueError: If the dataset does not exist.
        """
        ...

    def delete(self, name: str) -> None:
        """
        Delete a dataset.

        Args:
            name (str): Name of the dataset to delete.

        Raises:
            ValueError: If the dataset does not exist.
        """
        ...
