from __future__ import annotations
import hashlib
from typing import Callable, Generic, Iterable, Tuple, TypeVar, Any, TypedDict, cast
import pandas as pd
import numpy as np
import torch
import typing_extensions
from labchain.utils.utils import hash_normalize
from multimethod import multimethod
from scipy.sparse import spmatrix, csr_matrix, hstack, vstack
from typing import TypeAlias
from sklearn.model_selection import train_test_split
from dataclasses import dataclass, field

__all__ = ["XYData", "VData", "SkVData", "IncEx", "TypePlugable"]

# Type Definitions
Float = float | np.float16 | np.float32 | np.float64
"""Type alias for float values, including numpy float types."""

IncEx: typing_extensions.TypeAlias = (
    "set[int] | set[str] | dict[int, Any] | dict[str, Any] | None"
)
"""Type alias for inclusion/exclusion specifications in data processing."""

VData: TypeAlias = np.ndarray | pd.DataFrame | spmatrix | list | torch.Tensor
"""Type alias for various data structures used in the framework."""

SkVData: TypeAlias = np.ndarray | pd.DataFrame | spmatrix | csr_matrix
"""Type alias for scikit-learn compatible data structures."""

TypePlugable = TypeVar("TypePlugable")
"""Generic type variable for pluggable types in the framework."""

TxyData = TypeVar("TxyData", SkVData, VData)
"""Type variable constrained to SkVData or VData for use in XYData."""


class JsonEncoderkwargs(TypedDict, total=False):
    exclude: IncEx | None
    by_alias: bool
    exclude_unset: bool
    exclude_defaults: bool
    exclude_none: bool
    sqlalchemy_safe: bool


@dataclass(slots=True)
class XYData(Generic[TxyData]):
    """
    A dataclass representing data for machine learning tasks, typically features (X) or targets (Y).

    This class is immutable and uses slots for memory efficiency. It provides a standardized
    way to handle various types of data used in machine learning pipelines.

    Attributes:
        _hash (str): A unique identifier or hash for the data.
        _path (str): The path where the data is stored or retrieved from.
        _value (TxyData | Callable[..., TxyData]): The actual data or a callable that returns the data.

    Methods:
        train_test_split: Split the data into training and testing sets.
        split: Create a new XYData instance with specified indices.
        mock: Create a mock XYData instance for testing or placeholder purposes.
        concat: Concatenate a list of data along the specified axis.
        ensure_dim: Ensure the input data has at least two dimensions.
        as_iterable: Convert the data to an iterable form.

    Example:
        ```python
        import numpy as np
        from framework3.base.base_types import XYData

        # Create a mock XYData instance with random data
        features = np.random.rand(100, 5)
        labels = np.random.randint(0, 2, 100)

        x_data = XYData.mock(features, hash="feature_data", path="/data/features")
        y_data = XYData.mock(labels, hash="label_data", path="/data/labels")

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = x_data.train_test_split(x_data.value, y_data.value, test_size=0.2)

        # Access the data
        print(f"Training features shape: {X_train.value.shape}")
        print(f"Training labels shape: {y_train.value.shape}")

        # Create a subset of the data
        subset = x_data.split(range(50))
        print(f"Subset shape: {subset.value.shape}")
        ```

    Note:
        This class is designed to work with various data types including numpy arrays,
        pandas DataFrames, scipy sparse matrices, and PyTorch tensors.
    """

    _hash: str = field(init=True)
    _path: str = field(init=True)
    _value: TxyData | Callable[..., TxyData] = field(init=True, repr=False)

    def train_test_split(
        self, x: TxyData, y: TxyData | None, test_size: float, random_state: int = 42
    ) -> Tuple[XYData, XYData, XYData, XYData]:
        """
        Split the data into training and testing sets.

        This method uses sklearn's train_test_split function to divide the data
        into training and testing sets for both features (X) and targets (Y).

        Args:
            x (TxyData): The feature data to split.
            y (TxyData | None): The target data to split. Can be None for unsupervised learning.
            test_size (float): The proportion of the data to include in the test split (0.0 to 1.0).
            random_state (int, optional): Seed for the random number generator. Defaults to 42.

        Returns:
            Tuple[XYData, XYData, XYData, XYData]: A tuple containing (X_train, X_test, y_train, y_test),
            each wrapped in an XYData instance.

        Example:
            ```python
            data = XYData.mock(np.random.rand(100, 5))
            labels = XYData.mock(np.random.randint(0, 2, 100))
            X_train, X_test, y_train, y_test = data.train_test_split(data.value, labels.value, test_size=0.2)
            ```
        """
        X_train, X_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size, random_state=random_state
        )

        return (
            XYData.mock(X_train, hash=f"{self._hash} X train", path="/dataset"),
            XYData.mock(X_test, hash=f"{self._hash} X test", path="/dataset"),
            XYData.mock(y_train, hash=f"{self._hash} y train", path="/dataset"),
            XYData.mock(y_test, hash=f"{self._hash} y test", path="/dataset"),
        )

    def split(self, indices: Iterable[int]) -> XYData:
        """
        Split the data into a new XYData instance with the specified indices.

        This method creates a new XYData instance containing only the data
        corresponding to the provided indices.

        Args:
            indices (Iterable[int]): The indices to select from the data.

        Returns:
            XYData: A new XYData instance containing the selected data.

        Example:
            ```python
            data = XYData.mock(np.random.rand(100, 5))
            subset = data.split(range(50, 100))  # Select second half of the data
            ```
        """

        def split_data(self, indices: Iterable[int]) -> VData:
            value = self.value

            match value:
                case np.ndarray():
                    return value[list(indices)]

                case list():
                    return [value[i] for i in indices]

                case torch.Tensor():
                    return value[list(indices)]

                # mypy: disable-next-line[misc]
                # case spmatrix():
                #     return cast(spmatrix, csr_matrix(value)[indices])

                case pd.DataFrame():
                    return value.iloc[list(indices)]

                case _:
                    if isinstance(value, spmatrix):
                        return cast(spmatrix, csr_matrix(value)[indices])
                    raise TypeError(
                        f"Unsupported data type for splitting: {type(value)}"
                    )

        indices_hash = hashlib.sha1(str(list(indices)).encode()).hexdigest()
        return XYData(
            _hash=f"{self._hash}[{indices_hash}]",
            _path=self._path,
            _value=lambda: split_data(self, indices),
        )

    @staticmethod
    def mock(
        value: TxyData | Callable[..., TxyData],
        hash: str | None = None,
        path: str | None = None,
    ) -> XYData:
        """
        Create a mock XYData instance for testing or placeholder purposes.

        This static method allows for easy creation of XYData instances,
        particularly useful in testing scenarios or when placeholder data is needed.

        Args:
            value (TxyData | Callable[..., TxyData]): The data or a callable that returns the data.
            hash (str | None, optional): A hash string for the data. Defaults to "Mock" if None.
            path (str | None, optional): A path string for the data. Defaults to "/tmp" if None.

        Returns:
            XYData: A new XYData instance with the provided or default values.

        Example:
            ```python
            mock_data = XYData.mock(np.random.rand(10, 5), hash="test_data", path="/data/test")
            ```
        """
        if hash is None:
            norm = hash_normalize(value)
            data = repr(norm).encode()
            hash = hashlib.sha256(data).hexdigest()

        if path is None:
            path = "/tmp"

        return XYData(_hash=hash, _path=path, _value=value)

    @property
    def value(self) -> TxyData:
        """
        Property to access the actual data.

        This property ensures that if _value is a callable, it is called to retrieve the data.
        Otherwise, it returns the data directly.

        Returns:
            TxyData: The actual data (numpy array, pandas DataFrame, scipy sparse matrix, etc.).

        Note:
            This property may modify the _value attribute if it's initially a callable.
        """
        self._value = self._value() if callable(self._value) else self._value
        return self._value

    @staticmethod
    def concat(x: list[TxyData], axis: int = -1) -> XYData:
        """
        Concatenate a list of data along the specified axis.

        This static method handles concatenation for various data types,
        including sparse matrices and other array-like structures.

        Args:
            x (list[TxyData]): List of data to concatenate.
            axis (int, optional): Axis along which to concatenate. Defaults to -1.

        Returns:
            XYData: A new XYData instance with the concatenated data.

        Raises:
            ValueError: If an invalid axis is specified for sparse matrix concatenation.

        Example:
            ```python
            data1 = np.random.rand(10, 5)
            data2 = np.random.rand(10, 5)
            combined = XYData.concat([data1, data2], axis=1)
            ```
        """
        if all(isinstance(item, spmatrix) for item in x):
            if axis == 1:
                return XYData.mock(value=cast(spmatrix, hstack(x)))
            elif axis == 0:
                return XYData.mock(value=cast(spmatrix, vstack(x)))
            raise ValueError("Invalid axis for concatenating sparse matrices")
        return concat(x, axis=axis)

    @staticmethod
    def ensure_dim(x: list | np.ndarray) -> list | np.ndarray:
        """
        Ensure the input data has at least two dimensions.

        This static method is a wrapper around the ensure_dim function,
        which adds a new axis to 1D arrays or lists.

        Args:
            x (list | np.ndarray): Input data to ensure dimensions.

        Returns:
            list | np.ndarray: Data with at least two dimensions.

        Example:
            ```python
            data = [1, 2, 3, 4, 5]
            two_dim_data = XYData.ensure_dim(data)
            ```
        """
        return ensure_dim(x)

    def as_iterable(self) -> Iterable:
        """
        Convert the `_value` attribute to an iterable, regardless of its underlying type.

        This method provides a consistent way to iterate over the data,
        handling different data types appropriately.

        Returns:
            Iterable: An iterable version of `_value`.

        Raises:
            TypeError: If the value type is not compatible with iteration.

        Example:
            ```python
            data = XYData.mock(np.random.rand(10, 5))
            for item in data.as_iterable():
                print(item)
            ```
        """
        value = self.value

        # Maneja diferentes tipos de datos
        if isinstance(value, np.ndarray):
            return value  # Los arrays numpy ya son iterables
        elif isinstance(value, pd.DataFrame):
            return value.iterrows()  # Devuelve un iterable sobre las filas
        elif isinstance(value, spmatrix):
            return value.toarray()  # type: ignore # Convierte la matriz dispersa a un array denso
        elif isinstance(value, torch.Tensor):
            return value
        else:
            raise TypeError(f"El tipo {type(value)} no es compatible con iteración.")


@multimethod
def concat(x: Any, axis: int) -> "XYData":
    """
    Base multimethod for concatenation. Raises an error for unsupported types.

    Args:
        x (Any): Data to concatenate.
        axis (int): Axis along which to concatenate.

    Raises:
        TypeError: Always raised as this is the base method for unsupported types.
    """
    raise TypeError(f"Cannot concatenate this type of data, only {VData} compatible")


@concat.register  # type: ignore
def _(x: list[np.ndarray], axis: int = -1) -> "XYData":
    """
    Concatenate a list of numpy arrays.

    Args:
        x (list[np.ndarray]): List of numpy arrays to concatenate.
        axis (int, optional): Axis along which to concatenate. Defaults to -1.

    Returns:
        XYData: A new XYData instance with the concatenated numpy array.
    """
    return XYData.mock(np.concatenate(x, axis=axis))


@concat.register  # type: ignore
def _(x: list[pd.DataFrame], axis: int = -1) -> "XYData":
    """
    Concatenate a list of pandas DataFrames.

    Args:
        x (list[pd.DataFrame]): List of pandas DataFrames to concatenate.
        axis (int, optional): Axis along which to concatenate. Defaults to -1.

    Returns:
        XYData: A new XYData instance with the concatenated pandas DataFrame.
    """
    return XYData.mock(pd.concat(x, axis=axis))  # type: ignore


@concat.register  # type: ignore
def _(x: list[torch.Tensor], axis: int = -1) -> "XYData":
    """
    Concatenate a list of PyTorch tensors.

    Args:
        x (list[torch.Tensor]): List of PyTorch tensors to concatenate.
        axis (int, optional): Axis along which to concatenate. Defaults to -1.

    Returns:
        XYData: A new XYData instance with the concatenated PyTorch tensor.
    """
    return XYData.mock(torch.cat(x, axis=axis))  # type: ignore


@multimethod
def ensure_dim(x: Any) -> SkVData | VData:
    """
    Base multimethod for ensuring dimensions. Raises an error for unsupported types.

    Args:
        x (Any): Data to ensure dimensions for.

    Raises:
        TypeError: Always raised as this is the base method for unsupported types.
    """
    raise TypeError(
        f"Cannot concatenate this type of data, only {VData} or {SkVData} compatible"
    )


@ensure_dim.register  # type: ignore
def _(x: np.ndarray) -> SkVData:
    """
    Ensure that a numpy array has at least two dimensions.

    Args:
        x (np.ndarray): Input numpy array.

    Returns:
        SkVData: The input array with at least two dimensions.
    """
    if x.ndim == 1:  # Verifica si es unidimensional
        return x[:, None]  # Agrega una nueva dimensión
    return x  # No cambia el array si tiene más dimensiones


@ensure_dim.register  # type: ignore
def _(x: torch.Tensor) -> VData:
    """
    Ensure that a PyTorch tensor has at least two dimensions.

    Args:
        x (torch.Tensor): Input PyTorch tensor.

    Returns:
        VData: The input tensor with at least two dimensions.
    """
    if x.ndim == 1:  # Verifica si es unidimensional
        return x.unsqueeze(-1)
    return x  # No cambia el tensor si tiene más dimensiones


@ensure_dim.register  # type: ignore
def _(x: list) -> SkVData:
    """
    Ensure that a list has at least two dimensions by converting it to a numpy array.

    Args:
        x (list): Input list.

    Returns:
        SkVData: A numpy array with at least two dimensions.
    """
    return ensure_dim(np.array(x))
