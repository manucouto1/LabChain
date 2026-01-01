from typing import Callable, Optional, Tuple, cast
from labchain.container.container import Container
from labchain.base import BaseFilter
from labchain.base import BaseStorage
from labchain.base import XYData, VData

from rich import print as rprint
import pickle

from labchain.utils.utils import method_is_overridden

__all__ = ["Cached"]


@Container.bind()
class Cached(BaseFilter):
    """
    A filter that manages the storage of models and data in a BaseStorage type.

    This class extends BaseFilter to provide caching capabilities for both the filter model
    and the processed data. It allows for efficient reuse of previously computed results
    and trained models.

    Key Features:
        - Caches both filter models and processed data
        - Supports various storage backends through BaseStorage
        - Allows for overwriting existing cached data
        - Provides methods for managing the cache

    Usage:
        The Cached filter can be used to wrap any BaseFilter, providing caching capabilities:

        ```python
        from framework3.storage import LocalStorage
        from framework3.container import Container
        from your_custom_filter import CustomFilter

        # Configure storage
        Container.storage = LocalStorage(storage_path='cache')

        # Create a custom filter and wrap it with Cached
        custom_filter = CustomFilter()
        cached_filter = Cached(
            filter=custom_filter,
            cache_data=True,
            cache_filter=True,
            overwrite=False
        )

        # Use the cached filter
        X = XYData(_hash='input_data', _path='/datasets', _value=input_data)
        y = XYData(_hash='target_data', _path='/datasets', _value=target_data)

        cached_filter.fit(X, y)
        predictions = cached_filter.predict(X)

        # Clear the cache if needed
        cached_filter.clear_cache()
        ```

    Args:
        filter (BaseFilter): The underlying filter to be cached.
        cache_data (bool): Whether to cache the processed data.
        cache_filter (bool): Whether to cache the trained filter.
        overwrite (bool): Whether to overwrite existing cached data/models.
        storage (BaseStorage|None): The storage backend for caching.

    Attributes:
        filter (BaseFilter): The underlying filter being cached.
        cache_data (bool): Flag indicating whether to cache processed data.
        cache_filter (bool): Flag indicating whether to cache the trained filter.
        overwrite (bool): Flag indicating whether to overwrite existing cached data/models.
        _storage (BaseStorage): The storage backend used for caching.
        _lambda_filter (Callable[..., BaseFilter] | None): Lambda function for lazy loading of cached filter.

    Methods:
        init(): Initialize the cached filter.
        fit(x: XYData, y: Optional[XYData]): Fit the filter to the input data, caching the model if necessary.
        predict(x: XYData) -> XYData: Make predictions using the filter, caching the results if necessary.
        clear_cache(): Clear the cache in the storage.

    Note:
        The caching behavior can be customized by adjusting the cache_data, cache_filter, and overwrite parameters.
        The storage backend can be changed by providing a different BaseStorage implementation.
    """

    def __init__(
        self,
        filter: BaseFilter,
        cache_data: bool = True,
        cache_filter: bool = True,
        overwrite: bool = False,
        storage: BaseStorage | None = None,
    ):
        """
        Initialize a new Cached filter instance.

        This constructor sets up the Cached filter with the specified parameters and
        initializes the underlying filter and storage backend.

        Args:
            filter (BaseFilter): The underlying filter to be cached.
            cache_data (bool, optional): Whether to cache the processed data. Defaults to True.
            cache_filter (bool, optional): Whether to cache the trained filter. Defaults to True.
            overwrite (bool, optional): Whether to overwrite existing cached data/models. Defaults to False.
            storage (BaseStorage | None, optional): The storage backend for caching. If None, uses the Container's storage. Defaults to None.

        Note:
            If no storage is provided, the method will use the storage defined in the Container.
            The _lambda_filter attribute is initialized as None and will be set later if needed.
        """
        super().__init__(
            filter=filter,
            cache_data=cache_data,
            cache_filter=cache_filter,
            overwrite=overwrite,
            storage=storage,
        )
        self.filter: BaseFilter = filter
        self.cache_data = cache_data
        self.cache_filter = cache_filter
        self.overwrite = overwrite
        self._storage: BaseStorage = Container.storage if storage is None else storage
        self._lambda_filter: Callable[..., BaseFilter] | None = None

    def verbose(self, value: bool):
        super().verbose(value)
        self._storage._verbose = value
        self.filter.verbose(value)

    # def init(self) -> None:
    #     """
    #     Initialize the cached filter.

    #     This method initializes both the underlying filter and the Cached filter itself.
    #     """
    #     self.filter.init()
    #     super().init()

    def _pre_fit_wrapp(self, x: XYData, y: Optional[XYData]) -> float | None | dict:
        """
        Wrapper method for the pre-fit stage.

        Args:
            x (XYData): The input data.
            y (Optional[XYData]): The target data, if any.

        Returns:
            float | None: The result of the original fit method.
        """
        return self._original_fit(x, y)

    def _pre_predict_wrapp(self, x: XYData) -> XYData:
        """
        Wrapper method for the pre-predict stage.

        Args:
            x (XYData): The input data for prediction.

        Returns:
            XYData: The result of the original predict method.
        """
        return self._original_predict(x)

    def _get_model_name(self) -> str:
        """
        Get the name of the underlying filter's model.

        Returns:
            str: The model name.
        """
        return self.filter._get_model_name()

    def _get_model_key(self, data_hash: str) -> Tuple[str, str]:
        """
        Generate the model key based on the input data hash.

        Args:
            data_hash (str): The hash of the input data.

        Returns:
            Tuple[str, str]: A tuple containing the model hash and its string representation.
        """
        return BaseFilter._get_model_key(self.filter, data_hash)

    def _get_data_key(self, model_str: str, data_hash: str) -> Tuple[str, str]:
        """
        Generate the data key based on the model and input data hash.

        Args:
            model_str (str): The string representation of the model.
            data_hash (str): The hash of the input data.

        Returns:
            Tuple[str, str]: A tuple containing the data hash and its string representation.
        """
        return BaseFilter._get_data_key(self.filter, model_str, data_hash)

    def fit(self, x: XYData, y: Optional[XYData]) -> None:
        """
        Fit the filter to the input data, caching the model if necessary.

        This method checks if a cached model exists and uses it if available.
        If not, it trains the model and caches it if caching is enabled.

        Args:
            x (XYData): The input data.
            y (Optional[XYData]): The target data, if any.
        """

        f_m_hash = self.filter._m_hash
        f_m_path = self.filter._m_path
        f_m_str = self.filter._m_str

        try:
            self.filter._pre_fit(x, y)
            if (
                not self._storage.check_if_exists(
                    hashcode="model",
                    context=f"{self._storage.get_root_path()}{self.filter._m_path}",
                )
                or self.overwrite
            ):
                if self._verbose:
                    rprint(
                        f"\t - El filtro {self.filter} con hash {self.filter._m_hash} No existe, se va a entrenar."
                    )
                self.filter._original_fit(x, y)

                if self.cache_filter and method_is_overridden(
                    self.filter.__class__, "fit"
                ):
                    if self._verbose:
                        rprint(f"\t - El filtro {self.filter} Se cachea.")
                    self._storage.upload_file(
                        file=pickle.dumps(self.filter),
                        file_name="model",
                        context=f"{self._storage.get_root_path()}{self.filter._m_path}",
                    )
            else:
                if self._verbose:
                    rprint(
                        f"\t - El filtro {self.filter} Existe, se carga del storage."
                    )

                self._lambda_filter = lambda: cast(
                    BaseFilter,
                    self._storage.download_file(
                        "model", f"{self._storage.get_root_path()}{self.filter._m_path}"
                    ),
                )
        except Exception as e:
            self.filter._m_hash = f_m_hash
            self.filter._m_path = f_m_path
            self.filter._m_str = f_m_str
            raise e

    def predict(self, x: XYData) -> XYData:
        """
        Make predictions using the filter, caching the results if necessary.

        This method checks if cached predictions exist and uses them if available.
        If not, it makes new predictions and caches them if caching is enabled.

        Args:
            x (XYData): The input data for prediction.

        Returns:
            XYData: The prediction results.
        """
        x = self.filter._pre_predict(x)

        if (
            not self._storage.check_if_exists(
                x._hash, context=f"{self._storage.get_root_path()}{x._path}"
            )
            or self.overwrite
        ):
            if self._verbose:
                rprint(f"\t - El dato {x} No existe, se va a crear.")

            if self._lambda_filter is not None:
                if self._verbose:
                    rprint(
                        "\t - Existe un Lambda por lo que se recupera el filtro del storage."
                    )
                self.filter = self._lambda_filter()

            value = XYData(
                _hash=x._hash,
                _path=x._path,
                _value=self.filter._original_predict(x)._value,
            )
            if self.cache_data:
                if self._verbose:
                    rprint(f"\t - El dato {x} Se cachea.")

                self._storage.upload_file(
                    file=pickle.dumps(value.value),
                    file_name=x._hash,
                    context=f"{self._storage.get_root_path()}{x._path}",
                )
        else:
            if self._verbose:
                rprint(f"\t - El dato {x} Existe, se carga del storage.")

            value = XYData(
                _hash=x._hash,
                _path=x._path,
                _value=lambda: cast(
                    VData,
                    self._storage.download_file(
                        x._hash, f"{self._storage.get_root_path()}{x._path}"
                    ),
                ),
            )
        return value

    def clear_cache(self):
        """
        Clear the cache in the storage.

        This method should implement the logic to clear all cached data and models
        associated with this filter from the storage backend.

        Note:
            This method is not yet implemented.
        """
        # Implementa la lógica para limpiar el caché en el almacenamiento
        raise NotImplementedError("El método clear_cache no está implementado.")
        pass
