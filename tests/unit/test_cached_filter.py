import types
from typing import cast
import pytest
import numpy as np
import pickle
from unittest.mock import ANY, MagicMock
from framework3.base import BaseFilter, BaseStorage
from framework3.base import XYData
from framework3.base.exceptions import NotTrainableFilterError
from framework3.plugins.filters.cache.cached_filter import Cached
from numpy.typing import ArrayLike


class NonTrainableFilter(BaseFilter):
    def __init__(self):
        super().__init__()

    def predict(self, x):
        return XYData(
            _hash="output_hash", _path="/output/path", _value=np.array([7, 8, 9])
        )


@pytest.fixture
def non_trainable_filter():
    return NonTrainableFilter()


# Implementación simple de BaseFilter para testing
class SimpleFilter(BaseFilter):
    def fit(self, x, y):
        pass

    def predict(self, x):
        return XYData(
            _hash="output_hash", _path="/output/path", _value=np.array([7, 8, 9])
        )


@pytest.fixture
def mock_storage():
    return MagicMock(spec=BaseStorage)


@pytest.fixture
def simple_filter():
    return SimpleFilter()


def test_cache_filter_model_when_not_exists(mock_storage, simple_filter):
    mock_storage.check_if_exists.return_value = False
    mock_storage.get_root_path.return_value = "root/"

    x = XYData(_hash="input_hash", _path="/input/path", _value=np.array(range(100)))
    y = XYData(_hash="target_hash", _path="/target/path", _value=np.array(range(100)))

    cached_filter = Cached(
        filter=simple_filter, cache_filter=True, storage=mock_storage
    )
    cached_filter.fit(x, y)

    mock_storage.upload_file.assert_called_once()
    calls = mock_storage.upload_file.mock_calls

    assert calls[0].kwargs["file_name"] == "model"
    assert (
        calls[0].kwargs["context"]
        == f"root/{cached_filter._get_model_name()}/{cached_filter.filter._m_hash}"
    )


def test_use_cached_filter_model_when_exists(mock_storage, simple_filter):
    mock_storage.check_if_exists.return_value = True
    mock_storage.get_root_path.return_value = "root/"
    mock_storage.download_file.return_value = simple_filter

    x = XYData(_hash="input_hash", _path="/input/path", _value=np.array([1, 2, 3]))
    y = XYData(_hash="target_hash", _path="/target/path", _value=np.array([4, 5, 6]))

    cached_filter = Cached(
        filter=simple_filter, cache_filter=True, storage=mock_storage
    )

    cached_filter.fit(x, y)

    mock_storage.upload_file.assert_not_called()
    assert cached_filter._lambda_filter is not None

    mock_storage.download_file.return_value = np.array([1, 2, 3])

    # Trigger the lambda_filter execution
    result = cached_filter.predict(x)

    assert np.array_equal(
        cast(ArrayLike, result.value), np.array(object=np.array([1, 2, 3]))
    )

    mock_storage.download_file.assert_called_once_with(
        result._hash,
        f"root/{cached_filter._get_model_name()}/{cached_filter.filter._m_hash}",
    )
    assert cached_filter.filter == simple_filter


def test_cache_processed_data_when_cache_data_is_true(mock_storage, simple_filter):
    mock_storage.check_if_exists.return_value = False
    mock_storage.get_root_path.return_value = "root/"

    x = XYData(_hash="input_hash", _path="/input/path", _value=np.array([1, 2, 3]))

    cached_filter = Cached(
        filter=simple_filter, cache_data=True, cache_filter=False, storage=mock_storage
    )

    cached_filter.init()

    result = cached_filter.predict(x)
    mock_storage.upload_file.assert_called_once()

    calls = mock_storage.upload_file.mock_calls
    assert calls[0].kwargs["file_name"] == result._hash
    assert (
        calls[0].kwargs["context"]
        == f"root/{cached_filter._get_model_name()}/{cached_filter.filter._m_hash}"
    )
    assert np.array_equal(pickle.loads(calls[0].kwargs["file"]), np.array([7, 8, 9]))
    assert result._hash == result._hash
    assert (
        result._path
        == f"{cached_filter._get_model_name()}/{cached_filter.filter._m_hash}"
    )
    assert np.array_equal(cast(ArrayLike, result.value), np.array([7, 8, 9]))


def test_use_cached_data_when_exists(mock_storage, simple_filter):
    mock_storage.check_if_exists.return_value = True
    mock_storage.get_root_path.return_value = "root/"
    cached_data = np.array([7, 8, 9])
    mock_storage.download_file.return_value = cached_data

    x = XYData(_hash="input_hash", _path="/input/path", _value=np.array([1, 2, 3]))

    cached_filter = Cached(filter=simple_filter, cache_data=True, storage=mock_storage)
    cached_filter.init()
    result = cached_filter.predict(x)

    assert isinstance(result._value, types.FunctionType)

    mock_storage.upload_file.assert_not_called()
    mock_storage.check_if_exists.assert_called_once_with(
        result._hash,
        context=f"root/{cached_filter._get_model_name()}/{cached_filter.filter._m_hash}",
    )

    assert callable(result._value)

    assert np.array_equal(cast(ArrayLike, result.value), cached_data)
    mock_storage.download_file.assert_called_once_with(
        result._hash,
        f"root/{cached_filter._get_model_name()}/{cached_filter.filter._m_hash}",
    )


def test_overwrite_existing_cached_data(mock_storage, simple_filter):
    mock_storage.check_if_exists.return_value = True
    mock_storage.get_root_path.return_value = "root/"

    x = XYData(_hash="input_hash", _path="/input/path", _value=np.array([1, 2, 3]))
    y = XYData(_hash="target_hash", _path="/target/path", _value=np.array([4, 5, 6]))

    cached_filter = Cached(
        filter=simple_filter,
        cache_data=True,
        cache_filter=True,
        overwrite=True,
        storage=mock_storage,
    )

    # Simulate existing cached model
    mock_storage.download_file.return_value = simple_filter

    cached_filter.fit(x, y)

    # Verify that fit was called even though the model exists (due to overwrite=True)
    assert mock_storage.check_if_exists.called
    assert mock_storage.upload_file.called

    # Simulate existing cached data
    mock_storage.download_file.return_value = np.array([10, 11, 12])

    result = cached_filter.predict(x)

    # Verify that predict was called and new data was cached (due to overwrite=True)
    assert mock_storage.upload_file.called

    assert result._path == f"{cached_filter._get_model_name()}/{simple_filter._m_hash}"
    assert np.array_equal(
        cast(ArrayLike, result.value), np.array([7, 8, 9])
    )  # simple_filter always returns [7, 8, 9]

    # Verify the number of calls
    assert mock_storage.upload_file.call_count == 2  # Once for model, once for data


def test_predict_with_untrained_model(mock_storage, simple_filter):
    x = XYData(_hash="input_hash", _path="/input/path", _value=np.array([1, 2, 3]))

    cached_filter = Cached(filter=simple_filter, cache_data=True, storage=mock_storage)
    setattr(simple_filter, "_m_hash", "")
    # Ensure the filter doesn't have the required attributes

    with pytest.raises(ValueError) as excinfo:
        cached_filter.predict(x)

    assert str(excinfo.value) == "Trainable filter model not trained or loaded"


def test_create_lambda_filter_when_exists(mock_storage, simple_filter):
    mock_storage.check_if_exists.return_value = True
    mock_storage.get_root_path.return_value = "root/"

    x = XYData(_hash="input_hash", _path="/input/path", _value=np.array([1, 2, 3]))
    y = XYData(_hash="target_hash", _path="/target/path", _value=np.array([4, 5, 6]))

    cached_filter = Cached(
        filter=simple_filter, cache_filter=True, storage=mock_storage
    )
    setattr(simple_filter, "_m_hash", "model_hash")
    setattr(simple_filter, "_m_path", "model_path")

    cached_filter.fit(x, y)

    assert cached_filter._lambda_filter is not None
    assert callable(cached_filter._lambda_filter)
    mock_storage.check_if_exists.assert_called_once_with(
        hashcode="model",
        context=f"root/{cached_filter._get_model_name()}/{cached_filter.filter._m_hash}",
    )
    mock_storage.upload_file.assert_not_called()


def test_fit_with_x_and_y(mock_storage, simple_filter):
    mock_storage.check_if_exists.return_value = False
    mock_storage.get_root_path.return_value = "root/"

    x = XYData(_hash="input_hash", _path="/input/path", _value=np.array([1, 2, 3]))
    y = XYData(_hash="target_hash", _path="/target/path", _value=np.array([4, 5, 6]))

    cached_filter = Cached(
        filter=simple_filter, cache_filter=True, storage=mock_storage
    )

    cached_filter.fit(x, y)

    assert hasattr(simple_filter, "_m_hash")
    assert hasattr(simple_filter, "_m_path")
    assert hasattr(simple_filter, "_m_str")
    mock_storage.upload_file.assert_called_once()
    calls = mock_storage.upload_file.mock_calls
    assert calls[0].kwargs["file_name"] == "model"
    assert "input_hash, target_hash" in simple_filter._m_str


def test_fit_with_only_x(mock_storage, simple_filter):
    mock_storage.check_if_exists.return_value = False
    mock_storage.get_root_path.return_value = "root/"

    x = XYData(_hash="input_hash", _path="/input/path", _value=np.array([1, 2, 3]))

    cached_filter = Cached(
        filter=simple_filter, cache_filter=True, storage=mock_storage
    )
    setattr(simple_filter, "_m_hash", "model_hash")
    setattr(simple_filter, "_m_path", "model_path")

    cached_filter.fit(x, None)

    mock_storage.upload_file.assert_called_once()
    calls = mock_storage.upload_file.mock_calls
    assert calls[0].kwargs["file_name"] == "model"
    assert (
        calls[0].kwargs["context"]
        == f"root/{cached_filter._get_model_name()}/{cached_filter.filter._m_hash}"
    )
    assert cached_filter._lambda_filter is None


def test_manage_storage_paths_for_different_models_and_data(
    mock_storage, simple_filter
):
    mock_storage1 = mock_storage()
    mock_storage2 = mock_storage()
    mock_storage1.get_root_path.return_value = "root1/"

    x1 = XYData(_hash="input_hash1", _path="/input/path1", _value=np.array([1, 2, 3]))
    y1 = XYData(_hash="target_hash1", _path="/target/path1", _value=np.array([4, 5, 6]))

    x2 = XYData(_hash="input_hash2", _path="/input/path2", _value=np.array([7, 8, 9]))
    y2 = XYData(
        _hash="target_hash2", _path="/target/path2", _value=np.array([10, 11, 12])
    )

    cached_filter1 = Cached(
        filter=simple_filter, cache_data=True, cache_filter=True, storage=mock_storage1
    )

    # First model
    mock_storage1.check_if_exists.return_value = False
    setattr(simple_filter, "_m_hash", "model_hash1")
    setattr(simple_filter, "_m_path", "model_path1")
    setattr(simple_filter, "_m_str", "model_str1")

    cached_filter1.fit(x1, y1)
    mock_storage1.upload_file.assert_called_with(
        file=ANY,
        file_name="model",
        context=f"root1/{cached_filter1._get_model_name()}/{cached_filter1.filter._m_hash}",
    )

    ret1: XYData = cached_filter1.predict(x1)
    mock_storage1.upload_file.assert_called_with(
        file=ANY,
        file_name=ret1._hash,
        context=f"root1/{cached_filter1._get_model_name()}/{cached_filter1.filter._m_hash}",
    )

    # Second model
    mock_storage1.check_if_exists.return_value = False
    setattr(simple_filter, "_m_hash", "model_hash2")
    setattr(simple_filter, "_m_path", "model_path2")
    setattr(simple_filter, "_m_str", "model_str2")

    cached_filter1.fit(x2, y2)
    mock_storage1.upload_file.assert_called_with(
        file=ANY,
        file_name="model",
        context=f"root1/{cached_filter1._get_model_name()}/{cached_filter1.filter._m_hash}",
    )

    val = cached_filter1.predict(x2)
    mock_storage1.upload_file.assert_called_with(
        file=ANY,
        file_name=val._hash,
        context=f"root1/{cached_filter1._get_model_name()}/{cached_filter1.filter._m_hash}",
    )

    assert mock_storage1.upload_file.call_count == 4

    mock_storage2.check_if_exists.return_value = False
    setattr(simple_filter, "_m_hash", "model_hash1")
    setattr(simple_filter, "_m_path", "model_path1")
    setattr(simple_filter, "_m_str", "model_str1")

    cached_filter2 = Cached(
        filter=simple_filter, cache_data=True, cache_filter=True, storage=mock_storage1
    )
    mock_storage2.get_root_path.return_value = "root2/"
    cached_filter2.fit(x1, y1)
    mock_storage2.upload_file.assert_called_with(
        file=ANY,
        file_name="model",
        context=f"root2/{cached_filter2._get_model_name()}/{cached_filter2.filter._m_hash}",
    )

    ret2 = cached_filter2.predict(x1)
    mock_storage2.upload_file.assert_called_with(
        file=ANY,
        file_name=ret2._hash,
        context=f"root2/{cached_filter2._get_model_name()}/{cached_filter2.filter._m_hash}",
    )

    # Second model
    mock_storage2.check_if_exists.return_value = False
    setattr(simple_filter, "_m_hash", "model_hash2")
    setattr(simple_filter, "_m_path", "model_path2")
    setattr(simple_filter, "_m_str", "model_str2")

    cached_filter2.fit(x2, y2)
    mock_storage2.upload_file.assert_called_with(
        file=ANY,
        file_name="model",
        context=f"root2/{cached_filter2._get_model_name()}/{cached_filter2.filter._m_hash}",
    )

    ret2 = cached_filter2.predict(x2)
    mock_storage2.upload_file.assert_called_with(
        file=ANY,
        file_name=ret2._hash,
        context=f"root2/{cached_filter2._get_model_name()}/{cached_filter2.filter._m_hash}",
    )

    assert mock_storage2.upload_file.call_count == 8


def test_non_trainable_filter_initialization(non_trainable_filter, mock_storage):
    x = XYData(_hash="input_hash", _path="/input/path", _value=np.array([1, 2, 3]))

    # Sin inicialización
    with pytest.raises(
        ValueError, match="Trainable filter model not trained or loaded"
    ):
        non_trainable_filter.predict(x)

    # Con inicialización
    non_trainable_filter.init()
    result = non_trainable_filter.predict(x)
    assert np.array_equal(result.value, np.array([7, 8, 9]))


def test_cached_non_trainable_filter(non_trainable_filter, mock_storage):
    x = XYData(_hash="input_hash", _path="/input/path", _value=np.array([1, 2, 3]))
    y = XYData(_hash="target_hash", _path="/target/path", _value=np.array([4, 5, 6]))

    mock_storage.check_if_exists.return_value = False

    cached_filter = Cached(
        filter=non_trainable_filter, cache_filter=True, storage=mock_storage
    )

    # Intentar entrenar el filtro no entrenable
    with pytest.raises(NotTrainableFilterError):
        cached_filter.fit(x, y)

    # Verificar que no se ha intentado guardar el modelo
    mock_storage.upload_file.assert_not_called()

    # Inicializar el filtro
    cached_filter.init()

    # La predicción debería funcionar ahora
    cached_filter.predict(x)


def test_cached_non_trainable_filter_init():
    non_trainable_filter = NonTrainableFilter()
    cached_filter = Cached(filter=non_trainable_filter, cache_filter=False)

    # Verificar que se puede llamar a init() sin errores
    cached_filter.init()

    assert cached_filter.filter._m_hash is not None
    assert cached_filter.filter._m_str is not None
    assert cached_filter.filter._m_path is not None


def test_cached_non_trainable_filter_fit():
    non_trainable_filter = NonTrainableFilter()
    cached_filter = Cached(filter=non_trainable_filter, cache_filter=False)

    x = XYData.mock([1, 2, 3])
    y = XYData.mock([4, 5, 6])

    # Verificar que se puede llamar a fit() sin errores
    with pytest.raises(NotTrainableFilterError):
        cached_filter.fit(x, y)

    non_trainable_filter2 = NonTrainableFilter()

    non_trainable_filter2.init()

    # Asegurarse de que el filtro interno no ha cambiado
    assert cached_filter.filter._m_hash == non_trainable_filter2._m_hash
    assert cached_filter.filter._m_str == non_trainable_filter2._m_str
    assert cached_filter.filter._m_path == non_trainable_filter2._m_path


def test_cached_non_trainable_filter_predict():
    non_trainable_filter = NonTrainableFilter()
    cached_filter = Cached(filter=non_trainable_filter, cache_filter=False)

    x = XYData.mock([1, 2, 3])

    cached_filter.init()

    # Verificar que se puede llamar a predict() sin errores
    result = cached_filter.predict(x)

    assert result is not None  # El filtro no trainable devuelve los mismos datos


def test_cached_non_trainable_filter_in_pipeline():
    from framework3.plugins.pipelines import F3Pipeline

    non_trainable_filter = NonTrainableFilter()
    cached_filter = Cached(filter=non_trainable_filter, cache_filter=False)

    pipeline = F3Pipeline(filters=[cached_filter])

    x = XYData.mock([1, 2, 3])
    y = XYData.mock([4, 5, 6])

    # Verificar que se puede inicializar y ajustar el pipeline sin errores
    pipeline.init()
    pipeline.fit(x, y)

    # Verificar que se puede predecir sin errores
    result = pipeline.predict(x)

    assert result is not None  # El filtro no trainable devuelve los mismos datos


def test_pipeline_with_not_trainable_filters():
    from framework3.plugins.pipelines import F3Pipeline

    non_trainable_filter = NonTrainableFilter()
    cached_filter = Cached(filter=non_trainable_filter, cache_filter=False)

    pipeline = F3Pipeline(filters=[cached_filter])

    x = XYData.mock([1, 2, 3])

    # Verificar que se puede inicializar y ajustar el pipeline sin errores
    pipeline.init()

    # Verificar que se puede predecir sin errores
    result = pipeline.predict(x)

    assert any(result.value != x.value)
    assert result is not None  # El filtro no trainable devuelve los mismos datos
