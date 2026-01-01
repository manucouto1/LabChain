import pytest
import numpy as np
from sklearn import datasets
from unittest.mock import MagicMock, ANY
from labchain import MonoPipeline
from labchain.plugins.filters.cache.cached_filter import Cached
from labchain.base import BaseFilter, XYData
from labchain.plugins.filters.transformation.pca import PCAPlugin
from labchain.plugins.filters.transformation.scaler import StandardScalerPlugin
from labchain.plugins.pipelines.parallel.parallel_hpc_pipeline import HPCPipeline
from labchain.plugins.pipelines.sequential.f3_pipeline import F3Pipeline
from labchain.plugins.filters.classification.svm import ClassifierSVMPlugin
from labchain.plugins.filters.classification.knn import KnnFilter
from labchain.plugins.storage.local_storage import LocalStorage
from rich import print


@pytest.fixture
def test_data():
    iris = datasets.load_iris()
    X = XYData(
        _hash="Iris X data",
        _path="datasets",
        _value=iris.data,  # type: ignore
    )
    y = XYData(
        _hash="Iris y data",
        _path="datasets",
        _value=iris.target,  # type: ignore
    )
    return X, y


@pytest.fixture
def mock_storage():
    storage = MagicMock(spec=LocalStorage)
    storage.get_root_path.return_value = "root/"
    return storage


@pytest.fixture
def store_cached_pipelines(test_data, mock_storage):
    pipeline1 = Cached(
        filter=F3Pipeline(
            filters=[StandardScalerPlugin(), PCAPlugin(n_components=2), KnnFilter()],
            metrics=[],
        ),
        cache_data=True,
        cache_filter=True,
        overwrite=True,
        storage=mock_storage,
    )
    pipeline1.fit(*test_data)
    print("_" * 200)
    predictions1 = pipeline1.predict(test_data[0])
    return pipeline1, predictions1, mock_storage


def test_cached_pipeline_storage_interactions(store_cached_pipelines, test_data):
    pipeline, predictions, mock_storage = store_cached_pipelines

    x = test_data[0]
    y = test_data[1]
    # Comprobar que se han guardado los filtros
    filter = pipeline.filter
    # prefit
    _, m_path, _ = filter._pre_fit(x, y)
    # pre-predict
    filter._pre_predict(x)

    mock_storage.upload_file.assert_any_call(
        file=ANY, context=f"root/{m_path}", file_name="model"
    )


def test_cached_squential_pipeline_data_is_the_same_as_last_filter(
    store_cached_pipelines, test_data
):
    cached, predictions, mock_storage = store_cached_pipelines
    x = test_data[0]
    y = test_data[1]

    pipeline = cached.filter
    filters_x = x

    for f in pipeline.filters:
        f._pre_fit(filters_x, y)
        filters_x = f._pre_predict(filters_x)

    filters_x = x
    for f in pipeline.filters:
        filters_x = f._pre_predict(filters_x)

    assert filters_x._hash == predictions._hash
    assert filters_x._path == predictions._path
    assert pipeline.filters[-1]._m_hash != pipeline._m_hash


def test_cached_parallel_pipeline_data_is_correct(store_cached_pipelines, test_data):
    cached_pipeline, predictions, mock_storage = store_cached_pipelines
    x, y = test_data

    parallel_pipeline = HPCPipeline(
        filters=[
            F3Pipeline(
                filters=[
                    StandardScalerPlugin(),
                    PCAPlugin(n_components=3),
                    ClassifierSVMPlugin(),
                ],
                metrics=[],
            ),
            F3Pipeline(
                filters=[
                    StandardScalerPlugin(),
                    PCAPlugin(n_components=3),
                    ClassifierSVMPlugin(),
                ],
                metrics=[],
            ),
        ],
        app_name="test_parallel",
    )

    cached_parallel = Cached(
        filter=parallel_pipeline,
        cache_data=True,
        cache_filter=True,
        overwrite=True,
        storage=mock_storage,
    )

    cached_parallel.fit(x, y)
    parallel_predictions = cached_parallel.predict(x)

    # Check that each filter in each pipeline has correct _hash and _path
    generated_data = []
    for pipeline in parallel_pipeline.filters:
        pipe_data = []
        assert hasattr(pipeline, "_m_hash")
        assert hasattr(pipeline, "_m_path")

        for filter in pipeline.filters:
            assert hasattr(filter, "_m_hash")
            assert hasattr(filter, "_m_path")
            pipe_data.append(
                (
                    pipeline._m_str,
                    pipeline._m_hash,
                    pipeline._m_path,
                    filter._m_str,
                    filter._m_hash,
                    filter._m_path,
                )
            )

        generated_data.append(pipe_data)

    assert (
        generated_data[0] == generated_data[1]
    )  # Check that the data is the same across the pipelines so they are generated in parallel order

    print(x.value.shape)
    print(parallel_predictions.value.shape)
    assert (
        parallel_predictions.value.shape[1] == 2
    )  # Check that the predictions have the same shape as the input data


def test_cached_filter_in_parallel_pipeline_consistency():
    class DummyFilter(BaseFilter):
        def __init__(self, name):
            super().__init__()
            self.name = name

        def predict(self, x: XYData) -> XYData:
            return XYData.mock(x.value * 2)

    # Mock storage
    mock_storage = MagicMock(spec=LocalStorage)

    # Set to store hashcodes
    stored_hashcodes = set()

    # Configure mock_storage to return XYData compatible objects
    def mock_download_file(*args, **kwargs):
        return np.array([[2, 4, 6]])

    # Simulate upload_file
    def mock_upload_file(file, file_name, context, direct_stream=False):
        print("TTTTT--- Uploading file: ", file_name, " ---TTTTT")
        stored_hashcodes.add(file_name)
        return file_name

    # Simulate check_if_exists
    def mock_check_if_exists(hashcode, context):
        return hashcode in stored_hashcodes

    mock_storage.upload_file.side_effect = mock_upload_file
    mock_storage.check_if_exists.side_effect = mock_check_if_exists
    mock_storage.download_file.side_effect = mock_download_file

    pipeline_a = F3Pipeline(
        filters=[
            Cached(
                DummyFilter("one"),
                cache_data=True,
                cache_filter=False,
                storage=mock_storage,
            ),
            DummyFilter("two_a"),
        ]
    )
    pipeline_b = F3Pipeline(
        filters=[
            Cached(
                DummyFilter("one"),
                cache_data=True,
                cache_filter=False,
                storage=mock_storage,
            ),
            DummyFilter("two_b"),
        ]
    )

    pipeline_c = F3Pipeline(
        filters=[
            Cached(
                DummyFilter("one"),
                cache_data=True,
                cache_filter=False,
                storage=mock_storage,
            ),
            DummyFilter("two_c"),
        ]
    )

    parallel_mono_1 = MonoPipeline(filters=[pipeline_a, pipeline_b])

    parallel_mono_2 = MonoPipeline(
        filters=[
            pipeline_b,
            pipeline_a,
        ]
    )
    parallel_mono_3 = MonoPipeline(
        filters=[
            pipeline_c,
            pipeline_b,
            pipeline_a,
        ]
    )

    x = XYData.mock(np.array([[1, 2, 3]]))

    parallel_mono_1.predict(x)
    parallel_mono_2.predict(x)
    parallel_mono_3.predict(x)

    # Check that upload was called only once for the cached filter
    assert mock_storage.upload_file.call_count == 1

    # Reset mock to clear call history
    mock_storage.reset_mock()

    # Second set of predictions
    parallel_mono_1.predict(x)
    parallel_mono_2.predict(x)
    parallel_mono_3.predict(x)

    # Check that upload was not called again
    assert mock_storage.upload_file.call_count == 0

    # Check that get_file was called instead
    assert mock_storage.download_file.call_count == 7  # Once for each MonoPipeline
