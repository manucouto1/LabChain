import pytest
import typeguard

from labchain.base import BasePipeline
from labchain.plugins.filters.cache.cached_filter import Cached
from labchain.plugins.metrics.classification import F1
from labchain.plugins.pipelines.parallel.parallel_hpc_pipeline import HPCPipeline
from labchain.plugins.pipelines.sequential.f3_pipeline import F3Pipeline
from tests.unit.test_combiner_pipeline import DummyFilter
from sklearn.datasets import load_iris
from labchain.base import XYData
from labchain.plugins.filters.classification import ClassifierSVMPlugin, KnnFilter
from labchain.plugins.filters.transformation.pca import PCAPlugin
from labchain.plugins.filters.transformation.scaler import StandardScalerPlugin
from rich import print as rprint


def test_map_reduce_combiner_pipeline_initialization():
    filters = [DummyFilter(), DummyFilter()]
    app_name = "test_app"
    master = "local[2]"

    pipeline = HPCPipeline(filters=filters, app_name=app_name, master=master)

    assert isinstance(pipeline, HPCPipeline)
    assert isinstance(pipeline, BasePipeline)
    assert len(pipeline.filters) == 2
    assert all(isinstance(f, DummyFilter) for f in pipeline.filters)


def test_map_reduce_combiner_pipeline_invalid_filters():
    invalid_filters = [1, "not a filter", {}]  # Invalid filters
    app_name = "test_app"

    with pytest.raises(typeguard.TypeCheckError) as excinfo:
        HPCPipeline(filters=invalid_filters, app_name=app_name)

    assert "is not an instance of labchain.base.base_clases.BaseFilter" in str(
        excinfo.value
    )


def test_map_reduce_combiner_pipeline_fit():
    # Load Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target  # type: ignore

    # Create real filters
    pipeline = HPCPipeline(
        filters=[
            F3Pipeline(
                filters=[
                    StandardScalerPlugin(),
                    PCAPlugin(n_components=2),
                    KnnFilter(),
                ],
                metrics=[],
            ),
            F3Pipeline(
                filters=[
                    StandardScalerPlugin(),
                    PCAPlugin(n_components=1),
                    KnnFilter(),
                ],
                metrics=[],
            ),
            F3Pipeline(
                filters=[
                    StandardScalerPlugin(),
                    PCAPlugin(n_components=2),
                    ClassifierSVMPlugin(),
                ],
                metrics=[],
            ),
            F3Pipeline(
                filters=[
                    StandardScalerPlugin(),
                    PCAPlugin(n_components=1),
                    ClassifierSVMPlugin(),
                ],
                metrics=[],
            ),
        ],
        app_name="test_app",
    )

    print(pipeline)

    # Create XYData objects
    x = XYData(_hash="input_hash", _path="/input/path", _value=X)
    y = XYData(_hash="target_hash", _path="/target/path", _value=y)

    pipeline.fit(x, y)

    assert len(pipeline.filters) == 4

    # Check if fit was called on all filters
    for filter in pipeline.filters:
        assert filter.filters

        for f in filter.filters:
            assert hasattr(f, "_m_hash")  # This attribute is set after fitting
            assert hasattr(f, "_m_path")  # This attribute is set after fitting
            assert hasattr(f, "_m_str")  # This attribute is set after fitting

    # Test prediction
    result = pipeline.predict(x)

    assert isinstance(result, XYData)
    assert result.value is not None
    assert result.value.shape == (150, 4)  # We used 10 samples for prediction

    final_pieline = F3Pipeline(
        filters=[pipeline, ClassifierSVMPlugin()], metrics=[F1(average="weighted")]
    )

    final_pieline.fit(x, y)

    result = final_pieline.predict(x)

    rprint(final_pieline.evaluate(x, y, result))


def test_caching_map_reduce_combiner_pipeline():
    iris = load_iris()
    X, y = iris.data, iris.target  # type: ignore

    # Create real filters
    pipeline = HPCPipeline(
        filters=[
            F3Pipeline(
                filters=[
                    StandardScalerPlugin(),
                    PCAPlugin(n_components=2),
                    KnnFilter(),
                ],
                metrics=[],
            ),
            F3Pipeline(
                filters=[
                    StandardScalerPlugin(),
                    PCAPlugin(n_components=1),
                    KnnFilter(),
                ],
                metrics=[],
            ),
            F3Pipeline(
                filters=[
                    StandardScalerPlugin(),
                    PCAPlugin(n_components=2),
                    ClassifierSVMPlugin(),
                ],
                metrics=[],
            ),
            F3Pipeline(
                filters=[
                    StandardScalerPlugin(),
                    PCAPlugin(n_components=1),
                    ClassifierSVMPlugin(),
                ],
                metrics=[],
            ),
        ],
        app_name="test_app",
    )

    print(pipeline)

    pipeline = Cached(
        filter=pipeline,
        cache_data=True,
        cache_filter=True,
        overwrite=True,
    )

    # Create XYData objects
    x = XYData(_hash="input_hash", _path="/input/path", _value=X)
    y = XYData(_hash="target_hash", _path="/target/path", _value=y)

    pipeline.fit(x, y)

    for filter in pipeline.filter.filters:
        assert filter.filters

        for f in filter.filters:
            assert hasattr(f, "_m_hash")  # This attribute is set after fitting
            assert hasattr(f, "_m_path")  # This attribute is set after fitting
            assert hasattr(f, "_m_str")  # This attribute is set after fitting

    # Test prediction
    result = pipeline.predict(x)

    print(result.value.shape)
