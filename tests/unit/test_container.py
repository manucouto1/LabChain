import pytest
from labchain.container.container import Container
from labchain.base import (
    BaseFilter,
    BasePipeline,
    BaseMetric,
    BaseStorage,
)


@pytest.fixture
def container():
    return Container()


def test_bind_unregistered_decorator(container):
    class UnregisteredClass:
        pass

    with pytest.raises(NotImplementedError) as excinfo:
        Container.bind()(UnregisteredClass)

    assert str(excinfo.value) == "No decorator registered for UnregisteredClass"


def test_bind_base_filter(container):
    @Container.bind()
    class TestFilter(BaseFilter):
        def fit(self, x, y):
            pass

        def predict(self, x):
            return x

    assert "TestFilter" in Container.ff
    assert "TestFilter" in Container.pif
    assert Container.ff["TestFilter"] == TestFilter
    assert Container.pif["TestFilter"] == TestFilter


def test_bind_base_pipeline(container):
    @Container.bind()
    class TestPipeline(BasePipeline):
        def fit(self, x, y):
            pass

        def predict(self, x):
            return x

        def init(self):
            pass

        def start(self, x, y, X_):
            return None

        def log_metrics(self):
            pass

        def finish(self):
            pass

        def evaluate(self, x_data, y_true, y_pred):
            return {}

    assert "TestPipeline" in Container.pf
    assert "TestPipeline" in Container.pif
    assert Container.pf["TestPipeline"] == TestPipeline
    assert Container.pif["TestPipeline"] == TestPipeline


def test_bind_base_metric(container):
    @Container.bind()
    class TestMetric(BaseMetric):
        def evaluate(self, x_data, y_true, y_pred):
            return 0.5

    assert "TestMetric" in Container.mf
    assert "TestMetric" in Container.pif
    assert Container.mf["TestMetric"] == TestMetric
    assert Container.pif["TestMetric"] == TestMetric


def test_bind_base_storage(container):
    @Container.bind()
    class TestStorage(BaseStorage):
        def upload_file(self, file, file_name, context, direct_stream=False):
            pass

        def download_file(self, hashcode, context):
            pass

        def list_stored_files(self, context):
            return []

        def delete_file(self, hashcode, context):
            pass

        def check_if_exists(self, hashcode, context):
            return False

        def get_file_by_hashcode(self, hashcode, context):
            pass

    assert "TestStorage" in Container.sf
    assert "TestStorage" in Container.pif
    assert Container.sf["TestStorage"] == TestStorage
    assert Container.pif["TestStorage"] == TestStorage


def test_bind_returns_original_function(container):
    @Container.bind()
    class TestFilter(BaseFilter):
        def fit(self, x, y):
            pass

        def predict(self, x):
            return x

    assert isinstance(TestFilter, type)
    assert issubclass(TestFilter, BaseFilter)
    assert "TestFilter" in Container.ff
    assert "TestFilter" in Container.pif
    assert Container.ff["TestFilter"] is TestFilter
    assert Container.pif["TestFilter"] is TestFilter


def test_multiple_bindings_of_different_types(container):
    @Container.bind()
    class TestFilter(BaseFilter):
        def fit(self, x, y):
            pass

        def predict(self, x):
            return x

    @Container.bind()
    class TestPipeline(BasePipeline):
        def fit(self, x, y):
            pass

        def predict(self, x):
            return x

        def init(self):
            pass

        def start(self, x, y, X_):
            return None

        def log_metrics(self):
            pass

        def finish(self):
            pass

        def evaluate(self, x_data, y_true, y_pred):
            return {}

    @Container.bind()
    class TestMetric(BaseMetric):
        def evaluate(self, x_data, y_true, y_pred):
            return 0.5

    @Container.bind()
    class TestStorage(BaseStorage):
        def upload_file(self, file, file_name, context, direct_stream=False):
            pass

        def download_file(self, hashcode, context):
            pass

        def list_stored_files(self, context):
            return []

        def delete_file(self, hashcode, context):
            pass

        def check_if_exists(self, hashcode, context):
            return False

        def get_file_by_hashcode(self, hashcode, context):
            pass

    assert "TestFilter" in Container.ff and Container.ff["TestFilter"] == TestFilter
    assert (
        "TestPipeline" in Container.pf and Container.pf["TestPipeline"] == TestPipeline
    )
    assert "TestMetric" in Container.mf and Container.mf["TestMetric"] == TestMetric
    assert "TestStorage" in Container.sf and Container.sf["TestStorage"] == TestStorage

    assert "TestFilter" in Container.pif and Container.pif["TestFilter"] == TestFilter
    assert (
        "TestPipeline" in Container.pif
        and Container.pif["TestPipeline"] == TestPipeline
    )
    assert "TestMetric" in Container.pif and Container.pif["TestMetric"] == TestMetric
    assert (
        "TestStorage" in Container.pif and Container.pif["TestStorage"] == TestStorage
    )


def test_separate_factories_for_base_types():
    @Container.bind()
    class TestFilter(BaseFilter):
        def fit(self, x, y):
            pass

        def predict(self, x):
            return x

    @Container.bind()
    class TestPipeline(BasePipeline):
        def fit(self, x, y):
            pass

        def predict(self, x):
            return x

        def init(self):
            pass

        def start(self, x, y, X_):
            return None

        def log_metrics(self):
            pass

        def finish(self):
            pass

        def evaluate(self, x_data, y_true, y_pred):
            return {}

    @Container.bind()
    class TestMetric(BaseMetric):
        def evaluate(self, x_data, y_true, y_pred):
            return 0.5

    @Container.bind()
    class TestStorage(BaseStorage):
        def upload_file(self, file, file_name, context, direct_stream=False):
            pass

        def download_file(self, hashcode, context):
            pass

        def list_stored_files(self, context):
            return []

        def delete_file(self, hashcode, context):
            pass

        def check_if_exists(self, hashcode, context):
            return False

        def get_file_by_hashcode(self, hashcode, context):
            pass

    assert (
        "TestFilter" in Container.ff
        and "TestFilter" not in Container.pf
        and "TestFilter" not in Container.mf
        and "TestFilter" not in Container.sf
    )
    assert (
        "TestPipeline" in Container.pf
        and "TestPipeline" not in Container.ff
        and "TestPipeline" not in Container.mf
        and "TestPipeline" not in Container.sf
    )
    assert (
        "TestMetric" in Container.mf
        and "TestMetric" not in Container.ff
        and "TestMetric" not in Container.pf
        and "TestMetric" not in Container.sf
    )
    assert (
        "TestStorage" in Container.sf
        and "TestStorage" not in Container.ff
        and "TestStorage" not in Container.pf
        and "TestStorage" not in Container.mf
    )
    assert all(
        name in Container.pif
        for name in ["TestFilter", "TestPipeline", "TestMetric", "TestStorage"]
    )
