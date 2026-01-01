from typing import Optional
import numpy as np
from sklearn import datasets
from labchain import HPCPipeline, MonoPipeline
from labchain.base.base_types import XYData
from labchain.base.exceptions import NotTrainableFilterError
from labchain.container import Container
from labchain.container.container import BaseFilter
from labchain.plugins.filters.classification.svm import ClassifierSVMPlugin
from labchain.plugins.filters.grid_search.cv_grid_search import GridSearchCVPlugin
from labchain.plugins.filters.transformation.pca import PCAPlugin
from labchain.plugins.metrics.classification import F1, Precission, Recall
from labchain.plugins.pipelines.sequential.f3_pipeline import F3Pipeline


class NonTrainableFilter(BaseFilter):
    def __init__(self):
        super().__init__()

    def predict(self, x: XYData) -> XYData:
        return x


def test_pipeline_iris_dataset():
    iris = datasets.load_iris()

    pipeline = F3Pipeline(
        filters=[
            PCAPlugin(n_components=1),
            GridSearchCVPlugin(
                ClassifierSVMPlugin,
                ClassifierSVMPlugin.item_grid(C=[1.0, 10], kernel=["rbf"]),
                scoring="f1_weighted",
                cv=2,
            ),
        ],
        metrics=[
            F1(average="weighted"),
            Precission(average="weighted"),
            Recall(average="weighted"),
        ],
    )

    X = XYData(
        _hash="Iris X data",
        _path=Container.storage.get_root_path(),
        _value=iris.data,  # type: ignore
    )
    y = XYData(
        _hash="Iris y data",
        _path=Container.storage.get_root_path(),
        _value=iris.target,  # type: ignore
    )

    pipeline.fit(X, y)
    prediction = pipeline.predict(x=X)
    print(prediction.value)
    evaluate = pipeline.evaluate(X, y, prediction)

    assert isinstance(prediction.value, np.ndarray)
    assert prediction.value.shape == (150,)
    assert isinstance(evaluate, dict)
    assert "F1" in evaluate
    assert "Precission" in evaluate
    assert "Recall" in evaluate
    assert all(0 <= score <= 1 for score in evaluate.values())


def test_pipeline_different_feature_counts():
    # Create datasets with different numbers of features
    iris = datasets.load_iris()

    X_full = XYData(
        _hash="Iris X data",
        _path=Container.storage.get_root_path(),
        _value=iris.data,  # type: ignore
    )
    X_reduced = XYData(
        _hash="Iris X reduced data",
        _path=Container.storage.get_root_path(),
        _value=iris.data[:, :3],  # type: ignore
    )

    y = XYData(
        _hash="Iris y data",
        _path=Container.storage.get_root_path(),
        _value=iris.target,  # type: ignore
    )

    pipeline = F3Pipeline(
        filters=[
            PCAPlugin(n_components=1),
            GridSearchCVPlugin(
                ClassifierSVMPlugin,
                ClassifierSVMPlugin.item_grid(C=[1.0, 10], kernel=["rbf"]),
                scoring="f1_weighted",
                cv=2,
            ),
        ],
        metrics=[
            F1(average="weighted"),
            Precission(average="weighted"),
            Recall(average="weighted"),
        ],
    )

    # Test with full dataset
    pipeline.fit(X_full, y=y)
    prediction_full = pipeline.predict(x=X_full)

    evaluate_full = pipeline.evaluate(X_full, y, y_pred=prediction_full)

    # Test with reduced dataset

    pipeline.fit(X_reduced, y)
    prediction_reduced = pipeline.predict(x=X_reduced)
    evaluate_reduced = pipeline.evaluate(X_reduced, y, y_pred=prediction_reduced)

    assert isinstance(prediction_full.value, np.ndarray)
    assert isinstance(prediction_reduced.value, np.ndarray)
    assert prediction_full.value.shape == prediction_reduced.value.shape == (150,)
    assert isinstance(evaluate_full, dict)
    assert isinstance(evaluate_reduced, dict)
    assert (
        set(evaluate_full.keys())
        == set(evaluate_reduced.keys())
        == {"F1", "Precission", "Recall"}
    )
    assert all(0 <= score <= 1 for score in evaluate_full.values())
    assert all(0 <= score <= 1 for score in evaluate_reduced.values())


def test_grid_search_with_specified_parameters():
    iris = datasets.load_iris()
    X = XYData(
        _hash="Iris X data",
        _path=Container.storage.get_root_path(),
        _value=iris.data,  # type: ignore
    )
    y = XYData(
        _hash="Iris y data",
        _path=Container.storage.get_root_path(),
        _value=iris.target,  # type: ignore
    )

    pipeline = F3Pipeline(
        filters=[
            PCAPlugin(n_components=1),
            GridSearchCVPlugin(
                filterx=ClassifierSVMPlugin,
                param_grid=ClassifierSVMPlugin.item_grid(C=[1.0, 10], kernel=["rbf"]),
                scoring="f1_weighted",
                cv=2,
            ),
        ],
        metrics=[
            F1(average="weighted"),
            Precission(average="weighted"),
            Recall(average="weighted"),
        ],
    )

    pipeline.fit(X, y)

    # Check if the GridSearchCVPlugin is present in the pipeline
    grid_search_plugin = next(
        (
            plugin
            for plugin in pipeline.filters
            if isinstance(plugin, GridSearchCVPlugin)
        ),
        None,
    )
    assert (
        grid_search_plugin is not None
    ), "GridSearchCVPlugin not found in the pipeline"

    print(grid_search_plugin._clf)  # type: ignore

    # Check if the grid search parameters are correctly set
    param_grid = grid_search_plugin._clf.param_grid  # type: ignore
    assert "ClassifierSVMPlugin__C" in param_grid, "C parameter not found in param_grid"
    assert param_grid["ClassifierSVMPlugin__C"] == [
        1.0,
        10,
    ], "C parameter values are incorrect"
    assert (
        "ClassifierSVMPlugin__kernel" in param_grid
    ), "kernel parameter not found in param_grid"
    assert param_grid["ClassifierSVMPlugin__kernel"] == [
        "rbf"
    ], "kernel parameter values are incorrect"

    # Check if the scoring and cv parameters are correctly set
    assert (
        grid_search_plugin._clf.scoring == "f1_weighted"  # type: ignore
    ), "Incorrect scoring parameter"  # type: ignore
    assert grid_search_plugin._clf.cv == 2, "Incorrect cv parameter"  # type: ignore

    # Verify that the best parameters have been found
    assert hasattr(
        grid_search_plugin._clf, "best_params_"
    ), "Best parameters not found after fitting"
    assert isinstance(
        grid_search_plugin._clf.best_params_, dict
    ), "Best parameters should be a dictionary"
    assert (
        "ClassifierSVMPlugin__C" in grid_search_plugin._clf.best_params_
    ), "C parameter not found in best_params_"
    assert (
        "ClassifierSVMPlugin__kernel" in grid_search_plugin._clf.best_params_
    ), "kernel parameter not found in best_params_"


def test_f3_pipeline_with_non_trainable_filter():
    """Test que filtros no-entrenables funcionan en F3Pipeline con lazy initialization"""

    class NonTrainableFilterLocal(BaseFilter):
        def __init__(self):
            super().__init__()

        def fit(self, x: XYData, y: Optional[XYData]) -> None:
            raise NotTrainableFilterError("This filter is not trainable")

        def predict(self, x: XYData) -> XYData:
            return x

    non_trainable_filter = NonTrainableFilterLocal()
    pipeline = F3Pipeline(filters=[non_trainable_filter], metrics=[])

    x = XYData.mock([1, 2, 3])
    y = XYData.mock([4, 5, 6])

    # El filtro está inicializado automáticamente (lazy)
    assert hasattr(
        non_trainable_filter, "_m_hash"
    ), "Filter should be initialized automatically"
    assert non_trainable_filter._m_hash != "", "Hash should not be empty"

    # fit() debería manejar el NotTrainableFilterError correctamente
    pipeline.fit(x, y)

    # predict() debería funcionar
    result = pipeline.predict(x)
    assert np.array_equal(
        result.value, x.value
    ), "Non-trainable filter should return input unchanged"


def test_f3_pipeline_hash_changes_after_fit():
    """Test que el hash de filtros entrenables cambia después de fit()"""

    class TrainableFilterLocal(BaseFilter):
        def __init__(self):
            super().__init__()
            self.is_fitted = False

        def fit(self, x: XYData, y: Optional[XYData]) -> None:
            self.is_fitted = True

        def predict(self, x: XYData) -> XYData:
            if not self.is_fitted:
                raise ValueError("Must fit before predict")
            return x

    trainable_filter = TrainableFilterLocal()
    pipeline = F3Pipeline(filters=[trainable_filter], metrics=[])

    x = XYData.mock([1, 2, 3])
    y = XYData.mock([4, 5, 6])

    # Guardar hash inicial
    initial_hash = trainable_filter._m_hash
    assert initial_hash != "", "Initial hash should not be empty"

    # Después de fit, el hash debería cambiar
    pipeline.fit(x, y)
    assert trainable_filter._m_hash != initial_hash, "Hash should change after fit()"
    assert trainable_filter.is_fitted, "Filter should be fitted"

    # predict() debería funcionar
    result = pipeline.predict(x)
    assert result is not None


def test_parallel_mono_pipeline_with_non_trainable_filter():
    """Test MonoPipeline con filtros no-entrenables usando lazy initialization"""

    class NonTrainableFilterLocal(BaseFilter):
        def __init__(self):
            super().__init__()

        def predict(self, x: XYData) -> XYData:
            return x

    non_trainable_filter = NonTrainableFilterLocal()
    pipeline = MonoPipeline(filters=[non_trainable_filter, non_trainable_filter])

    x = XYData.mock([1, 2, 3])
    y = XYData.mock([4, 5, 6])

    # El filtro está inicializado automáticamente
    assert hasattr(
        non_trainable_filter, "_m_hash"
    ), "Filter should be initialized automatically"
    assert non_trainable_filter._m_hash != "", "Hash should not be empty"

    # fit() debería manejar filtros no-entrenables
    pipeline.fit(x, y)

    # predict() debería funcionar
    result = pipeline.predict(x)

    assert (
        result.value.shape[-1] == 2
    ), "Non-trainable filter should return input doubled last dimension"


def test_parallel_hpc_pipeline_with_non_trainable_filter():
    """Test HPCPipeline con filtros no-entrenables usando lazy initialization"""
    non_trainable_filter = NonTrainableFilter()
    pipeline = HPCPipeline(
        app_name="test_parallel", filters=[non_trainable_filter, non_trainable_filter]
    )

    x = XYData.mock([1, 2, 3])
    y = XYData.mock([4, 5, 6])

    # El filtro está inicializado automáticamente
    assert hasattr(
        non_trainable_filter, "_m_hash"
    ), "Filter should be initialized automatically"
    assert non_trainable_filter._m_hash != "", "Hash should not be empty"

    # fit() debería manejar filtros no-entrenables
    pipeline.fit(x, y)

    # predict() debería funcionar
    result = pipeline.predict(x)

    assert (
        result.value.shape[-1] == 2
    ), "Non-trainable filter should return input doubled last dimension"


def test_pipeline_mixed_trainable_and_non_trainable_filters():
    """Test pipeline con mezcla de filtros entrenables y no-entrenables"""

    class TrainableFilterLocal(BaseFilter):
        def __init__(self):
            super().__init__()
            self.is_fitted = False

        def fit(self, x: XYData, y: Optional[XYData]) -> None:
            self.is_fitted = True

        def predict(self, x: XYData) -> XYData:
            if not self.is_fitted:
                raise ValueError("Must fit before predict")
            # Doblar valores para verificar que se ejecutó
            return XYData(
                _hash=x._hash + "_doubled",
                _path=x._path,
                _value=list(map(lambda i: i * 2, x.value)),
            )

    class NonTrainableFilterLocal(BaseFilter):
        def __init__(self):
            super().__init__()

        # def fit(self, x: XYData, y: Optional[XYData]) -> None:
        #     raise NotTrainableFilterError("This filter is not trainable")

        def predict(self, x: XYData) -> XYData:
            # Sumar 1 para verificar que se ejecutó
            return XYData(
                _hash=x._hash + "_plus_one",
                _path=x._path,
                _value=list(map(lambda i: i + 1, x.value)),
            )

    trainable = TrainableFilterLocal()
    non_trainable = NonTrainableFilterLocal()

    pipeline = F3Pipeline(filters=[trainable, non_trainable], metrics=[])

    x = XYData.mock([1, 2, 3])
    y = XYData.mock([4, 5, 6])

    # Ambos filtros están inicializados automáticamente
    assert trainable._m_hash != ""
    assert non_trainable._m_hash != ""

    initial_trainable_hash = trainable._m_hash
    initial_non_trainable_hash = non_trainable._m_hash

    # fit() debería entrenar solo el filtro entrenable
    pipeline.fit(x, y)

    # El hash del filtro entrenable debería cambiar
    assert trainable._m_hash != initial_trainable_hash
    # El hash del filtro no-entrenable NO debería cambiar
    assert non_trainable._m_hash == initial_non_trainable_hash

    # predict() debería pasar por ambos filtros
    result = pipeline.predict(x)

    # Verificar que ambos filtros se ejecutaron: (x * 2) + 1
    expected = list(map(lambda x: x * 2 + 1, x.value))
    assert np.array_equal(
        result.value, expected
    ), "Both filters should have been applied"


def test_filter_hash_consistency_across_instances():
    """Test que diferentes instancias del mismo filtro tienen el mismo hash inicial"""
    filter1 = NonTrainableFilter()
    filter2 = NonTrainableFilter()

    # Mismo hash inicial (misma clase, mismos atributos)
    assert filter1._m_hash == filter2._m_hash
    assert filter1._m_str == filter2._m_str
    assert filter1._m_path == filter2._m_path


def test_pipeline_works_without_explicit_init():
    """Test que pipelines funcionan sin llamar explícitamente a init()"""
    non_trainable = NonTrainableFilter()
    pipeline = F3Pipeline(filters=[non_trainable], metrics=[])

    x = XYData.mock([1, 2, 3])
    y = XYData.mock([4, 5, 6])

    # fit() debería funcionar
    pipeline.fit(x, y)

    # predict() debería funcionar
    result = pipeline.predict(x)
    assert np.array_equal(result.value, x.value)


def test_trainable_filter_hash_includes_training_data():
    """Test que el hash de filtros entrenables incluye información de los datos de entrenamiento"""

    class TrainableFilterLocal(BaseFilter):
        def __init__(self):
            super().__init__()

        def fit(self, x: XYData, y: Optional[XYData]) -> None:
            pass

        def predict(self, x: XYData) -> XYData:
            return x

    filter1 = TrainableFilterLocal()
    filter2 = TrainableFilterLocal()

    # Hashes iniciales iguales
    assert filter1._m_hash == filter2._m_hash

    # Entrenar con datos diferentes
    x1 = XYData.mock([1, 2, 3])
    x2 = XYData.mock([4, 5, 6])
    y1 = XYData.mock([7, 8, 9])
    y2 = XYData.mock([10, 11, 12])

    filter1.fit(x1, y1)
    filter2.fit(x2, y2)

    # Hashes deberían ser diferentes (incluyen datos de entrenamiento)
    assert filter1._m_hash != filter2._m_hash

    # Entrenar tercer filtro con mismos datos que filter1
    filter3 = TrainableFilterLocal()
    filter3.fit(x1, y1)

    # Debería tener el mismo hash que filter1 (mismos datos)
    assert filter1._m_hash == filter3._m_hash
