import pytest
from sklearn import datasets
from rich import print

from labchain import (
    F1,
    Cached,
    F3Pipeline,
    KFoldSplitter,
    StratifiedKFoldSplitter,
    KnnFilter,
    Precission,
    Recall,
    StandardScalerPlugin,
)
from labchain.base.base_clases import XYData
from labchain.plugins.optimizer.grid_optimizer import GridOptimizer


def load_iris_data():
    """Helper function to load and wrap the Iris dataset in XYData."""
    iris = datasets.load_iris()
    X = XYData(
        _hash="Iris X data",
        _path="/datasets",
        _value=iris.data,  # type: ignore
    )
    y = XYData(
        _hash="Iris y data",
        _path="/datasets",
        _value=iris.target,  # type: ignore
    )
    return X, y


def build_pipeline(splitter):
    """Helper function to create a pipeline with GridSearch and the provided splitter."""
    return (
        F3Pipeline(
            filters=[
                Cached(StandardScalerPlugin()),
                KnnFilter().grid({"n_neighbors": (2, 6)}),
            ],
            metrics=[
                F1(average="weighted"),
                Precission(average="weighted"),
                Recall(average="weighted"),
            ],
        )
        .splitter(splitter)
        .optimizer(GridOptimizer(scorer=F1(average="weighted")))
    )


@pytest.mark.parametrize("splitter_class", [KFoldSplitter, StratifiedKFoldSplitter])
def test_pipeline_with_grid_search_and_splitter(splitter_class):
    """
    Test that the pipeline works correctly with both KFold and StratifiedKFold splitter classes.
    """
    X, y = load_iris_data()

    splitter = splitter_class(n_splits=2, shuffle=True, random_state=42)
    pipeline = build_pipeline(splitter)

    # Fit the pipeline
    pipeline.fit(X, y)

    # Check that 2 folds were evaluated
    print(pipeline._results)
    assert len(list(pipeline._results.items())) == 10

    # Predict on training data (sanity check)
    prediction = pipeline.predict(x=X)
    assert prediction.value.shape[0] == X.value.shape[0]

    # Fake prediction to evaluate the metrics
    y_pred = XYData.mock(prediction.value)
    evaluation = pipeline.pipeline.evaluate(X, y, y_pred=y_pred)

    # Sanity check for evaluation output
    assert isinstance(evaluation, dict)
    assert all(metric in evaluation for metric in ["F1", "Precission", "Recall"])

    print(f"\n[bold green]{splitter_class.__name__} results:[/bold green]")
    print(evaluation)
    print(pipeline._results)
