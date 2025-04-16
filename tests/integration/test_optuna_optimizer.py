from sklearn import datasets
from framework3 import (
    F1,
    F3Pipeline,
    KnnFilter,
    Precission,
    Recall,
    StandardScalerPlugin,
)

from rich import print

from framework3.base.base_clases import XYData
from framework3.plugins.optimizer.optuna_optimizer import OptunaOptimizer
from framework3.plugins.splitter.cross_validation_splitter import KFoldSplitter


def test_optuna_pipeline_init_and_fit():
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

    wandb_pipeline = (
        F3Pipeline(
            filters=[StandardScalerPlugin(), KnnFilter().grid({"n_neighbors": (2, 6)})],
            metrics=[F1(), Precission(), Recall()],
        )
        .splitter(
            KFoldSplitter(
                n_splits=2,
                shuffle=True,
                random_state=42,
            )
        )
        .optimizer(
            OptunaOptimizer(
                direction="maximize",
                study_name="Iris-KNN",
                reset_study=True,
                n_trials=100,
                storage="sqlite:///optuna_estudios.db",
            )
        )
    )

    wandb_pipeline.fit(X, y)

    prediction = wandb_pipeline.predict(x=X)

    y_pred = XYData.mock(prediction.value)

    evaluate = wandb_pipeline.pipeline.evaluate(X, y, y_pred=y_pred)

    print(wandb_pipeline)

    print(evaluate)
