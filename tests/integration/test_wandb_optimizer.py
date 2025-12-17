from unittest.mock import MagicMock
from rich import print
from sklearn import datasets
import pytest
import typeguard

from framework3 import (
    F1,
    Cached,
    F3Pipeline,
    KnnFilter,
    Precission,
    Recall,
    StandardScalerPlugin,
    WandbOptimizer,
)
from framework3.plugins.metrics.classification import XYData
from framework3.plugins.splitter.cross_validation_splitter import KFoldSplitter


from framework3.utils.wandb import WandbSweepManager
from framework3.base import BaseFilter, XYData, BaseMetric
import numpy as np


class DummyFilter(BaseFilter):
    def __init__(self, param_a: float = 0.5, param_b: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.param_a = param_a
        self.param_b = param_b

        self._grid = {
            "param_a": {
                "distribution": "uniform",
                "min": 0.1,
                "max": 0.9,
            },
            "param_b": {
                "distribution": "log_uniform_values",
                "min": 1e-3,
                "max": 1.0,
            },
        }

    def fit(self, x, y):
        pass

    def predict(self, x):
        return XYData.mock(np.random.rand(len(x.value)))


class DummyMetric(BaseMetric):
    higher_better = True

    def evaluate(
        self, x_data: XYData, y_true: XYData | None, y_pred: XYData
    ) -> float | np.ndarray:
        return float(np.random.rand())


def test_bayesian_sweep():
    """Test that Bayesian sweep config is generated correctly."""

    pipeline = DummyFilter()
    scorer = DummyMetric()
    x = XYData.mock(np.random.rand(100, 10))
    y = XYData.mock(np.random.rand(100))

    manager = WandbSweepManager()

    # Generate config (don't actually create sweep)
    sweep_config = manager.generate_config_for_pipeline(pipeline)
    print(sweep_config)
    sweep_config["method"] = "bayes"
    sweep_config["run_cap"] = 10

    # Assertions
    assert sweep_config["method"] == "bayes"
    assert sweep_config["run_cap"] == 10

    params = sweep_config["parameters"]["filters"]["parameters"]["DummyFilter"][
        "parameters"
    ]

    # Check param_a (uniform)
    assert params["param_a"]["distribution"] == "uniform"
    assert params["param_a"]["min"] == 0.1
    assert params["param_a"]["max"] == 0.9

    # Check param_b (log_uniform_values)
    assert params["param_b"]["distribution"] == "log_uniform_values"
    assert params["param_b"]["min"] == 1e-3
    assert params["param_b"]["max"] == 1.0

    print("✅ Test passed: Bayesian config generated correctly")


def test_wandb_pipeline_init_raises_value_error():
    from framework3.base import BaseMetric

    with pytest.raises(
        ValueError, match="Either pipeline or sweep_id must be provided"
    ):
        WandbOptimizer(
            project="test_project",
            pipeline=None,
            sweep_id=None,
            scorer=MagicMock(spec=BaseMetric),
        ).fit(XYData.mock([]), XYData.mock([]))


def test_wandb_pipeline_init_raises_value_error_for_invalid_project():
    from framework3.base import BaseMetric

    with pytest.raises(
        ValueError, match="Either pipeline or sweep_id must be provided"
    ):
        WandbOptimizer(
            project="",
            pipeline=None,
            sweep_id=None,
            scorer=MagicMock(spec=BaseMetric),
        ).fit(XYData.mock([]), XYData.mock([]))

    with pytest.raises(typeguard.TypeCheckError):
        WandbOptimizer(
            project=None,  # type: ignore
            pipeline=MagicMock(),
            sweep_id=None,
            scorer=MagicMock(spec=BaseMetric),
        )


def test_wandb_pipeline_init_with_valid_parameters():
    from framework3.base import BaseMetric

    mock_pipeline = MagicMock()
    mock_scorer = MagicMock(spec=BaseMetric)

    wandb_pipeline = WandbOptimizer(
        project="test_project",
        pipeline=mock_pipeline,
        sweep_id=None,
        scorer=mock_scorer,
    )

    assert wandb_pipeline.project == "test_project"
    assert wandb_pipeline.pipeline == mock_pipeline
    assert wandb_pipeline.sweep_id is None
    assert wandb_pipeline.scorer == mock_scorer


def test_wandb_pipeline_init_and_fit():
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
            filters=[
                Cached(StandardScalerPlugin()),
                KnnFilter().grid({"n_neighbors": [3, 5]}),
            ],
            metrics=[
                F1(average="weighted"),
                Precission(average="weighted"),
                Recall(average="weighted"),
            ],
        )
        .splitter(
            KFoldSplitter(
                n_splits=2,
                shuffle=True,
                random_state=42,
            )
        )
        .optimizer(
            WandbOptimizer(
                project="test_project",
                sweep_id=None,
                scorer=F1(average="weighted"),
            )
        )
    )

    print("______________________PIPELINE_____________________")
    print(wandb_pipeline)
    print("_____________________________________________________")

    assert wandb_pipeline.sweep_id is None

    try:
        wandb_pipeline.fit(X, y)
        prediction = wandb_pipeline.predict(x=X)

        y_pred = XYData.mock(prediction.value)

        evaluate = wandb_pipeline.evaluate(X, y, y_pred)

        print(wandb_pipeline)

        print(evaluate)

    except Exception as e:
        print(f"Error during pipeline execution: {e}")
        assert False
