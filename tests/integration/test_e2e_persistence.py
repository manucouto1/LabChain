# tests/integration/test_e2e_persistence.py

from typing import cast
import pytest
import tempfile
import shutil
import numpy as np

from labchain.container import Container
from labchain import LocalStorage
from labchain.base import BaseFilter, XYData, BasePlugin
from labchain import F3Pipeline


class TestEndToEndWorkflows:
    """Complete end-to-end workflow tests."""

    @pytest.fixture(autouse=True)
    def setup_storage(self):
        """Setup shared storage."""
        self.tmpdir = tempfile.mkdtemp(prefix="e2e_test_")
        self.storage = LocalStorage(self.tmpdir)

        yield

        shutil.rmtree(self.tmpdir)

    def test_ml_workflow_train_then_deploy(self):
        """
        Complete ML workflow:
        1. Train on machine A
        2. Save config
        3. Deploy on machine B (without source code)
        """

        # === MACHINE A: Training ===
        Container.storage = self.storage
        # Container.pcm = None  # type: ignore
        # Container.ppif = None  # type: ignore

        @Container.bind(persist=True)
        class Normalizer(BaseFilter):
            def __init__(self):
                super().__init__()
                self._mean = None
                self._std = None

            def fit(self, x: XYData, y=None):
                self._mean = np.mean(x.value, axis=0)
                self._std = np.std(x.value, axis=0)

            def predict(self, x: XYData) -> XYData:
                return XYData.mock((x.value - self._mean) / (self._std + 1e-8))  # type: ignore

        @Container.bind(persist=True)
        class FeatureSelector(BaseFilter):
            def __init__(self, indices: list):
                super().__init__(indices=indices)

            def predict(self, x: XYData) -> XYData:
                return XYData.mock(x.value[:, self.indices])

        # Train
        pipeline = F3Pipeline(filters=[Normalizer(), FeatureSelector(indices=[0, 2])])

        X_train = XYData.mock(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
        pipeline.fit(X_train, None)

        # Save config
        config = pipeline.item_dump()

        # Push classes to storage
        Container.ppif.push_all()

        # === MACHINE B: Deployment (simulate by clearing memory) ===
        Container.ppif._foundry.clear()
        Container.ppif._version_control.clear()

        # Load pipeline from config (classes auto-loaded from storage)
        deployed_pipeline = cast(
            F3Pipeline, BasePlugin.build_from_dump(config, Container.ppif)
        )

        deployed_pipeline.fit(X_train, None)

        # Use in production
        X_new = XYData.mock(np.array([[10, 11, 12]]))
        result = deployed_pipeline.predict(X_new)

        # Verify it works
        assert result.value.shape == (1, 2)  # Selected 2 features

    def test_distributed_training_workflow(self):
        """
        Distributed training scenario:
        1. Define model on coordinator
        2. Push to shared storage
        3. Workers pull and train
        4. Results aggregated
        """

        Container.storage = self.storage
        # Container.pcm = None  # type: ignore
        # Container.ppif = None  # type: ignore

        # Coordinator defines model
        @Container.bind(persist=True, auto_push=True)
        class DistributedModel(BaseFilter):
            def __init__(self, worker_id: int = 0):
                super().__init__(worker_id=worker_id)
                self._trained = False

            def fit(self, x: XYData, y=None):
                # Simulate training
                self._trained = True

            def predict(self, x: XYData) -> XYData:
                return XYData.mock(x.value * (self.worker_id + 1))

        # Worker 1 (simulate by creating new instance)
        Container.ppif._foundry.clear()
        Worker1Model = Container.ppif["DistributedModel"]
        worker1 = Worker1Model(worker_id=1)

        # Worker 2
        worker2 = Worker1Model(worker_id=2)

        # Both workers can use the same class definition
        X = XYData.mock(np.array([10]))

        result1 = worker1.predict(X)
        result2 = worker2.predict(X)

        assert result1.value[0] == 20  # 10 * (1 + 1)
        assert result2.value[0] == 30  # 10 * (2 + 1)
