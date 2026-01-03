# tests/unit/test_pet_container_advanced.py

import pytest
import tempfile
import shutil
import warnings
import numpy as np
from typing import Optional, cast

from labchain.container import Container
from labchain import LocalStorage
from labchain.base import BaseFilter, BasePlugin, XYData
from labchain import F3Pipeline
from typeguard import InstrumentationWarning


class TestAdvancedPersistence:
    """Advanced test cases for persistence functionality."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup and teardown for each test."""
        warnings.filterwarnings("ignore", category=InstrumentationWarning)

        self.tmpdir = tempfile.mkdtemp(prefix="advanced_persist_test_")
        self.storage = LocalStorage(self.tmpdir)
        Container.storage = self.storage
        # Container.pcm = None  # type: ignore
        # Container.ppif = None  # type: ignore

        yield

        if Container.ppif is not None:
            Container.ppif._foundry.clear()

        try:
            shutil.rmtree(self.tmpdir)
        except Exception as e:
            print(f"Warning: Could not clean up {self.tmpdir}: {e}")

    def test_nested_pipeline_persistence(self):
        """Test persisting a pipeline with nested filters."""

        @Container.bind(persist=True)
        class ScalerFilter(BaseFilter):
            def __init__(self, scale: float = 1.0):
                super().__init__(scale=scale)
                self._mean = None

            def fit(self, x: XYData, y: Optional[XYData] = None):
                self._mean = np.mean(x.value)

            def predict(self, x: XYData) -> XYData:
                if self._mean is None:
                    return x
                return XYData.mock((x.value - self._mean) * self.scale)

        @Container.bind(persist=True)
        class MultiplierFilter(BaseFilter):
            def __init__(self, factor: int = 2):
                super().__init__(factor=factor)

            def predict(self, x: XYData) -> XYData:
                return XYData.mock(x.value * self.factor)

        # Create pipeline with both filters
        pipeline = F3Pipeline(
            filters=[ScalerFilter(scale=2.0), MultiplierFilter(factor=3)]
        )
        # Fit pipeline
        X = XYData.mock(np.array([[1, 2], [3, 4], [5, 6]]))
        pipeline.fit(X, None)
        # Get prediction
        result1 = pipeline.predict(X)
        # Dump pipeline
        pipeline_dump = pipeline.item_dump()
        # Push all to storage
        Container.ppif.push_all()
        # Clear memory
        Container.ppif._foundry.clear()

        reconstructed = cast(
            F3Pipeline, BasePlugin.build_from_dump(pipeline_dump, Container.ppif)
        )
        reconstructed.fit(
            X, None
        )  # Reconstructed pipeline should have the same behavior as the original pipeline
        # Verify it works
        result2 = reconstructed.predict(X)
        np.testing.assert_array_almost_equal(result1.value, result2.value)

    def test_version_rollback(self):
        """Test rolling back to previous version of a class."""

        # Version 1
        @Container.bind(persist=True)
        class EvolvingFilter(BaseFilter):  # type: ignore
            def predict(self, x: XYData) -> XYData:
                return XYData.mock(x.value * 1)  # V1: multiply by 1

        Container.ppif.push_all()
        hash_v1 = Container.pcm.get_class_hash(EvolvingFilter)

        # Create config with V1
        instance_v1 = EvolvingFilter()
        config_v1 = instance_v1.item_dump()

        print(config_v1)
        print(hash_v1)

        # Version 2 - modify the class
        @Container.bind(persist=True)
        class EvolvingFilter(BaseFilter):  # noqa: F811
            def predict(self, x: XYData) -> XYData:
                return XYData.mock(x.value * 10)  # V2: multiply by 10

        hash_v2 = Container.pcm.get_class_hash(EvolvingFilter)
        print(hash_v2)

        Container.ppif.push_all()

        # Current version should be V2
        current = Container.ppif["EvolvingFilter"]()
        result_v2 = current.predict(XYData.mock(np.array([5])))
        assert result_v2.value[0] == 50  # V2 behavior

        # But we can still reconstruct V1 from its config
        reconstructed_v1 = cast(
            F3Pipeline, BasePlugin.build_from_dump(config_v1, Container.ppif)
        )
        result_v1 = reconstructed_v1.predict(XYData.mock(np.array([5])))
        assert result_v1.value[0] == 5  # V1 behavior

    def test_multiple_users_same_storage(self):
        """Simulate multiple users/processes sharing same storage."""

        # User 1: Create and push a filter
        @Container.bind(persist=True, auto_push=True)
        class SharedFilter(BaseFilter):
            def __init__(self, value: int = 42):
                super().__init__(value=value)

            def predict(self, x: XYData) -> XYData:
                return XYData.mock(x.value + self.value)

        # User 2: Simulate by clearing local memory
        Container.ppif._foundry.clear()
        Container.ppif._version_control.clear()

        # User 2 should be able to access the class
        LoadedFilter = Container.ppif["SharedFilter"]
        instance = LoadedFilter(value=100)
        result = instance.predict(XYData.mock(np.array([10])))

        assert result.value[0] == 110

    def test_concurrent_version_updates(self):
        """Test handling of concurrent version updates."""

        @Container.bind(persist=True)
        class ConcurrentFilter(BaseFilter):  # type: ignore
            def predict(self, x: XYData) -> XYData:
                return x

        # Push V1
        Container.ppif.push_all()
        hash_v1 = Container.pcm.get_class_hash(ConcurrentFilter)

        # Modify and push V2
        @Container.bind(persist=True)
        class ConcurrentFilter(BaseFilter):  # noqa: F811
            def predict(self, x: XYData) -> XYData:
                return XYData.mock(x.value * 2)

        Container.ppif.push_all()
        hash_v2 = Container.pcm.get_class_hash(ConcurrentFilter)

        # Both versions should be accessible
        assert hash_v1 != hash_v2

        v1 = Container.ppif.get_version("ConcurrentFilter", hash_v1)
        v2 = Container.ppif.get_version("ConcurrentFilter", hash_v2)

        # Test both versions work correctly
        test_data = XYData.mock(np.array([5]))

        result_v1 = v1().predict(test_data)
        assert result_v1.value[0] == 5  # V1: unchanged

        result_v2 = v2().predict(test_data)
        assert result_v2.value[0] == 10  # V2: doubled

    def test_mixed_persistent_and_standard_classes(self):
        """Test mixing persistent and standard (non-persistent) classes."""

        # Standard class (not persistent)
        @Container.bind()
        class StandardFilter(BaseFilter):
            def predict(self, x: XYData) -> XYData:
                return x

        # Persistent class
        @Container.bind(persist=True)
        class PersistentFilter(BaseFilter):
            def predict(self, x: XYData) -> XYData:
                return XYData.mock(x.value * 2)

        # Standard should only be in regular factories
        assert "StandardFilter" in Container.ff._foundry
        assert "StandardFilter" in Container.pif._foundry
        assert Container.ppif is not None
        assert "StandardFilter" not in Container.ppif._foundry

        # Persistent should be in both
        assert "PersistentFilter" in Container.ff._foundry
        assert "PersistentFilter" in Container.pif._foundry
        assert "PersistentFilter" in Container.ppif._foundry

    def test_status_checking(self):
        """Test checking sync status of persistent classes."""

        @Container.bind(persist=True)
        class StatusFilter(BaseFilter):  # type: ignore
            def predict(self, x: XYData) -> XYData:
                return x

        # Initially untracked
        status = Container.pcm.check_status(StatusFilter)
        assert status == "untracked"

        # Push to storage
        Container.ppif.push_all()

        # Now synced
        status = Container.pcm.check_status(StatusFilter)
        assert status == "synced"

        # Modify class
        @Container.bind(persist=True)
        class StatusFilter(BaseFilter):  # noqa: F811
            def predict(self, x: XYData) -> XYData:
                return XYData.mock(x.value * 999)

        # Now out of sync
        status = Container.pcm.check_status(StatusFilter)
        assert status == "out_of_sync"

    def test_large_class_persistence(self):
        """Test persisting a class with significant state."""

        @Container.bind(persist=True)
        class LargeStateFilter(BaseFilter):
            def __init__(self, size: int = 1000):
                super().__init__(size=size)
                self._weights = None

            def fit(self, x: XYData, y: Optional[XYData] = None):
                # Simulate large state
                self._weights = np.random.randn(self.size, self.size)

            def predict(self, x: XYData) -> XYData:
                if self._weights is None:
                    return x
                # Simple transformation using weights
                return XYData.mock(x.value * self._weights[0, 0])

        # Create and fit
        filter_instance = LargeStateFilter(size=100)
        X = XYData.mock(np.array([[1, 2, 3]]))
        filter_instance.fit(X)

        # Get result
        filter_instance.predict(X)

        # Dump (note: this dumps the instance, not the class)
        # For class persistence, we're testing the class can be recovered
        Container.ppif.push_all()

        # Clear and reload
        Container.ppif._foundry.clear()

        recovered_class = Container.ppif["LargeStateFilter"]
        recovered_instance = recovered_class(size=100)
        recovered_instance.fit(X)
        result2 = recovered_instance.predict(X)

        # Should work (but different random weights, so different results)
        assert result2 is not None

    def test_error_handling_corrupted_storage(self):
        """Test handling of corrupted storage files."""

        @Container.bind(persist=True, auto_push=True)
        class TestFilter(BaseFilter):
            def predict(self, x: XYData) -> XYData:
                return x

        # Corrupt the latest.json file
        latest_path = f"{self.tmpdir}/plugins/TestFilter/latest.json"
        with open(latest_path, "w") as f:
            f.write("corrupted json {{{")

        # Clear memory
        Container.ppif._foundry.clear()

        # Should handle corruption gracefully
        # (either raise meaningful error or return None)
        meta = Container.pcm._get_remote_latest_meta("TestFilter")
        assert meta is None  # Should return None for corrupted file

    def test_auto_push_false_then_manual_push(self):
        """Test workflow: bind without auto_push, then manual push later."""

        @Container.bind(persist=True, auto_push=False)
        class DelayedPushFilter(BaseFilter):
            def predict(self, x: XYData) -> XYData:
                return x

        # Should be registered but not pushed
        assert "DelayedPushFilter" in Container.ppif._foundry
        meta = Container.pcm._get_remote_latest_meta("DelayedPushFilter")
        assert meta is None

        # Manual push later
        Container.pcm.push(DelayedPushFilter)

        # Now should be in storage
        meta = Container.pcm._get_remote_latest_meta("DelayedPushFilter")
        assert meta is not None
