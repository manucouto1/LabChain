# tests/unit/test_pet_container.py

from typing import cast
import warnings
import pytest
import tempfile
import shutil
import numpy as np

from labchain.container.persistent.pet_class_manager import PetClassManager
from labchain.container.persistent.pet_factory import PetFactory
from labchain import Container
from labchain import LocalStorage
from labchain.base import BaseFilter, BasePlugin, XYData
from typeguard import InstrumentationWarning


@Container.bind(persist=True)
class FilterTest(BaseFilter):
    def predict(self, x: XYData) -> XYData:
        return x


@Container.bind(persist=True)
class FilterTest1(BaseFilter):
    def predict(self, x: XYData) -> XYData:
        return x


@Container.bind(persist=True)
class FilterTest2(BaseFilter):
    def predict(self, x: XYData) -> XYData:
        return XYData.mock(x.value * 2)


@Container.bind(persist=True)
class FilterTestNotTracked(BaseFilter):
    def predict(self, x: XYData) -> XYData:
        return x


@Container.bind(persist=True)
class FilterTestModified(BaseFilter):
    def predict(self, x: XYData) -> XYData:
        return XYData.mock(x.value * 2)


@Container.bind(persist=True)
class FilterTestV2(BaseFilter):
    def predict(self, x: XYData) -> XYData:
        return XYData.mock(x.value * 2)  # V2: doubles values


@Container.bind(persist=True)
class FilterTestV1(BaseFilter):
    def predict(self, x: XYData) -> XYData:
        return x  # V1: returns unchanged


@Container.bind(persist=True)
class SimpleFilterTest(BaseFilter):
    """Simple filter for testing."""

    def predict(self, x: XYData) -> XYData:
        return x


class TestPetClassManager:
    """Test suite for PetClassManager functionality."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup and teardown for each test."""
        # Setup: Create fresh temp storage
        self.tmpdir = tempfile.mkdtemp()
        self.temp_storage = LocalStorage(self.tmpdir)
        self.manager = PetClassManager(self.temp_storage)

        # Run the test
        yield

        # Teardown: Clean up
        shutil.rmtree(self.tmpdir)

    def test_get_class_hash_deterministic(self):
        """Test that hash generation is deterministic."""

        hash1 = self.manager.get_class_hash(FilterTest)
        hash2 = self.manager.get_class_hash(FilterTest)

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex digest

    def test_get_class_hash_changes_with_code(self):
        """Test that hash changes when class code changes."""

        hash1 = self.manager.get_class_hash(FilterTest1)
        hash2 = self.manager.get_class_hash(FilterTest2)

        assert hash1 != hash2

    def test_persist_class(self):
        """Test persisting a class to storage."""

        code_hash = self.manager.persist_class(FilterTest)

        assert code_hash is not None
        assert len(code_hash) == 64

        # Verify file exists in storage
        path = f"FilterTest/{code_hash}.pkl"
        assert self.temp_storage.check_if_exists(path, context="plugins")

    def test_persist_class_idempotent(self):
        """Test that persisting same class twice doesn't duplicate."""

        hash1 = self.manager.persist_class(FilterTest)
        hash2 = self.manager.persist_class(FilterTest)

        assert hash1 == hash2

    def test_push_creates_latest_pointer(self):
        """Test that push creates a latest.json pointer."""

        self.manager.push(FilterTest)

        # Check that latest.json exists
        latest_path = "FilterTest/latest.json"
        assert self.temp_storage.check_if_exists(latest_path, context="plugins")

        # Check metadata content
        meta = self.manager._get_remote_latest_meta("FilterTest")
        assert meta is not None
        assert "hash" in meta
        assert meta["class_name"] == "FilterTest"

    def test_check_status_untracked(self):
        """Test status check for untracked class."""

        status = self.manager.check_status(FilterTestNotTracked)
        assert status == "untracked"

    def test_check_status_synced(self):
        """Test status check for synced class."""

        self.manager.push(FilterTest)
        status = self.manager.check_status(FilterTest)
        assert status == "synced"

    def test_check_status_out_of_sync(self):
        """Test status check for out of sync class."""

        # Push original version
        self.manager.push(FilterTest)

        # Change the name to match original
        FilterTestModified.__name__ = "FilterTest"

        status = self.manager.check_status(FilterTestModified)
        assert status == "out_of_sync"

    def test_pull_latest(self):
        """Test pulling latest version from storage."""

        class FilterTest(BaseFilter):
            def __init__(self, factor: float = 1.0):
                super().__init__(factor=factor)

            def predict(self, x: XYData) -> XYData:
                return XYData.mock(x.value * self.factor)

        # Push
        self.manager.push(FilterTest)

        # Pull
        recovered = self.manager.pull("FilterTest")

        assert recovered.__name__ == "FilterTest"

        # Test functionality
        instance = recovered(factor=3.0)
        result = instance.predict(XYData.mock(np.array([1, 2, 3])))
        np.testing.assert_array_equal(result.value, np.array([3, 6, 9]))

    def test_pull_specific_version(self):
        """Test pulling a specific version by hash and verifying functionality."""

        # Push V1
        self.manager.push(FilterTestV1)
        hash_v1 = self.manager.get_class_hash(FilterTestV1)

        # Create and push V2

        FilterTestV2.__name__ = "FilterTestV1"
        self.manager.push(FilterTestV2)

        # Pull V1 by hash
        recovered_v1 = self.manager.pull("FilterTestV1", code_hash=hash_v1)

        # Verify name
        assert recovered_v1.__name__ == "FilterTestV1"

        # Verify functionality (V1 behavior: returns unchanged)
        instance = recovered_v1()
        test_data = XYData.mock(np.array([1, 2, 3]))
        result = instance.predict(test_data)

        # V1 should return unchanged values
        np.testing.assert_array_equal(result.value, np.array([1, 2, 3]))

        # Pull V2 (latest) to verify it's different
        recovered_v2 = self.manager.pull("FilterTestV1")  # Without hash = latest
        instance_v2 = recovered_v2()
        result_v2 = instance_v2.predict(test_data)

        # V2 should double the values
        np.testing.assert_array_equal(result_v2.value, np.array([2, 4, 6]))

    def test_pull_nonexistent_raises(self):
        """Test that pulling nonexistent class raises error."""

        with pytest.raises(ValueError, match="No remote versions found"):
            self.manager.pull("NonexistentClass")

    def test_recover_class(self):
        """Test direct class recovery by hash and verifying functionality."""

        @Container.bind(persist=True)
        class FilterTest(BaseFilter):
            def __init__(self, multiplier: int = 1):
                super().__init__(multiplier=multiplier)

            def predict(self, x: XYData) -> XYData:
                return XYData.mock(x.value * self.multiplier)

        code_hash = self.manager.persist_class(FilterTest)
        recovered = self.manager.recover_class("FilterTest", code_hash)

        # Verify name
        assert recovered.__name__ == "FilterTest"

        # Verify functionality
        instance = recovered(multiplier=5)
        result = instance.predict(XYData.mock(np.array([1, 2, 3])))
        np.testing.assert_array_equal(result.value, np.array([5, 10, 15]))


class TestPetFactory:
    """Test suite for PetFactory functionality."""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage for testing."""
        tmpdir = tempfile.mkdtemp()
        storage = LocalStorage(tmpdir)
        yield storage
        shutil.rmtree(tmpdir)

    @pytest.fixture
    def factory(self, temp_storage: LocalStorage):
        """Create a PetFactory instance."""
        manager = PetClassManager(temp_storage)
        return PetFactory(manager, Container.pif)

    def test_setitem_computes_hash(self, factory: PetFactory):
        """Test that setting item computes and stores hash."""

        @Container.bind(persist=True)
        class FilterTest(BaseFilter):
            def predict(self, x: XYData) -> XYData:
                return x

        factory["FilterTest"] = FilterTest

        assert "FilterTest" in factory._version_control
        assert len(factory._version_control["FilterTest"]) == 64

    def test_getitem_from_memory(self, factory: PetFactory):
        """Test retrieving class from memory."""

        @Container.bind(persist=True)
        class FilterTest(BaseFilter):
            def predict(self, x: XYData) -> XYData:
                return x

        factory["FilterTest"] = FilterTest
        retrieved = factory["FilterTest"]

        assert retrieved == FilterTest

    def test_getitem_lazy_load_from_storage(self, factory: PetFactory):
        """Test automatic lazy loading from storage."""

        @Container.bind(persist=True)
        class FilterTest(BaseFilter):
            def predict(self, x: XYData) -> XYData:
                return x

        factory["FilterTest"] = FilterTest
        factory._manager.push(FilterTest)

        factory._foundry.clear()
        factory._version_control.clear()

        # Should auto-load from storage
        retrieved = factory["FilterTest"]

        assert retrieved.__name__ == "FilterTest"
        assert "FilterTest" in factory._version_control

    def test_getitem_raises_if_not_found(self, factory: PetFactory):
        """Test that accessing nonexistent class raises error."""

        with pytest.raises(
            AttributeError, match="not found locally and no remote version"
        ):
            factory["NonexistentClass"]

    def test_get_with_default(self, factory: PetFactory):
        """Test get method with default value."""

        @Container.bind(persist=True)
        class DefaultFilter(BaseFilter):
            def predict(self, x: XYData) -> XYData:
                return x

        result = factory.get("NonexistentClass", DefaultFilter)
        assert result == DefaultFilter

    def test_get_without_default_returns_none(self, factory: PetFactory):
        """Test get method without default returns None."""

        result = factory.get("NonexistentClass")
        assert result is None

    def test_get_existing_class(self, factory: PetFactory):
        """Test get method for existing class."""

        @Container.bind(persist=True)
        class FilterTest(BaseFilter):
            def predict(self, x: XYData) -> XYData:
                return x

        factory["FilterTest"] = FilterTest
        result = factory.get("FilterTest")

        assert result == FilterTest

    def test_get_version_specific(self, factory: PetFactory):
        """Test retrieving specific version by hash."""

        # Register and push V1
        factory._manager.push(FilterTest)
        hash_v1 = factory._manager.get_class_hash(FilterTest)

        factory._manager.push(FilterTest)

        recovered_v1 = factory.get_version(FilterTest.__name__, hash_v1)

        assert recovered_v1.__name__ == "FilterTest"
        assert factory._manager.get_class_hash(recovered_v1) == hash_v1

    def test_get_version_from_memory(self, factory: PetFactory):
        """Test get_version returns from memory if hash matches."""

        @Container.bind(persist=True)
        class FilterTest(BaseFilter):
            def predict(self, x: XYData) -> XYData:
                return x

        factory["FilterTest"] = FilterTest
        code_hash = factory._manager.get_class_hash(FilterTest)

        # Should return from memory without storage access
        result = factory.get_version("FilterTest", code_hash)
        assert result == FilterTest

    def test_push_all(self, factory: PetFactory, temp_storage: LocalStorage):
        """Test pushing all registered classes."""

        @Container.bind(persist=True)
        class Filter1(BaseFilter):
            def predict(self, x: XYData) -> XYData:
                return x

        @Container.bind(persist=True)
        class Filter2(BaseFilter):
            def predict(self, x: XYData) -> XYData:
                return x

        factory["Filter1"] = Filter1
        factory["Filter2"] = Filter2

        factory.push_all()

        # Verify both were pushed
        assert temp_storage.check_if_exists("Filter1/latest.json", context="plugins")
        assert temp_storage.check_if_exists("Filter2/latest.json", context="plugins")

    def test_push_all_empty_factory(
        self, factory: PetFactory, capsys: pytest.CaptureFixture[str]
    ):
        """Test push_all on empty factory doesn't error."""

        factory.push_all()
        # Should complete without error


class TestPersistentContainer:
    """Test suite for Container persistence features."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup and teardown for each test."""
        warnings.filterwarnings("ignore", category=InstrumentationWarning)

        # Setup: Create temp storage and configure Container
        self.tmpdir = tempfile.mkdtemp(prefix="container_persist_test_")
        self.storage = LocalStorage(self.tmpdir)
        Container.storage = self.storage

        # Reset persistent components
        # Container.pcm = None  # type: ignore
        # Container.ppif = None  # type: ignore

        yield

        # Cleanup
        if Container.ppif is not None:
            Container.ppif._foundry.clear()

        try:
            shutil.rmtree(self.tmpdir)
        except Exception as e:
            print(f"Warning: Could not clean up {self.tmpdir}: {e}")

    def test_bind_without_persist(self):
        """Test standard bind (persist=False) doesn't create persistent components."""

        @Container.bind()
        class StandardFilter(BaseFilter):
            def predict(self, x: XYData) -> XYData:
                return x

        # Should be in standard factories
        assert "StandardFilter" in Container.ff._foundry
        assert "StandardFilter" in Container.pif._foundry

    def test_bind_with_persist(self):
        """Test bind with persist=True creates persistent components."""

        @Container.bind(persist=True)
        class PersistentFilter(BaseFilter):
            def predict(self, x: XYData) -> XYData:
                return x

        # Should be in standard factories
        assert "PersistentFilter" in Container.ff._foundry
        assert "PersistentFilter" in Container.pif._foundry

        # Should ALSO be in persistent factory
        assert Container.ppif is not None
        assert "PersistentFilter" in Container.ppif._foundry
        assert "PersistentFilter" in Container.ppif._version_control

    def test_bind_with_auto_push(self):
        """Test bind with persist=True and auto_push=True."""

        @Container.bind(persist=True, auto_push=True)
        class AutoPushFilter(BaseFilter):
            def predict(self, x: XYData) -> XYData:
                return x

        # Should be registered
        assert "AutoPushFilter" in Container.ppif._foundry

        # Should be pushed to storage
        meta = Container.pcm._get_remote_latest_meta("AutoPushFilter")
        assert meta is not None
        assert meta["class_name"] == "AutoPushFilter"

    def test_manual_push_workflow(self):
        """Test manual push workflow."""

        @Container.bind(persist=True)
        class ManualPushFilter(BaseFilter):
            def predict(self, x: XYData) -> XYData:
                return x

        # Not pushed yet
        meta = Container.pcm._get_remote_latest_meta("ManualPushFilter")
        assert meta is None

        # Manual push
        Container.ppif.push_all()

        # Now it should be in storage
        meta = Container.pcm._get_remote_latest_meta("ManualPushFilter")
        assert meta is not None

    def test_lazy_loading_from_storage(self):
        """Test lazy loading of persistent classes."""

        # Use module-level class to avoid serialization issues
        Container.bind(persist=True)(SimpleFilterTest)

        # Push to storage
        Container.ppif.push_all()

        # Clear from memory
        Container.ppif._foundry.clear()
        Container.ppif._version_control.clear()

        # Should auto-load from storage
        recovered = Container.ppif["SimpleFilterTest"]
        assert recovered.__name__ == "SimpleFilterTest"

        # Test functionality
        instance = recovered()
        result = instance.predict(XYData.mock(np.array([1, 2, 3])))
        np.testing.assert_array_equal(result.value, np.array([1, 2, 3]))

    def test_build_from_dump_with_persistent_factory(self):
        """Test BasePlugin.build_from_dump with persistent factory."""

        @Container.bind(persist=True)
        class DumpFilterTest(BaseFilter):
            def __init__(self, factor: int = 1):
                super().__init__(factor=factor)

            def predict(self, x: XYData) -> XYData:
                return XYData.mock(x.value * self.factor)

        # Create instance and dump
        instance = DumpFilterTest(factor=5)
        dump = instance.item_dump()

        # Push to storage
        Container.ppif.push_all()

        # Clear from memory
        Container.ppif._foundry.clear()

        # Reconstruct from dump (should auto-load from storage)
        reconstructed: DumpFilterTest = cast(
            DumpFilterTest, BasePlugin.build_from_dump(dump, Container.ppif)
        )

        assert reconstructed.__class__.__name__ == "DumpFilterTest"
        assert reconstructed.factor == 5

        # Test functionality
        result = reconstructed.predict(XYData.mock(np.array([2, 4])))
        np.testing.assert_array_equal(result.value, np.array([10, 20]))
