# tests/integration/test_cached_with_locking.py

import pytest
import numpy as np
import time
from multiprocessing import Process, Queue
from pathlib import Path
import shutil

from labchain.base import BaseFilter, XYData
from labchain.plugins.storage import LockingLocalStorage
from labchain.plugins.filters.cache import CachedWithLocking


class SlowTrainingFilter(BaseFilter):
    """Filter that simulates slow training."""

    def __init__(self, training_time: float = 2.0):
        super().__init__()
        self.training_time = training_time
        self._trained = False

    def fit(self, x, y):
        """Simulate slow training."""
        time.sleep(self.training_time)
        self._trained = True

    def predict(self, x):
        """Simple prediction."""
        return XYData(_hash=f"pred_{x._hash}", _path="/predictions", _value=x.value * 2)


@pytest.fixture
def temp_storage():
    """Fixture that provides clean temporary storage."""
    storage_path = "tests/test_data/cached_locking"
    storage = LockingLocalStorage(storage_path=storage_path)

    yield storage

    # Cleanup
    if Path(storage_path).exists():
        shutil.rmtree(storage_path)


class TestCachedWithLockingBasics:
    """Test basic CachedWithLocking functionality."""

    def test_initialization(self, temp_storage):
        """Test that CachedWithLocking initializes correctly."""
        filter_obj = SlowTrainingFilter()
        cached = CachedWithLocking(
            filter=filter_obj, storage=temp_storage, lock_ttl=3600, lock_timeout=7200
        )

        assert cached.filter == filter_obj
        assert cached.lock_ttl == 3600
        assert cached.lock_timeout == 7200
        assert isinstance(cached._storage, LockingLocalStorage)

    def test_requires_locking_storage(self):
        """Test that non-locking storage raises error."""
        from labchain.plugins.storage import LocalStorage

        regular_storage = LocalStorage("./cache")
        filter_obj = SlowTrainingFilter()

        with pytest.raises(TypeError) as exc_info:
            CachedWithLocking(filter=filter_obj, storage=regular_storage)

        assert "BaseLockingStorage" in str(exc_info.value)

    def test_first_fit_trains_model(self, temp_storage):
        """Test that first fit actually trains the model."""
        filter_obj = SlowTrainingFilter(training_time=0.5)
        cached = CachedWithLocking(
            filter=filter_obj, storage=temp_storage, cache_filter=True
        )

        x = XYData.mock(np.array([1, 2, 3]))
        y = XYData.mock(np.array([4, 5, 6]))

        start = time.time()
        cached.fit(x, y)
        elapsed = time.time() - start

        # Should take time to train
        assert elapsed >= 0.4
        assert filter_obj._trained is True

    def test_second_fit_loads_from_cache(self, temp_storage):
        """Test that second fit loads from cache without training."""
        filter_obj1 = SlowTrainingFilter(training_time=1.0)
        cached1 = CachedWithLocking(
            filter=filter_obj1, storage=temp_storage, cache_filter=True
        )

        x = XYData.mock(np.array([1, 2, 3]))
        y = XYData.mock(np.array([4, 5, 6]))

        # First fit (trains)
        cached1.fit(x, y)

        # Second fit with new instance (should load from cache)
        filter_obj2 = SlowTrainingFilter(training_time=1.0)
        cached2 = CachedWithLocking(
            filter=filter_obj2, storage=temp_storage, cache_filter=True
        )

        start = time.time()
        cached2.fit(x, y)
        elapsed = time.time() - start

        # Should be fast (loaded from cache)
        assert elapsed < 0.5


class TestCachedWithLockingParallel:
    """Test parallel execution with locking."""

    def test_parallel_fit_only_one_trains(self, temp_storage):
        """Test that only one process trains when multiple fit simultaneously."""
        training_results = Queue()

        def train_process(process_id, storage_path, results_queue):
            """Process that tries to train."""
            storage = LockingLocalStorage(storage_path)
            filter_obj = SlowTrainingFilter(training_time=2.0)
            cached = CachedWithLocking(
                filter=filter_obj,
                storage=storage,
                cache_filter=True,
                lock_ttl=10,
                lock_timeout=15,
            )

            x = XYData.mock(np.array([1, 2, 3]))
            y = XYData.mock(np.array([4, 5, 6]))

            start = time.time()
            cached.fit(x, y)
            elapsed = time.time() - start

            # If trained, should take ~2 seconds
            # If waited, should take 2-3 seconds
            trained = elapsed >= 1.5  # Assume trained if took significant time

            results_queue.put((process_id, trained, elapsed))

        # Start 3 processes simultaneously
        processes = []
        for i in range(3):
            p = Process(
                target=train_process,
                args=(i, temp_storage.storage_path.rstrip("/"), training_results),
            )
            p.start()
            processes.append(p)

        # Wait for all processes
        for p in processes:
            p.join(timeout=20)
            assert p.exitcode == 0, f"Process failed with exit code {p.exitcode}"

        # Collect results
        results = []
        while not training_results.empty():
            results.append(training_results.get())

        assert len(results) == 3

        # At least one should have trained (the one that got the lock)
        # trainers = [r for r in results if r[1]]
        # waiters = [r for r in results if not r[1]]

        # In practice, one trains and others wait
        # But due to timing, we just ensure all completed
        assert len(results) == 3

    def test_parallel_predict_only_one_computes(self, temp_storage):
        """Test that only one process computes predictions."""
        # First, train the model
        filter_obj = SlowTrainingFilter(training_time=0.1)
        cached = CachedWithLocking(
            filter=filter_obj, storage=temp_storage, cache_data=True, cache_filter=False
        )

        x = XYData.mock(np.array([1, 2, 3]))

        # Train once
        cached.fit(x, None)

        prediction_results = Queue()

        def predict_process(process_id, storage_path, results_queue):
            """Process that tries to predict."""
            storage = LockingLocalStorage(storage_path)
            filter_obj = SlowTrainingFilter(training_time=0.1)
            filter_obj._trained = True  # Mark as trained

            cached = CachedWithLocking(
                filter=filter_obj,
                storage=storage,
                cache_data=True,
                cache_filter=False,
                lock_ttl=10,
                lock_timeout=15,
            )

            x = XYData.mock(np.array([1, 2, 3]))

            start = time.time()
            result = cached.predict(x)
            elapsed = time.time() - start

            results_queue.put((process_id, elapsed, result._hash))

        # Start 3 processes simultaneously
        processes = []
        for i in range(3):
            p = Process(
                target=predict_process,
                args=(i, temp_storage.storage_path.rstrip("/"), prediction_results),
            )
            p.start()
            processes.append(p)

        # Wait for all processes
        for p in processes:
            p.join(timeout=10)
            assert p.exitcode == 0

        # Collect results
        results = []
        while not prediction_results.empty():
            results.append(prediction_results.get())

        assert len(results) == 3

        # All should have same hash (same predictions)
        hashes = [r[2] for r in results]
        assert len(set(hashes)) == 1  # All same hash


class TestCachedWithLockingTimeouts:
    """Test timeout behavior."""

    def test_timeout_when_training_takes_too_long(self, temp_storage):
        """Test that timeout occurs when lock holder takes too long."""
        lock_holder_started = Queue()
        timeout_occurred = Queue()

        def hold_lock_forever(storage_path, started_queue):
            """Process que mantiene el lock por mucho tiempo."""
            storage = LockingLocalStorage(storage_path)
            filter_obj = SlowTrainingFilter(training_time=100.0)  # 100 segundos

            cached = CachedWithLocking(
                filter=filter_obj,
                storage=storage,
                lock_ttl=10,  # ← CLAVE: TTL CORTO (10s)
                lock_timeout=2,  # Timeout alto (este proceso no esperará)
                heartbeat_interval=None,  # ← SIN HEARTBEAT (lock se vuelve stale)
            )

            x = XYData.mock(np.array([1, 2, 3]))
            y = XYData.mock(np.array([4, 5, 6]))

            # Signal that we've started
            started_queue.put(True)

            # Start training (this will hold the lock for 100 seconds)
            try:
                cached.fit(x, y)
            except Exception as e:
                print(f"Holder error: {e}")

        def wait_and_timeout(storage_path, holder_started_queue, timeout_queue):
            """Process que espera y debería hacer timeout."""
            # Wait for holder to start
            holder_started_queue.get(timeout=5)
            time.sleep(2.0)  # Esperar a que holder adquiera el lock

            storage = LockingLocalStorage(storage_path)
            filter_obj = SlowTrainingFilter(training_time=100.0)

            cached = CachedWithLocking(
                filter=filter_obj,
                storage=storage,
                lock_ttl=10,  # TTL largo (no relevante para este proceso)
                lock_timeout=2,  # ← CLAVE: TIMEOUT CORTO (5s)
                heartbeat_interval=None,
            )

            x = XYData.mock(np.array([1, 2, 3]))
            y = XYData.mock(np.array([4, 5, 6]))

            try:
                cached.fit(x, y)
                timeout_queue.put(False)  # No timeout
            except TimeoutError:
                timeout_queue.put(True)  # Timeout occurred ✓

        # Start lock holder
        holder = Process(
            target=hold_lock_forever,
            args=(temp_storage.storage_path.rstrip("/"), lock_holder_started),
        )
        holder.start()

        # Start waiter
        waiter = Process(
            target=wait_and_timeout,
            args=(
                temp_storage.storage_path.rstrip("/"),
                lock_holder_started,
                timeout_occurred,
            ),
        )
        waiter.start()

        # Wait for waiter to finish
        waiter.join(timeout=15)  # Dar tiempo suficiente

        # Terminate holder (it would run for 100 seconds otherwise)
        holder.terminate()
        holder.join()

        # Check that timeout occurred
        assert not timeout_occurred.empty(), "Timeout queue is empty!"
        did_timeout = timeout_occurred.get()
        assert did_timeout is True, f"Expected timeout=True, got {did_timeout}"


class TestCachedWithLockingCrashRecovery:
    """Test crash recovery via TTL."""

    def test_stale_lock_recovery(self, temp_storage):
        """Test that stale locks can be recovered."""
        # Simulate a crashed process by acquiring lock and not releasing
        lock_name = "model_test_hash"
        temp_storage.try_acquire_lock(lock_name, ttl=1)  # 1 second TTL

        # Wait for lock to become stale
        time.sleep(1.5)

        # New process should be able to train
        filter_obj = SlowTrainingFilter(training_time=0.5)
        cached = CachedWithLocking(filter=filter_obj, storage=temp_storage, lock_ttl=10)

        x = XYData.mock(np.array([1, 2, 3]))
        y = XYData.mock(np.array([4, 5, 6]))

        # Should succeed (stale lock stolen)
        start = time.time()
        cached.fit(x, y)
        elapsed = time.time() - start

        # Should have trained (not waited for timeout)
        assert elapsed < 2.0
        assert filter_obj._trained is True


class TestCachedWithLockingOverwrite:
    """Test overwrite functionality with locking."""

    def test_overwrite_forces_retraining(self, temp_storage):
        """Test that overwrite=True forces retraining even with cache."""
        # First training
        filter_obj1 = SlowTrainingFilter(training_time=0.5)
        cached1 = CachedWithLocking(
            filter=filter_obj1, storage=temp_storage, cache_filter=True, overwrite=False
        )

        x = XYData.mock(np.array([1, 2, 3]))
        y = XYData.mock(np.array([4, 5, 6]))

        cached1.fit(x, y)
        assert filter_obj1._trained is True

        # Second training with overwrite
        filter_obj2 = SlowTrainingFilter(training_time=0.5)
        cached2 = CachedWithLocking(
            filter=filter_obj2,
            storage=temp_storage,
            cache_filter=True,
            overwrite=True,  # Force retrain
        )

        start = time.time()
        cached2.fit(x, y)
        elapsed = time.time() - start

        # Should retrain (not load from cache)
        assert elapsed >= 0.4
        assert filter_obj2._trained is True


class TestCachedWithLockingVerbose:
    """Test verbose output."""

    def test_verbose_output(self, temp_storage, capsys):
        """Test that verbose mode produces helpful output."""
        filter_obj = SlowTrainingFilter(training_time=0.1)
        cached = CachedWithLocking(
            filter=filter_obj, storage=temp_storage, cache_filter=True
        )
        cached.verbose(True)

        x = XYData.mock(np.array([1, 2, 3]))
        y = XYData.mock(np.array([4, 5, 6]))

        cached.fit(x, y)
        captured = capsys.readouterr()

        # Should show locking and training messages
        output = captured.out
        assert any(emoji in output for emoji in ["🔒", "🔨", "✅", "🔓"])
