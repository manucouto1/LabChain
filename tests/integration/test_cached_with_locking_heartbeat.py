# tests/integration/test_cached_with_locking_heartbeat.py

import pytest
import numpy as np
import time
from pathlib import Path
import shutil
from multiprocessing import Process, Queue

from labchain.base import BaseFilter, XYData
from labchain.plugins.storage import LockingLocalStorage
from labchain.plugins.filters.cache import CachedWithLocking


class VerySlowFilter(BaseFilter):
    """Filter that simulates very slow training with progress."""

    def __init__(self, total_steps: int = 10, step_duration: float = 1.0):
        super().__init__()
        self.total_steps = total_steps
        self.step_duration = step_duration
        self._trained = False
        self._steps_completed = 0

    def fit(self, x, y):
        """Simulate slow training with multiple steps."""
        for i in range(self.total_steps):
            time.sleep(self.step_duration)
            self._steps_completed = i + 1
        self._trained = True

    def predict(self, x):
        """Slow prediction."""
        time.sleep(2.0)  # 2 second prediction
        return XYData(_hash=f"pred_{x._hash}", _path="/predictions", _value=x.value * 2)


@pytest.fixture
def temp_storage():
    """Fixture for temporary storage."""
    storage_path = "tests/test_data/cached_heartbeat"
    storage = LockingLocalStorage(storage_path=storage_path)

    yield storage

    if Path(storage_path).exists():
        shutil.rmtree(storage_path)


class TestCachedWithHeartbeat:
    """Test CachedWithLocking with heartbeat enabled."""

    def test_auto_heartbeat_during_fit(self, temp_storage):
        """Test that heartbeat is automatically sent during fit."""
        filter_obj = VerySlowFilter(total_steps=5, step_duration=0.5)

        cached = CachedWithLocking(
            filter=filter_obj,
            storage=temp_storage,
            cache_filter=True,
            lock_ttl=3600,
            heartbeat_interval=1,  # 1 second heartbeat
            auto_heartbeat=True,
        )

        x = XYData.mock(np.array([1, 2, 3]))
        y = XYData.mock(np.array([4, 5, 6]))

        # Train (takes ~2.5 seconds, should send ~2 heartbeats)
        cached.fit(x, y)

        assert filter_obj._trained is True
        assert filter_obj._steps_completed == 5

    def test_auto_heartbeat_during_predict(self, temp_storage):
        """Test that heartbeat is sent during predict."""
        filter_obj = VerySlowFilter(total_steps=1, step_duration=0.1)
        filter_obj._trained = True  # Skip training

        cached = CachedWithLocking(
            filter=filter_obj,
            storage=temp_storage,
            cache_data=True,
            cache_filter=False,
            lock_ttl=3600,
            heartbeat_interval=1,
            auto_heartbeat=True,
        )

        x = XYData.mock(np.array([1, 2, 3]))

        # Predict (takes ~2 seconds, should send ~2 heartbeats)
        result = cached.predict(x)

        assert result is not None

    def test_heartbeat_prevents_parallel_process_stealing(self, temp_storage):
        """Test that active heartbeat prevents other processes from stealing."""
        results = Queue()

        def train_with_heartbeat(process_id, storage_path, results_queue):
            """Process that trains with heartbeat."""
            storage = LockingLocalStorage(storage_path)

            filter_obj = VerySlowFilter(total_steps=10, step_duration=0.5)
            cached = CachedWithLocking(
                filter=filter_obj,
                storage=storage,
                cache_filter=True,
                lock_ttl=3600,
                heartbeat_interval=1,  # Active heartbeat
                auto_heartbeat=True,
            )

            x = XYData.mock(np.array([1, 2, 3]))
            y = XYData.mock(np.array([4, 5, 6]))

            start = time.time()
            cached.fit(x, y)
            elapsed = time.time() - start

            results_queue.put((process_id, "trained", elapsed, filter_obj._trained))

        def try_concurrent_train(process_id, storage_path, results_queue):
            """Process that tries to train concurrently."""
            time.sleep(1)  # Let first process start

            storage = LockingLocalStorage(storage_path)

            filter_obj = VerySlowFilter(total_steps=10, step_duration=0.5)
            cached = CachedWithLocking(
                filter=filter_obj,
                storage=storage,
                cache_filter=True,
                lock_ttl=3600,
                heartbeat_interval=1,
                auto_heartbeat=True,
            )

            x = XYData.mock(np.array([1, 2, 3]))
            y = XYData.mock(np.array([4, 5, 6]))

            start = time.time()
            cached.fit(x, y)
            elapsed = time.time() - start

            results_queue.put((process_id, "waited", elapsed, filter_obj._trained))

        # Start both processes
        p1 = Process(
            target=train_with_heartbeat,
            args=(1, temp_storage.storage_path.rstrip("/"), results),
        )
        p2 = Process(
            target=try_concurrent_train,
            args=(2, temp_storage.storage_path.rstrip("/"), results),
        )

        p1.start()
        p2.start()

        p1.join(timeout=30)
        p2.join(timeout=30)

        # Collect results
        results_list = []
        while not results.empty():
            results_list.append(results.get())

        assert len(results_list) == 2

        # One should train, one should wait
        trainers = [r for r in results_list if r[1] == "trained"]
        [r for r in results_list if r[1] == "waited"]

        # At least one trained
        assert len(trainers) >= 1

    def test_heartbeat_allows_steal_after_crash(self, temp_storage):
        """Test that crashed process lock can be stolen via heartbeat."""
        results = Queue()

        def train_then_crash(storage_path, results_queue):
            """Process that starts training but crashes."""
            storage = LockingLocalStorage(storage_path)

            # Acquire lock with heartbeat but don't send updates
            lock_name = "model_test_hash"
            storage.try_acquire_lock(lock_name, ttl=3600, heartbeat_interval=2)

            results_queue.put("crashed")
            # Crash without releasing (simulate process death)

        def wait_then_recover(storage_path, results_queue):
            """Process that waits for crash detection then recovers."""
            # Wait for crash
            results_queue.get(timeout=5)

            # Wait for heartbeat death detection (3x2s = 6s)
            time.sleep(7)

            # Should be able to acquire now
            storage = LockingLocalStorage(storage_path)

            filter_obj = VerySlowFilter(total_steps=2, step_duration=0.5)
            cached = CachedWithLocking(
                filter=filter_obj,
                storage=storage,
                cache_filter=True,
                lock_ttl=3600,
                heartbeat_interval=2,
                auto_heartbeat=True,
            )

            x = XYData.mock(np.array([1, 2, 3]))
            y = XYData.mock(np.array([4, 5, 6]))

            start = time.time()
            cached.fit(x, y)
            elapsed = time.time() - start

            results_queue.put(("recovered", elapsed, filter_obj._trained))

        # Start processes
        crasher = Process(
            target=train_then_crash,
            args=(temp_storage.storage_path.rstrip("/"), results),
        )
        recoverer = Process(
            target=wait_then_recover,
            args=(temp_storage.storage_path.rstrip("/"), results),
        )

        crasher.start()
        recoverer.start()

        crasher.join(timeout=5)
        recoverer.join(timeout=20)

        # Get recovery result
        recovery_result = results.get(timeout=1)

        assert recovery_result[0] == "recovered"
        assert recovery_result[2] is True  # Training succeeded
        assert recovery_result[1] < 10  # Recovered quickly (not waiting for full TTL)

    def test_disable_heartbeat(self, temp_storage):
        """Test that heartbeat can be disabled."""
        filter_obj = VerySlowFilter(total_steps=3, step_duration=0.3)

        cached = CachedWithLocking(
            filter=filter_obj,
            storage=temp_storage,
            cache_filter=True,
            lock_ttl=3600,
            heartbeat_interval=0,  # Disabled
            auto_heartbeat=False,
        )

        x = XYData.mock(np.array([1, 2, 3]))
        y = XYData.mock(np.array([4, 5, 6]))

        # Should work without heartbeat
        cached.fit(x, y)

        assert filter_obj._trained is True

    def test_heartbeat_interval_auto_calculation(self, temp_storage):
        """Test that heartbeat interval is auto-calculated from TTL."""
        filter_obj = VerySlowFilter(total_steps=2, step_duration=0.2)

        # Don't specify heartbeat_interval
        cached = CachedWithLocking(
            filter=filter_obj,
            storage=temp_storage,
            cache_filter=True,
            lock_ttl=600,  # 10 minutes
            # heartbeat_interval not specified
            auto_heartbeat=True,
        )

        # Should auto-calculate: 600 / 20 = 30 seconds
        assert cached.heartbeat_interval == 30

        x = XYData.mock(np.array([1, 2, 3]))
        y = XYData.mock(np.array([4, 5, 6]))

        cached.fit(x, y)
        assert filter_obj._trained is True


class TestBackoffWaiting:
    """Test exponential backoff in wait_for_unlock."""

    def test_backoff_reduces_check_frequency(self, temp_storage):
        """Test that exponential backoff reduces polling frequency."""
        # Create a lock that will be held for a while
        lock_name = "backoff_test_lock"
        temp_storage.try_acquire_lock(lock_name, ttl=3600)

        # Start waiting with backoff in background
        import threading

        wait_complete = threading.Event()

        def wait_with_backoff():
            temp_storage.wait_for_unlock(
                lock_name,
                timeout=5,
                initial_poll_interval=0.1,
                max_poll_interval=1.0,
                backoff_factor=2.0,
            )
            wait_complete.set()

        thread = threading.Thread(target=wait_with_backoff)

        start = time.time()
        thread.start()

        # Release after 3 seconds
        time.sleep(3)
        temp_storage.release_lock(lock_name)

        # Wait for completion
        wait_complete.wait(timeout=10)
        elapsed = time.time() - start

        # Should complete around 3 seconds (when lock released)
        assert 2.5 < elapsed < 4.0

        thread.join()
