# tests/unit/storage/test_locking_local_storage.py

import pytest
import os
import time
import json
from pathlib import Path
from multiprocessing import Process, Queue
from labchain.plugins.storage import LockingLocalStorage


@pytest.fixture
def locking_storage():
    """Fixture that provides a clean LockingLocalStorage instance."""
    storage_path = "tests/test_data/locking"
    storage = LockingLocalStorage(storage_path=storage_path)

    yield storage

    # Cleanup
    import shutil

    if Path(storage_path).exists():
        shutil.rmtree(storage_path)


class TestLockingLocalStorageBasics:
    """Test basic locking operations."""

    def test_initialization(self, locking_storage):
        """Test that storage initializes correctly with locks directory."""
        assert locking_storage.storage_path.endswith("locking/")
        assert locking_storage._locks_dir.exists()
        assert locking_storage._locks_dir.name == "locks"

    def test_acquire_lock_success(self, locking_storage):
        """Test successful lock acquisition."""
        lock_name = "test_lock"

        result = locking_storage.try_acquire_lock(lock_name, ttl=3600)

        assert result is True
        lock_file = locking_storage._locks_dir / f"{lock_name}.lock"
        assert lock_file.exists()

        # Verify lock metadata
        with open(lock_file, "r") as f:
            lock_data = json.load(f)

        assert "owner" in lock_data
        assert "pid" in lock_data
        assert "created_at" in lock_data
        assert "ttl" in lock_data  # ← Actualizado: ahora incluye TTL
        assert lock_data["pid"] == os.getpid()
        assert lock_data["ttl"] == 3600

        # Cleanup
        locking_storage.release_lock(lock_name)

    def test_acquire_lock_fail_when_locked(self, locking_storage):
        """Test that acquiring a locked lock fails."""
        lock_name = "test_lock"

        # First acquisition succeeds
        assert locking_storage.try_acquire_lock(lock_name) is True

        # Second acquisition fails
        assert locking_storage.try_acquire_lock(lock_name) is False

        # Cleanup
        locking_storage.release_lock(lock_name)

    def test_release_lock(self, locking_storage):
        """Test lock release."""
        lock_name = "test_lock"

        locking_storage.try_acquire_lock(lock_name)
        lock_file = locking_storage._locks_dir / f"{lock_name}.lock"
        assert lock_file.exists()

        locking_storage.release_lock(lock_name)
        assert not lock_file.exists()

    def test_release_nonexistent_lock(self, locking_storage):
        """Test that releasing a non-existent lock doesn't raise error."""
        # Should not raise any exception
        locking_storage.release_lock("nonexistent_lock")

    def test_multiple_different_locks(self, locking_storage):
        """Test acquiring multiple different locks simultaneously."""
        locks = ["lock1", "lock2", "lock3"]

        # Acquire all locks
        for lock_name in locks:
            assert locking_storage.try_acquire_lock(lock_name) is True

        # Verify all locks exist
        for lock_name in locks:
            lock_file = locking_storage._locks_dir / f"{lock_name}.lock"
            assert lock_file.exists()

        # Release all locks
        for lock_name in locks:
            locking_storage.release_lock(lock_name)

        # Verify all locks released
        for lock_name in locks:
            lock_file = locking_storage._locks_dir / f"{lock_name}.lock"
            assert not lock_file.exists()


class TestLocalStorageTTLStorage:
    """Test that TTL is correctly stored and used in LocalStorage."""

    def test_ttl_stored_in_lock_metadata(self, locking_storage):
        """Test that TTL is saved in the lock file."""
        lock_name = "test_lock"
        ttl = 1800  # 30 minutes

        # Acquire lock with specific TTL
        assert locking_storage.try_acquire_lock(lock_name, ttl=ttl) is True

        # Read lock file and verify TTL is stored
        lock_file = locking_storage._locks_dir / f"{lock_name}.lock"
        with open(lock_file, "r") as f:
            lock_data = json.load(f)

        assert "ttl" in lock_data
        assert lock_data["ttl"] == ttl
        assert "created_at" in lock_data
        assert "owner" in lock_data
        assert "pid" in lock_data

        locking_storage.release_lock(lock_name)

    def test_different_ttls_for_different_locks(self, locking_storage):
        """Test that different locks can have different TTLs."""
        locks = [
            ("lock_short", 60),
            ("lock_medium", 1800),
            ("lock_long", 7200),
        ]

        # Acquire all locks with different TTLs
        for lock_name, ttl in locks:
            assert locking_storage.try_acquire_lock(lock_name, ttl=ttl) is True

        # Verify each lock has correct TTL
        for lock_name, expected_ttl in locks:
            lock_file = locking_storage._locks_dir / f"{lock_name}.lock"
            with open(lock_file, "r") as f:
                lock_data = json.load(f)

            assert lock_data["ttl"] == expected_ttl

        # Cleanup
        for lock_name, _ in locks:
            locking_storage.release_lock(lock_name)

    def test_staleness_uses_stored_ttl(self, locking_storage):
        """Test that staleness check uses the TTL from lock metadata."""
        lock_name = "ttl_test_lock"
        short_ttl = 1  # 1 second

        # Acquire lock with short TTL
        assert locking_storage.try_acquire_lock(lock_name, ttl=short_ttl) is True

        # Immediately - should not be stale
        assert locking_storage._is_locked(lock_name) is True

        # Wait for TTL to expire
        time.sleep(short_ttl + 0.5)

        # Now should be stale
        assert locking_storage._is_locked(lock_name) is False

        # Should be able to steal it
        assert locking_storage.try_acquire_lock(lock_name, ttl=3600) is True

        locking_storage.release_lock(lock_name)

    def test_lock_without_ttl_treated_as_stale(self, locking_storage):
        """Test that locks without TTL field are treated as stale."""
        lock_name = "old_format_lock"
        lock_file = locking_storage._locks_dir / f"{lock_name}.lock"

        # Create lock file in old format (without TTL)
        old_format_data = {
            "owner": "old_system",
            "pid": 99999,
            "created_at": time.time(),
            # No TTL field
        }

        lock_file.parent.mkdir(parents=True, exist_ok=True)
        with open(lock_file, "w") as f:
            json.dump(old_format_data, f)

        # Should be treated as stale
        assert locking_storage._is_locked(lock_name) is False

        # Should be able to acquire
        assert locking_storage.try_acquire_lock(lock_name, ttl=3600) is True

        # Now should have TTL
        with open(lock_file, "r") as f:
            new_data = json.load(f)
        assert "ttl" in new_data

        locking_storage.release_lock(lock_name)

    def test_multiple_acquires_each_with_own_ttl(self, locking_storage):
        """Test that re-acquiring a lock updates the TTL."""
        lock_name = "changing_ttl_lock"

        # First acquisition with short TTL
        assert locking_storage.try_acquire_lock(lock_name, ttl=1) is True

        lock_file = locking_storage._locks_dir / f"{lock_name}.lock"
        with open(lock_file, "r") as f:
            data1 = json.load(f)
        assert data1["ttl"] == 1

        # Wait for it to become stale
        time.sleep(1.5)

        # Second acquisition with long TTL
        assert locking_storage.try_acquire_lock(lock_name, ttl=3600) is True

        with open(lock_file, "r") as f:
            data2 = json.load(f)
        assert data2["ttl"] == 3600

        # Should not be stale now (has long TTL)
        assert locking_storage._is_locked(lock_name) is True

        locking_storage.release_lock(lock_name)


class TestLockingLocalStorageWaiting:
    """Test wait_for_unlock functionality with exponential backoff."""

    def test_wait_for_unlock_success(self, locking_storage):
        """Test waiting for a lock that gets released."""
        lock_name = "wait_test_lock"

        # Acquire lock
        locking_storage.try_acquire_lock(lock_name)

        # Release lock in background after 1 second
        def release_after_delay():
            time.sleep(1)
            locking_storage.release_lock(lock_name)

        import threading

        thread = threading.Thread(target=release_after_delay)
        thread.start()

        # Wait for unlock (should succeed within 3 seconds)
        start = time.time()
        result = locking_storage.wait_for_unlock(
            lock_name,
            timeout=3,
            initial_poll_interval=0.1,  # ← Actualizado
            max_poll_interval=0.1,
            backoff_factor=1.0,
        )
        elapsed = time.time() - start

        assert result is True
        assert 0.9 < elapsed < 2.0  # Should take ~1 second

        thread.join()

    def test_wait_for_unlock_timeout(self, locking_storage):
        """Test timeout when waiting for a lock."""
        lock_name = "timeout_test_lock"

        # Acquire lock (and never release)
        locking_storage.try_acquire_lock(lock_name)

        # Wait should timeout
        start = time.time()
        result = locking_storage.wait_for_unlock(
            lock_name,
            timeout=1,
            initial_poll_interval=0.1,  # ← Actualizado
            max_poll_interval=0.1,
            backoff_factor=1.0,
        )
        elapsed = time.time() - start

        assert result is False
        assert 0.9 < elapsed < 1.5  # Should take ~1 second

        # Cleanup
        locking_storage.release_lock(lock_name)

    def test_wait_for_unlock_immediate_if_not_locked(self, locking_storage):
        """Test that waiting returns immediately if lock doesn't exist."""
        lock_name = "nonexistent_lock"

        start = time.time()
        result = locking_storage.wait_for_unlock(lock_name, timeout=5)
        elapsed = time.time() - start

        assert result is True
        assert elapsed < 0.5  # Should be immediate

    def test_wait_with_exponential_backoff(self, locking_storage):
        """Test that exponential backoff reduces polling frequency."""
        lock_name = "backoff_test_lock"

        # Create a lock
        locking_storage.try_acquire_lock(lock_name, ttl=3600)

        # Release after 2 seconds in background
        def release_delayed():
            time.sleep(2)
            locking_storage.release_lock(lock_name)

        import threading

        thread = threading.Thread(target=release_delayed)
        thread.start()

        # Wait with exponential backoff
        start = time.time()
        result = locking_storage.wait_for_unlock(
            lock_name,
            timeout=10,
            initial_poll_interval=0.1,
            max_poll_interval=2.0,
            backoff_factor=1.5,
        )
        elapsed = time.time() - start

        assert result is True
        assert 1.9 < elapsed < 3.0  # Should detect release around 2 seconds

        thread.join()


class TestLockingLocalStorageMultiprocessing:
    """Test locking behavior across multiple processes."""

    def test_multiprocess_lock_exclusion(self, locking_storage):
        """Test that only one process can acquire a lock."""
        lock_name = "multiprocess_lock"
        results = Queue()

        def try_acquire(storage_path, lock_name, process_id, results_queue):
            """Process function that tries to acquire lock."""
            storage = LockingLocalStorage(storage_path)
            acquired = storage.try_acquire_lock(lock_name, ttl=5)

            if acquired:
                # Hold lock for 2 seconds
                time.sleep(2)
                storage.release_lock(lock_name)

            results_queue.put((process_id, acquired))

        # Start 5 processes trying to acquire the same lock
        processes = []
        for i in range(5):
            p = Process(
                target=try_acquire,
                args=(locking_storage.storage_path.rstrip("/"), lock_name, i, results),
            )
            p.start()
            processes.append(p)

        # Wait for all processes
        for p in processes:
            p.join()

        # Collect results
        acquisitions = []
        while not results.empty():
            acquisitions.append(results.get())

        # Verify only ONE process acquired the lock
        successful = [r for r in acquisitions if r[1] is True]
        assert (
            len(successful) == 1
        ), f"Expected 1 successful acquisition, got {len(successful)}"

        failed = [r for r in acquisitions if r[1] is False]
        assert len(failed) == 4, f"Expected 4 failed acquisitions, got {len(failed)}"

    def test_multiprocess_sequential_access(self, locking_storage):
        """Test that multiple processes can sequentially access a resource."""
        lock_name = "sequential_lock"
        shared_counter_file = Path(locking_storage.storage_path) / "counter.txt"
        shared_counter_file.parent.mkdir(parents=True, exist_ok=True)
        shared_counter_file.write_text("0")

        def increment_counter(storage_path, lock_name, process_id):
            """Process function that increments a shared counter."""
            storage = LockingLocalStorage(storage_path)
            counter_file = Path(storage_path) / "counter.txt"

            # Try to acquire lock with retries
            max_retries = 10
            for _ in range(max_retries):
                if storage.try_acquire_lock(lock_name, ttl=5):
                    try:
                        # Read current value
                        current = int(counter_file.read_text())
                        time.sleep(0.1)  # Simulate work
                        # Write incremented value
                        counter_file.write_text(str(current + 1))
                    finally:
                        storage.release_lock(lock_name)
                    return
                time.sleep(0.2)

            raise Exception(f"Process {process_id} couldn't acquire lock")

        # Start 10 processes
        processes = []
        for i in range(10):
            p = Process(
                target=increment_counter,
                args=(locking_storage.storage_path.rstrip("/"), lock_name, i),
            )
            p.start()
            processes.append(p)

        # Wait for all processes
        for p in processes:
            p.join()
            assert p.exitcode == 0, f"Process failed with exit code {p.exitcode}"

        # Verify counter reached 10 (all increments succeeded)
        final_value = int(shared_counter_file.read_text())
        assert final_value == 10, f"Expected counter=10, got {final_value}"


class TestLockingLocalStorageIntegration:
    """Test integration with existing LocalStorage functionality."""

    def test_storage_operations_work_with_locking(self, locking_storage):
        """Test that regular storage operations still work."""
        # Upload file
        test_data = {"key": "value"}
        result = locking_storage.upload_file(
            test_data, "test.pkl", locking_storage.get_root_path()
        )
        assert result == "test.pkl"

        # Check exists
        assert (
            locking_storage.check_if_exists("test.pkl", locking_storage.get_root_path())
            is True
        )

        # Download file
        downloaded = locking_storage.download_file(
            "test.pkl", locking_storage.get_root_path()
        )
        assert downloaded == test_data

        # Delete file
        locking_storage.delete_file("test.pkl", locking_storage.get_root_path())
        assert (
            locking_storage.check_if_exists("test.pkl", locking_storage.get_root_path())
            is False
        )

    def test_locks_directory_separate_from_data(self, locking_storage):
        """Test that locks are stored separately from data files."""
        # Upload data
        locking_storage.upload_file("test", "data.pkl", "")

        # Acquire lock
        locking_storage.try_acquire_lock("lock1")

        # Verify structure
        data_file = Path(locking_storage.storage_path) / "data.pkl"
        lock_file = locking_storage._locks_dir / "lock1.lock"

        assert data_file.exists()
        assert lock_file.exists()
        assert data_file.parent != lock_file.parent

        # Cleanup
        locking_storage.release_lock("lock1")
        locking_storage.delete_file("data.pkl", "")


class TestLockingLocalStorageEdgeCases:
    """Test edge cases and error conditions."""

    def test_lock_with_special_characters(self, locking_storage):
        """Test locks with special characters in name."""
        lock_names = [
            "lock_with_underscore",
            "lock-with-dash",
            "lock.with.dots",
            "lock123numbers",
        ]

        for lock_name in lock_names:
            assert locking_storage.try_acquire_lock(lock_name) is True
            assert locking_storage._is_locked(lock_name) is True
            locking_storage.release_lock(lock_name)
            assert locking_storage._is_locked(lock_name) is False

    def test_very_short_ttl(self, locking_storage):
        """Test locks with very short TTL."""
        lock_name = "short_ttl_lock"
        ttl = 1  # 1 second

        assert locking_storage.try_acquire_lock(lock_name, ttl=ttl) is True
        time.sleep(1.2)

        # Should be stale
        assert locking_storage._is_locked(lock_name) is False

        # Should be able to reacquire
        assert locking_storage.try_acquire_lock(lock_name, ttl=ttl) is True

        locking_storage.release_lock(lock_name)

    def test_acquire_release_many_times(self, locking_storage):
        """Test acquiring and releasing the same lock many times."""
        lock_name = "repeated_lock"

        for i in range(100):
            assert locking_storage.try_acquire_lock(lock_name) is True
            locking_storage.release_lock(lock_name)

        # Should still work after 100 iterations
        assert locking_storage.try_acquire_lock(lock_name) is True
        locking_storage.release_lock(lock_name)

    def test_verbose_output(self, locking_storage, capsys):
        """Test that verbose mode produces output."""
        locking_storage._verbose = True
        lock_name = "verbose_lock"

        locking_storage.try_acquire_lock(lock_name)
        captured = capsys.readouterr()
        assert "Lock acquired" in captured.out or "🔒" in captured.out

        locking_storage.release_lock(lock_name)
        captured = capsys.readouterr()
        assert "Lock released" in captured.out or "🔓" in captured.out
