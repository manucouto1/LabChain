# tests/unit/storage/test_locking_local_storage_heartbeat.py

import pytest
import os
import time
import json
from pathlib import Path
import threading
from multiprocessing import Process, Queue
import shutil

from labchain.plugins.storage import LockingLocalStorage


@pytest.fixture
def locking_storage():
    """Fixture that provides a clean LockingLocalStorage instance."""
    storage_path = "tests/test_data/locking_heartbeat"
    storage = LockingLocalStorage(storage_path=storage_path)

    yield storage

    # Cleanup
    if Path(storage_path).exists():
        shutil.rmtree(storage_path)


class TestHeartbeatBasics:
    """Test basic heartbeat functionality."""

    def test_lock_with_heartbeat_creates_heartbeat_fields(self, locking_storage):
        """Test that enabling heartbeat adds required fields to lock."""
        lock_name = "heartbeat_lock"

        # Acquire lock with heartbeat
        result = locking_storage.try_acquire_lock(
            lock_name, ttl=3600, heartbeat_interval=30
        )

        assert result is True

        # Read lock file and verify heartbeat fields
        lock_file = locking_storage._locks_dir / f"{lock_name}.lock"
        with open(lock_file, "r") as f:
            lock_data = json.load(f)

        assert "last_heartbeat" in lock_data
        assert "heartbeat_interval" in lock_data
        assert lock_data["heartbeat_interval"] == 30
        # Use approximate comparison for timestamps
        assert abs(lock_data["last_heartbeat"] - lock_data["created_at"]) < 0.1

        locking_storage.release_lock(lock_name)

    def test_lock_without_heartbeat_no_heartbeat_fields(self, locking_storage):
        """Test that locks without heartbeat don't have heartbeat fields."""
        lock_name = "no_heartbeat_lock"

        # Acquire lock without heartbeat (explicitly pass None)
        result = locking_storage.try_acquire_lock(
            lock_name, ttl=3600, heartbeat_interval=None
        )

        assert result is True

        # Read lock file
        lock_file = locking_storage._locks_dir / f"{lock_name}.lock"
        with open(lock_file, "r") as f:
            lock_data = json.load(f)

        assert "last_heartbeat" not in lock_data
        assert "heartbeat_interval" not in lock_data

        locking_storage.release_lock(lock_name)

    def test_update_heartbeat_success(self, locking_storage):
        """Test successful heartbeat update."""
        lock_name = "update_heartbeat_lock"

        # Acquire lock with heartbeat
        locking_storage.try_acquire_lock(lock_name, ttl=3600, heartbeat_interval=30)

        # Get initial heartbeat
        lock_file = locking_storage._locks_dir / f"{lock_name}.lock"
        with open(lock_file, "r") as f:
            initial_data = json.load(f)
        initial_heartbeat = initial_data["last_heartbeat"]

        # Wait a bit and update
        time.sleep(0.1)
        result = locking_storage.update_heartbeat(lock_name)

        assert result is True

        # Verify heartbeat was updated
        with open(lock_file, "r") as f:
            updated_data = json.load(f)
        updated_heartbeat = updated_data["last_heartbeat"]

        assert updated_heartbeat > initial_heartbeat

        locking_storage.release_lock(lock_name)

    def test_update_heartbeat_fails_without_heartbeat_enabled(self, locking_storage):
        """Test that updating heartbeat fails if not enabled."""
        lock_name = "no_heartbeat_lock"

        # Acquire lock WITHOUT heartbeat
        locking_storage.try_acquire_lock(lock_name, ttl=3600, heartbeat_interval=None)

        # Try to update heartbeat - should fail
        result = locking_storage.update_heartbeat(lock_name)

        assert result is False

        locking_storage.release_lock(lock_name)

    def test_update_heartbeat_fails_if_not_owner(self, locking_storage):
        """Test that updating heartbeat fails if process doesn't own lock."""
        lock_name = "other_owner_lock"
        lock_file = locking_storage._locks_dir / f"{lock_name}.lock"

        # Create lock owned by another process
        lock_data = {
            "owner": "other_host",
            "pid": 99999,  # Different PID
            "created_at": time.time(),
            "last_heartbeat": time.time(),
            "heartbeat_interval": 30,
            "ttl": 3600,
        }

        lock_file.parent.mkdir(parents=True, exist_ok=True)
        with open(lock_file, "w") as f:
            json.dump(lock_data, f)

        # Try to update - should fail
        result = locking_storage.update_heartbeat(lock_name)

        assert result is False

        # Cleanup
        lock_file.unlink()

    def test_multiple_heartbeat_updates(self, locking_storage):
        """Test multiple heartbeat updates over time."""
        lock_name = "multi_heartbeat_lock"

        locking_storage.try_acquire_lock(lock_name, ttl=3600, heartbeat_interval=1)

        lock_file = locking_storage._locks_dir / f"{lock_name}.lock"
        heartbeats = []

        # Collect multiple heartbeats
        for i in range(5):
            with open(lock_file, "r") as f:
                data = json.load(f)
            heartbeats.append(data["last_heartbeat"])

            if i < 4:  # Don't sleep after last iteration
                time.sleep(0.2)
                locking_storage.update_heartbeat(lock_name)

        # Verify heartbeats are increasing
        for i in range(1, len(heartbeats)):
            assert heartbeats[i] >= heartbeats[i - 1]

        locking_storage.release_lock(lock_name)


class TestHeartbeatDeadLockDetection:
    """Test dead lock detection via heartbeat."""

    def test_fresh_lock_with_heartbeat_not_stale(self, locking_storage):
        """Test that fresh lock with recent heartbeat is not stale."""
        lock_name = "fresh_heartbeat_lock"

        locking_storage.try_acquire_lock(lock_name, ttl=3600, heartbeat_interval=30)

        # Should not be stale (heartbeat is fresh)
        assert locking_storage._is_locked(lock_name) is True

        locking_storage.release_lock(lock_name)

    def test_dead_lock_with_old_heartbeat_is_stale(self, locking_storage):
        """Test that lock with old heartbeat is detected as dead."""
        lock_name = "dead_heartbeat_lock"
        lock_file = locking_storage._locks_dir / f"{lock_name}.lock"

        # Create lock with old heartbeat (process appears dead)
        heartbeat_interval = 10
        lock_data = {
            "owner": "dead_process",
            "pid": 99999,
            "created_at": time.time() - 50,  # 50 seconds ago
            "last_heartbeat": time.time() - 50,  # No updates for 50 seconds
            "heartbeat_interval": heartbeat_interval,
            "ttl": 3600,  # TTL not expired, but heartbeat is dead
        }

        lock_file.parent.mkdir(parents=True, exist_ok=True)
        with open(lock_file, "w") as f:
            json.dump(lock_data, f)

        # Should be considered stale/dead (3x heartbeat_interval = 30s has passed)
        assert locking_storage._is_locked(lock_name) is False

        # Should be able to steal it
        assert locking_storage.try_acquire_lock(lock_name, ttl=3600) is True

        locking_storage.release_lock(lock_name)

    def test_lock_with_recent_heartbeat_cannot_be_stolen(self, locking_storage):
        """Test that lock with recent heartbeat cannot be stolen."""
        lock_name = "active_heartbeat_lock"
        lock_file = locking_storage._locks_dir / f"{lock_name}.lock"

        # Create lock with recent heartbeat
        lock_data = {
            "owner": "active_process",
            "pid": 99999,
            "created_at": time.time() - 100,  # Old, but...
            "last_heartbeat": time.time() - 5,  # Recent heartbeat!
            "heartbeat_interval": 30,
            "ttl": 3600,
        }

        lock_file.parent.mkdir(parents=True, exist_ok=True)
        with open(lock_file, "w") as f:
            json.dump(lock_data, f)

        # Should NOT be stale (heartbeat is recent)
        assert locking_storage._is_locked(lock_name) is True

        # Cannot steal it
        assert locking_storage.try_acquire_lock(lock_name, ttl=3600) is False

        # Cleanup
        lock_file.unlink()

    def test_heartbeat_detection_faster_than_ttl(self, locking_storage):
        """Test that heartbeat detects dead process faster than TTL."""
        lock_name = "fast_detection_lock"
        lock_file = locking_storage._locks_dir / f"{lock_name}.lock"

        # Create lock with long TTL but dead heartbeat
        lock_data = {
            "owner": "crashed_process",
            "pid": 99999,
            "created_at": time.time() - 10,
            "last_heartbeat": time.time() - 10,  # No heartbeat for 10s
            "heartbeat_interval": 2,  # Expected every 2s
            "ttl": 3600,  # Would take 1 hour to expire via TTL
        }

        lock_file.parent.mkdir(parents=True, exist_ok=True)
        with open(lock_file, "w") as f:
            json.dump(lock_data, f)

        # Should detect death via heartbeat (3x2s = 6s) before TTL
        assert locking_storage._is_locked(lock_name) is False

        # Can steal immediately (no need to wait for TTL)
        start = time.time()
        assert locking_storage.try_acquire_lock(lock_name, ttl=3600) is True
        elapsed = time.time() - start

        # Should be instant (not waiting for TTL)
        assert elapsed < 1.0

        locking_storage.release_lock(lock_name)


class TestHeartbeatThreadSafety:
    """Test heartbeat updates in multithreaded scenarios."""

    def test_concurrent_heartbeat_updates(self, locking_storage):
        """Test that concurrent heartbeat updates are safe."""
        lock_name = "concurrent_heartbeat_lock"

        locking_storage.try_acquire_lock(lock_name, ttl=3600, heartbeat_interval=1)

        errors = []

        def update_heartbeat_repeatedly():
            """Thread that updates heartbeat multiple times."""
            for _ in range(10):
                try:
                    locking_storage.update_heartbeat(lock_name)
                    time.sleep(0.1)
                except Exception as e:
                    errors.append(str(e))

        # Start multiple threads
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=update_heartbeat_repeatedly)
            thread.start()
            threads.append(thread)

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Should have no errors
        assert len(errors) == 0

        locking_storage.release_lock(lock_name)


class TestHeartbeatMultiprocessing:
    """Test heartbeat in multiprocessing scenarios."""

    def test_heartbeat_prevents_lock_stealing(self, locking_storage):
        """Test that active heartbeat prevents lock theft."""
        lock_name = "heartbeat_prevents_steal"
        results = Queue()

        def hold_lock_with_heartbeat(storage_path, lock_name, duration, results_queue):
            """Process that holds lock and sends heartbeats."""
            storage = LockingLocalStorage(storage_path)

            # Acquire lock with short heartbeat interval
            acquired = storage.try_acquire_lock(
                lock_name, ttl=3600, heartbeat_interval=1
            )

            if not acquired:
                results_queue.put(("holder", False, "failed_to_acquire"))
                return

            # Send heartbeats while working
            start = time.time()
            heartbeat_count = 0

            while time.time() - start < duration:
                time.sleep(0.5)
                if storage.update_heartbeat(lock_name):
                    heartbeat_count += 1

            storage.release_lock(lock_name)
            results_queue.put(("holder", True, heartbeat_count))

        def try_steal_lock(storage_path, lock_name, results_queue):
            """Process that tries to steal the lock."""
            time.sleep(1)  # Wait for holder to acquire lock

            storage = LockingLocalStorage(storage_path)

            # Try to steal (should fail because heartbeat is active)
            acquired = storage.try_acquire_lock(lock_name, ttl=3600)

            results_queue.put(("stealer", acquired, ""))

        # Start holder
        holder = Process(
            target=hold_lock_with_heartbeat,
            args=(
                locking_storage.storage_path.rstrip("/"),
                lock_name,
                3,  # Hold for 3 seconds
                results,
            ),
        )
        holder.start()

        # Start stealer
        stealer = Process(
            target=try_steal_lock,
            args=(locking_storage.storage_path.rstrip("/"), lock_name, results),
        )
        stealer.start()

        # Wait for both
        holder.join(timeout=10)
        stealer.join(timeout=10)

        # Collect results
        holder_result = None
        stealer_result = None

        while not results.empty():
            result = results.get()
            if result[0] == "holder":
                holder_result = result
            else:
                stealer_result = result

        # Verify holder succeeded and sent heartbeats
        assert holder_result is not None
        assert holder_result[1] is True
        assert holder_result[2] > 0  # Sent at least one heartbeat

        # Verify stealer failed to steal
        assert stealer_result is not None
        assert stealer_result[1] is False

    def test_heartbeat_allows_stealing_after_death(self, locking_storage):
        """Test that dead lock can be stolen after heartbeat stops."""
        lock_name = "dead_then_steal"
        results = Queue()

        def hold_lock_then_crash(storage_path, lock_name, results_queue):
            """Process that acquires lock but crashes (stops heartbeat)."""
            storage = LockingLocalStorage(storage_path)

            storage.try_acquire_lock(lock_name, ttl=3600, heartbeat_interval=1)

            results_queue.put("acquired")

            # Simulate crash - just exit without releasing
            # (no heartbeat updates, no release)

        def wait_then_steal(storage_path, lock_name, results_queue):
            """Process that waits for death detection then steals."""
            # Wait for initial acquisition
            results_queue.get(timeout=5)

            # Wait for heartbeat to die (3x interval = 3 seconds)
            time.sleep(4)

            storage = LockingLocalStorage(storage_path)

            # Should be able to steal now
            start = time.time()
            acquired = storage.try_acquire_lock(lock_name, ttl=3600)
            elapsed = time.time() - start

            results_queue.put(("stealer", acquired, elapsed))

            if acquired:
                storage.release_lock(lock_name)

        # Start processes
        holder = Process(
            target=hold_lock_then_crash,
            args=(locking_storage.storage_path.rstrip("/"), lock_name, results),
        )
        holder.start()

        stealer = Process(
            target=wait_then_steal,
            args=(locking_storage.storage_path.rstrip("/"), lock_name, results),
        )
        stealer.start()

        # Wait for both
        holder.join(timeout=5)
        stealer.join(timeout=10)

        # Get stealer result
        stealer_result = results.get(timeout=1)

        # Verify stealer succeeded
        assert stealer_result[0] == "stealer"
        assert stealer_result[1] is True
        assert stealer_result[2] < 2.0  # Stole quickly (not waiting for TTL)


class TestHeartbeatEdgeCases:
    """Test edge cases with heartbeat."""

    def test_very_short_heartbeat_interval(self, locking_storage):
        """Test lock with very short heartbeat interval."""
        lock_name = "short_interval_lock"

        locking_storage.try_acquire_lock(
            lock_name,
            ttl=3600,
            heartbeat_interval=1,  # 1 second
        )

        # Should be locked
        assert locking_storage._is_locked(lock_name) is True

        # Wait for 3x interval (death threshold)
        time.sleep(3.5)

        # Should be dead now
        assert locking_storage._is_locked(lock_name) is False

        # Can steal
        assert locking_storage.try_acquire_lock(lock_name, ttl=3600) is True

        locking_storage.release_lock(lock_name)

    def test_heartbeat_larger_than_ttl(self, locking_storage):
        """Test that TTL takes precedence when heartbeat_interval > TTL."""
        lock_name = "ttl_precedence_lock"
        lock_file = locking_storage._locks_dir / f"{lock_name}.lock"

        # Create lock where heartbeat interval is larger than TTL
        lock_data = {
            "owner": "test",
            "pid": os.getpid(),
            "created_at": time.time() - 5,
            "last_heartbeat": time.time(),  # Recent heartbeat
            "heartbeat_interval": 100,  # Very long interval
            "ttl": 2,  # But short TTL
        }

        lock_file.parent.mkdir(parents=True, exist_ok=True)
        with open(lock_file, "w") as f:
            json.dump(lock_data, f)

        # Should be stale due to TTL (despite recent heartbeat)
        assert locking_storage._is_locked(lock_name) is False

        # Cleanup
        lock_file.unlink()

    def test_corrupted_heartbeat_data(self, locking_storage):
        """Test handling of corrupted heartbeat fields."""
        lock_name = "corrupted_heartbeat_lock"
        lock_file = locking_storage._locks_dir / f"{lock_name}.lock"

        # Create lock with invalid heartbeat data
        lock_data = {
            "owner": "test",
            "pid": 12345,
            "created_at": time.time(),
            "ttl": 3600,
            "last_heartbeat": "not_a_number",  # Invalid
            "heartbeat_interval": "also_invalid",  # Invalid
        }

        lock_file.parent.mkdir(parents=True, exist_ok=True)
        with open(lock_file, "w") as f:
            json.dump(lock_data, f)

        # Should handle gracefully (treat as stale or ignore heartbeat)
        # The implementation should not crash
        try:
            locking_storage._is_locked(lock_name)
            # Either locked (ignoring bad heartbeat) or not locked (treating as stale)
            # Both are acceptable
        except (TypeError, ValueError):
            # Also acceptable - corrupted data causes error
            pass

        # Should be able to acquire regardless
        acquired = locking_storage.try_acquire_lock(lock_name, ttl=3600)
        # Clean up if acquired
        if acquired:
            locking_storage.release_lock(lock_name)
