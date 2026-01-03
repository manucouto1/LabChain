import os
import json
import time
import socket
from pathlib import Path
from labchain.plugins.storage import LocalStorage
from labchain.base.base_storage import BaseLockingStorage

__all__ = ["LockingLocalStorage"]


class LockingLocalStorage(LocalStorage, BaseLockingStorage):
    """Local filesystem storage with atomic file-based locking.

        Extends LocalStorage to add distributed locking capabilities using atomic
        file operations. Lock atomicity is guaranteed through O_CREAT | O_EXCL flags,
        which provide kernel-level atomic file creation.

        This implementation is ideal for:
        - Single-machine parallel processing
        - Development and testing
        - CI/CD pipelines
        - Local experimentation
        - Multi-process training on the same machine

        The locking mechanism is safe for multiple processes on the same machine
        accessing the same filesystem (including NFS with proper configuration).

        Attributes:
            storage_path: Root directory for all storage operations (inherited).
            locks_dir: Directory where lock files are stored.

        Examples:
            Basic usage with locking:
    ```python
            from framework3.plugins.storage import LockingLocalStorage

            # Initialize storage
            storage = LockingLocalStorage(storage_path='./cache')

            # Use storage operations (same as LocalStorage)
            storage.upload_file(model, "model.pkl", "models/")

            # Use locking for concurrent operations
            lock_name = "train_model_abc123"
            if storage.try_acquire_lock(lock_name, ttl=3600):
                try:
                    # Train model - exclusive access guaranteed
                    train_expensive_model()
                    storage.upload_file(model, "model.pkl", "models/")
                finally:
                    storage.release_lock(lock_name)
            else:
                # Another process is training
                storage.wait_for_unlock(lock_name)
                model = storage.download_file("model.pkl", "models/")
    ```

            Parallel processing example:
    ```python
            from multiprocessing import Pool

            def train_model(model_id):
                storage = LockingLocalStorage("./cache")
                lock_name = f"model_{model_id}"

                if storage.try_acquire_lock(lock_name):
                    try:
                        print(f"🔨 Training {model_id}")
                        # Train model (only one process enters here)
                        train()
                    finally:
                        storage.release_lock(lock_name)
                else:
                    print(f"⏳ Waiting for {model_id}")
                    storage.wait_for_unlock(lock_name)
                    # Load cached model

            # Run 4 processes in parallel
            # Each unique model trains only once
            with Pool(4) as p:
                p.map(train_model, ["model_1", "model_1", "model_2", "model_2"])
    ```

        Note:
            Lock files are stored in a dedicated "locks/" subdirectory and contain
            metadata (hostname, PID, timestamp) for debugging and stale detection.
    """

    def __init__(self, storage_path: str = "cache/"):
        """Initialize locking-enabled local storage.

                Creates the base directory and locks subdirectory if they don't exist.

                Args:
                    storage_path: Root directory for storage. Can be relative or absolute.
                        Defaults to "cache/". Will be created if it doesn't exist.

                Examples:
        ```python
                    # Default cache directory
                    storage = LockingLocalStorage()

                    # Custom path
                    storage = LockingLocalStorage("./ml_cache")

                    # Absolute path
                    storage = LockingLocalStorage("/var/ml-experiments/cache")

                    # User home directory
                    storage = LockingLocalStorage("~/projects/cache")
        ```
        """
        super().__init__(storage_path=storage_path)
        self._locks_dir = Path(self.storage_path) / "locks"
        self._locks_dir.mkdir(parents=True, exist_ok=True)

    def try_acquire_lock(
        self, lock_name: str, ttl: int = 3600, heartbeat_interval: int | None = 30
    ) -> bool:
        """Acquire lock using atomic file creation (O_CREAT | O_EXCL).

                Uses POSIX O_EXCL flag to guarantee atomic lock acquisition at the
                kernel level. This operation is safe across multiple processes and
                works on network filesystems like NFS (with proper configuration).

                Stale locks (older than TTL) are automatically detected and stolen,
                ensuring the system recovers from crashed processes.

                Args:
                    lock_name: Unique lock identifier (e.g., "model_abc123").
                    ttl: Lock validity duration in seconds. After this time, the lock
                        is considered stale and can be stolen. Default 3600 (1 hour).
                    heartbeat_interval: If provided, enables heartbeat-based crash
                        detection. Should be much smaller than ttl (e.g., ttl/10).
                        Default None (no heartbeat).


                Returns:
                    True if lock was acquired successfully, False if another process
                    holds the lock.

                Examples:
        ```python
                    # Short-lived lock for quick operations
                    if storage.try_acquire_lock("cache_data", ttl=300):
                        process_data()
                        storage.release_lock("cache_data")

                    # Long-lived lock for training
                    if storage.try_acquire_lock("train_model", ttl=7200):
                        train_for_hours()
                        storage.release_lock("train_model")

                    # Always use try-finally for safety
                    lock_acquired = storage.try_acquire_lock("my_lock")
                    if lock_acquired:
                        try:
                            critical_operation()
                        finally:
                            storage.release_lock("my_lock")
        ```

                Note:
                    Lock metadata is stored as JSON for debugging:
        ```json
                    {
                        "owner": "hostname.local",
                        "pid": 12345,
                        "created_at": 1735562400.123
                    }
        ```

                    This helps identify which process holds the lock and when it
                    was acquired, useful for debugging deadlocks or stale locks.
        """
        lock_path = self._locks_dir / f"{lock_name}.lock"

        # Check if lock exists and is stale
        if lock_path.exists():
            if self._is_lock_stale_or_dead(lock_path, heartbeat_interval):
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    pass
            else:
                return False

        # Create lock with heartbeat info
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)

            lock_data = {
                "owner": socket.gethostname(),
                "pid": os.getpid(),
                "created_at": time.time(),
                "ttl": ttl,
            }

            if heartbeat_interval is not None:
                lock_data["last_heartbeat"] = time.time()
                lock_data["heartbeat_interval"] = heartbeat_interval

            os.write(fd, json.dumps(lock_data).encode())
            os.close(fd)

            if self._verbose:
                hb_info = f", HB: {heartbeat_interval}s" if heartbeat_interval else ""
                print(f"\t🔒 Lock acquired: {lock_name} (TTL: {ttl}s{hb_info})")

            return True

        except FileExistsError:
            return False

    def update_heartbeat(self, lock_name: str) -> bool:
        """Update the heartbeat timestamp of a held lock.

                Should be called periodically by the lock holder to indicate
                it's still alive and working.

                Args:
                    lock_name: Lock identifier.

                Returns:
                    True if heartbeat updated successfully, False if lock doesn't exist
                    or is held by another process.

                Examples:
        ```python
                    if storage.try_acquire_lock("train_model"):
                        try:
                            for epoch in range(100):
                                train_epoch()
                                # Update heartbeat every epoch
                                if epoch % 10 == 0:
                                    storage.update_heartbeat("train_model")
                        finally:
                            storage.release_lock("train_model")
        ```
        """
        lock_path = self._locks_dir / f"{lock_name}.lock"

        try:
            # Read current lock data
            with open(lock_path, "r") as f:
                lock_data = json.load(f)

            # Verify we own this lock
            if lock_data.get("pid") != os.getpid():
                if self._verbose:
                    print(
                        f"\t⚠️ Cannot update heartbeat: lock owned by PID {lock_data.get('pid')}"
                    )
                return False

            # Check if heartbeat is enabled for this lock
            if "last_heartbeat" not in lock_data:
                if self._verbose:
                    print(f"\t⚠️ Heartbeat not enabled for lock: {lock_name}")
                return False

            # Update heartbeat
            lock_data["last_heartbeat"] = time.time()

            # Write back atomically
            temp_path = lock_path.with_suffix(".lock.tmp")
            with open(temp_path, "w") as f:
                json.dump(lock_data, f)

            temp_path.replace(lock_path)

            if self._verbose:
                age = time.time() - lock_data["created_at"]
                print(f"\t💓 Heartbeat updated: {lock_name} (age: {age:.0f}s)")

            return True

        except (FileNotFoundError, json.JSONDecodeError, KeyError, OSError) as e:
            if self._verbose:
                print(f"\t⚠️ Failed to update heartbeat: {e}")
            return False

    def release_lock(self, lock_name: str) -> None:
        """Release lock by deleting the lock file.

                Removes the lock file to make the lock available for other processes.
                Safe to call multiple times or if lock doesn't exist.

                Args:
                    lock_name: Lock identifier to release.

                Examples:
        ```python
                    # Standard pattern with try-finally
                    if storage.try_acquire_lock("my_lock"):
                        try:
                            do_critical_work()
                        finally:
                            storage.release_lock("my_lock")

                    # Safe to call even if not locked
                    storage.release_lock("might_not_exist")  # No error
        ```

                Note:
                    Always release locks in a finally block to prevent deadlocks,
                    even if an exception occurs during the critical section.
        """
        lock_path = self._locks_dir / f"{lock_name}.lock"
        try:
            lock_path.unlink()
            if self._verbose:
                print(f"\t🔓 Lock released: {lock_name}")
        except FileNotFoundError:
            pass  # Already released

    def _is_locked(self, lock_name: str) -> bool:
        """Check if lock exists and is not stale.

        Internal method used by wait_for_unlock to check lock status.
        Considers TTL when determining if lock is valid.

        Args:
            lock_name: Lock identifier to check.

        Returns:
            True if lock exists and is valid (not expired), False otherwise.
        """
        lock_path = self._locks_dir / f"{lock_name}.lock"
        if not lock_path.exists():
            return False
        return not self._is_lock_stale_or_dead(lock_path)

    def _is_lock_stale_or_dead(
        self, lock_path: Path, max_heartbeat_age: int | None = None
    ) -> bool:
        """Check if lock is stale (TTL expired) or dead (heartbeat stopped).

        A lock is considered dead if:
        1. Its heartbeat is older than 3 * heartbeat_interval
        2. The process is no longer running (optional check)

        Args:
            lock_path: Path to lock file.
            max_heartbeat_age: Maximum age for heartbeat. If None, uses 3x the
                lock's heartbeat_interval.

        Returns:
            True if lock should be considered stale or dead.
        """
        try:
            with open(lock_path, "r") as f:
                lock_data = json.load(f)

            current_time = time.time()

            # Check 1: TTL expiry
            ttl = lock_data.get("ttl")
            if ttl is None:
                return True

            created_at = lock_data["created_at"]
            age = current_time - created_at

            if age > ttl:
                if self._verbose:
                    print(f"\t⏰ Lock expired (age: {age:.0f}s > TTL: {ttl}s)")
                return True

            # Check 2: Heartbeat death detection (if enabled)
            last_heartbeat = lock_data.get("last_heartbeat")
            if last_heartbeat is not None:
                heartbeat_interval = lock_data.get("heartbeat_interval", 30)
                heartbeat_age = current_time - last_heartbeat

                max_heartbeat_age = heartbeat_interval * 3

                if heartbeat_age > max_heartbeat_age:
                    if self._verbose:
                        print(
                            f"\t💀 Lock appears dead (heartbeat age: {heartbeat_age:.0f}s)"
                        )
                    return True

            return False

        except (
            json.JSONDecodeError,
            KeyError,
            FileNotFoundError,
            OSError,
            TypeError,
            ValueError,
        ):
            # Cualquier problema con el lock → tratarlo como stale
            return True
