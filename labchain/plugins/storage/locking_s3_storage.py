import json
import time
import socket
import os
from botocore.exceptions import ClientError
from labchain.plugins.storage import S3Storage
from labchain.base.base_storage import BaseLockingStorage

__all__ = ["LockingS3Storage"]


class LockingS3Storage(S3Storage, BaseLockingStorage):
    """AWS S3 storage with atomic S3-native distributed locking.

        Extends S3Storage to add distributed locking capabilities using S3's
        atomic operations. Provides true distributed coordination across any
        number of machines accessing the same S3 bucket.

        This implementation is ideal for:
        - Multi-machine distributed training
        - Cloud-based ML pipelines
        - Cross-datacenter synchronization
        - Production ML systems
        - Kubernetes/container environments

        Lock atomicity is achieved through S3's PUT operations, which are
        atomic at the object level. Works with any S3-compatible service
        (AWS S3, MinIO, DigitalOcean Spaces, etc.).

        Attributes:
            bucket: S3 bucket name (inherited).
            storage_path: Prefix for all keys (inherited).
            _client: Boto3 S3 client (inherited).

        Examples:
            Basic usage with locking:
    ```python
            from framework3.plugins.storage import LockingS3Storage

            # Initialize storage
            storage = LockingS3Storage(
                bucket='ml-training-cache',
                region_name='us-west-2',
                access_key_id='YOUR_KEY',
                access_key='YOUR_SECRET'
            )

            # Use storage operations (same as S3Storage)
            storage.upload_file(model, "model.pkl", "models/")

            # Use locking for distributed coordination
            lock_name = "train_model_abc123"
            if storage.try_acquire_lock(lock_name, ttl=7200):
                try:
                    # Only ONE EC2 instance trains
                    train_model()
                    storage.upload_file(model, "model.pkl", "models/")
                finally:
                    storage.release_lock(lock_name)
            else:
                # All other instances wait
                storage.wait_for_unlock(lock_name)
                model = storage.download_file("model.pkl", "models/")
    ```

            Distributed training across EC2 instances:
    ```python
            # This code runs on multiple EC2 instances simultaneously
            storage = LockingS3Storage(
                bucket='company-ml-cache',
                region_name='us-east-1',
                access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
            )

            model_hash = compute_model_hash(params)
            lock_name = f"model_{model_hash}"

            if storage.try_acquire_lock(lock_name, ttl=2*3600):
                # Only ONE instance across ALL machines enters here
                try:
                    print(f"🔨 Instance {os.getenv('HOSTNAME')} training...")
                    model = train_expensive_model()
                    storage.upload_file(model, f"{model_hash}.pkl", "models/")
                    print("✅ Model trained and uploaded")
                finally:
                    storage.release_lock(lock_name)
            else:
                # All other instances wait here
                print(f"⏳ Instance {os.getenv('HOSTNAME')} waiting...")
                if storage.wait_for_unlock(lock_name, timeout=3*3600):
                    model = storage.download_file(f"{model_hash}.pkl", "models/")
                    print("📥 Model downloaded from cache")
                else:
                    raise TimeoutError("Training timeout exceeded")
    ```

            MinIO (S3-compatible) configuration:
    ```python
            # Works with MinIO or any S3-compatible service
            storage = LockingS3Storage(
                bucket='ml-cache',
                region_name='us-east-1',
                access_key_id='minioadmin',
                access_key='minioadmin',
                endpoint_url='http://minio.company.com:9000'
            )
    ```

        Note:
            Requires boto3 and proper AWS credentials. Credentials can be provided
            via constructor arguments, environment variables, ~/.aws/credentials,
            or IAM roles (recommended for EC2/ECS).
    """

    def __init__(
        self,
        bucket: str,
        region_name: str,
        access_key_id: str,
        access_key: str,
        endpoint_url: str | None = None,
        storage_path: str = "",
    ):
        """Initialize S3 storage with locking capabilities.

                Args:
                    bucket: S3 bucket name. Must already exist.
                    region_name: AWS region name (e.g., 'us-west-2').
                    access_key_id: AWS access key ID.
                    access_key: AWS secret access key.
                    endpoint_url: Optional endpoint URL for S3-compatible services
                        (e.g., MinIO). Defaults to None (uses AWS S3).
                    storage_path: Optional prefix for all keys. Useful for organizing
                        different projects/teams in the same bucket. Defaults to "".

                Examples:
        ```python
                    # AWS S3
                    storage = LockingS3Storage(
                        bucket='my-bucket',
                        region_name='us-west-2',
                        access_key_id='AKIAIOSFODNN7EXAMPLE',
                        access_key='wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY'
                    )

                    # With prefix for organization
                    storage = LockingS3Storage(
                        bucket='company-ml',
                        region_name='eu-west-1',
                        access_key_id='...',
                        access_key='...',
                        storage_path='team-vision/experiments'
                    )
                    # Keys will be: company-ml/team-vision/experiments/models/...

                    # MinIO (S3-compatible)
                    storage = LockingS3Storage(
                        bucket='ml-cache',
                        region_name='us-east-1',
                        access_key_id='minioadmin',
                        access_key='minioadmin',
                        endpoint_url='http://localhost:9000'
                    )
        ```
        """
        super().__init__(
            bucket=bucket,
            region_name=region_name,
            access_key_id=access_key_id,
            access_key=access_key,
            endpoint_url=endpoint_url,
            storage_path=storage_path,
        )

    def try_acquire_lock(
        self, lock_name: str, ttl: int = 3600, heartbeat_interval: int | None = None
    ) -> bool:
        """Acquire distributed lock using S3 PUT operation.

                The TTL is stored as part of the lock metadata in S3 and will be
                used to determine if the lock has become stale.

                Args:
                    lock_name: Unique lock identifier (e.g., "model_abc123").
                    ttl: Lock validity in seconds. This value is stored with the
                        lock in S3 and used later to detect stale locks.
                        Default 3600 (1 hour).
                    heartbeat_interval: If provided, enables heartbeat-based crash
                        detection. Should be much smaller than ttl. Default None.


                Returns:
                    True if lock was acquired, False if held by another process.

                Examples:
        ```python
                    # Distributed training coordination
                    lock_name = f"train_{experiment_id}"

                    if storage.try_acquire_lock(lock_name, ttl=7200):
                        try:
                            # Only one instance trains
                            train_model()
                            upload_artifacts()
                        finally:
                            storage.release_lock(lock_name)
                    else:
                        # Other instances wait
                        storage.wait_for_unlock(lock_name, timeout=7200)
                        download_artifacts()
        ```

                Note:
                    The lock is stored as an S3 object with metadata including TTL:
        ```
                    s3://bucket/storage_path/locks/lock_name.lock
        ```

                    Lock content (JSON):
        ```json
                    {
                        "owner": "ip-10-0-1-42.ec2.internal",
                        "pid": 1234,
                        "created_at": 1735562400.5,
                        "ttl": 7200
                    }
        ```
        """
        lock_path = self._get_full_lock_path(lock_name)

        # Check if lock exists and is stale/dead
        try:
            response = self._client.get_object(Bucket=self.bucket, Key=lock_path)
            lock_data = json.loads(response["Body"].read().decode())

            if not self._is_lock_stale_or_dead(lock_data):
                return False  # Lock is fresh

            # Lock is stale/dead, will try to steal it

        except ClientError as e:
            if e.response["Error"]["Code"] != "NoSuchKey":
                if self._verbose:
                    print(f"\t⚠️ Error checking lock: {e}")
                return False
            # Lock doesn't exist, proceed to create

        # Create lock with metadata
        lock_data = {
            "owner": socket.gethostname(),
            "pid": os.getpid(),
            "created_at": time.time(),
            "ttl": ttl,
        }

        # Add heartbeat fields if enabled
        if heartbeat_interval is not None:
            lock_data["last_heartbeat"] = time.time()
            lock_data["heartbeat_interval"] = heartbeat_interval

        try:
            self._client.put_object(
                Bucket=self.bucket,
                Key=lock_path,
                Body=json.dumps(lock_data).encode(),
            )

            if self._verbose:
                hb_info = f", HB: {heartbeat_interval}s" if heartbeat_interval else ""
                print(f"\t🔒 Lock acquired: {lock_name} (TTL: {ttl}s{hb_info})")

            return True

        except ClientError as e:
            if self._verbose:
                print(f"\t⚠️ Failed to acquire lock: {e}")
            return False

    def update_heartbeat(self, lock_name: str) -> bool:
        """Update the heartbeat timestamp in S3.

        Args:
            lock_name: Lock identifier.

        Returns:
            True if heartbeat updated successfully, False otherwise.

        Note:
            This operation involves read-modify-write in S3, which is not
            atomic. However, since only the lock owner should call this,
            race conditions are unlikely.
        """
        lock_path = self._get_full_lock_path(lock_name)

        try:
            # Read current lock data
            response = self._client.get_object(Bucket=self.bucket, Key=lock_path)
            lock_data = json.loads(response["Body"].read().decode())

            # Verify we own this lock
            if lock_data.get("pid") != os.getpid():
                if self._verbose:
                    print(
                        f"\t⚠️ Cannot update heartbeat: lock owned by PID {lock_data.get('pid')}"
                    )
                return False

            # Check if heartbeat is enabled
            if "last_heartbeat" not in lock_data:
                if self._verbose:
                    print(f"\t⚠️ Heartbeat not enabled for lock: {lock_name}")
                return False

            # Update heartbeat
            lock_data["last_heartbeat"] = time.time()

            # Write back to S3
            self._client.put_object(
                Bucket=self.bucket,
                Key=lock_path,
                Body=json.dumps(lock_data).encode(),
            )

            if self._verbose:
                age = time.time() - lock_data["created_at"]
                print(f"\t💓 Heartbeat updated: {lock_name} (age: {age:.0f}s)")

            return True

        except (ClientError, json.JSONDecodeError, KeyError) as e:
            if self._verbose:
                print(f"\t⚠️ Failed to update heartbeat: {e}")
            return False

    def release_lock(self, lock_name: str) -> None:
        """Release lock by deleting the S3 object.

                Removes the lock object from S3 to make it available for other
                processes. Safe to call even if lock doesn't exist.

                Args:
                    lock_name: Lock identifier to release.

                Examples:
        ```python
                    # Standard pattern with try-finally
                    if storage.try_acquire_lock("my_lock"):
                        try:
                            expensive_computation()
                        finally:
                            storage.release_lock("my_lock")

                    # Safe to call multiple times
                    storage.release_lock("my_lock")
                    storage.release_lock("my_lock")  # No error
        ```
        """
        lock_path = self._get_full_lock_path(lock_name)
        try:
            self._client.delete_object(Bucket=self.bucket, Key=lock_path)
            if self._verbose:
                print(f"\t🔓 Lock released: {lock_name}")
        except ClientError:
            pass  # Already released or doesn't exist

    def _is_locked(self, lock_name: str) -> bool:
        """Check if lock exists and is not stale.

        Reads the lock from S3 and checks if it has exceeded its own TTL.

        Args:
            lock_name: Lock identifier.

        Returns:
            True if lock exists and is valid (not expired), False otherwise.
        """
        lock_path = self._get_full_lock_path(lock_name)
        try:
            response = self._client.get_object(Bucket=self.bucket, Key=lock_path)
            lock_data = json.loads(response["Body"].read().decode())

            # Check staleness using lock's own TTL
            return not self._is_lock_stale_or_dead(lock_data)

        except ClientError:
            return False

    def _is_lock_stale_or_dead(self, lock_data: dict) -> bool:
        """Check if lock data indicates a stale lock.

        Uses the TTL stored in the lock metadata to determine staleness.

        Args:
            lock_data: Dictionary containing lock metadata.

        Returns:
            True if lock is stale, False if still valid.

        Note:
            Returns True if TTL is missing (old format) or if any required
            fields are missing.
        """
        try:
            current_time = time.time()

            # Check 1: TTL expiry
            ttl = lock_data.get("ttl")
            if ttl is None:
                return True  # Old format

            created_at = lock_data["created_at"]
            age = current_time - created_at

            if age > ttl:
                if self._verbose:
                    print(f"\t⏰ Lock expired (age: {age:.0f}s > TTL: {ttl}s)")
                return True

            # Check 2: Heartbeat death detection
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

        except (KeyError, TypeError):
            return True

    def _get_full_lock_path(self, lock_name: str) -> str:
        """Get full S3 key for lock file including prefix.

                Combines storage_path prefix with locks directory and lock name.

                Args:
                    lock_name: Lock identifier.

                Returns:
                    Full S3 key for the lock object.

                Examples:
        ```python
                    storage = LockingS3Storage(..., storage_path="exp1")
                    path = storage._get_full_lock_path("model_abc")
                    # Returns: "exp1/locks/model_abc.lock"
        ```
        """
        lock_relative_path = self._get_lock_path(lock_name)
        if self.storage_path:
            return f"{self.storage_path}{lock_relative_path}"
        return lock_relative_path
