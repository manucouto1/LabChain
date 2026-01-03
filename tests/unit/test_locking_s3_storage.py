# tests/unit/storage/test_locking_storage.py

import pytest
import boto3
import json
import time
from moto import mock_aws
from labchain.plugins.storage import LockingS3Storage


@pytest.fixture
def s3_client():
    """Fixture that provides a mocked S3 client."""
    with mock_aws():
        client = boto3.client(
            "s3",
            region_name="us-east-1",
            aws_access_key_id="fake_access_key",
            aws_secret_access_key="fake_secret_key",
        )
        yield client


@pytest.fixture
def locking_storage(s3_client):
    """Fixture that provides a LockingS3Storage instance."""
    bucket_name = "test-locking-bucket"
    s3_client.create_bucket(Bucket=bucket_name)

    storage = LockingS3Storage(
        bucket=bucket_name,
        region_name="us-east-1",
        access_key_id="fake_access_key",
        access_key="fake_secret_key",
    )

    return storage


class TestLockingS3StorageBasics:
    """Test basic locking operations for S3."""

    def test_initialization(self, locking_storage):
        """Test that S3 storage initializes correctly."""
        assert locking_storage.bucket == "test-locking-bucket"
        assert locking_storage.storage_path == ""

    def test_initialization_with_prefix(self):
        """Test initialization with storage prefix."""
        with mock_aws():
            client = boto3.client("s3", region_name="us-east-1")
            client.create_bucket(Bucket="test-bucket")

            storage = LockingS3Storage(
                bucket="test-bucket",
                region_name="us-east-1",
                access_key_id="fake",
                access_key="fake",
                storage_path="my-prefix/",
            )

            assert storage.storage_path == "my-prefix/"

    def test_acquire_lock_success(self, locking_storage):
        """Test successful lock acquisition in S3."""
        lock_name = "test_s3_lock"

        result = locking_storage.try_acquire_lock(lock_name, ttl=3600)

        assert result is True

        # Verify lock object exists in S3
        lock_key = locking_storage._get_full_lock_path(lock_name)
        response = locking_storage._client.get_object(
            Bucket=locking_storage.bucket, Key=lock_key
        )

        lock_data = json.loads(response["Body"].read().decode())
        assert "owner" in lock_data
        assert "pid" in lock_data
        assert "created_at" in lock_data

        # Cleanup
        locking_storage.release_lock(lock_name)

    def test_acquire_lock_fail_when_locked(self, locking_storage):
        """Test that acquiring a locked S3 lock fails."""
        lock_name = "test_s3_lock"

        # First acquisition succeeds
        assert locking_storage.try_acquire_lock(lock_name) is True

        # Second acquisition fails
        assert locking_storage.try_acquire_lock(lock_name) is False

        # Cleanup
        locking_storage.release_lock(lock_name)

    def test_release_lock(self, locking_storage):
        """Test S3 lock release."""
        lock_name = "test_s3_lock"

        locking_storage.try_acquire_lock(lock_name)

        # Verify lock exists
        assert locking_storage._is_locked(lock_name) is True

        locking_storage.release_lock(lock_name)

        # Verify lock removed
        assert locking_storage._is_locked(lock_name) is False

    def test_release_nonexistent_lock(self, locking_storage):
        """Test that releasing a non-existent S3 lock doesn't raise error."""
        # Should not raise any exception
        locking_storage.release_lock("nonexistent_s3_lock")

    def test_multiple_different_locks(self, locking_storage):
        """Test acquiring multiple different S3 locks simultaneously."""
        locks = ["s3_lock1", "s3_lock2", "s3_lock3"]

        # Acquire all locks
        for lock_name in locks:
            assert locking_storage.try_acquire_lock(lock_name) is True

        # Verify all locks exist
        for lock_name in locks:
            assert locking_storage._is_locked(lock_name) is True

        # Release all locks
        for lock_name in locks:
            locking_storage.release_lock(lock_name)

        # Verify all locks released
        for lock_name in locks:
            assert locking_storage._is_locked(lock_name) is False


class TestS3StorageTTLStorage:
    """Test that TTL is correctly stored and used in S3Storage."""

    def test_ttl_stored_in_s3_object(self, locking_storage):
        """Test that TTL is saved in the S3 lock object."""
        lock_name = "test_s3_lock"
        ttl = 2400  # 40 minutes

        # Acquire lock with specific TTL
        assert locking_storage.try_acquire_lock(lock_name, ttl=ttl) is True

        # Read lock object and verify TTL is stored
        lock_key = locking_storage._get_full_lock_path(lock_name)
        response = locking_storage._client.get_object(
            Bucket=locking_storage.bucket, Key=lock_key
        )
        lock_data = json.loads(response["Body"].read().decode())

        assert "ttl" in lock_data
        assert lock_data["ttl"] == ttl
        assert "created_at" in lock_data
        assert "owner" in lock_data
        assert "pid" in lock_data

        locking_storage.release_lock(lock_name)

    def test_different_ttls_for_different_s3_locks(self, locking_storage):
        """Test that different S3 locks can have different TTLs."""
        locks = [
            ("s3_lock_short", 120),
            ("s3_lock_medium", 3600),
            ("s3_lock_long", 14400),
        ]

        # Acquire all locks with different TTLs
        for lock_name, ttl in locks:
            assert locking_storage.try_acquire_lock(lock_name, ttl=ttl) is True

        # Verify each lock has correct TTL
        for lock_name, expected_ttl in locks:
            lock_key = locking_storage._get_full_lock_path(lock_name)
            response = locking_storage._client.get_object(
                Bucket=locking_storage.bucket, Key=lock_key
            )
            lock_data = json.loads(response["Body"].read().decode())

            assert lock_data["ttl"] == expected_ttl

        # Cleanup
        for lock_name, _ in locks:
            locking_storage.release_lock(lock_name)

    def test_s3_staleness_uses_stored_ttl(self, locking_storage):
        """Test that S3 staleness check uses the TTL from object metadata."""
        lock_name = "ttl_test_s3_lock"
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

    def test_s3_lock_without_ttl_treated_as_stale(self, locking_storage):
        """Test that S3 locks without TTL field are treated as stale."""
        lock_name = "old_format_s3_lock"
        lock_key = locking_storage._get_full_lock_path(lock_name)

        # Create lock object in old format (without TTL)
        old_format_data = {
            "owner": "old_system",
            "pid": 99999,
            "created_at": time.time(),
            # No TTL field
        }

        locking_storage._client.put_object(
            Bucket=locking_storage.bucket,
            Key=lock_key,
            Body=json.dumps(old_format_data).encode(),
        )

        # Should be treated as stale
        assert locking_storage._is_locked(lock_name) is False

        # Should be able to acquire
        assert locking_storage.try_acquire_lock(lock_name, ttl=3600) is True

        # Now should have TTL
        response = locking_storage._client.get_object(
            Bucket=locking_storage.bucket, Key=lock_key
        )
        new_data = json.loads(response["Body"].read().decode())
        assert "ttl" in new_data

        locking_storage.release_lock(lock_name)


class TestLockingS3StorageWaiting:
    """Test wait_for_unlock functionality for S3."""

    def test_wait_for_unlock_success(self, locking_storage):
        """Test waiting for an S3 lock that gets released."""
        lock_name = "wait_test_s3_lock"

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
            initial_poll_interval=0.1,  # ← Cambiado de poll_interval
            max_poll_interval=0.1,  # ← Añadido
            backoff_factor=1.0,  # ← Añadido (sin backoff)
        )
        elapsed = time.time() - start

        assert result is True
        assert 0.9 < elapsed < 2.0  # Should take ~1 second

        thread.join()

    def test_wait_for_unlock_timeout(self, locking_storage):
        """Test timeout when waiting for an S3 lock."""
        lock_name = "timeout_test_s3_lock"

        # Acquire lock (and never release)
        locking_storage.try_acquire_lock(lock_name)

        # Wait should timeout
        start = time.time()
        result = locking_storage.wait_for_unlock(
            lock_name,
            timeout=1,
            initial_poll_interval=0.1,  # ← Cambiado de poll_interval
            max_poll_interval=0.1,  # ← Añadido
            backoff_factor=1.0,  # ← Añadido (sin backoff)
        )
        elapsed = time.time() - start

        assert result is False
        assert 0.9 < elapsed < 1.5  # Should take ~1 second

        # Cleanup
        locking_storage.release_lock(lock_name)


class TestLockingS3StorageIntegration:
    """Test integration with existing S3Storage functionality."""

    def test_storage_operations_work_with_locking(self, locking_storage):
        """Test that regular S3 storage operations still work."""
        # Upload file
        test_data = {"key": "value"}
        result = locking_storage.upload_file(test_data, "test.pkl", "")
        assert result == "test.pkl"

        # Check exists
        assert locking_storage.check_if_exists("test.pkl", "") is True

        # Download file
        downloaded = locking_storage.download_file("test.pkl", "")
        assert downloaded == test_data

        # Delete file
        locking_storage.delete_file("test.pkl", "")
        assert locking_storage.check_if_exists("test.pkl", "") is False

    def test_locks_stored_separately_from_data(self, locking_storage, s3_client):
        """Test that locks are stored in separate S3 keys."""
        # Upload data
        locking_storage.upload_file("test", "data.pkl", "")

        # Acquire lock
        locking_storage.try_acquire_lock("lock1")

        # List all objects
        response = s3_client.list_objects_v2(Bucket=locking_storage.bucket)
        keys = [obj["Key"] for obj in response.get("Contents", [])]

        assert "data.pkl" in keys
        assert "locks/lock1.lock" in keys

        # Cleanup
        locking_storage.release_lock("lock1")
        locking_storage.delete_file("data.pkl", "")

    def test_storage_with_prefix(self):
        """Test that locks work correctly with storage prefix."""
        with mock_aws():
            client = boto3.client("s3", region_name="us-east-1")
            client.create_bucket(Bucket="test-bucket")

            storage = LockingS3Storage(
                bucket="test-bucket",
                region_name="us-east-1",
                access_key_id="fake",
                access_key="fake",
                storage_path="exp1/",
            )

            # Acquire lock
            assert storage.try_acquire_lock("my_lock") is True

            # Verify lock key includes prefix
            response = client.list_objects_v2(Bucket="test-bucket")
            keys = [obj["Key"] for obj in response.get("Contents", [])]

            assert "exp1/locks/my_lock.lock" in keys

            storage.release_lock("my_lock")


class TestLockingS3StorageEdgeCases:
    """Test edge cases for S3 locking."""

    def test_lock_with_special_characters(self, locking_storage):
        """Test S3 locks with special characters in name."""
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
        """Test S3 locks with very short TTL."""
        lock_name = "short_ttl_s3_lock"
        ttl = 1  # 100ms

        assert locking_storage.try_acquire_lock(lock_name, ttl=ttl) is True
        time.sleep(2)

        # Should be stale
        assert locking_storage._is_locked(lock_name) is False

        # Should be able to reacquire
        assert locking_storage.try_acquire_lock(lock_name, ttl=ttl) is True

        locking_storage.release_lock(lock_name)
