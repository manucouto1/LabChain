# tests/unit/storage/test_locking_s3_storage_heartbeat.py

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
    bucket_name = "test-locking-heartbeat-bucket"
    s3_client.create_bucket(Bucket=bucket_name)

    storage = LockingS3Storage(
        bucket=bucket_name,
        region_name="us-east-1",
        access_key_id="fake_access_key",
        access_key="fake_secret_key",
    )

    return storage


class TestS3HeartbeatBasics:
    """Test basic S3 heartbeat functionality."""

    def test_s3_lock_with_heartbeat_creates_fields(self, locking_storage):
        """Test that S3 lock with heartbeat has required fields."""
        lock_name = "s3_heartbeat_lock"

        result = locking_storage.try_acquire_lock(
            lock_name, ttl=3600, heartbeat_interval=30
        )

        assert result is True

        # Read lock from S3
        lock_key = locking_storage._get_full_lock_path(lock_name)
        response = locking_storage._client.get_object(
            Bucket=locking_storage.bucket, Key=lock_key
        )
        lock_data = json.loads(response["Body"].read().decode())

        assert "last_heartbeat" in lock_data
        assert "heartbeat_interval" in lock_data
        assert lock_data["heartbeat_interval"] == 30
        # Approximate comparison for timestamps
        assert abs(lock_data["last_heartbeat"] - lock_data["created_at"]) < 0.1

        locking_storage.release_lock(lock_name)

    def test_s3_lock_without_heartbeat_no_fields(self, locking_storage):
        """Test that S3 lock without heartbeat has no heartbeat fields."""
        lock_name = "s3_no_heartbeat_lock"

        result = locking_storage.try_acquire_lock(
            lock_name, ttl=3600, heartbeat_interval=None
        )

        assert result is True

        # Read lock from S3
        lock_key = locking_storage._get_full_lock_path(lock_name)
        response = locking_storage._client.get_object(
            Bucket=locking_storage.bucket, Key=lock_key
        )
        lock_data = json.loads(response["Body"].read().decode())

        assert "last_heartbeat" not in lock_data
        assert "heartbeat_interval" not in lock_data

        locking_storage.release_lock(lock_name)

    def test_s3_update_heartbeat_success(self, locking_storage):
        """Test successful S3 heartbeat update."""
        lock_name = "s3_update_heartbeat"

        locking_storage.try_acquire_lock(lock_name, ttl=3600, heartbeat_interval=30)

        # Get initial heartbeat
        lock_key = locking_storage._get_full_lock_path(lock_name)
        response = locking_storage._client.get_object(
            Bucket=locking_storage.bucket, Key=lock_key
        )
        initial_data = json.loads(response["Body"].read().decode())
        initial_heartbeat = initial_data["last_heartbeat"]

        # Wait and update
        time.sleep(0.1)
        result = locking_storage.update_heartbeat(lock_name)

        assert result is True

        # Verify update
        response = locking_storage._client.get_object(
            Bucket=locking_storage.bucket, Key=lock_key
        )
        updated_data = json.loads(response["Body"].read().decode())
        updated_heartbeat = updated_data["last_heartbeat"]

        assert updated_heartbeat > initial_heartbeat

        locking_storage.release_lock(lock_name)

    def test_s3_update_heartbeat_fails_without_enabled(self, locking_storage):
        """Test that S3 heartbeat update fails if not enabled."""
        lock_name = "s3_no_heartbeat"

        # Acquire without heartbeat
        locking_storage.try_acquire_lock(lock_name, ttl=3600, heartbeat_interval=None)

        # Try to update - should fail
        result = locking_storage.update_heartbeat(lock_name)

        assert result is False

        locking_storage.release_lock(lock_name)

    def test_s3_update_heartbeat_fails_wrong_owner(self, locking_storage):
        """Test that S3 heartbeat update fails if not owner."""
        lock_name = "s3_other_owner"
        lock_key = locking_storage._get_full_lock_path(lock_name)

        # Create lock owned by another process
        lock_data = {
            "owner": "other_host",
            "pid": 99999,
            "created_at": time.time(),
            "last_heartbeat": time.time(),
            "heartbeat_interval": 30,
            "ttl": 3600,
        }

        locking_storage._client.put_object(
            Bucket=locking_storage.bucket,
            Key=lock_key,
            Body=json.dumps(lock_data).encode(),
        )

        # Try to update - should fail
        result = locking_storage.update_heartbeat(lock_name)

        assert result is False

        # Cleanup
        locking_storage.release_lock(lock_name)

    def test_s3_dead_lock_detection(self, locking_storage):
        """Test S3 dead lock detection via heartbeat."""
        lock_name = "s3_dead_lock"
        lock_key = locking_storage._get_full_lock_path(lock_name)

        # Create dead lock
        lock_data = {
            "owner": "dead_process",
            "pid": 99999,
            "created_at": time.time() - 50,
            "last_heartbeat": time.time() - 50,  # 50 seconds old
            "heartbeat_interval": 10,
            "ttl": 3600,
        }

        locking_storage._client.put_object(
            Bucket=locking_storage.bucket,
            Key=lock_key,
            Body=json.dumps(lock_data).encode(),
        )

        # Should be detected as dead
        assert locking_storage._is_locked(lock_name) is False

        # Can steal
        assert locking_storage.try_acquire_lock(lock_name, ttl=3600) is True

        locking_storage.release_lock(lock_name)

    def test_s3_recent_heartbeat_prevents_stealing(self, locking_storage):
        """Test that S3 lock with recent heartbeat cannot be stolen."""
        lock_name = "s3_active_lock"
        lock_key = locking_storage._get_full_lock_path(lock_name)

        # Create lock with recent heartbeat
        lock_data = {
            "owner": "active_process",
            "pid": 99999,
            "created_at": time.time() - 100,
            "last_heartbeat": time.time() - 5,  # Recent
            "heartbeat_interval": 30,
            "ttl": 3600,
        }

        locking_storage._client.put_object(
            Bucket=locking_storage.bucket,
            Key=lock_key,
            Body=json.dumps(lock_data).encode(),
        )

        # Should NOT be stale
        assert locking_storage._is_locked(lock_name) is True

        # Cannot steal
        assert locking_storage.try_acquire_lock(lock_name, ttl=3600) is False

        # Cleanup
        locking_storage.release_lock(lock_name)


class TestS3HeartbeatWithBackoff:
    """Test S3 heartbeat with exponential backoff waiting."""

    def test_wait_for_unlock_with_heartbeat_stops_early(self, locking_storage):
        """Test that wait_for_unlock stops when heartbeat indicates death."""
        lock_name = "s3_heartbeat_wait"
        lock_key = locking_storage._get_full_lock_path(lock_name)

        # Create lock with heartbeat that will die soon
        lock_data = {
            "owner": "dying_process",
            "pid": 99999,
            "created_at": time.time() - 8,
            "last_heartbeat": time.time() - 8,  # 8 seconds old
            "heartbeat_interval": 2,  # Interval is 2s, so 3x = 6s
            "ttl": 3600,  # Long TTL
        }

        locking_storage._client.put_object(
            Bucket=locking_storage.bucket,
            Key=lock_key,
            Body=json.dumps(lock_data).encode(),
        )

        # Wait for unlock - should detect death quickly
        start = time.time()
        result = locking_storage.wait_for_unlock(
            lock_name,
            timeout=100,  # Long timeout
            initial_poll_interval=0.5,
            max_poll_interval=5.0,
        )
        elapsed = time.time() - start

        # Should succeed
        assert result is True

        # Should be quick (not waiting for full TTL)
        assert elapsed < 10.0


class TestS3HeartbeatEdgeCases:
    """Test S3 heartbeat edge cases."""

    def test_s3_heartbeat_with_prefix(self):
        """Test S3 heartbeat works with storage prefix."""
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

            # Acquire lock with heartbeat
            assert (
                storage.try_acquire_lock("my_lock", ttl=3600, heartbeat_interval=30)
                is True
            )

            # Verify lock key includes prefix and has heartbeat
            lock_key = storage._get_full_lock_path("my_lock")
            assert lock_key == "exp1/locks/my_lock.lock"

            response = client.get_object(Bucket="test-bucket", Key=lock_key)
            lock_data = json.loads(response["Body"].read().decode())

            assert "last_heartbeat" in lock_data
            assert "heartbeat_interval" in lock_data

            storage.release_lock("my_lock")

    def test_s3_very_short_heartbeat_interval(self, locking_storage):
        """Test S3 lock with very short heartbeat interval."""
        lock_name = "s3_short_interval"

        locking_storage.try_acquire_lock(lock_name, ttl=3600, heartbeat_interval=1)

        # Should be locked
        assert locking_storage._is_locked(lock_name) is True

        # Wait for death threshold
        time.sleep(3.5)

        # Should be dead
        assert locking_storage._is_locked(lock_name) is False

        locking_storage.release_lock(lock_name)

    def test_s3_corrupted_heartbeat_data(self, locking_storage):
        """Test S3 handling of corrupted heartbeat data."""
        lock_name = "s3_corrupted"
        lock_key = locking_storage._get_full_lock_path(lock_name)

        # Create lock with invalid heartbeat
        lock_data = {
            "owner": "test",
            "pid": 12345,
            "created_at": time.time(),
            "ttl": 3600,
            "last_heartbeat": "invalid",
            "heartbeat_interval": "also_invalid",
        }

        locking_storage._client.put_object(
            Bucket=locking_storage.bucket,
            Key=lock_key,
            Body=json.dumps(lock_data).encode(),
        )

        # Should handle gracefully
        try:
            locking_storage._is_locked(lock_name)
            # Either result is acceptable
        except (TypeError, ValueError):
            # Also acceptable
            pass

        # Should be able to acquire
        acquired = locking_storage.try_acquire_lock(lock_name, ttl=3600)
        if acquired:
            locking_storage.release_lock(lock_name)
