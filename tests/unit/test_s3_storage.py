import pytest
import boto3
import pickle
import io

from moto import mock_s3

from labchain.plugins.storage import S3Storage


@pytest.fixture
def s3_client():
    with mock_s3():
        client = boto3.client(
            "s3",
            region_name="us-east-1",
            aws_access_key_id="fake_access_key",
            aws_secret_access_key="fake_secret_key",
        )
        yield client


@pytest.fixture
def s3_storage(s3_client):
    bucket_name = "test-bucket"
    s3_client.create_bucket(Bucket=bucket_name)
    return S3Storage(
        bucket=bucket_name,
        region_name="us-east-1",
        access_key_id="fake_access_key",
        access_key="fake_secret_key",
    )


def test_upload_file(s3_storage):
    test_data = {"key": "value"}
    file_name = "test_file.pkl"

    result = s3_storage.upload_file(test_data, file_name, "")
    assert result == file_name
    # Verify the file was uploaded
    response = s3_storage._client.get_object(
        Bucket=s3_storage.bucket, Key=f"{file_name}"
    )
    content = pickle.loads(response["Body"].read())
    assert content == test_data


def test_upload_file_error(s3_storage, mocker):
    test_data = {"key": "value"}
    file_name = "test_file.pkl"
    context = ""
    # Mock the put_object method to raise an exception
    mocker.patch.object(
        s3_storage._client, "put_object", side_effect=Exception("Mocked error")
    )
    with pytest.raises(Exception) as e:
        s3_storage.upload_file(test_data, file_name, context)
        assert e.value == "Mocked error"


def test_list_stored_files(s3_storage, s3_client):
    # Upload some test files
    test_files = ["file1.txt", "file2.txt", "file3.txt"]
    for file_name in test_files:
        s3_client.put_object(
            Bucket=s3_storage.bucket, Key=file_name, Body=b"test content"
        )
    # Call the method to list stored files
    stored_files = s3_storage.list_stored_files(context="")
    # Check if the returned list contains the correct number of files
    assert len(stored_files) == len(test_files)
    # Check if all uploaded files are in the list
    for file_name in test_files:
        assert file_name in stored_files


def test_get_file_by_hashcode(s3_storage, s3_client):
    file_id = "test_file.txt"
    file_content = b"This is a test file content"
    # Upload a test file
    s3_client.put_object(Bucket=s3_storage.bucket, Key=file_id, Body=file_content)
    # Retrieve the file content using get_file_by_id
    retrieved_content = s3_storage.get_file_by_hashcode(file_id, context="")
    # Assert that the retrieved content matches the original content
    assert retrieved_content == file_content


def test_check_if_exists_true(s3_storage, s3_client):
    file_id = "existing_file.txt"
    s3_client.put_object(
        Bucket=s3_storage.bucket, Key=f"{file_id}", Body=b"test content"
    )
    result = s3_storage.check_if_exists(file_id, "")
    assert result is True


def test_check_if_exists_false(s3_storage, s3_client):
    non_existent_file_id = "non_existent_file.txt"
    result = s3_storage.check_if_exists(non_existent_file_id, "/test")
    assert result is False


def test_download_file(s3_storage, s3_client):
    file_id = "test_file.pkl"
    test_data = {"key": "value"}
    # Upload a test file
    s3_client.put_object(
        Bucket=s3_storage.bucket, Key=f"{file_id}", Body=pickle.dumps(test_data)
    )
    # Download and deserialize the file
    downloaded_data = s3_storage.download_file(file_id, "")
    # Assert that the downloaded data matches the original data
    assert downloaded_data == test_data


def test_delete_existing_file(s3_storage, s3_client):
    file_id = "existing_file.txt"
    # s3_client.put_object(Bucket=s3_storage.bucket, Key=file_id, Body=b"test content")
    s3_storage.upload_file("test content", file_id, "")
    s3_storage.delete_file(file_id, context="")
    # Verify the file no longer exists
    with pytest.raises(s3_client.exceptions.NoSuchKey):
        s3_client.get_object(Bucket=s3_storage.bucket, Key=file_id)
    # Verify that check_if_exists returns False
    assert s3_storage.check_if_exists(file_id, "test") is False


def test_delete_non_existent_file(s3_storage):
    non_existent_file = "non_existent_file.txt"
    with pytest.raises(FileExistsError) as excinfo:
        s3_storage.delete_file(non_existent_file, context="/test")
    assert str(excinfo.value) == "No existe en el bucket"


def test_upload_large_file(s3_storage, mocker):
    # Create a large file-like object
    large_data = b"0" * 1024 * 1024 * 100  # 100 MB of data
    large_file = io.BytesIO(large_data)
    file_name = "large_file.bin"
    # Mock the put_object method to avoid actual S3 operations
    mock_put_object = mocker.patch.object(s3_storage._client, "put_object")
    # Call the upload_file method
    result = s3_storage.upload_file(large_file, file_name, "test")
    # Assert that the file was uploaded successfully
    assert result == file_name
    # Verify that put_object was called with the correct arguments
    mock_put_object.assert_called_once()
    call_args = mock_put_object.call_args[1]
    assert call_args["Bucket"] == s3_storage.bucket
    assert call_args["Key"] == f"test/{file_name}"
    assert isinstance(call_args["Body"], io.BytesIO)
    # Verify that the uploaded data matches the original data
    # uploaded_data = pickle.loads(call_args['Body'].getvalue())
    uploaded_data = call_args["Body"].getvalue()
    large_file.seek(0)  # Reset the file pointer to the beginning
    original_data = large_file.getvalue()
    assert uploaded_data == original_data
