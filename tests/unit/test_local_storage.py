from pathlib import Path
import pytest
import os
import pickle
from labchain.plugins.storage import LocalStorage


@pytest.fixture
def local_storage():
    storage_path = "tests/test_data"
    storage = LocalStorage(storage_path=storage_path)
    Path(storage.get_root_path()).mkdir(parents=True, exist_ok=True)

    yield storage
    for file in os.listdir(storage.get_root_path()):
        os.remove(os.path.join(storage.get_root_path(), file))
    os.rmdir(storage.get_root_path())


def test_upload_file(local_storage):
    test_data = {"key": "value"}
    file_name = "test_file.pkl"

    result = local_storage.upload_file(
        file=test_data, file_name=file_name, context=local_storage.get_root_path()
    )

    assert result == file_name
    assert os.path.exists(os.path.join(local_storage.get_root_path(), file_name))

    with open(os.path.join(local_storage.get_root_path(), file_name), "rb") as f:
        loaded_data = pickle.load(f)

    assert loaded_data == test_data


def test_upload_file_exception(local_storage, mocker):
    test_data = {"key": "value"}
    file_name = "test_file.pkl"

    # Mock the pickle.dump function to raise an exception
    mocker.patch("pickle.dump", side_effect=Exception("Mocked error"))

    result = local_storage.upload_file(
        test_data, file_name, local_storage.get_root_path()
    )

    assert result is None
    assert os.path.exists(os.path.join(local_storage.get_root_path(), file_name))


def test_list_stored_files(local_storage):
    # Create some test files
    test_files = ["file1.txt", "file2.pkl", "file3.bin"]
    for file_name in test_files:
        local_storage.upload_file("", file_name, local_storage.get_root_path())

    # Call the method to list stored files
    stored_files = local_storage.list_stored_files(local_storage.get_root_path())

    # Check if the returned list contains all the test files
    assert set(stored_files) == set(test_files)

    # Check if the number of files matches
    assert len(stored_files) == len(test_files)

    # Clean up the test files
    for file_name in test_files:
        os.remove(os.path.join(local_storage.get_root_path(), file_name))


def test_get_file_by_hashcode(local_storage):
    # Create a test file with binary content
    test_file_name = "test_file.bin"
    test_content = b"\x00\x01\x02\x03\x04\x05"

    # Upload the binary file
    with open(os.path.join(local_storage.get_root_path(), test_file_name), "wb") as f:
        f.write(test_content)

    # Retrieve the file using get_file_by_hashcode
    retrieved_file = local_storage.get_file_by_hashcode(
        test_file_name, local_storage.get_root_path()
    )

    # Assert that the retrieved file is a file object
    assert hasattr(retrieved_file, "read")

    # Read the content of the retrieved file as bytes
    retrieved_content = retrieved_file.read()

    # Assert that the binary content matches the original
    assert retrieved_content == test_content

    # Close the file object
    retrieved_file.close()

    # Clean up the test file
    os.remove(os.path.join(local_storage.get_root_path(), test_file_name))


def test_get_non_existent_file(local_storage):
    non_existent_file = "non_existent_file.txt"
    with pytest.raises(FileNotFoundError) as excinfo:
        local_storage.get_file_by_hashcode(
            non_existent_file, local_storage.get_root_path()
        )
        print(str(excinfo.value))
        print(
            f"Couldn't find file {non_existent_file} in path {local_storage.get_root_path()}"
        )
    assert (
        str(excinfo.value)
        == f"Couldn't find file {non_existent_file} in path {local_storage.get_root_path()}"
    )


def test_check_if_exists_true(local_storage):
    # Create a test file
    test_file = "test_file.txt"
    test_content = "Test content"
    local_storage.upload_file(test_content, test_file, local_storage.get_root_path())

    # Check if the file exists using its name as hashcode
    result = local_storage.check_if_exists(test_file, local_storage.get_root_path())

    # Assert that the file exists
    assert result is True

    # Clean up the test file
    os.remove(os.path.join(local_storage.get_root_path(), test_file))


def test_check_if_exists_false(local_storage):
    non_existent_file = "non_existent_file.txt"
    result = local_storage.check_if_exists(
        non_existent_file, local_storage.get_root_path()
    )
    assert result is False


def test_download_file(local_storage):
    # Create a test file
    test_data = {"key": "value"}
    file_name = "test_file.pkl"

    # Upload the test file
    local_storage.upload_file(test_data, file_name, local_storage.get_root_path())

    # Download and deserialize the file
    downloaded_data = local_storage.download_file(
        file_name, local_storage.get_root_path()
    )

    # Assert that the downloaded data matches the original data
    assert downloaded_data == test_data

    # Clean up the test file
    os.remove(os.path.join(local_storage.get_root_path(), file_name))


def test_delete_existing_file(local_storage):
    # Create a test file
    test_file = "test_file.txt"
    test_content = "Test content"
    local_storage.upload_file(test_content, test_file, local_storage.get_root_path())

    # Delete the file
    local_storage.delete_file(test_file, local_storage.get_root_path())

    # Assert that the file no longer exists
    assert not os.path.exists(os.path.join(local_storage.get_root_path(), test_file))

    # Verify that check_if_exists returns False
    assert (
        local_storage.check_if_exists(test_file, local_storage.get_root_path()) is False
    )


def test_delete_non_existent_file(local_storage):
    non_existent_file = "non_existent_file.txt"
    with pytest.raises(FileExistsError) as excinfo:
        local_storage.delete_file(non_existent_file, local_storage.get_root_path())
    assert str(excinfo.value) == "No existe en la carpeta"
