import pytest
from unittest.mock import Mock, patch
from labchain.container import Container
from labchain.plugins.ingestion import DatasetManager
from labchain.base.base_types import XYData


@pytest.fixture
def mock_local_storage():
    with patch("labchain.container.LocalStorage") as mock_storage:
        yield mock_storage


@pytest.fixture
def container(mock_local_storage):
    with patch(
        "labchain.container.Container.storage", new_callable=Mock
    ) as mock_container_storage:
        container = Container()
        container.ds = DatasetManager()
        yield container


def test_list(container):
    container.storage.list_stored_files.return_value = ["dataset1", "dataset2"]
    result = container.ds.list()
    assert result == ["dataset1", "dataset2"]
    container.storage.list_stored_files.assert_called_once()


def test_save(container):
    data = Mock(spec=XYData)
    container.storage.check_if_exists.return_value = False
    container.ds.save("new", data)
    container.storage.upload_file.assert_called_once()


def test_save_already_exists(container):
    container.storage.check_if_exists.return_value = True
    data = Mock(spec=XYData)
    with pytest.raises(ValueError):
        container.ds.save("existing", data)


def test_load(container):
    mock_data = Mock(spec=XYData)
    container.storage.check_if_exists.return_value = True
    container.storage.download_file.return_value = mock_data
    result = container.ds.load("existing")
    assert result.value == mock_data
    container.storage.download_file.assert_called_once()


def test_load_not_exists(container):
    container.storage.check_if_exists.return_value = False
    with pytest.raises(ValueError):
        container.ds.load("non_existing")


def test_delete(container):
    container.storage.check_if_exists.return_value = True
    container.ds.delete("existing")
    container.storage.delete_file.assert_called_once_with(
        "existing", f"{container.storage.get_root_path()}/datasets"
    )


def test_delete_not_exists(container):
    container.storage.check_if_exists.return_value = False
    with pytest.raises(ValueError):
        container.ds.delete("non_existing")


def test_update_existing_dataset(container):
    data = Mock(spec=XYData)
    container.storage.check_if_exists.return_value = True
    container.ds.update("existing", data)
    container.storage.upload_file.assert_called_once()


def test_update_non_existing_dataset(container):
    data = Mock(spec=XYData)
    container.storage.check_if_exists.return_value = False
    with pytest.raises(ValueError, match="Dataset 'non_existing' does not exist"):
        container.ds.update("non_existing", data)


def test_update_with_different_data(container):
    old_data = Mock(spec=XYData)
    new_data = Mock(spec=XYData)
    container.storage.check_if_exists.return_value = True
    container.storage.download_file.return_value = old_data
    container.ds.update("existing", new_data)
    container.storage.upload_file.assert_called_once()
