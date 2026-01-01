from labchain.container.container import *  # noqa: F403
from labchain.container.overload import *  # noqa: F403
from labchain.container.container import Container
from labchain.plugins.storage import LocalStorage
from labchain.plugins.ingestion import DatasetManager


Container.storage = LocalStorage()
Container.ds = DatasetManager()
Container.bind()(LocalStorage)
