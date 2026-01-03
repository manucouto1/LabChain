from labchain.container.container import *  # noqa: F403
from labchain.container.overload import *  # noqa: F403
from labchain.container.container import Container
from labchain.plugins.ingestion import DatasetManager
from labchain.plugins.storage import LocalStorage, S3Storage


Container.bind()(LocalStorage)
Container.bind()(S3Storage)
Container.ds = DatasetManager()
