from typing import Any, Callable, Optional, Tuple, cast
from labchain.container.container import Container
from labchain.base import BaseFilter
from labchain.base import BaseStorage
from labchain.base.base_storage import BaseLockingStorage
from labchain.base import XYData, VData

from rich import print as rprint
import pickle
import threading

from labchain.utils.utils import method_is_overridden

__all__ = ["CachedWithLocking"]


class CachedWithLocking(BaseFilter):
    """Cached filter with distributed locking for race-condition-free caching.

        Extends the Cached pattern with distributed locks to ensure only one process
        computes each cached artifact (model or predictions). Other processes wait
        and reuse the result, preventing redundant computation and race conditions
        in parallel pipelines.

        Key Features:
            - **Single-flight caching**: Only one process computes each artifact
            - **Wait-and-reuse**: Other processes wait and load cached results
            - **Crash recovery**: TTL-based automatic recovery from failed processes
            - **Storage agnostic**: Works with LockingLocalStorage and LockingS3Storage
            - **Zero redundant work**: Eliminates duplicate training/prediction

        This decorator is essential for:
            - Parallel hyperparameter tuning
            - Distributed cross-validation
            - Multi-process/multi-machine training
            - Grid search with caching
            - Any scenario with concurrent pipeline execution

        Attributes:
            filter: The underlying filter being cached.
            cache_data: Whether to cache prediction results.
            cache_filter: Whether to cache trained models.
            overwrite: Whether to overwrite existing cache.
            lock_ttl: Lock validity duration in seconds.
            lock_timeout: Maximum wait time for locks in seconds.
            _storage: Storage backend (must support locking).

        Examples:
            Basic usage with local storage:
    ```python
            from framework3.plugins.storage import LockingLocalStorage
            from framework3.filters import MyMLFilter
            from framework3.container import Container

            # Configure locking storage
            Container.storage = LockingLocalStorage(storage_path='./cache')

            # Wrap filter with locking cache
            model = MyMLFilter()
            cached_model = CachedWithLocking(
                filter=model,
                cache_filter=True,
                cache_data=True,
                lock_ttl=3600,
                lock_timeout=7200
            )

            # Use normally - locking is automatic
            cached_model.fit(X_train, y_train)
            predictions = cached_model.predict(X_test)
    ```

            Parallel grid search without redundant training:
    ```python
            from multiprocessing import Pool
            from framework3.plugins.storage import LockingLocalStorage

            def train_with_params(params):
                storage = LockingLocalStorage('./cache')
                model = MyMLFilter(**params)
                cached = CachedWithLocking(model, storage=storage)

                # If another process is training this config,
                # this process waits and loads the result
                cached.fit(X_train, y_train)
                return cached.predict(X_test)

            # Run grid search with 8 processes
            # Identical configs only train once!
            param_grid = [
                {'C': 0.1, 'kernel': 'rbf'},
                {'C': 0.1, 'kernel': 'rbf'},  # Duplicate
                {'C': 1.0, 'kernel': 'rbf'},
                {'C': 1.0, 'kernel': 'rbf'},  # Duplicate
            ]

            with Pool(8) as p:
                results = p.map(train_with_params, param_grid)
            # Only 2 trainings occurred, not 4!
    ```

            Distributed training across EC2 instances:
    ```python
            from framework3.plugins.storage import LockingS3Storage

            # Same code runs on all EC2 instances
            storage = LockingS3Storage(
                bucket='ml-training-cache',
                region_name='us-west-2',
                access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
            )

            model = HeavyDeepLearningModel()
            cached = CachedWithLocking(
                filter=model,
                storage=storage,
                lock_ttl=2*3600,  # 2 hour training time
                lock_timeout=3*3600  # 3 hour max wait
            )

            # Only ONE EC2 instance trains
            # All others wait and download the trained model
            cached.fit(huge_dataset)
    ```

            Custom TTL for long operations:
    ```python
            # Training takes 6 hours, predictions take 1 hour
            cached = CachedWithLocking(
                model,
                storage,
                lock_ttl=7*3600,      # 7 hour lock validity
                lock_timeout=8*3600,  # 8 hour max wait
                cache_filter=True,
                cache_data=True
            )
    ```

        Note:
            The storage backend MUST be a BaseLockingStorage (either
            LockingLocalStorage or LockingS3Storage). Regular LocalStorage
            or S3Storage will raise an error.

            Console output shows process coordination:
    ```
            🔨 Training model abc12345...
            ✅ Model abc12345 cached
    ```

            Or for waiting processes:
    ```
            ⏳ Waiting for model abc12345 to be trained...
            📥 Loaded cached model abc12345
    ```
    """

    def __init__(
        self,
        filter: BaseFilter,
        cache_data: bool = True,
        cache_filter: bool = True,
        overwrite: bool = False,
        storage: BaseStorage | None = None,
        lock_ttl: int = 3600,
        lock_timeout: int = 7200,
        heartbeat_interval: int | None = 30,
        auto_heartbeat: bool = True,
    ):
        """Initialize cached filter with distributed locking.

                Args:
                    filter: The ML filter/model to wrap. Must have fit() and predict().
                    cache_data: Whether to cache prediction results. Default True.
                    cache_filter: Whether to cache trained models. Default True.
                    overwrite: Whether to force recomputation even if cached. Default False.
                    storage: Storage backend. Must be a BaseLockingStorage instance
                        (LockingLocalStorage or LockingS3Storage). If None, uses
                        Container.storage.
                    lock_ttl: Lock validity duration in seconds. Should be longer than
                        expected fit() duration. Default 3600 (1 hour).
                    lock_timeout: Maximum time to wait for another process. Should be
                        longer than lock_ttl. Default 7200 (2 hours).
                    heartbeat_interval: Seconds between heartbeat updates. Default 30.
                        Should be much smaller than lock_ttl (e.g., ttl/10).
                    auto_heartbeat: If True, automatically update heartbeat during
                        long operations. Default True.


                Raises:
                    TypeError: If storage is not a BaseLockingStorage instance.

                Examples:
        ```python
                    # Quick experiments (short TTL)
                    cached = CachedWithLocking(
                        model, storage,
                        lock_ttl=600,      # 10 min
                        lock_timeout=900   # 15 min
                    )

                    # Production training (long TTL)
                    cached = CachedWithLocking(
                        model, storage,
                        lock_ttl=4*3600,   # 4 hours
                        lock_timeout=5*3600  # 5 hours
                    )

                    # No model caching, only predictions
                    cached = CachedWithLocking(
                        model, storage,
                        cache_filter=False,
                        cache_data=True
                    )

                    # Force recomputation
                    cached = CachedWithLocking(
                        model, storage,
                        overwrite=True
                    )
        ```
        """
        super().__init__(
            filter=filter,
            cache_data=cache_data,
            cache_filter=cache_filter,
            overwrite=overwrite,
            storage=storage,
        )

        self.filter: BaseFilter = filter
        self.cache_data = cache_data
        self.cache_filter = cache_filter
        self.overwrite = overwrite
        self.lock_ttl = lock_ttl
        self.lock_timeout = lock_timeout
        self.heartbeat_interval = heartbeat_interval
        self.auto_heartbeat = auto_heartbeat

        if heartbeat_interval is None:
            # Default: 1/20 of TTL (e.g., 180s for 3600s TTL)
            self.heartbeat_interval = max(30, lock_ttl // 20)
        elif heartbeat_interval == 0:
            # Explicitly disabled
            self.heartbeat_interval = None
        else:
            self.heartbeat_interval = heartbeat_interval

        # Get storage and validate it supports locking
        self._storage: BaseStorage = Container.storage if storage is None else storage

        if not isinstance(self._storage, BaseLockingStorage):
            raise TypeError(
                f"Storage must be a BaseLockingStorage instance "
                f"(LockingLocalStorage or LockingS3Storage), "
                f"got {type(self._storage).__name__}"
            )

        self._lambda_filter: Callable[..., BaseFilter] | None = None

    def verbose(self, value: bool):
        """Set verbosity for this filter, storage, and wrapped filter."""
        super().verbose(value)
        self._storage._verbose = value
        self.filter.verbose(value)

    def _run_with_heartbeat(
        self,
        lock_name: str,
        operation: Callable[[], Any],
        operation_name: str = "operation",
    ) -> Any:
        """Run an operation with automatic heartbeat updates.

        Args:
            lock_name: Lock identifier for heartbeat updates.
            operation: Callable to execute.
            operation_name: Name for logging purposes.

        Returns:
            Result of the operation.
        """
        if not self.auto_heartbeat or self.heartbeat_interval is None:
            # Heartbeat disabled, run directly
            return operation()

        # Check if storage supports heartbeat
        if not hasattr(self._storage, "update_heartbeat"):
            return operation()

        # Setup heartbeat thread
        stop_heartbeat = threading.Event()
        heartbeat_errors = []

        def heartbeat_worker():
            """Background thread that sends heartbeats."""
            while not stop_heartbeat.is_set():
                try:
                    success = self._storage.update_heartbeat(lock_name)
                    if not success and self._verbose:
                        heartbeat_errors.append("Failed to update heartbeat")
                except Exception as e:
                    heartbeat_errors.append(str(e))

                # Wait for next heartbeat or stop signal
                stop_heartbeat.wait(timeout=self.heartbeat_interval)

        # Start heartbeat thread
        heartbeat_thread = threading.Thread(
            target=heartbeat_worker, daemon=True, name=f"Heartbeat-{lock_name[:8]}"
        )
        heartbeat_thread.start()

        if self._verbose:
            print(
                f"\t💓 Heartbeat started for {operation_name} (interval: {self.heartbeat_interval}s)"
            )

        try:
            # Run the actual operation
            result = operation()
            return result
        finally:
            # Stop heartbeat
            stop_heartbeat.set()
            heartbeat_thread.join(timeout=2.0)

            if self._verbose:
                print(f"\t💓 Heartbeat stopped for {operation_name}")

            if heartbeat_errors and self._verbose:
                print(f"\t⚠️ Heartbeat errors: {len(heartbeat_errors)} occurred")

    def _get_model_name(self) -> str:
        """Get the name of the underlying filter's model."""
        return self.filter._get_model_name()

    def _get_model_key(self, data_hash: str) -> Tuple[str, str]:
        """Generate the model key based on input data hash."""
        return BaseFilter._get_model_key(self.filter, data_hash)

    def _get_data_key(self, model_str: str, data_hash: str) -> Tuple[str, str]:
        """Generate the data key based on model and input data hash."""
        return BaseFilter._get_data_key(self.filter, model_str, data_hash)

    def fit(self, x: XYData, y: Optional[XYData]) -> None:
        """Fit the filter with distributed locking to prevent race conditions.

                Coordinates with other processes using distributed locks to ensure only
                one process trains each unique model. Uses model hash for identification.

                Process flow:
                1. Check if model exists in cache (fast path)
                2. Try to acquire lock for this model hash
                3. If acquired: train model, save to cache, release lock
                4. If not acquired: wait for other process, then load from cache

                Args:
                    x: Training input data (XYData with _hash attribute).
                    y: Training target data (optional for unsupervised learning).

                Raises:
                    TimeoutError: If waiting for another process exceeds lock_timeout.
                    TypeError: If storage doesn't support locking.

                Examples:
        ```python
                    # Normal training - uses cache and locking
                    cached.fit(X_train, y_train)

                    # Force retraining (ignores cache)
                    cached_overwrite = CachedWithLocking(model, storage, overwrite=True)
                    cached_overwrite.fit(X_train, y_train)

                    # In parallel processes - only one trains
                    from multiprocessing import Pool

                    def train(process_id):
                        cached = CachedWithLocking(model, storage)
                        cached.fit(X_train, y_train)  # Safe!

                    with Pool(4) as p:
                        p.map(train, range(4))  # Only 1 training occurs
        ```

                Note:
                    Console output indicates which process is training:
        ```
                    🔒 Lock acquired: model_abc12345
                    🔨 Training model abc12345...
                    ✅ Model abc12345 cached
                    🔓 Lock released: model_abc12345
        ```

                    Or waiting:
        ```
                    ⏳ Waiting for model abc12345 to be trained...
                    📥 Loaded cached model abc12345
        ```
        """
        f_m_hash = self.filter._m_hash
        f_m_path = self.filter._m_path
        f_m_str = self.filter._m_str

        try:
            self.filter._pre_fit(x, y)

            model_path = f"{self._storage.get_root_path()}{self.filter._m_path}"
            lock_name = f"model_{self.filter._m_hash}"

            # Fast path: model already exists
            if (
                self._storage.check_if_exists(hashcode="model", context=model_path)
                and not self.overwrite
            ):
                if self._verbose:
                    rprint(f"\t📥 Model {self.filter._m_hash[:8]} already cached")
                self._lambda_filter = lambda: cast(
                    BaseFilter,
                    self._storage.download_file("model", model_path),
                )
                return

            # Try to acquire lock with heartbeat
            acquired = self._storage.try_acquire_lock(
                lock_name, ttl=self.lock_ttl, heartbeat_interval=self.heartbeat_interval
            )

            if acquired:
                try:
                    # Double-check after acquiring lock
                    if (
                        self._storage.check_if_exists(
                            hashcode="model", context=model_path
                        )
                        and not self.overwrite
                    ):
                        if self._verbose:
                            rprint(
                                f"\t📥 Model {self.filter._m_hash[:8]} cached by another process"
                            )
                        self._lambda_filter = lambda: cast(
                            BaseFilter,
                            self._storage.download_file("model", model_path),
                        )
                        return

                    # We're the producer: train with heartbeat
                    if self._verbose:
                        rprint(f"\t🔨 Training model {self.filter._m_hash[:8]}...")

                    # Train with automatic heartbeat
                    def train_operation():
                        self.filter._original_fit(x, y)

                    self._run_with_heartbeat(
                        lock_name, train_operation, operation_name="model training"
                    )

                    # Cache the trained model
                    if self.cache_filter and method_is_overridden(
                        self.filter.__class__, "fit"
                    ):
                        if self._verbose:
                            rprint(f"\t💾 Caching model {self.filter._m_hash[:8]}...")

                        self._storage.upload_file(
                            file=pickle.dumps(self.filter),
                            file_name="model",
                            context=model_path,
                        )

                        if self._verbose:
                            rprint(f"\t✅ Model {self.filter._m_hash[:8]} cached")

                finally:
                    self._storage.release_lock(lock_name)

            else:
                # We're a consumer: wait for the producer
                if self._verbose:
                    rprint(f"\t⏳ Waiting for model {self.filter._m_hash[:8]}...")

                wait_success = self._storage.wait_for_unlock(
                    lock_name, timeout=self.lock_timeout
                )

                if not wait_success:
                    raise TimeoutError(
                        f"Timeout waiting for model {self.filter._m_hash} "
                        f"(waited {self.lock_timeout}s)"
                    )

                # Load the model
                if self._verbose:
                    rprint(f"\t📥 Loading cached model {self.filter._m_hash[:8]}...")

                self._lambda_filter = lambda: cast(
                    BaseFilter,
                    self._storage.download_file("model", model_path),
                )

        except Exception as e:
            self.filter._m_hash = f_m_hash
            self.filter._m_path = f_m_path
            self.filter._m_str = f_m_str
            raise e

    def predict(self, x: XYData) -> XYData:
        """Make predictions with distributed locking to prevent race conditions.

                Coordinates with other processes using distributed locks to ensure only
                one process computes predictions for each unique input. Uses data hash
                for identification.

                Process flow:
                1. Check if predictions exist in cache (fast path)
                2. Try to acquire lock for this data hash
                3. If acquired: compute predictions, save to cache, release lock
                4. If not acquired: wait for other process, then load from cache

                Args:
                    x: Input data to predict on (XYData with _hash attribute).

                Returns:
                    Predictions as XYData object.

                Raises:
                    TimeoutError: If waiting for another process exceeds lock_timeout.

                Examples:
        ```python
                    # Normal prediction - uses cache and locking
                    predictions = cached.predict(X_test)

                    # In cross-validation - same fold predictions reused
                    from sklearn.model_selection import KFold

                    kf = KFold(n_splits=5)
                    for train_idx, test_idx in kf.split(X):
                        cached.fit(X[train_idx], y[train_idx])
                        # Multiple folds with same test data:
                        # First process computes, others wait and reuse
                        pred = cached.predict(X[test_idx])

                    # In parallel grid search
                    from joblib import Parallel, delayed

                    def evaluate(params):
                        model = CachedWithLocking(MyFilter(**params), storage)
                        model.fit(X_train, y_train)
                        return model.predict(X_val)  # Cached per unique X_val

                    Parallel(n_jobs=8)(
                        delayed(evaluate)(p) for p in param_grid
                    )
        ```

                Note:
                    Console output indicates computation status:
        ```
                    🔒 Lock acquired: data_xyz78901
                    🔨 Computing predictions xyz78901...
                    ✅ Predictions xyz78901 cached
                    🔓 Lock released: data_xyz78901
        ```

                    Or waiting:
        ```
                    ⏳ Waiting for predictions xyz78901...
                    📥 Loaded cached predictions xyz78901
        ```
        """
        x = self.filter._pre_predict(x)

        data_path = f"{self._storage.get_root_path()}{x._path}"
        lock_name = f"data_{x._hash}"

        # Fast path: predictions already exist
        if (
            self._storage.check_if_exists(hashcode=x._hash, context=data_path)
            and not self.overwrite
        ):
            if self._verbose:
                rprint(f"\t📥 Predictions {x._hash[:8]} already cached")

            return XYData(
                _hash=x._hash,
                _path=x._path,
                _value=lambda: cast(
                    VData,
                    self._storage.download_file(x._hash, data_path),
                ),
            )

        # Try to acquire lock with heartbeat
        acquired = self._storage.try_acquire_lock(
            lock_name, ttl=self.lock_ttl, heartbeat_interval=self.heartbeat_interval
        )

        if acquired:
            try:
                # Double-check after acquiring lock
                if self._storage.check_if_exists(hashcode=x._hash, context=data_path):
                    if self._verbose:
                        rprint(
                            f"\t📥 Predictions {x._hash[:8]} cached by another process"
                        )

                    return XYData(
                        _hash=x._hash,
                        _path=x._path,
                        _value=lambda: cast(
                            VData,
                            self._storage.download_file(x._hash, data_path),
                        ),
                    )

                # We're the producer: compute predictions with heartbeat
                if self._verbose:
                    rprint(f"\t🔨 Computing predictions {x._hash[:8]}...")

                # Load filter if needed
                if self._lambda_filter is not None:
                    if self._verbose:
                        rprint("\t📥 Loading filter from storage...")
                    self.filter = self._lambda_filter()

                # Compute predictions with automatic heartbeat
                def predict_operation():
                    return self.filter._original_predict(x)._value

                result_value = self._run_with_heartbeat(
                    lock_name, predict_operation, operation_name="predictions"
                )

                value = XYData(
                    _hash=x._hash,
                    _path=x._path,
                    _value=result_value,
                )

                # Cache predictions
                if self.cache_data:
                    if self._verbose:
                        rprint(f"\t💾 Caching predictions {x._hash[:8]}...")

                    self._storage.upload_file(
                        file=pickle.dumps(value.value),
                        file_name=x._hash,
                        context=data_path,
                    )

                    if self._verbose:
                        rprint(f"\t✅ Predictions {x._hash[:8]} cached")

                return value

            finally:
                self._storage.release_lock(lock_name)

        else:
            # We're a consumer: wait for the producer
            if self._verbose:
                rprint(f"\t⏳ Waiting for predictions {x._hash[:8]}...")

            wait_success = self._storage.wait_for_unlock(
                lock_name, timeout=self.lock_timeout
            )

            if not wait_success:
                raise TimeoutError(
                    f"Timeout waiting for predictions {x._hash} "
                    f"(waited {self.lock_timeout}s)"
                )

            # Load cached predictions
            if self._verbose:
                rprint(f"\t📥 Loading cached predictions {x._hash[:8]}...")

            return XYData(
                _hash=x._hash,
                _path=x._path,
                _value=lambda: cast(
                    VData,
                    self._storage.download_file(x._hash, data_path),
                ),
            )

    def clear_cache(self):
        """Clear the cache in the storage.

        Note:
            This method is not yet implemented. Should implement logic to
            clear all cached data and models, and potentially remove locks.
        """
        raise NotImplementedError("El método clear_cache no está implementado.")
