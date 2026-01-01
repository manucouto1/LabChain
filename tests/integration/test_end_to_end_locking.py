# tests/integration/test_end_to_end_locking.py

import pytest
from sklearn.datasets import load_iris
from multiprocessing import Process, Queue
from pathlib import Path
import shutil

from labchain.base import XYData
from labchain.plugins.storage import LockingLocalStorage
from labchain.plugins.filters.cache import CachedWithLocking
from labchain.plugins.filters.transformation.pca import PCAPlugin
from labchain.plugins.filters.transformation.scaler import StandardScalerPlugin
from labchain.plugins.filters.classification.knn import KnnFilter
from labchain.plugins.pipelines.sequential.f3_pipeline import F3Pipeline


@pytest.fixture
def temp_storage_e2e():
    """Fixture for end-to-end tests."""
    storage_path = "tests/test_data/e2e_locking"
    storage = LockingLocalStorage(storage_path=storage_path)

    yield storage

    # Cleanup
    if Path(storage_path).exists():
        shutil.rmtree(storage_path)


class TestEndToEndMLPipeline:
    """Test complete ML pipeline with locking."""

    def test_simple_pipeline_with_locking(self, temp_storage_e2e):
        """Test a simple ML pipeline with CachedWithLocking."""
        # Load data
        iris = load_iris()
        X = XYData.mock(iris.data)  # type: ignore
        y = XYData.mock(iris.target)  # type: ignore

        # Create pipeline with caching
        pipeline = F3Pipeline(
            filters=[StandardScalerPlugin(), PCAPlugin(n_components=2), KnnFilter()],
            metrics=[],
        )

        cached_pipeline = CachedWithLocking(
            filter=pipeline,
            storage=temp_storage_e2e,
            cache_filter=True,
            cache_data=True,
        )

        # Train
        cached_pipeline.fit(X, y)

        # Predict
        predictions = cached_pipeline.predict(X)

        assert predictions is not None
        assert predictions.value.shape == (150,)

    def test_parallel_pipeline_execution(self, temp_storage_e2e):
        """Test multiple processes running the same pipeline."""
        results_queue = Queue()

        def run_pipeline(process_id, storage_path, results_queue):
            """Process function that runs pipeline."""
            from sklearn.datasets import load_iris

            storage = LockingLocalStorage(storage_path)

            iris = load_iris()
            X = XYData.mock(iris.data)  # type: ignore
            y = XYData.mock(iris.target)  # type: ignore

            pipeline = F3Pipeline(
                filters=[
                    StandardScalerPlugin(),
                    PCAPlugin(n_components=2),
                    KnnFilter(),
                ],
                metrics=[],
            )

            cached_pipeline = CachedWithLocking(
                filter=pipeline,
                storage=storage,
                cache_filter=True,
                cache_data=True,
                lock_ttl=30,
                lock_timeout=60,
            )

            # Train and predict
            import time

            start = time.time()
            cached_pipeline.fit(X, y)
            predictions = cached_pipeline.predict(X)
            elapsed = time.time() - start

            results_queue.put(
                {
                    "process_id": process_id,
                    "elapsed": elapsed,
                    "predictions_shape": predictions.value.shape,
                    "success": True,
                }
            )

        # Run 4 processes in parallel
        processes = []
        for i in range(4):
            p = Process(
                target=run_pipeline,
                args=(i, temp_storage_e2e.storage_path.rstrip("/"), results_queue),
            )
            p.start()
            processes.append(p)

        # Wait for all processes
        for p in processes:
            p.join(timeout=120)
            assert p.exitcode == 0, "Process failed"

        # Collect results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())

        assert len(results) == 4
        assert all(r["success"] for r in results)

        # All should produce same predictions
        shapes = [r["predictions_shape"] for r in results]
        assert all(s == (150,) for s in shapes)


class TestRealWorldScenario:
    """Test realistic ML workflow scenarios."""

    def test_grid_search_simulation(self, temp_storage_e2e):
        """Simulate grid search where same configs are tried multiple times."""
        results_queue = Queue()

        configs = [
            {"n_components": 2, "n_neighbors": 3},
            {"n_components": 2, "n_neighbors": 3},  # Duplicate
            {"n_components": 2, "n_neighbors": 5},
            {"n_components": 3, "n_neighbors": 3},
            {"n_components": 2, "n_neighbors": 3},  # Duplicate
        ]

        def train_config(config_idx, config, storage_path, results_queue):
            """Train with specific config."""
            from sklearn.datasets import load_iris
            import time

            storage = LockingLocalStorage(storage_path)

            iris = load_iris()
            X = XYData.mock(iris.data)  # type: ignore
            y = XYData.mock(iris.target)  # type: ignore

            pipeline = F3Pipeline(
                filters=[
                    StandardScalerPlugin(),
                    PCAPlugin(n_components=config["n_components"]),
                    KnnFilter(n_neighbors=config["n_neighbors"]),
                ],
                metrics=[],
            )

            cached_pipeline = CachedWithLocking(
                filter=pipeline,
                storage=storage,
                cache_filter=True,
                cache_data=True,
                lock_ttl=60,
                lock_timeout=120,
            )

            start = time.time()
            cached_pipeline.fit(X, y)
            cached_pipeline.predict(X)
            elapsed = time.time() - start

            results_queue.put(
                {
                    "config_idx": config_idx,
                    "config": config,
                    "elapsed": elapsed,
                    "model_hash": cached_pipeline.filter._m_hash,
                }
            )

        # Run all configs in parallel
        processes = []
        for idx, config in enumerate(configs):
            p = Process(
                target=train_config,
                args=(
                    idx,
                    config,
                    temp_storage_e2e.storage_path.rstrip("/"),
                    results_queue,
                ),
            )
            p.start()
            processes.append(p)

        # Wait for all
        for p in processes:
            p.join(timeout=180)
            assert p.exitcode == 0

        # Analyze results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())

        assert len(results) == 5

        # Configs with same parameters should have same hash
        hashes_by_config = {}
        for r in results:
            key = (r["config"]["n_components"], r["config"]["n_neighbors"])
            if key not in hashes_by_config:
                hashes_by_config[key] = []
            hashes_by_config[key].append(r["model_hash"])

        # Same configs should produce same hashes
        for key, hashes in hashes_by_config.items():
            assert (
                len(set(hashes)) == 1
            ), f"Config {key} produced different hashes: {hashes}"
