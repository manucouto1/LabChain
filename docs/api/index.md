---
icon: material/bookshelf
---

# LabChain API Documentation

Welcome to the API documentation for LabChain. This comprehensive guide details the modules, classes, and functions that form the backbone of LabChain, enabling you to build, extend, and customize ML experimentation workflows efficiently.

---

## Table of Contents

- [Base Classes](#base-classes)
- [Container & Dependency Injection](#container-dependency-injection)
- [Persistent Storage & Remote Injection](#persistent-storage-remote-injection) ⚡ New
- [Plugins](#plugins)
  - [Pipelines](#pipelines)
  - [Filters](#filters)
  - [Metrics](#metrics)
  - [Optimizers](#optimizers)
  - [Splitters](#splitters)
  - [Storage](#storage)
- [Utilities](#utilities)
- [Using the API](#using-the-api)

---

## Base Classes

The foundation of LabChain is built on these abstract base classes:

- [Types](base/base_types.md) - Core data structures and type definitions.
- [Classes](base/base_plugin.md) - Abstract base class for all components.
- [Pipeline](base/base_pipelines.md) - Base class for creating pipelines.
- [Filter](base/base_filter.md) - Abstract class for all filter implementations.
- [Metric](base/base_metric.md) - Base class for metric implementations.
- [Optimizer](base/base_optimizer.md) - Abstract base for optimization algorithms.
- [Splitter](base/base_splitter.md) - Base class for data splitting strategies.
- [Factory](base/base_factory.md) - Factory classes for component creation.
- [Storage](base/base_storage.md) - Abstract base for storage implementations.

## Container & Dependency Injection

The core of LabChain's component management:

- [Container](container/container.md) - Main class for dependency injection and component management.
- [Overload](container/overload.md) - Utilities for method overloading in the container.

## Persistent Storage & Remote Injection

!!! warning "Experimental Feature"
    Remote Injection is currently an experimental feature. See the [Remote Injection Guide](../remote_injection/index.md) for important limitations and best practices.

Classes and systems for persistent class storage with version control:

- **[Remote Injection Guide](../remote_injection/index.md)** ⚡ - Complete guide to deploying pipelines without source code
- [PetClassManager](container/pet_class_manager.md) - Manager for class serialization and storage operations
- [PetFactory](container/pet_factory.md) - Persistent factory with automatic version tracking and lazy loading

### Quick Example
```python
from labchain import Container
from labchain.base import BaseFilter

# Enable persistence for custom classes
@Container.bind(persist=True)
class MyCustomFilter(BaseFilter):
    def predict(self, x):
        return x * 2

# Push to storage
Container.ppif.push_all()

# On remote server (no source code needed!)
from labchain.base import BasePlugin
pipeline = BasePlugin.build_from_dump(config, Container.ppif)
```

## Plugins

### Pipelines

Pipelines orchestrate the data flow through various processing steps:

- **Parallel Pipelines**
  - [MonoPipeline](plugins/pipelines/parallel/mono_pipeline.md) - For parallel processing of independent tasks.
  - [HPCPipeline](plugins/pipelines/parallel/hpc_pipeline.md) - Optimized for high-performance computing environments.
- **Sequential Pipeline**
  - [F3Pipeline](plugins/pipelines/sequential/f3_pipeline.md) - The basic sequential pipeline.

### Filters

Modular processing units that can be composed together within pipelines:

- [Classification Filters](plugins/filters/classification.md)
- [Clustering Filters](plugins/filters/clustering.md)
- [Regression Filters](plugins/filters/regression.md)
- [Transformation Filters](plugins/filters/transformation.md)
- [Text Processing Filters](plugins/filters/text_processing.md)
- [Cache Filters](plugins/filters/cache.md)
  - [CachedFilter](plugins/filters/cache.md)
- [Grid Search Filters](plugins/filters/grid_search.md)
  - [GridSearchCVFilter](plugins/filters/grid_search.md)

### Metrics

Metrics evaluate model performance across various tasks:

- [Classification Metrics](plugins/metrics/classification.md)
- [Clustering Metrics](plugins/metrics/clustering.md)
- [Coherence Metrics](plugins/metrics/coherence.md)

### Optimizers

Optimizers help fine-tune hyperparameters for optimal performance:

- [SklearnOptimizer](plugins/optimizers/sklearn_optimizer.md)
- [OptunaOptimizer](plugins/optimizers/optuna_optimizer.md)
- [WandbOptimizer](plugins/optimizers/wandb_optimizer.md)
- [GridOptimizer](plugins/optimizers/grid_optimizer.md)

### Splitters

Splitters divide the dataset for cross-validation and evaluation:

- [KFoldSplitter](plugins/splitters/kfold_splitter.md)
- [StratifiedKFoldSplitter](plugins/splitters/stratified_kfold_splitter.md)

### Storage

Storage plugins for data persistence and remote class storage:

- [Local Storage](plugins/storage/local.md) - Local filesystem storage
- [S3 Storage](plugins/storage/s3.md) - Amazon S3 cloud storage

## Utilities

Additional utility functions and helpers that support the framework:

- [PySpark Utilities](utils/pyspark.md)
- [Weights & Biases Integration](utils/wandb.md)
- [Typeguard for Notebooks](utils/typeguard.md)
- [Scikit-learn Estimator Utilities](utils/sklearn.md)
- [General Utilities](utils/utils.md)

---

## Using the API

### Standard Component Registration

To utilize any component of LabChain, import it from the respective module and register it with the Container:
```python
from labchain.container import Container
from labchain.base import BaseFilter, BasePipeline, BaseMetric

@Container.bind()
class MyFilter(BaseFilter):
    def fit(self, x, y):
        pass

    def predict(self, x):
        return x

@Container.bind()
class MyPipeline(BasePipeline):
    # Custom pipeline implementation
    pass

@Container.bind()
class MyMetric(BaseMetric):
    def evaluate(self, x_data, y_true, y_pred):
        return 0.95

# Retrieve components
my_filter = Container.ff["MyFilter"]()
my_pipeline = Container.pf["MyPipeline"]()
my_metric = Container.mf["MyMetric"]()
```

### Persistent Component Registration (Experimental)

For components that need to be deployed remotely or shared across environments:
```python
from labchain.container import Container
from labchain.base import BaseFilter
from labchain.storage import S3Storage

# Configure shared storage
Container.storage = S3Storage(bucket="my-ml-models")

# Register with persistence enabled
@Container.bind(persist=True)
class MyPersistentFilter(BaseFilter):
    def __init__(self, threshold: float = 0.5):
        super().__init__(threshold=threshold)

    def predict(self, x):
        return x.value > self.threshold

# Push to storage
Container.ppif.push_all()

# Later, on any machine with access to the same storage
my_filter = Container.ppif["MyPersistentFilter"]()
# Class automatically loaded from storage if not in memory
```

---

## API Organization

### By Functionality

- **Data Processing**: [Filters](plugins/filters/classification.md), [Transformations](plugins/filters/transformation.md)
- **Model Evaluation**: [Metrics](plugins/metrics/classification.md), [Splitters](plugins/splitters/kfold_splitter.md)
- **Workflow Orchestration**: [Pipelines](plugins/pipelines/sequential/f3_pipeline.md), [Optimizers](plugins/optimizers/wandb_optimizer.md)
- **Persistence & Storage**: [Storage Backends](plugins/storage/local.md), [Remote Injection](../remote_injection/index.md)
- **Infrastructure**: [Container](container/container.md), [Base Classes](base/base_plugin.md)

### By Use Case

- **Classification Tasks**: [Classification Filters](plugins/filters/classification.md), [Classification Metrics](plugins/metrics/classification.md)
- **Clustering Tasks**: [Clustering Filters](plugins/filters/clustering.md), [Clustering Metrics](plugins/metrics/clustering.md)
- **Hyperparameter Tuning**: [GridOptimizer](plugins/optimizers/grid_optimizer.md), [OptunaOptimizer](plugins/optimizers/optuna_optimizer.md)
- **Distributed Computing**: [HPCPipeline](plugins/pipelines/parallel/hpc_pipeline.md), [PySpark Utilities](utils/pyspark.md)
- **Remote Deployment**: [Remote Injection](../remote_injection/index.md), [S3 Storage](plugins/storage/s3.md)

---

## Quick Reference

### Most Common Operations

| Operation | Code |
|-----------|------|
| Register a filter | `@Container.bind()`<br>`class MyFilter(BaseFilter): ...` |
| Create a pipeline | `F3Pipeline(filters=[...], metrics=[...])` |
| Enable persistence | `@Container.bind(persist=True)` |
| Push to storage | `Container.ppif.push_all()` |
| Load from storage | `Container.ppif["ClassName"]` |
| Reconstruct pipeline | `BasePlugin.build_from_dump(config, Container.ppif)` |
| Check version status | `Container.pcm.check_status(MyClass)` |
| Get class hash | `Container.pcm.get_class_hash(MyClass)` |

### Import Shortcuts
```python
# Core functionality
from labchain import Container
from labchain.base import BaseFilter, BasePipeline, BaseMetric, XYData

# Common pipelines
from labchain.pipeline import F3Pipeline, MonoPipeline, HPCPipeline

# Storage
from labchain.storage import LocalStorage, S3Storage

# Common filters
from labchain.plugins.filters import (
    StandardScalerPlugin,
    PCAPlugin,
    KnnFilter,
    ClassifierSVMPlugin
)

# Common metrics
from labchain.plugins.metrics import F1, Precision, Recall
```

---

## Additional Resources

- 📘 [Quick Start Guide](../quick_start/index.md)
- 🎓 [Tutorials & Examples](../examples/index.md)
- 🏗️ [Architecture Overview](../architecture/index.md)
- ⚡ [Remote Injection Guide](../remote_injection/index.md) (Experimental)

---

!!! tip "Contributing to the Documentation"
    Found an error or want to improve the documentation? Contributions are welcome!

    - 📝 Edit on [GitHub](https://github.com/manucouto1/LabChain/docs)
    - 🐛 Report issues on [GitHub Issues](https://github.com/manucouto1/LabChain/issues)
