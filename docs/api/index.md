---
icon: material/bookshelf
---

# LabChain API Documentation

Welcome to the API documentation for LabChain. This comprehensive guide details the modules, classes, and functions that form the backbone of LabChain, enabling you to build, extend, and customize ML experimentation workflows efficiently.

---

## Table of Contents

- [Base Classes](#base-classes)
- [Container & Dependency Injection](#container-dependency-injection)
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
- [KFoldSplitter](plugins/splitters/stratified_kfold_splitter.md)

### Storage

Storage plugins for data persistence:

- [Local Storage](plugins/storage/local.md)
- [S3 Storage](plugins/storage/s3.md)

## Utilities

Additional utility functions and helpers that support the framework:

- [PySpark Utilities](utils/pyspark.md)
- [Weights & Biases Integration](utils/wandb.md)
- [Typeguard for Notebooks](utils/typeguard.md)
- [Scikit-learn Estimator Utilities](utils/sklearn.md)
- [General Utilities](utils/utils.md)

---

## Using the API

To utilize any component of LabChain, import it from the respective module and register it with the Container if necessary. For example:

```python
from labchain.container import Container
from labchain.base import BaseFilter, BasePipeline, BaseMetric

@Container.bind()
class MyFilter(BaseFilter):
    # Custom filter implementation

@Container.bind()
class MyPipeline(BasePipeline):
    # Custom pipeline implementation

@Container.bind()
class MyMetric(BaseMetric):
    # Custom metric implementation

# Retrieve components
my_filter = Container.ff["MyFilter"]()
my_pipeline = Container.pf["MyPipeline"]()
my_metric = Container.mf["MyMetric"]()

```
