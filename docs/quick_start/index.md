---
icon: material/rocket-launch
---

# Quick Start Guide

This comprehensive guide demonstrates LabChain's core capabilities through practical examples. You'll learn how to build pipelines, add caching, perform cross-validation, optimize hyperparameters, and coordinate distributed experiments.

## Installation

Install LabChain via pip:

```bash
pip install framework3
```

Verify the installation:

```python
import labchain
print(f"LabChain version: {labchain.__version__}")
```

## Core Concepts

Before diving into examples, understand LabChain's key abstractions:

- **XYData**: Container that wraps data with metadata and content-addressable hashing
- **BaseFilter**: Any transformation with `fit()` and `predict()` methods (preprocessing, models, etc.)
- **BasePipeline**: Orchestrates multiple filters in sequence, parallel, or MapReduce patterns
- **BaseMetric**: Evaluation functions that know optimization direction (higher/lower is better)
- **Cached**: Wrapper that adds automatic caching to any filter
- **Container**: Dependency injection system that manages component registration

## Example 1: Basic Pipeline

Let's start with a simple classification pipeline using the Iris dataset.

### Step 1: Load and Prepare Data

```python
from labchain import F3Pipeline, XYData
from labchain.plugins.filters import StandardScalerPlugin, KnnFilter
from labchain.plugins.metrics import F1, Precission, Recall
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Wrap in XYData containers
x_train = XYData("iris_train_X", "/datasets/iris", X_train)
y_train = XYData("iris_train_y", "/datasets/iris", y_train)
x_test = XYData("iris_test_X", "/datasets/iris", X_test)
y_test = XYData("iris_test_y", "/datasets/iris", y_test)
```

**What's happening:**

- `XYData` wraps your numpy arrays with metadata
- The first argument is a descriptive name
- The second argument is a logical path (used for cache organization)
- The third argument is the actual data

### Step 2: Create a Pipeline

```python
# Create pipeline with preprocessing and classification
pipeline = F3Pipeline(
    filters=[
        StandardScalerPlugin(),  # Normalize features
        KnnFilter(n_neighbors=5) # K-Nearest Neighbors classifier
    ],
    metrics=[F1(), Precision(), Recall()]
)

# Train the pipeline
pipeline.fit(x_train, y_train)

# Make predictions
predictions = pipeline.predict(x_test)

# Evaluate performance
results = pipeline.evaluate(x_test, y_test, predictions)
print(results)
# {'F1': 0.9666..., 'Precision': 0.9722..., 'Recall': 0.9666...}
```

**Key points:**

- Filters execute sequentially in the order specified
- Each filter's output becomes the next filter's input
- Metrics are computed during evaluation only

## Example 2: Adding Smart Caching

Now let's add caching to avoid recomputing expensive operations. This is especially valuable when experimenting with different models on the same preprocessed data.

### Step 1: Configure Storage

```python
from labchain import Container
from labchain.plugins.storage import LocalStorage

# Configure where cache will be stored
Container.storage = LocalStorage(storage_path='./cache')
```

### Step 2: Wrap Expensive Filters

```python
from labchain.plugins.filters import Cached

# Wrap the preprocessing step with caching
pipeline = F3Pipeline(
    filters=[
        Cached(
            filter=StandardScalerPlugin(),
            cache_data=True,      # Cache the scaled data
            cache_filter=True,    # Cache the fitted scaler
            overwrite=False       # Reuse existing cache
        ),
        KnnFilter(n_neighbors=5)
    ],
    metrics=[F1()]
)

# First run: computes and caches scaling
print("First run (with caching):")
pipeline.fit(x_train, y_train)
predictions_1 = pipeline.predict(x_test)

# Second run: loads from cache (much faster!)
print("\nSecond run (from cache):")
pipeline.fit(x_train, y_train)
predictions_2 = pipeline.predict(x_test)
```

### Step 3: Test Different Classifiers with Cached Preprocessing

```python
from labchain.plugins.filters.classification.svm import ClassifierSVMPlugin

# Change classifier, keep preprocessing cached
pipeline_svm = F3Pipeline(
    filters=[
        Cached(
            filter=StandardScalerPlugin(),
            cache_data=True,
            cache_filter=True
        ),
        ClassifierSVMPlugin(kernel='rbf', C=1.0)  # Different classifier
    ],
    metrics=[F1(), Precision(), Recall()]
)

# Preprocessing loads from cache, only SVM trains
pipeline_svm.fit(x_train, y_train)
predictions_svm = pipeline_svm.predict(x_test)
results_svm = pipeline_svm.evaluate(x_test, y_test, predictions_svm)
print(results_svm)
```

**Benefits:**

- Preprocessing computed once, reused for all classifiers
- Dramatically faster experimentation
- Cache is content-addressable (automatic invalidation when data changes)

## Example 3: Cross-Validation for Robust Evaluation

Cross-validation provides more reliable performance estimates. LabChain makes it trivial.

### Using K-Fold Cross-Validation

```python
from labchain.plugins.splitter import KFoldSplitter

# Create pipeline with cross-validation
pipeline_cv = F3Pipeline(
    filters=[
        StandardScalerPlugin(),
        KnnFilter(n_neighbors=5)
    ],
    metrics=[F1(), Precision(), Recall()]
).splitter(
    KFoldSplitter(
        n_splits=5,        # 5-fold CV
        shuffle=True,      # Shuffle before splitting
        random_state=42    # Reproducibility
    )
)

# Fit performs 5-fold cross-validation automatically
results_cv = pipeline_cv.fit(x_train, y_train)

print("Cross-validation results:")
print(f"F1: {results_cv['F1']:.3f} ± {results_cv['F1_std']:.3f}")
print(f"Precision: {results_cv['Precision']:.3f} ± {results_cv['Precision_std']:.3f}")
print(f"Recall: {results_cv['Recall']:.3f} ± {results_cv['Recall_std']:.3f}")

# Access individual fold scores
print(f"\nIndividual fold F1 scores: {results_cv['F1_scores']}")
```

**What you get:**

- Mean score across all folds
- Standard deviation (measures stability)
- Individual fold scores for detailed analysis

### Stratified K-Fold for Imbalanced Data

```python
from labchain.plugins.splitter import StratifiedKFoldSplitter

# Use stratified splitting to maintain class distribution
pipeline_stratified = F3Pipeline(
    filters=[StandardScalerPlugin(), KnnFilter()],
    metrics=[F1()]
).splitter(
    StratifiedKFoldSplitter(
        n_splits=5,
        shuffle=True,
        random_state=42
    )
)

results_stratified = pipeline_stratified.fit(x_train, y_train)
```

## Example 4: Hyperparameter Optimization

Find the best hyperparameters automatically using different optimization strategies.

### Grid Search (Exhaustive)

```python
from labchain.plugins.optimizer import GridOptimizer

# Define parameter grid on the filter
knn_with_grid = KnnFilter().grid({
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance']
})

# Create pipeline with optimization
pipeline_grid = F3Pipeline(
    filters=[
        StandardScalerPlugin(),
        knn_with_grid
    ],
    metrics=[F1(), Precision(), Recall()]
).splitter(
    KFoldSplitter(n_splits=5, shuffle=True)
).optimizer(
    GridOptimizer(scorer=F1())
)

# This performs:
# - 10 configurations (5 n_neighbors × 2 weights)
# - 5-fold CV per configuration (50 model trainings)
# - Returns best configuration
best_results = pipeline_grid.fit(x_train, y_train)

print(f"Best configuration found:")
print(f"Best F1: {best_results['best_score']:.3f}")
print(f"Best params: {best_results['best_params']}")
```

### Bayesian Optimization (Efficient)

```python
from labchain.plugins.optimizer.optuna_optimizer import OptunaOptimizer

# Define search space
knn_optuna = KnnFilter().grid({
    'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
    'weights': ['uniform', 'distance'],
    'p': [1, 2]  # Manhattan or Euclidean distance
})

# Use Bayesian optimization
pipeline_optuna = F3Pipeline(
    filters=[StandardScalerPlugin(), knn_optuna],
    metrics=[F1()]
).splitter(
    KFoldSplitter(n_splits=5)
).optimizer(
    OptunaOptimizer(
        direction="maximize",
        n_trials=30,              # Fewer trials than grid search
        study_name="knn_optimization",
        storage="sqlite:///optuna.db"  # Persist study
    )
)

best_optuna = pipeline_optuna.fit(x_train, y_train)
print(f"Optuna best F1: {best_optuna['best_score']:.3f}")
```

**Why Bayesian optimization:**

- Smarter search strategy (learns from previous trials)
- Fewer evaluations needed
- Can handle continuous and categorical parameters

### Weights & Biases Integration

```python
from labchain.plugins.optimizer.wandb_optimizer import WandbOptimizer

# Track experiments in W&B cloud
pipeline_wandb = F3Pipeline(
    filters=[
        StandardScalerPlugin(),
        KnnFilter().grid({
            'n_neighbors': [3, 5, 7, 9, 11]
        })
    ],
    metrics=[F1(), Precision(), Recall()]
).optimizer(
    WandbOptimizer(
        project="labchain-iris-classification",
        sweep_id=None,  # Creates new sweep
        scorer=F1()
    )
)

# Results tracked in W&B dashboard
best_wandb = pipeline_wandb.fit(x_train, y_train)
```

## Example 5: Combining Everything

Let's combine caching, cross-validation, and optimization for a production-grade workflow.

```python
from labchain.plugins.filters import Cached
from labchain.plugins.splitter import StratifiedKFoldSplitter
from labchain.plugins.optimizer.optuna_optimizer import OptunaOptimizer

# Configure storage for caching
Container.storage = LocalStorage('./ml_cache')

# Create comprehensive pipeline
production_pipeline = F3Pipeline(
    filters=[
        # Cache preprocessing (computed once)
        Cached(
            filter=StandardScalerPlugin(),
            cache_data=True,
            cache_filter=True
        ),
        # Optimize classifier
        KnnFilter().grid({
            'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
            'weights': ['uniform', 'distance'],
            'p': [1, 2]
        })
    ],
    metrics=[F1(), Precission(), Recall()]
).splitter(
    # Robust evaluation
    StratifiedKFoldSplitter(n_splits=5, shuffle=True, random_state=42)
).optimizer(
    # Smart hyperparameter search
    OptunaOptimizer(
        direction="maximize",
        n_trials=50,
        scorer=F1(),
        study_name="production_knn",
        storage="sqlite:///production_experiments.db"
    )
)

# Enable verbose output
production_pipeline.verbose(True)

# Run complete workflow
final_results = production_pipeline.fit(x_train, y_train)

print("\n=== FINAL RESULTS ===")
print(f"Best F1 Score: {final_results['best_score']:.4f}")
print(f"Best Parameters: {final_results['best_params']}")
print(f"Cross-validation std: {final_results.get('cv_std', 'N/A')}")
```

**What this pipeline does:**

1. **Preprocessing** (with caching):

    - Scales features once
    - Cached for all subsequent trials

2. **Cross-validation** (robust evaluation):

    - 5-fold stratified CV
    - Maintains class distribution
    - Reports mean ± std

3. **Optimization** (smart search):

    - 50 Bayesian trials
    - Learns from previous evaluations
    - Persists results to SQLite

4. **Result**: Best model configuration with reliable performance estimate

**Computation savings:**

- Without caching: 50 trials × 5 folds × preprocessing = 250 preprocessing operations
- With caching: 1 preprocessing operation (reused 249 times)

## Example 6: Custom Filters

Create domain-specific transformations by extending `BaseFilter`.

```python
from labchain import BaseFilter, Container
import numpy as np

@Container.bind()
class LogTransform(BaseFilter):
    """Apply log transformation to features."""

    def __init__(self, offset: float = 1.0):
        """
        Args:
            offset: Value to add before log (avoid log(0)).
        """
        super().__init__(offset=offset)
        self.offset = offset  # Public attribute (in hash)
        self._fitted = False   # Private attribute (not in hash)

    def fit(self, x: XYData, y: XYData | None):
        """Log transform doesn't require fitting."""
        self._fitted = True

    def predict(self, x: XYData) -> XYData:
        """Apply log transformation."""
        transformed = np.log(x.value + self.offset)
        return XYData.mock(transformed)

# Use your custom filter
pipeline_custom = F3Pipeline(
    filters=[
        LogTransform(offset=1.0),  # Your custom filter
        StandardScalerPlugin(),     # Built-in filter
        KnnFilter()                 # Built-in filter
    ],
    metrics=[F1()]
)

pipeline_custom.fit(x_train, y_train)
results_custom = pipeline_custom.evaluate(x_test, y_test,
                                         pipeline_custom.predict(x_test))
```

**Key points for custom filters:**

- **Public attributes** (e.g., `self.offset`): Must be constructor parameters, included in hash
- **Private attributes** (e.g., `self._fitted`): Internal state, excluded from hash
- Decorate with `@Container.bind()` for automatic registration

## Example 7: Pipeline Serialization

Save and restore complete pipelines as JSON for reproducibility.

```python
# Create and train pipeline
pipeline = F3Pipeline(
    filters=[
        StandardScalerPlugin(),
        KnnFilter(n_neighbors=7)
    ],
    metrics=[F1(), Precision()]
)
pipeline.fit(x_train, y_train)

# Serialize to JSON
config = pipeline.item_dump()

import json
with open('my_pipeline.json', 'w') as f:
    json.dump(config, f, indent=2)

print("Pipeline saved to my_pipeline.json")

# Later: Restore pipeline
with open('my_pipeline.json', 'r') as f:
    loaded_config = json.load(f)

from labchain.base import BasePlugin

restored_pipeline = BasePlugin.build_from_dump(
    loaded_config,
    Container.pif
)

# Use restored pipeline
predictions_restored = restored_pipeline.predict(x_test)
```

**Use cases:**

- Version control experiments
- Share configurations with team
- Reproduce published results
- Deploy to production

## Performance Tips

### 1. Cache Strategically

```python
# ✅ DO: Cache expensive operations
Cached(filter=BERTEmbeddings())       # Hours of computation
Cached(filter=TfidfVectorizer())      # Minutes for large corpora

# ❌ DON'T: Cache trivial operations
Cached(filter=StandardScalerPlugin())  # Milliseconds (minimal benefit)
```

### 2. Use Appropriate Storage

```python
# Individual work
Container.storage = LocalStorage('./cache')

# Team collaboration (shared filesystem)
Container.storage = LocalStorage('/shared/nfs/ml_cache')

# Distributed teams (cloud)
from labchain.plugins.storage import S3Storage
Container.storage = S3Storage(
    bucket='team-ml-cache',
    region='us-east-1'
)
```

### 3. Optimize Cross-Validation

```python
# Fewer folds for quick iteration
.splitter(KFoldSplitter(n_splits=3))

# More folds for final evaluation
.splitter(KFoldSplitter(n_splits=10))
```

### 4. Smart Hyperparameter Search

```python
# Start with grid search (small space)
.grid({'n_neighbors': [3, 5, 7]})

# Graduate to Bayesian (larger space)
.optimizer(OptunaOptimizer(n_trials=50))
```

## Troubleshooting

### Cache Not Hitting

```python
# Enable verbose to see cache activity
pipeline.verbose(True)

# Check filter hash
print(f"Filter hash: {filter._m_hash}")

# Check storage configuration
print(f"Storage path: {Container.storage.get_root_path()}")

# Force cache refresh
Cached(filter=MyFilter(), overwrite=True)
```

### Memory Issues

```python
# Don't cache large intermediate results
Cached(filter=HugeTransform(), cache_data=False, cache_filter=True)

# Use data generators for large datasets
# (LabChain supports lazy loading through XYData)
```

### Slow Optimization

```python
# Reduce cross-validation folds during search
.splitter(KFoldSplitter(n_splits=3))  # Instead of 5

# Use fewer optimization trials
.optimizer(OptunaOptimizer(n_trials=20))  # Instead of 50

# Cache preprocessing
Cached(filter=preprocessing_step)
```

## Next Steps

You've learned the fundamentals! Explore these resources:

- **[Caching Guide](../start_caching/index.md)** — Deep dive into local and distributed caching
- **[Architecture](../architecture/index.md)** — Understand LabChain's design
- **[Examples](../examples/index.md)** — Real-world case studies
- **[API Reference](../api/index.md)** — Complete API documentation

## Summary

LabChain provides a modular, cacheable, and reproducible framework for ML experiments:

- ✅ **Modular**: Build pipelines from composable filters
- ✅ **Cacheable**: Automatic result reuse with content-addressable hashing
- ✅ **Reproducible**: JSON serialization for exact experiment replay
- ✅ **Optimizable**: Built-in grid, Bayesian, and W&B integration
- ✅ **Validated**: Seamless cross-validation support
- ✅ **Extensible**: Create custom filters by inheriting `BaseFilter`

Start building better ML experiments today! 🚀
