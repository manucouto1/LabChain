# F3Pipeline

::: labchain.plugins.pipelines.sequential.f3_pipeline

## Overview

The F3Pipeline is a flexible and powerful pipeline implementation in the LabChain ecosystem. It allows you to chain multiple data processing steps, machine learning models, and evaluation metrics into a single, cohesive workflow.

## Key Features

- Seamless integration of multiple plugins (filters, transformers, models)
- Built-in support for various metrics
- Caching capabilities for improved performance
- Nested pipeline support for complex workflows

## Basic Usage

Here's a simple example of how to create and use an F3Pipeline:

```python
from labchain.plugins.pipelines import F3Pipeline
from labchain.plugins.filters.transformation import PCAPlugin
from labchain.plugins.filters.classification import SVMClassifier
from labchain.plugins.metrics import F1Score, Accuracy
from labchain.base.base_types import XYData
import numpy as np

# Create a pipeline
pipeline = F3Pipeline(
    plugins=[
        PCAPlugin(n_components=2),
        SVMClassifier(kernel='rbf')
    ],
    metrics=[F1Score(), Accuracy()]
)

# Generate some dummy data
X = XYData(value=np.random.rand(100, 10))
y = XYData(value=np.random.randint(0, 2, 100))

# Fit the pipeline
pipeline.fit(X, y)

# Make predictions
y_pred = pipeline.predict(X)

# Evaluate the pipeline
results = pipeline.evaluate(X, y, y_pred)
print(results)
```

## Advanced Usage

### Nested Pipelines

F3Pipeline supports nesting, allowing you to create more complex workflows:

```python
from labchain.plugins.pipelines import F3Pipeline
from labchain.plugins.filters.transformation import NormalizationPlugin
from labchain.plugins.filters.feature_selection import VarianceThresholdPlugin

# Create a sub-pipeline
feature_engineering = F3Pipeline(
    plugins=[
        NormalizationPlugin(),
        VarianceThresholdPlugin(threshold=0.1)
    ],
    metrics=[]
)

# Create the main pipeline
main_pipeline = F3Pipeline(
    plugins=[
        feature_engineering,
        SVMClassifier(kernel='linear')
    ],
    metrics=[F1Score(), Accuracy()]
)

# Use the main pipeline as before
main_pipeline.fit(X, y)
y_pred = main_pipeline.predict(X)
results = main_pipeline.evaluate(X, y, y_pred)
```

### Caching

F3Pipeline supports caching of intermediate results and fitted models for improved performance:

```python
from labchain.plugins.filters.cached_filter import Cached
from labchain.plugins.filters.transformation import PCAPlugin

pipeline = F3Pipeline(
    plugins=[
        Cached(
            filter=PCAPlugin(n_components=2),
            cache_data=True,
            cache_filter=True,
            overwrite=False
        ),
        SVMClassifier()
    ],
    metrics=[F1Score()]
)

# The PCA transformation will be cached after the first run
pipeline.fit(X, y)
```

## API Reference

### F3Pipeline

```python
class F3Pipeline(BasePipeline):
    def __init__(self, plugins: List[BasePlugin], metrics: List[BaseMetric], overwrite: bool = False, store: bool = False, log: bool = False) -> None:
        """
        Initialize the F3Pipeline.

        Args:
            plugins (List[BasePlugin]): List of plugins to be applied in the pipeline.
            metrics (List[BaseMetric]): List of metrics for evaluation.
            overwrite (bool, optional): Whether to overwrite existing data. Defaults to False.
            store (bool, optional): Whether to store intermediate results. Defaults to False.
            log (bool, optional): Whether to log pipeline operations. Defaults to False.
        """
```

#### Methods

- `fit(self, x: XYData, y: Optional[XYData])`: Fit the pipeline to the input data.
- `predict(self, x: XYData) -> XYData`: Make predictions using the fitted pipeline.
- `evaluate(self, x_data: XYData, y_true: XYData|None, y_pred: XYData) -> Dict[str, float]`: Evaluate the pipeline using the specified metrics.

## Best Practices

1. **Order Matters**: The order of plugins in the pipeline is crucial. Ensure that your data preprocessing steps come before your model.

2. **Caching**: Use caching for computationally expensive steps, especially when you're iterating on your pipeline design.

3. **Nested Pipelines**: Use nested pipelines to organize complex workflows into logical sub-components.

4. **Metrics**: Include multiple relevant metrics to get a comprehensive view of your pipeline's performance.

5. **Cross-Validation**: Consider using cross-validation techniques in conjunction with F3Pipeline for more robust model evaluation.

6. **Logging**: Enable logging to get insights into the pipeline's operation and to help with debugging.

7. **Parameter Tuning**: Use F3Pipeline in conjunction with hyperparameter tuning techniques to optimize your entire workflow.

## Conclusion

F3Pipeline provides a powerful and flexible way to build complex data processing and machine learning workflows in LabChain. By combining multiple plugins, nested pipelines, and caching capabilities, you can create efficient and maintainable pipelines for a wide range of tasks.
