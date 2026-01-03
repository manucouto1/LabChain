# MonoPipeline

The `MonoPipeline` class is part of the `labchain.plugins.pipelines.parallel` module and is designed to run multiple pipelines in parallel, combining their outputs to create new features. This pipeline is ideal for scenarios where different models or transformations need to be applied simultaneously to the same dataset.

## Module Contents
::: labchain.plugins.pipelines.parallel.parallel_mono_pipeline
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3

## Class Hierarchy
- MonoPipeline

## MonoPipeline
`MonoPipeline` extends `ParallelPipeline` and provides functionality for executing multiple pipelines in parallel, combining their outputs to enhance feature sets.

### Key Methods:
- `__init__(filters: Sequence[BaseFilter])`: Initializes the pipeline with a sequence of filters to be applied in parallel.
- `start(x: XYData, y: XYData|None, X_: XYData|None) -> XYData|None`: Starts the pipeline execution, fitting the data and making predictions.
- `fit(x: XYData, y: XYData|None = None)`: Fits all pipelines in parallel using the provided input data.
- `predict(x: XYData) -> XYData`: Runs predictions on all pipelines in parallel and combines their outputs.
- `evaluate(x_data: XYData, y_true: XYData|None, y_pred: XYData) -> Dict[str, Any]`: Evaluates the pipeline using the provided metrics.
- `combine_features(pipeline_outputs: list[XYData]) -> XYData`: Combines features from all pipeline outputs.

## Usage Examples

### Creating and Using MonoPipeline
```python
from labchain.plugins.pipelines.parallel.mono_pipeline import MonoPipeline
from labchain.base import XYData

# Define filters (assuming filters are defined elsewhere)
filters = [Filter1(), Filter2()]

# Initialize the MonoPipeline
mono_pipeline = MonoPipeline(filters=filters)

# Example data
x_data = XYData(_hash='x_data', _path='/tmp', _value=[[1, 2], [3, 4]])
y_data = XYData(_hash='y_data', _path='/tmp', _value=[0, 1])

# Start the pipeline
results = mono_pipeline.start(x_data, y_data, None)

# Evaluate the pipeline
evaluation_results = mono_pipeline.evaluate(x_data, y_data, results)
print(evaluation_results)
```

## Best Practices
1. Ensure that the filters used in the pipeline are compatible with the input data.
2. Define a clear combiner function if the default concatenation does not meet your needs.
3. Monitor the performance and resource utilization to optimize parallel execution.

## Conclusion
`MonoPipeline` provides a flexible and efficient way to execute multiple pipelines in parallel, enhancing feature sets through combined outputs. By following the best practices and examples provided, you can effectively integrate this pipeline into your machine learning workflows.
