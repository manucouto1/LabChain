# LabChain [![test_on_push](https://github.com/manucouto1/LabChain/actions/workflows/test_on_push_pull.yml/badge.svg)](https://github.com/manucouto1/LabChain/actions/workflows/test_on_push_pull.yml)
LabChain is an innovative platform designed to simplify and accelerate the development of machine learning models. It provides data scientists and machine learning engineers with a flexible and powerful tool to create, experiment with, and deploy models efficiently and in a structured manner. [https://manucouto1.github.io/LabChain](https://manucouto1.github.io/LabChain)

## Key Features

- Modular and flexible architecture
- Customizable pipelines for ML workflows
- Extensible plugin system for filters, metrics, and storage
- Support for distributed processing with MapReduce
- Integrated model evaluation and optimization tools

## Diagram
```mermaid
      classDiagram
      class BasePlugin {
            <<abstract>>
            +item_dump() : Dict
            +build_from_dump(dump_dict: Dict, factory: BaseFactory) : BasePlugin
      }

      class BaseFilter {
            <<abstract>>
            +fit(x: XYData, y: XYData|None) : float|None
            +predict(x: XYData) : XYData
            +evaluate(x_data: XYData, y_true: XYData|None, y_pred: XYData) : Dict[str, Any]
      }

      class BasePipeline {
            <<abstract>>
            -filters: List[BaseFilter]
            +evaluate(x_data: XYData, y_true: XYData|None, y_pred: XYData) : Dict[str, Any]
            +start(x: XYData, y: XYData|None, X_: XYData|None) : XYData|None
            +init()
            +get_types() : List[Type[BaseFilter]]
            +optimizer(optimizer: BaseOptimizer) : BaseOptimizer|None
            +splitter(splitter: BaseSplitter) : BaseSplitter|None
      }

      class BaseSplitter {
            <<abstract>>
            -pipeline: BasePipeline
            +split(pipeline: BasePipeline)
            +evaluate(x_data: XYData, y_true: XYData|None, y_pred: XYData) : Dict[str, Any]
            +start(x: XYData, y: XYData|None, X_: XYData|None) : XYData|None
            +unwrap() : BasePipeline
      }

      class BaseOptimizer {
            <<abstract>>
            -pipeline: BasePipeline
            +start(x: XYData, y: XYData|None, X_: XYData|None) : XYData|None
            +optimize(pipeline: BasePipeline)
      }

      class BaseMetric {
            <<abstract>>
            +evaluate(x_data: XYData, y_true: XYData|None, y_pred: XYData) : float
      }

      class BaseStorer {
            <<abstract>>
            +get_root_path() : str
            +upload_file(file_path: str, destination_path: str)
            +list_stored_files() : List[str]
            +get_file_by_hashcode(hashcode: str) : str
            +check_if_exists(file_path: str) : bool
            +download_file(file_path: str, destination_path: str)
            +delete_file(file_path: str)
      }

      class ParallelPipeline {
            <<abstract>>
      }

      class SequentialPipeline {
            <<abstract>>
      }

      class MonoPipeline {
      }

      class LocalThreadPipeline {
      }

      class HPCPipeline {
      }

      class F3Pipeline {
      }

      class KFoldSplitter {
      }

      class OptunaOptimizer {
      }

      class SklearnOptimizer {
      }

      class WandbOptimizer {
      }

      class LocalStorer {
      }

      class S3Storer {
      }

      class Container {
            +bind()
            +get()
      }

      BasePlugin <|-- BaseFilter
      BasePlugin <|-- BaseMetric
      BaseFilter <|-- BasePipeline
      BaseFilter <|-- BaseSplitter
      BaseFilter <|-- BaseOptimizer
      BasePlugin <|-- BaseStorer
      BasePipeline <|-- ParallelPipeline
      BasePipeline <|-- SequentialPipeline
      ParallelPipeline <|-- MonoPipeline
      ParallelPipeline <|-- LocalThreadPipeline
      ParallelPipeline <|-- HPCPipeline
      SequentialPipeline <|-- F3Pipeline
      BaseSplitter <|-- KFoldSplitter
      BaseOptimizer <|-- OptunaOptimizer
      BaseOptimizer <|-- SklearnOptimizer
      BaseOptimizer <|-- WandbOptimizer
      BaseStorer <|-- LocalStorer
      BaseStorer <|-- S3Storer

      BasePipeline "1" *-- "1..*" BaseFilter : contains
      BaseSplitter "1" *-- "1" BaseFilter : contains
      BaseOptimizer "1" *-- "1" BaseFilter : contains
      BasePipeline "1" *-- "0..*" BaseMetric : uses

      Container --> BasePlugin : contains
```

## Prerequisites

Before installing LabChain, ensure you have the following prerequisites:

1. Python 3.11 or higher
2. pip (Python package installer)

## Installation Options

You have two options to install LabChain:

### Option 1: Install from PyPI

The easiest way to install LabChain is directly from PyPI using pip:

```bash
pip install framework3
```

This will install the latest stable version of LabChain and its dependencies.

### Option 2: Install from Source

2. Clone the repository:
   ```
   git clone https://github.com/manucouto1/LabChain.git
   ```

3. Navigate to the project directory:
   ```
   cd LabChain
   ```

4. Install the dependencies using pip:
   ```
   pip install -r requirements.txt
   ```

## Basic Usage

Here's a basic example of how to use LabChain:

```python
from labchain.plugins.pipelines import F3Pipeline
from labchain.plugins.filters import KnnFilter
from labchain.plugins.metrics import F1, Precision, Recall

# Create a pipeline
pipeline = F3Pipeline(
    plugins=[KnnFilter()],
    metrics=[F1(), Precision(), Recall()]
)

# Fit the model
pipeline.fit(X_train, y_train)

# Make predictions
predictions = pipeline.predict(X_test)

# Evaluate the model
evaluation = pipeline.evaluate(X_test, y_test, y_pred=predictions)
print(evaluation)
```

## Documentation

For more detailed information on how to use LabChain, check out our complete documentation at:

[https://manucouto1.github.io/LabChain](https://manucouto1.github.io/LabChain)

## Contributing

Contributions are welcome. Please read our contribution guidelines before submitting pull requests.

## License

This project is licensed under the AGPL-3.0 license. See the `LICENSE` file for more details.

## Contact

If you have any questions or suggestions, don't hesitate to open an issue in this repository or contact the development team.

---

Thank you for your interest in LabChain! We hope this tool will be useful in your machine learning projects.
