<div align="center">

# 🔬 LabChain

### *The Modern ML Experimentation Framework*

[![test_on_push](https://github.com/manucouto1/LabChain/actions/workflows/test_on_push_pull.yml/badge.svg)](https://github.com/manucouto1/LabChain/actions/workflows/test_on_push_pull.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL%203.0-blue.svg)](https://opensource.org/licenses/AGPL-3.0)
[![PyPI version](https://badge.fury.io/py/framework3.svg)](https://badge.fury.io/py/framework3)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://manucouto1.github.io/LabChain)

*Build, experiment, and deploy ML pipelines with confidence*

[Documentation](https://manucouto1.github.io/LabChain) • [Quick Start](#-quick-start) • [Examples](#-examples) • [Contributing](#-contributing)

</div>

---

## 🎯 What is LabChain?

LabChain is a **production-ready ML experimentation framework** that combines the flexibility of research with the rigor of production deployment. Stop fighting with boilerplate code and focus on what matters: your models.

### ✨ Why LabChain?

<table>
<tr>
<td>

**🧩 Modular by Design**
- Compose pipelines from reusable filters
- Plug-and-play architecture
- No vendor lock-in

</td>
<td>

**🚀 Production Ready**
- Automatic caching and versioning
- Distributed processing support
- Cloud-native storage backends

</td>
</tr>
<tr>
<td>

**🔄 Reproducible**
- Version-controlled experiments
- Deterministic pipelines
- Full audit trails

</td>
<td>

**⚡ Experimental Features**
- Remote code injection
- Zero-deployment pipelines
- Automatic dependency management

</td>
</tr>
</table>

---

## 🚀 Quick Start

### Installation
```bash
pip install framework3
```

### Your First Pipeline (2 minutes)
```python
from labchain import Container
from labchain.pipeline import F3Pipeline
from labchain.plugins.filters import StandardScalerPlugin, KnnFilter
from labchain.plugins.metrics import F1, Precision, Recall
from labchain.base import XYData
from sklearn.datasets import load_iris

# Load data
iris = load_iris()
X = XYData.mock(iris.data)
y = XYData.mock(iris.target)

# Build pipeline
pipeline = F3Pipeline(
    filters=[
        StandardScalerPlugin(),
        KnnFilter(n_neighbors=5)
    ],
    metrics=[F1(), Precision(), Recall()]
)

# Train and evaluate
pipeline.fit(X, y)
predictions = pipeline.predict(X)
results = pipeline.evaluate(X, y, predictions)

print(results)
# {'F1': 0.96, 'Precision': 0.96, 'Recall': 0.96}
```

**That's it!** 🎉 You just built, trained, and evaluated an ML pipeline.

---

## 💡 Key Features

### 🏗️ Modular Architecture
```python
# Mix and match components like LEGO blocks
from labchain.plugins.filters import (
    PCAPlugin,
    StandardScalerPlugin,
    ClassifierSVMPlugin
)

pipeline = F3Pipeline(
    filters=[
        StandardScalerPlugin(),
        PCAPlugin(n_components=2),
        ClassifierSVMPlugin(kernel='rbf')
    ]
)
```

### 🔄 Smart Caching
```python
from labchain.plugins.filters import Cached

# Cache expensive operations automatically
pipeline = F3Pipeline(
    filters=[
        Cached(
            filter=ExpensivePreprocessor(),
            cache_data=True,
            cache_filter=True
        ),
        MyModel()
    ]
)
```

### 📊 Hyperparameter Optimization
```python
from labchain.plugins.optimizers import WandbOptimizer

# Optimize with Weights & Biases
optimizer = WandbOptimizer(
    project="my-experiment",
    scorer=F1(),
    method="bayes",
    n_trials=50
)

# Define search space
pipeline = F3Pipeline(
    filters=[
        KnnFilter().grid({
            'n_neighbors': [3, 5, 7, 9]
        })
    ]
)

optimizer.optimize(pipeline)
optimizer.fit(X_train, y_train)
```

### ⚡ Remote Injection (Experimental)

Deploy pipelines **without deploying code**:
```python
# On your laptop
@Container.bind(persist=True)
class MyCustomFilter(BaseFilter):
    def predict(self, x):
        return x * 2

Container.storage = S3Storage(bucket="my-models")
Container.ppif.push_all()

# On production server (no source code needed!)
from labchain.base import BasePlugin

pipeline = BasePlugin.build_from_dump(config, Container.ppif)
predictions = pipeline.predict(data)  # Just works! ✨
```

### 🌐 Distributed Processing (Experimental)
```python
from labchain.pipeline import HPCPipeline

# Automatic Spark distribution
pipeline = HPCPipeline(
    app_name="distributed-training",
    filters=[Filter1(), Filter2(), Filter3()]
)

pipeline.fit(large_dataset)
```

---

## 📚 Examples

<details>
<summary><b>Classification with Cross-Validation</b></summary>
```python
from labchain.pipeline import F3Pipeline
from labchain.plugins.splitters import KFoldSplitter
from labchain.plugins.filters import StandardScalerPlugin, ClassifierSVMPlugin
from labchain.plugins.metrics import F1, Precision, Recall

pipeline = F3Pipeline(
    filters=[
        StandardScalerPlugin(),
        ClassifierSVMPlugin(kernel='rbf', C=1.0)
    ],
    metrics=[F1(), Precision(), Recall()]
).splitter(
    KFoldSplitter(n_splits=5, shuffle=True, random_state=42)
)

pipeline.fit(X_train, y_train)
results = pipeline.evaluate(X_test, y_test, pipeline.predict(X_test))
```

</details>

<details>
<summary><b>Parallel Processing</b></summary>
```python
from labchain.pipeline import MonoPipeline
from labchain.plugins.filters import Filter1, Filter2, Filter3

# Process filters in parallel
pipeline = MonoPipeline(
    filters=[
        Filter1(),  # Runs in parallel
        Filter2(),  # Runs in parallel
        Filter3()   # Runs in parallel
    ]
)

# Results are concatenated automatically
predictions = pipeline.predict(X)
```

</details>

<details>
<summary><b>Custom Components</b></summary>
```python
from labchain import Container
from labchain.base import BaseFilter, XYData

@Container.bind()
class MyCustomFilter(BaseFilter):
    def __init__(self, threshold: float = 0.5):
        super().__init__(threshold=threshold)

    def fit(self, x: XYData, y: XYData = None):
        # Your training logic
        pass

    def predict(self, x: XYData) -> XYData:
        # Your prediction logic
        return XYData.mock(x.value > self.threshold)

# Use it like any other filter
pipeline = F3Pipeline(filters=[MyCustomFilter(threshold=0.7)])
```

</details>

<details>
<summary><b>Version Control & Rollback</b></summary>
```python
# Version 1
@Container.bind(persist=True)
class MyModel(BaseFilter):
    def predict(self, x):
        return x * 1

Container.ppif.push_all()
hash_v1 = Container.pcm.get_class_hash(MyModel)

# Version 2
@Container.bind(persist=True)
class MyModel(BaseFilter):
    def predict(self, x):
        return x * 2

Container.ppif.push_all()
hash_v2 = Container.pcm.get_class_hash(MyModel)

# Rollback to V1
ModelV1 = Container.ppif.get_version("MyModel", hash_v1)
```

</details>

---

## 🏗️ Architecture
```mermaid
graph LR
    A[Data Input] --> B[XYData]
    B --> C[Pipeline]
    C --> D[Filter 1]
    C --> E[Filter 2]
    C --> F[Filter N]
    D & E & F --> G[Predictions]
    G --> H[Metrics]
    H --> I[Evaluation Results]

    J[Storage] -.-> C
    K[Cache] -.-> D
    K -.-> E
    K -.-> F

    style C fill:#2374ab
    style G fill:#00a896
    style I fill:#ff6f59
```

<details>
<summary><b>View Full Class Diagram</b></summary>
```mermaid
classDiagram
    class BasePlugin {
        <<abstract>>
        +item_dump() Dict
        +build_from_dump() BasePlugin
    }

    class BaseFilter {
        <<abstract>>
        +fit(x, y) float
        +predict(x) XYData
        +evaluate(x, y_true, y_pred) Dict
    }

    class BasePipeline {
        <<abstract>>
        -filters List[BaseFilter]
        +optimizer() BaseOptimizer
        +splitter() BaseSplitter
    }

    class F3Pipeline {
        +Sequential execution
    }

    class MonoPipeline {
        +Parallel execution
    }

    class HPCPipeline {
        +Spark distribution
    }

    BasePlugin <|-- BaseFilter
    BaseFilter <|-- BasePipeline
    BasePipeline <|-- F3Pipeline
    BasePipeline <|-- MonoPipeline
    BasePipeline <|-- HPCPipeline
```

</details>

---

## 📖 Documentation

| Resource | Description |
|----------|-------------|
| [📘 Quick Start Guide](https://manucouto1.github.io/LabChain/quick_start/) | Get up and running in 5 minutes |
| [🎓 Tutorials](https://manucouto1.github.io/LabChain/examples/) | Step-by-step guides and examples |
| [📚 API Reference](https://manucouto1.github.io/LabChain/api/) | Complete API documentation |
| [⚡ Remote Injection](https://manucouto1.github.io/LabChain/api/remote-injection/) | Deploy without code (experimental) |
| [🏗️ Architecture](https://manucouto1.github.io/LabChain/architecture/) | Deep dive into design principles |
| [💡 Best Practices](https://manucouto1.github.io/LabChain/best_practices/) | Production-ready patterns |

---

## 🛠️ Supported Components

<table>
<tr>
<td width="50%">

### Filters
- ✅ Classification (SVM, KNN, Random Forest, etc.)
- ✅ Clustering (KMeans, DBSCAN, etc.)
- ✅ Transformation (PCA, StandardScaler, etc.)
- ✅ Text Processing (TF-IDF, Embeddings, etc.)
- ✅ Custom filters (extend `BaseFilter`)

### Pipelines
- ✅ **F3Pipeline**: Sequential execution
- ✅ **MonoPipeline**: Parallel execution
- ✅ **HPCPipeline**: Spark-based distribution

</td>
<td width="50%">

### Optimizers
- ✅ **Optuna**: Bayesian optimization
- ✅ **Weights & Biases**: Experiment tracking
- ✅ **Grid Search**: Exhaustive search
- ✅ **Sklearn**: Scikit-learn integration

### Storage
- ✅ **Local Storage**: Filesystem caching
- ✅ **S3 Storage**: Cloud-native storage
- ✅ **Custom backends**: Extend `BaseStorage`

</td>
</tr>
</table>

---

## 🚦 Roadmap

- [x] Core pipeline functionality
- [x] Automatic caching system
- [x] Hyperparameter optimization
- [x] Distributed processing (Spark)
- [x] Remote injection (experimental)
- [ ] Multi-cloud storage backends (GCS, Azure)
- [ ] Real-time inference API
- [ ] AutoML capabilities
- [ ] Model registry integration
- [ ] Kubernetes deployment templates

---

## 🤝 Contributing

We ❤️ contributions! Here's how you can help:

### Ways to Contribute

- 🐛 **Report bugs** by opening an issue
- 💡 **Suggest features** in discussions
- 📝 **Improve documentation**
- 🔧 **Submit pull requests**
- ⭐ **Star the repo** to show support

### Development Setup
```bash
# Clone the repository
git clone https://github.com/manucouto1/LabChain.git
cd LabChain

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/

# Build documentation
cd docs && mkdocs serve
```

### Guidelines

- Follow PEP 8 style guide
- Add tests for new features
- Update documentation
- Keep commits atomic and well-described

---

## 📊 Community & Support

<div align="center">

[![GitHub issues](https://img.shields.io/github/issues/manucouto1/LabChain)](https://github.com/manucouto1/LabChain/issues)
[![GitHub pull requests](https://img.shields.io/github/issues-pr/manucouto1/LabChain)](https://github.com/manucouto1/LabChain/pulls)
[![GitHub stars](https://img.shields.io/github/stars/manucouto1/LabChain?style=social)](https://github.com/manucouto1/LabChain/stargazers)

</div>

- 💬 [GitHub Discussions](https://github.com/manucouto1/LabChain/discussions) - Ask questions and share ideas
- 🐛 [Issue Tracker](https://github.com/manucouto1/LabChain/issues) - Report bugs and request features
- 📧 [Email](mailto:support@labchain.ai) - Contact the maintainers
- 📖 [Documentation](https://manucouto1.github.io/LabChain) - Comprehensive guides

---

## 📜 License

This project is licensed under the **AGPL-3.0 License** - see the [LICENSE](LICENSE) file for details.

### What this means:
- ✅ Use LabChain for free in your projects
- ✅ Modify and distribute the code
- ⚠️ If you modify and distribute LabChain, you must release your changes under AGPL-3.0
- ⚠️ If you use LabChain in a network service, you must make the source available

---

## 🙏 Acknowledgments

Built with ❤️ by the LabChain team and [amazing contributors](https://github.com/manucouto1/LabChain/graphs/contributors).

Special thanks to:
- The scikit-learn team for inspiration
- The MLOps community for feedback
- All our users and contributors

---

<div align="center">

**[⬆ back to top](#-labchain)**

Made with ☕ and Python

</div>
