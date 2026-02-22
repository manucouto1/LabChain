---
icon: material/download
---

# Installation Guide for LabChain

This guide will walk you through the process of installing LabChain using pip.

## Prerequisites

Before installing LabChain, ensure you have the following prerequisites:

1. Python 3.11 or higher
2. pip (Python package installer)

## Optional dependency groups

LabChain is split into optional extras so you only install what your use case requires:

| Extra | Included packages | Use case |
|---|---|---|
| *(none)* | `typeguard`, `multimethod`, `rich`, `cloudpickle`, `tqdm`, `dill` | Core pipeline engine |
| `data` | `pandas`, `scipy`, `scikit-learn` | Classical ML |
| `dl` | `torch`, `transformers`, `sentence-transformers` | Deep learning |
| `nlp` | `nltk`, `gensim` | NLP |
| `tracking` | `wandb`, `optuna` | Experiment tracking & optimization |
| `spark` | `pyspark` | Distributed processing |
| `aws` | `boto3`, `fastapi` | Cloud storage & serving |
| `dev` | `pytest`, `pytest-mock`, `pytest-cov`, `ruff`, `moto`, `ipykernel` | Development |

## Installation

### Step 1: Set Up a Virtual Environment (Recommended)

```bash
python -m venv venv
```

Activate the virtual environment:

- On Windows:
  ```bash
  venv\Scripts\activate
  ```
- On macOS and Linux:
  ```bash
  source venv/bin/activate
  ```

### Step 2: Install LabChain

**Core only** (pipeline engine, caching, serialization):

```bash
pip install framework3
```

**With specific extras:**

```bash
# Classical ML
pip install "framework3[data]"

# Deep learning
pip install "framework3[dl]"

# NLP
pip install "framework3[nlp]"

# Experiment tracking & optimization
pip install "framework3[tracking]"

# Distributed processing
pip install "framework3[spark]"

# Cloud storage & serving
pip install "framework3[aws]"
```

**Combining extras:**

```bash
# Mix and match what you need
pip install "framework3[data,dl,tracking]"

# Install everything
pip install "framework3[data,dl,nlp,tracking,spark,aws]"
```

## Verify Installation

```python
from labchain import __version__

print(f"LabChain version: {__version__}")
```

## Updating LabChain

```bash
pip install --upgrade framework3
```

To update while keeping your chosen extras:

```bash
pip install --upgrade "framework3[data,dl]"
```

## Troubleshooting

If you encounter any issues during installation:

1. Ensure your Python version is 3.11 or higher.
2. Make sure pip is up to date: `pip install --upgrade pip`
3. If you're using a virtual environment, ensure it's activated when installing and using LabChain.
4. If a class or plugin is not found, check that you installed the extra that provides it (see the table above).

For more detailed error messages, you can use the verbose mode when installing:

```bash
pip install -v "framework3[data]"
```

If problems persist, please check the project's [issue tracker on GitHub](https://github.com/manucouto1/LabChain/issues) or reach out to the maintainers for support.

## Next Steps

Now that you have LabChain installed, you can start using it in your projects. Check out the [Quick Start Guide](../quick_start/index.md) for an introduction to using LabChain, or explore the [API Documentation](../api/index.md) for more detailed information on available modules and functions.
