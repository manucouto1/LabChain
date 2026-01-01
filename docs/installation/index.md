---
icon: material/download
---

# Installation Guide for LabChain

This guide will walk you through the process of installing LabChain using pip.

## Prerequisites

Before installing LabChain, ensure you have the following prerequisites:

1. Python 3.11 or higher
2. pip (Python package installer)

## Installation

Installing LabChain is straightforward using pip. Follow these steps:

### Step 1: Set Up a Virtual Environment (Recommended)

It's recommended to use a virtual environment to avoid conflicts with other Python projects:

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

Install LabChain directly from PyPI using pip:

```bash
pip install labchain
```

This command will install the latest stable version of LabChain and all its dependencies.

## Verify Installation

To verify that LabChain is installed correctly, you can run a simple Python script:

```python
from labchain import __version__

print(f"LabChain version: {__version__}")
```

If this runs without errors and prints the version number, the installation was successful.

## Updating LabChain

To update LabChain to the latest version, simply run:

```bash
pip install --upgrade labchain
```

## Troubleshooting

If you encounter any issues during installation:

1. Ensure your Python version is 3.11 or higher.
2. Make sure pip is up to date: `pip install --upgrade pip`
3. If you're using a virtual environment, ensure it's activated when installing and using LabChain.

For more detailed error messages, you can use the verbose mode when installing:

```bash
pip install -v labchain
```

If problems persist, please check the project's [issue tracker on GitHub](https://github.com/manucouto1/LabChain/issues) or reach out to the maintainers for support.

## Next Steps

Now that you have LabChain installed, you can start using it in your projects. Check out the [Quick Start Guide](../quick_start/index.md) for an introduction to using LabChain, or explore the [API Documentation](../api/index.md) for more detailed information on available modules and functions.
