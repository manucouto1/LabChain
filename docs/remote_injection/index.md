---
title: Remote Injection (Experimental)
icon: material/cloud-sync
status: experimental
---

# 🌐 Remote Injection

!!! warning "Experimental Feature"
    Remote Injection is an **experimental feature** currently in beta. While it passes all test cases, it has not been extensively tested in production environments. Use at your own risk and consider the following:

    - **Test thoroughly** in your specific use case before deploying to production
    - **Version pin** your LabChain installation to avoid unexpected behavior
    - **Maintain backups** of critical models and pipelines
    - **Report issues** on [GitHub](https://github.com/manucouto1/LabChain/issues) to help improve stability

    This feature is under active development and the API may change in future releases.

Remote Injection is a powerful feature in LabChain that allows you to **execute pipelines on remote servers without deploying source code**. By using persistent class storage with deterministic version control, you can train models locally and deploy them remotely with full reproducibility.

## 🎯 What is Remote Injection?

Remote Injection enables you to:

- **Develop locally** with your custom filters and pipelines
- **Push class definitions** to cloud storage (S3, GCS, etc.)
- **Execute remotely** without copying Python files
- **Guarantee version consistency** through deterministic hashing
- **Roll back** to previous versions when needed

## 🔑 Key Concepts

### Persistent Classes

When you decorate a class with `@Container.bind(persist=True)`, LabChain:

1. **Computes a deterministic hash** from the class's bytecode, methods, and signature
2. **Serializes the class** using CloudPickle
3. **Stores it** in your configured storage backend with the hash as the key
4. **Tracks versions** for reproducibility

### Deterministic Version Hashing

Each class gets a unique SHA-256 hash based on its actual implementation:
```python
# Version 1
@Container.bind(persist=True)
class MyFilter(BaseFilter):
    def predict(self, x):
        return x * 2

# Hash computed from:
# - Module name and qualified name
# - Base classes
# - Method bytecode (co_code)
# - Method constants and names
# - Method signatures
# Result: abc123...

# Version 2 (different implementation = different hash)
@Container.bind(persist=True)
class MyFilter(BaseFilter):
    def predict(self, x):
        return x * 3  # Changed implementation

# Hash: def456... (different from V1)
```

!!! info "Hash Computation Details"
    The hash is computed from:

    - **Module and qualified name**: Ensures classes in different modules get different hashes
    - **Base classes**: Changes in inheritance hierarchy produce different hashes
    - **Method bytecode**: The actual compiled Python bytecode of each method
    - **Constants and names**: Literal values and variable names used in methods
    - **Method signatures**: Parameter names and types

    This approach is **deterministic** - the same source code will always produce the same hash, enabling reliable version tracking across different machines.

Both versions are stored and can be retrieved independently.

## ⚠️ Important Limitations

### 1. Hash Determinism Considerations

While the hashing system is designed to be deterministic, be aware of these edge cases:
```python
# ✅ GOOD: Deterministic across environments
@Container.bind(persist=True)
class MyFilter(BaseFilter):
    def __init__(self, threshold: float = 0.5):
        super().__init__(threshold=threshold)

    def predict(self, x: XYData) -> XYData:
        return XYData.mock(x.value > self.threshold)

# ⚠️ CAUTION: May have different hashes in different Python versions
@Container.bind(persist=True)
class VersionSensitiveFilter(BaseFilter):
    def predict(self, x: XYData) -> XYData:
        # Bytecode may differ between Python 3.10 and 3.11
        result = [item for item in x.value if item > 0]
        return XYData.mock(result)
```

**Best Practice**: Use the same Python version across all environments where you push/pull classes.

### 2. Import Context

Classes must have imports at module level to work correctly after deserialization:
```python
# ❌ BAD: Imports inside class/methods may fail after deserialization
@Container.bind(persist=True)
class BadFilter(BaseFilter):
    def predict(self, x: XYData) -> XYData:
        import numpy as np  # May cause issues
        return XYData.mock(np.array(x.value))

# ✅ GOOD: Module-level imports
import numpy as np

@Container.bind(persist=True)
class GoodFilter(BaseFilter):
    def predict(self, x: XYData) -> XYData:
        return XYData.mock(np.array(x.value))
```

### 3. Not Suitable For All Use Cases

Remote Injection is **not recommended** for:

- **Production-critical systems** (until extensively tested in your environment)
- **Highly regulated industries** (where code deployment must be fully auditable)
- **Classes with external dependencies** that can't be serialized (database connections, file handles, etc.)

Remote Injection **is suitable** for:

- **Research and experimentation** workflows
- **Rapid prototyping** across multiple machines
- **ML model versioning** and experimentation tracking
- **Development environments** with controlled infrastructure

## 📚 Quick Start

### 1. Configure Storage

First, configure a shared storage backend:

=== "S3 Storage"
```python
    from labchain import Container, S3Storage

    # Configure once at startup
    Container.storage = S3Storage(
        bucket="my-ml-models",
        region_name="us-east-1"
    )
```

=== "Local Storage (Testing)"
```python
    from labchain import Container, LocalStorage

    # For local testing/development
    Container.storage = LocalStorage("./shared_cache")
```

### 2. Create Persistent Classes

Define your custom classes with `persist=True`:
```python
from labchain import Container
from labchain.base import BaseFilter, XYData
import numpy as np  # Module-level import!

@Container.bind(persist=True)
class CustomNormalizer(BaseFilter):
    def __init__(self, scale: float = 1.0):
        super().__init__(scale=scale)
        self._mean = None
        self._std = None

    def fit(self, x: XYData, y=None):
        self._mean = x.value.mean(axis=0)
        self._std = x.value.std(axis=0)

    def predict(self, x: XYData) -> XYData:
        normalized = (x.value - self._mean) / (self._std + 1e-8)
        return XYData.mock(normalized * self.scale)

@Container.bind(persist=True)
class DomainSpecificFilter(BaseFilter):
    def __init__(self, threshold: float = 0.5):
        super().__init__(threshold=threshold)

    def predict(self, x: XYData) -> XYData:
        # Your custom business logic
        filtered = x.value[x.value > self.threshold]
        return XYData.mock(filtered)
```

### 3. Build and Serialize Pipeline

Create a pipeline and serialize its configuration:
```python
from labchain import F3Pipeline
from labchain.plugins.filters import StandardScalerPlugin
import numpy as np

# Create pipeline with your custom filters
pipeline = F3Pipeline(
    filters=[
        CustomNormalizer(scale=2.0),
        DomainSpecificFilter(threshold=0.3),
        StandardScalerPlugin()
    ]
)

# Train locally
X_train = XYData.mock(np.random.randn(100, 10))
pipeline.fit(X_train, None)

# Serialize configuration (includes version hashes)
config = pipeline.item_dump()

# Push classes to storage
Container.ppif.push_all()

# Save config to file or database
import json
with open('pipeline_config.json', 'w') as f:
    json.dump(config, f)
```

### 4. Deploy and Execute Remotely

On the remote server (without your custom class source files):
```python
from labchain import Container, S3Storage
from labchain.base import BasePlugin
import json

# Configure same storage
Container.storage = S3Storage(
    bucket="my-ml-models",
    region="us-east-1"
)

# Load configuration
with open('pipeline_config.json', 'r') as f:
    config = json.load(f)

# Reconstruct pipeline (auto-loads classes from storage)
pipeline = BasePlugin.build_from_dump(config, Container.ppif)

# Use immediately - no source code needed!
X_new = XYData.mock(np.random.randn(20, 10))
predictions = pipeline.predict(X_new)
```


## 🎓 Complete Tutorial: ML Workflow

### Scenario

You're building a custom NLP pipeline for sentiment analysis:

- **Development**: Local laptop
- **Training**: Cloud GPU instance
- **Inference**: Production API server

### Step 1: Local Development
```python
# sentiment_filters.py
from labchain import Container
from labchain.base import BaseFilter, XYData
import numpy as np

@Container.bind(persist=True)
class TextPreprocessor(BaseFilter):
    """Custom text preprocessing for your domain."""

    def __init__(self, lowercase: bool = True, remove_punctuation: bool = True):
        super().__init__(
            lowercase=lowercase,
            remove_punctuation=remove_punctuation
        )

    def predict(self, x: XYData) -> XYData:
        texts = x.value

        if self.lowercase:
            texts = [t.lower() for t in texts]

        if self.remove_punctuation:
            import string
            texts = [t.translate(str.maketrans('', '', string.punctuation))
                    for t in texts]

        return XYData.mock(np.array(texts))

@Container.bind(persist=True)
class DomainEmbedding(BaseFilter):
    """Custom embeddings trained on your domain."""

    def __init__(self, vocab_size: int = 10000, embedding_dim: int = 128):
        super().__init__(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim
        )
        self._embeddings = None

    def fit(self, x: XYData, y=None):
        # Your custom embedding training logic
        self._embeddings = np.random.randn(self.vocab_size, self.embedding_dim)

    def predict(self, x: XYData) -> XYData:
        # Transform texts to embeddings
        embedded = np.array([self._lookup_embedding(text) for text in x.value])
        return XYData.mock(embedded)

    def _lookup_embedding(self, text):
        # Simplified - real implementation would tokenize properly
        return self._embeddings[hash(text) % self.vocab_size]

@Container.bind(persist=True)
class SentimentClassifier(BaseFilter):
    """Final classification layer."""

    def __init__(self, num_classes: int = 3):
        super().__init__(num_classes=num_classes)
        self._model = None

    def fit(self, x: XYData, y: XYData):
        from sklearn.linear_model import LogisticRegression
        self._model = LogisticRegression()
        self._model.fit(x.value, y.value)

    def predict(self, x: XYData) -> XYData:
        predictions = self._model.predict(x.value)
        return XYData.mock(predictions)
```

### Step 2: Configure Storage and Build Pipeline
```python
# train_pipeline.py
from labchain import Container, S3Storage, F3Pipeline
from sentiment_filters import TextPreprocessor, DomainEmbedding, SentimentClassifier
import json

# Configure shared storage
Container.storage = S3Storage(
    bucket="my-sentiment-models",
    region_name="us-west-2"
)

# Build pipeline
pipeline = F3Pipeline(
    filters=[
        TextPreprocessor(lowercase=True, remove_punctuation=True),
        DomainEmbedding(vocab_size=10000, embedding_dim=128),
        SentimentClassifier(num_classes=3)
    ]
)

# Train locally (or on training server)
# ... load your training data ...
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
metrics = pipeline.evaluate(X_test, y_test, y_pred)
print(f"Accuracy: {metrics}")

# Serialize configuration
config = pipeline.item_dump()

# Push classes to S3
Container.ppif.push_all()

# Save config to S3 or your metadata store
with open('sentiment_pipeline_v1.json', 'w') as f:
    json.dump(config, f)

print("✅ Pipeline trained and pushed to S3!")
print(f"📝 Config saved to sentiment_pipeline_v1.json")
```

### Step 3: Deploy to Production API
```python
# api_server.py
from fastapi import FastAPI
from labchain import Container, S3Storage
from labchain.base import BasePlugin, XYData
import json
import numpy as np

app = FastAPI()

# Configure storage (same as training)
Container.storage = S3Storage(
    bucket="my-sentiment-models",
    region_name="us-west-2"
)

# Load pipeline configuration
with open('sentiment_pipeline_v1.json', 'r') as f:
    config = json.load(f)

# Reconstruct pipeline (auto-downloads classes from S3)
pipeline = BasePlugin.build_from_dump(config, Container.ppif)

print("🚀 Pipeline loaded successfully!")

@app.post("/predict")
async def predict_sentiment(texts: list[str]):
    """Predict sentiment for a batch of texts."""

    # Convert to XYData
    X = XYData.mock(np.array(texts))

    # Predict
    predictions = pipeline.predict(X)

    # Map to labels
    label_map = {0: "negative", 1: "neutral", 2: "positive"}
    results = [label_map[p] for p in predictions.value]

    return {"predictions": results}
```

### Step 4: Version Management
```python
# version_manager.py
from labchain import Container, S3Storage

Container.storage = S3Storage(bucket="my-sentiment-models", region="us-west-2")

# Check version status
status = Container.pcm.check_status(TextPreprocessor)
print(f"TextPreprocessor status: {status}")
# Output: "synced" | "out_of_sync" | "untracked"

# Get version hash
hash_v1 = Container.pcm.get_class_hash(TextPreprocessor)
print(f"Current version: {hash_v1[:8]}...")

# Retrieve specific version
TextPreprocessorV1 = Container.ppif.get_version("TextPreprocessor", hash_v1)

# Check what's in storage
meta = Container.pcm._get_remote_latest_meta("TextPreprocessor")
print(f"Latest in storage: {meta}")
```

## 🛡️ Best Practices for Experimental Feature

### 1. Always Use Version Control
```python
# ✅ Good: Track versions with metadata
config = pipeline.item_dump()
version_info = {
    'version_tag': 'v1.0.0',
    'timestamp': datetime.utcnow().isoformat(),
    'python_version': sys.version,
    'labchain_version': labchain.__version__,
    'config': config
}
with open(f'pipeline_v1.0.0.json', 'w') as f:
    json.dump(version_info, f)
Container.ppif.push_all()
```

### 2. Pin Python Version
```dockerfile
# Dockerfile - Pin exact Python version
FROM python:3.11.5-slim  # Exact version for hash consistency

RUN pip install labchain==x.y.z  # Pin framework version too
```

### 3. Test Reconstruction Before Deployment
```python
def verify_deployment(config):
    """Verify pipeline reconstruction works before deploying."""
    try:
        reconstructed = BasePlugin.build_from_dump(config, Container.ppif)

        # Test with sample data
        test_input = XYData.mock(np.random.randn(5, 10))
        test_output = reconstructed.predict(test_input)

        assert test_output is not None
        assert test_output.value.shape[0] > 0

        print("✅ Reconstruction verified")
        return True
    except Exception as e:
        print(f"❌ Reconstruction failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# ALWAYS verify before deploying
if verify_deployment(config):
    deploy_to_production(config)
else:
    print("Deployment aborted due to verification failure")
```

### 4. Use Staging Environment
```python
# Deploy to staging first
ENVIRONMENTS = {
    'staging': S3Storage(bucket='staging-models', region='us-east-1'),
    'production': S3Storage(bucket='prod-models', region='us-east-1')
}

# Test in staging
Container.storage = ENVIRONMENTS['staging']
Container.ppif.push_all()

# Verify in staging
if verify_in_staging():
    # Promote to production
    Container.storage = ENVIRONMENTS['production']
    Container.ppif.push_all()
```

### 5. Monitor and Log
```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_pipeline_safe(config_file: str):
    """Load pipeline with extensive logging."""
    try:
        logger.info(f"Loading config from {config_file}")
        with open(config_file) as f:
            config = json.load(f)

        logger.info(f"Config loaded. Classes: {config.get('clazz')}")
        logger.info(f"Version hash: {config.get('version_hash', 'N/A')[:8]}...")

        pipeline = BasePlugin.build_from_dump(config, Container.ppif)
        logger.info("✅ Pipeline reconstructed successfully")

        return pipeline
    except Exception as e:
        logger.error(f"❌ Failed to load pipeline: {e}", exc_info=True)
        raise
```

## 🐛 Troubleshooting

### Issue: `NameError` when reconstructing classes

**Cause**: Missing imports at module level.

**Solution**: Ensure all imports are at module scope:
```python
# ✅ Module-level imports
from labchain import XYData
import numpy as np

@Container.bind(persist=True)
class MyFilter(BaseFilter):
    def predict(self, x: XYData):
        return XYData.mock(np.array(x.value))
```

### Issue: Hash changes unexpectedly between environments

**Cause**: Different Python versions produce different bytecode.

**Solution**: Use identical Python versions:
```bash
# Check Python version
python --version

# On all machines, use exact same version
pyenv install 3.11.5
pyenv local 3.11.5
```

### Issue: Class not found in storage

**Cause**: Forgot to call `push_all()` or wrong storage configuration.

**Solution**:
```python
# Verify storage is configured
print(f"Storage: {Container.storage}")

# Check status before pushing
from labchain.container.persistent import PetClassManager
status = Container.pcm.check_status(MyFilter)
print(f"Status: {status}")  # Should be "untracked" before first push

# Push
Container.ppif.push_all()

# Verify
status = Container.pcm.check_status(MyFilter)
print(f"Status after push: {status}")  # Should be "synced"
```

### Issue: Deserialization fails with TypeGuard errors

**Cause**: TypeGuard wrapping interferes with CloudPickle deserialization.

**Solution**: This is handled automatically, but if you encounter issues:

```python
# The framework automatically unwraps TypeGuard during recovery
# If you see issues, report them on GitHub with:
# 1. Your class definition
# 2. Python version
# 3. LabChain version
# 4. Full traceback
```

## 📖 API Reference

### `@Container.bind(persist=True)`

Register a class for remote injection.

**Parameters:**

- `persist` (bool): Enable persistent storage. Default: `False`
- `auto_push` (bool): Automatically push to storage on registration. Default: `False`

**Example:**

```python
@Container.bind(persist=True, auto_push=True)
class MyFilter(BaseFilter):
    pass
```

### `Container.ppif.push_all()`

Push all registered persistent classes to storage.

**Example:**

```python
Container.ppif.push_all()
```

### `Container.pcm.check_status(class_obj)`

Check sync status of a class.

**Returns:** `"synced"` | `"out_of_sync"` | `"untracked"`

**Example:**

```python
status = Container.pcm.check_status(MyFilter)
```

### `Container.pcm.get_class_hash(class_obj)`

Get deterministic version hash for a class.

**Returns:** SHA-256 hex digest (64 characters)

**Details:** Hash is computed from class bytecode, method signatures, and implementation details.

**Example:**

```python
hash = Container.pcm.get_class_hash(MyFilter)
print(hash[:8])  # First 8 chars
```

### `Container.ppif.get_version(name, hash)`

Retrieve specific version of a class.

**Parameters:**

- `name` (str): Class name
- `hash` (str): Version hash

**Returns:** Class object

**Example:**

```python
MyFilterV1 = Container.ppif.get_version("MyFilter", "abc123...")
```

### `BasePlugin.build_from_dump(config, factory)`

Reconstruct instance from serialized configuration.

**Parameters:**

- `config` (dict): Serialized configuration (from `item_dump()`)
- `factory` (BaseFactory): Factory to use (typically `Container.ppif`)

**Returns:** Reconstructed instance

**Example:**

```python
pipeline = BasePlugin.build_from_dump(config, Container.ppif)
```

## 🚨 Known Limitations

1. **Python Version Sensitivity**: While hashing is deterministic within the same Python version, bytecode may differ across Python versions (e.g., 3.10 vs 3.11)

2. **Experimental Status**: Not extensively battle-tested in production environments

3. **Import Requirements**: All imports must be at module level

4. **CloudPickle Limitations**: Some objects cannot be serialized (open files, database connections, etc.)

5. **Storage Consistency**: Requires reliable storage backend with strong consistency guarantees

## 🎬 Next Steps

- Read [Storage Backends](../api/index.md#storage) documentation
- Learn about [Pipeline Management](../api/index.md#pipelines)
- Check [Advanced Caching](../start_caching/index.md) techniques

---

!!! question "Need Help?"
    - 🐛 Report issues on [GitHub](https://github.com/manucouto1/LabChain/issues)
    - ✉️ Email support at manuel.couto.pintos@usc.es

!!! tip "Contributing"
    This feature is experimental and your feedback is invaluable! If you use Remote Injection, please:

    - Share your use case and results
    - Report any issues or edge cases you discover
    - Suggest improvements to the hashing or serialization system
    - Help test across different Python versions and environments
