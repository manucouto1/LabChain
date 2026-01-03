---
title: PetClassManager
icon: material/database-cog
status: experimental
---

# PetClassManager

!!! warning "Experimental Feature"
    This component is part of the experimental Remote Injection system. Use with caution in production environments.

::: labchain.container.persistent.pet_class_manager.PetClassManager
    options:
        show_root_heading: true
        show_source: true
        heading_level: 2
        members:
            - __init__
            - get_class_hash
            - check_status
            - persist_class
            - push
            - pull
            - recover_class

---

## Overview

`PetClassManager` is the core component for managing persistent class storage with deterministic version tracking. It handles serialization, storage, and retrieval of plugin classes using CloudPickle and a hash-based versioning system.

### Key Responsibilities

- **Hash Computation**: Generates deterministic SHA-256 hashes from class bytecode
- **Serialization**: Converts classes to binary format using CloudPickle
- **Storage Management**: Handles upload/download of class binaries
- **Version Tracking**: Maintains `latest` pointers and version metadata
- **Class Recovery**: Deserializes and reconstructs classes from storage

---

## Storage Structure
```
storage/
└── plugins/
    ├── MyCustomFilter/
    │   ├── abc123...pkl      # Immutable class binary (version 1)
    │   ├── def456...pkl      # Immutable class binary (version 2)
    │   └── latest.json       # Pointer to current version
    ├── AnotherFilter/
    │   ├── 789xyz...pkl
    │   └── latest.json
    └── ...
```

Each class directory contains:

- **`{hash}.pkl`**: Immutable serialized class binary, named by its content hash
- **`latest.json`**: Development pointer with structure:
```json
  {
    "class_name": "MyCustomFilter",
    "hash": "abc123..."
  }
```

---

## Basic Usage

### Initialize Manager
```python
from labchain.storage import S3Storage
from labchain.container.persistent import PetClassManager

# Create storage backend
storage = S3Storage(bucket="my-ml-models", region="us-east-1")

# Initialize manager
manager = PetClassManager(storage)
```

### Check Class Status
```python
from labchain.base import BaseFilter

class MyFilter(BaseFilter):
    def predict(self, x):
        return x * 2

# Check if class is synced with storage
status = manager.check_status(MyFilter)
print(status)  # "untracked" | "synced" | "out_of_sync"
```

### Push Class to Storage
```python
# Compute hash and upload class
manager.push(MyFilter)

# Verify it's synced
status = manager.check_status(MyFilter)
print(status)  # "synced"
```

### Pull Class from Storage
```python
# Pull latest version
MyFilterClass = manager.pull("MyFilter")

# Pull specific version by hash
MyFilterV1 = manager.pull("MyFilter", code_hash="abc123...")

# Create instance
instance = MyFilterClass(param=42)
```

---

## Advanced Usage

### Version Tracking Workflow
```python
from labchain.base import BaseFilter
from labchain.container.persistent import PetClassManager
from labchain.storage import LocalStorage

storage = LocalStorage("./model_storage")
manager = PetClassManager(storage)

# Version 1
class MyModel(BaseFilter):
    def predict(self, x):
        return x * 1

hash_v1 = manager.get_class_hash(MyModel)
manager.push(MyModel)
print(f"V1 hash: {hash_v1[:8]}...")

# Modify and create Version 2
class MyModel(BaseFilter):  # Same name, different implementation
    def predict(self, x):
        return x * 2

hash_v2 = manager.get_class_hash(MyModel)
manager.push(MyModel)
print(f"V2 hash: {hash_v2[:8]}...")

# Both versions are now in storage
assert hash_v1 != hash_v2

# Pull specific versions
ModelV1 = manager.pull("MyModel", code_hash=hash_v1)
ModelV2 = manager.pull("MyModel", code_hash=hash_v2)

# Test both versions work correctly
import numpy as np
from labchain.base import XYData

test_data = XYData.mock(np.array([5]))
print(ModelV1().predict(test_data).value)  # [5]
print(ModelV2().predict(test_data).value)  # [10]
```

### Checking Remote Metadata
```python
# Get metadata for latest version
meta = manager._get_remote_latest_meta("MyFilter")

if meta:
    print(f"Latest version: {meta['hash'][:8]}...")
    print(f"Class name: {meta['class_name']}")
else:
    print("No remote version found")
```

### Recovery with Validation
```python
def safe_recover(manager, class_name, code_hash):
    """Safely recover a class with validation."""
    try:
        # Recover class
        recovered = manager.recover_class(class_name, code_hash)

        # Validate it's a proper class
        assert isinstance(recovered, type), "Not a class type"

        # Validate it has required methods
        assert hasattr(recovered, 'predict'), "Missing predict method"

        print(f"✅ Successfully recovered {class_name}")
        return recovered

    except Exception as e:
        print(f"❌ Recovery failed: {e}")
        raise

# Usage
MyFilter = safe_recover(manager, "MyFilter", "abc123...")
```

---

## Hash Computation Details

The `get_class_hash()` method computes a deterministic SHA-256 hash based on:

### Components Included in Hash

1. **Module and Qualified Name**
```python
   h.update(class_obj.__module__.encode())
   h.update(class_obj.__qualname__.encode())
```

2. **Base Classes** (order matters)
```python
   for base in class_obj.__bases__:
       h.update(base.__module__.encode())
       h.update(base.__qualname__.encode())
```

3. **Method Bytecode**
```python
   for name, method in sorted(class_obj.__dict__.items()):
       if isinstance(method, types.FunctionType):
           code = method.__code__
           h.update(code.co_code)          # Bytecode
           h.update(repr(code.co_consts))  # Constants
           h.update(repr(code.co_names))   # Names
           h.update(str(inspect.signature(method)))  # Signature
```

### Example: Hash Sensitivity
```python
import hashlib

class V1(BaseFilter):
    def predict(self, x):
        return x * 1

hash1 = manager.get_class_hash(V1)

# Change implementation
class V1(BaseFilter):
    def predict(self, x):
        return x * 2

hash2 = manager.get_class_hash(V1)

assert hash1 != hash2  # Different bytecode = different hash

# Change only whitespace
class V1(BaseFilter):
    def predict(self, x):
        return   x   *   2  # Extra spaces

hash3 = manager.get_class_hash(V1)

# Bytecode is identical despite whitespace
# (Python compiles to same bytecode)
assert hash2 == hash3
```

---

## Integration with Container

`PetClassManager` works seamlessly with the Container system:
```python
from labchain import Container
from labchain.storage import S3Storage

# Configure storage
Container.storage = S3Storage(bucket="my-models")

# Container.pcm is automatically initialized
# No need to create PetClassManager manually

# Use via Container
@Container.bind(persist=True)
class MyFilter(BaseFilter):
    def predict(self, x):
        return x

# Check status via Container
status = Container.pcm.check_status(MyFilter)

# Push via Container
Container.ppif.push_all()  # Uses Container.pcm internally
```

---

## Error Handling

### Common Exceptions
```python
from labchain.container.persistent import PetClassManager

manager = PetClassManager(storage)

# FileNotFoundError: Class not in storage
try:
    cls = manager.pull("NonExistentClass")
except ValueError as e:
    print(f"Class not found: {e}")

# TypeError: Invalid class object
try:
    manager.push("not_a_class")  # Wrong type
except (TypeError, AttributeError) as e:
    print(f"Invalid input: {e}")

# Connection errors
try:
    manager.push(MyFilter)
except Exception as e:
    print(f"Storage error: {e}")
    # Handle network issues, permissions, etc.
```

### Validation Helper
```python
def validate_and_push(manager, class_obj):
    """Push class with comprehensive validation."""

    # 1. Validate it's a class
    if not isinstance(class_obj, type):
        raise TypeError(f"{class_obj} is not a class")

    # 2. Check status
    status = manager.check_status(class_obj)
    if status == "synced":
        print(f"✓ {class_obj.__name__} already synced")
        return

    # 3. Get hash
    try:
        hash_value = manager.get_class_hash(class_obj)
        print(f"Hash: {hash_value[:8]}...")
    except Exception as e:
        raise ValueError(f"Could not compute hash: {e}")

    # 4. Push
    try:
        manager.push(class_obj)
        print(f"✓ Pushed {class_obj.__name__}")
    except Exception as e:
        raise RuntimeError(f"Push failed: {e}")

    # 5. Verify
    final_status = manager.check_status(class_obj)
    assert final_status == "synced", "Push verification failed"

# Usage
validate_and_push(manager, MyFilter)
```

---

## Best Practices

### 1. Always Check Status Before Push
```python
# ✅ Good: Check before pushing
status = manager.check_status(MyFilter)
if status != "synced":
    manager.push(MyFilter)
    print("Pushed new version")
else:
    print("Already synced, skipping")

# ❌ Bad: Blind push
manager.push(MyFilter)  # Redundant if already synced
```

### 2. Use Specific Versions in Production
```python
# ✅ Good: Pin to specific hash
production_hash = "abc123def456..."
MyFilter = manager.pull("MyFilter", code_hash=production_hash)

# ❌ Bad: Use 'latest' in production
MyFilter = manager.pull("MyFilter")  # May change unexpectedly
```

### 3. Version Tagging
```python
# Keep a mapping of semantic versions to hashes
VERSION_MANIFEST = {
    "MyFilter": {
        "v1.0.0": "abc123...",
        "v1.1.0": "def456...",
        "v2.0.0": "789xyz...",
    }
}

def get_version(class_name, version_tag):
    """Get class by semantic version."""
    hash_value = VERSION_MANIFEST[class_name][version_tag]
    return manager.pull(class_name, code_hash=hash_value)

# Usage
MyFilterV1 = get_version("MyFilter", "v1.0.0")
```

### 4. Monitor Hash Collisions (Extremely Rare)
```python
import logging

logger = logging.getLogger(__name__)

def safe_push(manager, class_obj):
    """Push with collision detection."""
    new_hash = manager.get_class_hash(class_obj)

    # Check if hash already exists
    try:
        existing = manager.recover_class(class_obj.__name__, new_hash)

        # If we can recover it, hash collision or duplicate
        logger.warning(
            f"Hash {new_hash[:8]} already exists for {class_obj.__name__}. "
            f"Skipping push (likely duplicate)."
        )
        return

    except FileNotFoundError:
        # Hash doesn't exist, safe to push
        manager.push(class_obj)
        logger.info(f"Pushed {class_obj.__name__} with hash {new_hash[:8]}")
```

---

## Performance Considerations

### Hash Computation Cost
```python
import time

class ComplexFilter(BaseFilter):
    # Many methods
    def method1(self, x): return x
    def method2(self, x): return x
    # ... 50+ methods

# Measure hash computation time
start = time.time()
hash_value = manager.get_class_hash(ComplexFilter)
duration = time.time() - start

print(f"Hash computed in {duration:.3f}s")
# Typically < 0.01s for most classes
```

**Optimization Tips:**

- Hash computation is fast (< 10ms for typical classes)
- Cached in `PetFactory._version_control` after first computation
- Only recomputed if class is modified

### Storage I/O
```python
# Minimize round-trips
classes = [Filter1, Filter2, Filter3, Filter4, Filter5]

# ❌ Bad: Multiple individual pushes
for cls in classes:
    manager.push(cls)  # 5 network round-trips

# ✅ Better: Batch check status
for cls in classes:
    if manager.check_status(cls) != "synced":
        manager.push(cls)

# ✅ Best: Use PetFactory.push_all()
# (Handles batching internally)
from labchain.container.persistent import PetFactory
factory = PetFactory(manager, Container.pif)
for cls in classes:
    factory[cls.__name__] = cls
factory.push_all()
```

---

## Troubleshooting

### Issue: Hash keeps changing

**Symptom:** Same class produces different hashes on different runs.

**Possible Causes:**

1. **Python version mismatch** - Bytecode differs between versions
2. **Dynamic class attributes** - Attributes set outside `__init__`
3. **Non-deterministic imports** - Conditional imports

**Solution:**
```python
# ✅ Deterministic class definition
import numpy as np  # Module-level import

class MyFilter(BaseFilter):
    def __init__(self, threshold: float):
        super().__init__(threshold=threshold)  # Stable params
        self._cache = None  # Private, not hashed

    def predict(self, x):
        return np.array(x.value > self.threshold)  # Stable logic
```

### Issue: Cannot recover class

**Symptom:** `FileNotFoundError` when calling `pull()` or `recover_class()`.

**Diagnostic Steps:**
```python
# 1. Verify storage connectivity
print(f"Storage: {manager.storage}")
print(f"Storage type: {type(manager.storage)}")

# 2. Check if metadata exists
meta = manager._get_remote_latest_meta("MyFilter")
print(f"Remote metadata: {meta}")

# 3. List files in storage (if LocalStorage)
if hasattr(manager.storage, 'storage_path'):
    import os
    path = f"{manager.storage.storage_path}/plugins/MyFilter"
    if os.path.exists(path):
        print(f"Files: {os.listdir(path)}")

# 4. Try explicit hash
if meta:
    try:
        cls = manager.recover_class("MyFilter", meta['hash'])
        print("✓ Recovery successful")
    except Exception as e:
        print(f"✗ Recovery failed: {e}")
```

---

## See Also

- [PetFactory](pet_factory.md) - Higher-level persistent factory interface
- [Remote Injection Guide](../../remote_injection/index.md) - Complete deployment workflow
- [Storage Backends](../../api/index.md#storage) - Available storage options
- [Container](../../api/index.md#container--dependency-injection) - Dependency injection system

---

## API Reference

::: labchain.container.persistent.pet_class_manager.PetClassManager
    options:
        show_root_heading: false
        show_source: true
        heading_level: 3
        members: true
        show_if_no_docstring: true
