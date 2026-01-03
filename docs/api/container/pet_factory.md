---
title: PetFactory
icon: material/factory
status: experimental
---

# PetFactory

!!! warning "Experimental Feature"
    This component is part of the experimental Remote Injection system. Use with caution in production environments.

::: labchain.container.persistent.pet_factory.PetFactory
    options:
        show_root_heading: true
        show_source: true
        heading_level: 2
        members:
            - __init__
            - __setitem__
            - __getitem__
            - get
            - get_version
            - push_all

---

## Overview

`PetFactory` is a persistent-aware factory that extends `BaseFactory` with automatic version tracking and lazy loading from storage. It provides a seamless interface for working with persistent classes while maintaining full compatibility with the standard factory API.

### Key Features

- **Automatic Hash Tracking**: Computes and stores version hashes on registration
- **Lazy Loading**: Automatically fetches classes from storage when not in memory
- **Version-Specific Retrieval**: Load exact versions by hash for reproducibility
- **Bulk Operations**: Push all registered classes with a single command
- **Transparent Fallback**: Falls back to standard `Container.pif` for framework classes

---

## Basic Usage

### Initialize Factory
```python
from labchain.storage import S3Storage
from labchain.container.persistent import PetClassManager, PetFactory
from labchain import Container

# Create storage and manager
storage = S3Storage(bucket="my-ml-models")
manager = PetClassManager(storage)

# Create factory (typically via Container)
factory = PetFactory(manager, Container.pif)

# Or use the pre-initialized Container.ppif
factory = Container.ppif
```

### Register Classes
```python
from labchain.base import BaseFilter

class MyFilter(BaseFilter):
    def __init__(self, threshold: float = 0.5):
        super().__init__(threshold=threshold)

    def predict(self, x):
        return x.value > self.threshold

# Register in factory (hash computed automatically)
factory["MyFilter"] = MyFilter

# Check version tracking
print(f"Tracked hash: {factory._version_control['MyFilter'][:8]}...")
```

### Lazy Loading
```python
# Class in memory - instant return
MyFilter = factory["MyFilter"]

# Class not in memory - auto-loads from storage
RemoteFilter = factory["RemoteFilter"]  # Pulls from storage automatically
```

### Push to Storage
```python
# Register multiple classes
factory["Filter1"] = Filter1
factory["Filter2"] = Filter2
factory["Filter3"] = Filter3

# Push all at once
factory.push_all()
```

---

## Advanced Usage

### Version-Specific Retrieval
```python
# Save current version
config = my_pipeline.item_dump()
version_hash = config['params']['filter']['version_hash']

# Later, get exact version
MyFilterV1 = factory.get_version("MyFilter", version_hash)

# Use old version alongside new
MyFilterV2 = factory["MyFilter"]  # Latest

instance_v1 = MyFilterV1(threshold=0.3)
instance_v2 = MyFilterV2(threshold=0.5)
```

### Safe Get with Default
```python
# Try to get class, fallback to default
MyFilter = factory.get("MyOptionalFilter", default=DefaultFilter)

if MyFilter is None:
    print("Filter not found and no default provided")
```

### Version Rollback Pattern
```python
class VersionedFilterFactory:
    """Wrapper for version management."""

    def __init__(self, factory):
        self.factory = factory
        self.version_history = {}

    def register_with_tag(self, cls, tag):
        """Register class and tag the version."""
        self.factory[cls.__name__] = cls
        hash_value = self.factory._version_control[cls.__name__]
        self.version_history[cls.__name__] = {
            **self.version_history.get(cls.__name__, {}),
            tag: hash_value
        }
        return hash_value

    def get_by_tag(self, class_name, tag):
        """Get specific version by semantic tag."""
        hash_value = self.version_history[class_name][tag]
        return self.factory.get_version(class_name, hash_value)

    def rollback(self, class_name, tag):
        """Rollback to a previous version."""
        old_version = self.get_by_tag(class_name, tag)
        self.factory[class_name] = old_version
        return old_version

# Usage
vf = VersionedFilterFactory(Container.ppif)

# Version 1
class MyFilter(BaseFilter):
    def predict(self, x): return x * 1

vf.register_with_tag(MyFilter, "v1.0.0")

# Version 2
class MyFilter(BaseFilter):
    def predict(self, x): return x * 2

vf.register_with_tag(MyFilter, "v2.0.0")

# Rollback to v1
MyFilterV1 = vf.rollback("MyFilter", "v1.0.0")
```

---

## Integration with Container

`PetFactory` is pre-initialized in `Container.ppif`:
```python
from labchain import Container

# Already available as Container.ppif
print(type(Container.ppif))  # <class 'PetFactory'>

# Register with decorator
@Container.bind(persist=True)
class MyFilter(BaseFilter):
    def predict(self, x):
        return x

# Automatically registered in Container.ppif
assert "MyFilter" in Container.ppif._foundry

# Push all persistent classes
Container.ppif.push_all()
```

### Automatic vs Manual Registration
```python
# Automatic (recommended)
@Container.bind(persist=True, auto_push=True)
class AutoFilter(BaseFilter):
    def predict(self, x): return x
# Automatically pushed to storage on registration

# Manual control
@Container.bind(persist=True)
class ManualFilter(BaseFilter):
    def predict(self, x): return x

# Explicitly push when ready
Container.ppif.push_all()
```

---

## Fallback Behavior

`PetFactory` automatically falls back to `Container.pif` for framework classes:
```python
from labchain.pipeline import F3Pipeline

# F3Pipeline is a framework class (not user-defined)
# It's in Container.pif but NOT in Container.ppif

# PetFactory automatically falls back
pipeline_cls = Container.ppif["F3Pipeline"]  # Works! Falls back to Container.pif

# This enables build_from_dump to work seamlessly
config = pipeline.item_dump()
reconstructed = BasePlugin.build_from_dump(config, Container.ppif)
# Framework classes loaded from pif, user classes from ppif
```

### Fallback Priority
```python
def __getitem__(self, name):
    # Priority order:
    # 1. Check local memory (self._foundry)
    # 2. Check persistent storage (self.manager.pull)
    # 3. Check fallback factory (self._fallback_factory)
    # 4. Raise AttributeError
    pass
```

---

## Working with build_from_dump

`PetFactory` integrates seamlessly with pipeline reconstruction:
```python
from labchain.base import BasePlugin
from labchain import Container

# Create and train pipeline
pipeline = F3Pipeline(filters=[MyCustomFilter(), StandardScalerPlugin()])
pipeline.fit(X_train, y_train)

# Serialize (includes version hashes)
config = pipeline.item_dump()

# Push custom classes
Container.ppif.push_all()

# Later, reconstruct (even on different machine)
reconstructed = BasePlugin.build_from_dump(config, Container.ppif)

# Uses get_version() internally to fetch exact versions
```

### How build_from_dump Works
```python
@staticmethod
def build_from_dump(dump_dict, factory):
    clazz_name = dump_dict["clazz"]
    v_hash = dump_dict.get("version_hash")

    if v_hash and hasattr(factory, "get_version"):
        # PetFactory has get_version - fetch specific version
        level_clazz = factory.get_version(clazz_name, v_hash)
    else:
        # Standard factory - get current version
        level_clazz = factory.get(clazz_name)

    # Recursively reconstruct nested plugins
    if "params" in dump_dict:
        params = {}
        for k, v in dump_dict["params"].items():
            if isinstance(v, dict) and "clazz" in v:
                params[k] = build_from_dump(v, factory)
            else:
                params[k] = v
        return level_clazz(**params)

    return level_clazz
```

---

## Performance Optimization

### Batch Registration
```python
# ❌ Inefficient: Compute hash multiple times
for i in range(100):
    class DynamicFilter(BaseFilter):
        def predict(self, x): return x

    Container.ppif[f"Filter{i}"] = DynamicFilter

# ✅ Better: Pre-define classes
filters = {
    f"Filter{i}": create_filter(i)
    for i in range(100)
}

for name, cls in filters.items():
    Container.ppif[name] = cls

# Single push
Container.ppif.push_all()
```

### Lazy Loading Benefits
```python
# Only loads classes when actually needed
# No upfront cost for unused classes

def load_pipeline_lazy(config):
    """Reconstruct pipeline without loading all classes."""
    # build_from_dump only loads referenced classes
    pipeline = BasePlugin.build_from_dump(config, Container.ppif)

    # Other classes in storage remain unloaded
    return pipeline
```

---

## Error Handling

### Registration Errors
```python
try:
    # Must have _hash attribute (set by @Container.bind(persist=True))
    Container.ppif["MyFilter"] = MyFilter
except ValueError as e:
    print(f"Registration failed: {e}")
    # Class must be decorated with @Container.bind(persist=True)
    # or have _hash attribute set manually
```

### Loading Errors
```python
try:
    cls = Container.ppif["MaybeExists"]
except AttributeError as e:
    print(f"Class not found: {e}")
    # Not in memory, storage, or fallback factory

# Safe alternative
cls = Container.ppif.get("MaybeExists", default=DefaultClass)
```

### Version Errors
```python
try:
    old_version = Container.ppif.get_version("MyFilter", "invalid_hash")
except ValueError as e:
    print(f"Version not found: {e}")
    # Hash doesn't exist in storage
```

---

## Best Practices

### 1. Use Container.ppif via Decorator
```python
# ✅ Recommended: Use decorator
@Container.bind(persist=True)
class MyFilter(BaseFilter):
    def predict(self, x): return x

# Automatically registered in Container.ppif

# ❌ Manual registration (more error-prone)
class MyFilter(BaseFilter):
    def predict(self, x): return x

MyFilter._hash = Container.pcm.get_class_hash(MyFilter)
Container.ppif["MyFilter"] = MyFilter
```

### 2. Push After Registration Batch
```python
# ✅ Good: Register multiple, then push once
@Container.bind(persist=True)
class Filter1(BaseFilter): pass

@Container.bind(persist=True)
class Filter2(BaseFilter): pass

@Container.bind(persist=True)
class Filter3(BaseFilter): pass

Container.ppif.push_all()  # Single batch upload

# ❌ Bad: Push after each registration
@Container.bind(persist=True, auto_push=True)  # Uploads immediately
class Filter1(BaseFilter): pass

@Container.bind(persist=True, auto_push=True)  # Another upload
class Filter2(BaseFilter): pass
# Multiple network round-trips
```

### 3. Version Verification
```python
def verify_version_consistency(factory, config):
    """Verify all version hashes in config exist in storage."""

    def extract_hashes(obj):
        """Recursively extract all version_hash values."""
        hashes = []
        if isinstance(obj, dict):
            if "version_hash" in obj:
                hashes.append((obj["clazz"], obj["version_hash"]))
            for v in obj.values():
                hashes.extend(extract_hashes(v))
        elif isinstance(obj, list):
            for item in obj:
                hashes.extend(extract_hashes(item))
        return hashes

    version_hashes = extract_hashes(config)

    for class_name, hash_value in version_hashes:
        try:
            factory.get_version(class_name, hash_value)
            print(f"✓ {class_name} {hash_value[:8]}... verified")
        except Exception as e:
            print(f"✗ {class_name} {hash_value[:8]}... MISSING")
            raise ValueError(f"Version verification failed: {e}")

    print("✓ All versions verified")

# Usage before deployment
verify_version_consistency(Container.ppif, pipeline_config)
```

---

## Comparison: Standard vs Persistent Factory

| Feature | BaseFactory | PetFactory |
|---------|-------------|------------|
| **Registration** | `factory[name] = cls` | `factory[name] = cls` (with hash) |
| **Retrieval** | Memory only | Memory → Storage → Fallback |
| **Versioning** | None | Automatic via hash |
| **Storage Sync** | None | `push_all()` |
| **Lazy Loading** | No | Yes |
| **Version Pinning** | No | `get_version(name, hash)` |
| **Fallback** | No | Yes (to Container.pif) |

---

## Troubleshooting

### Issue: "Class must have a hash attribute"

**Solution:** Use `@Container.bind(persist=True)` decorator:
```python
# ❌ Wrong
class MyFilter(BaseFilter): pass
Container.ppif["MyFilter"] = MyFilter  # Error!

# ✅ Correct
@Container.bind(persist=True)
class MyFilter(BaseFilter): pass
# Hash automatically set
```

### Issue: Lazy loading fails

**Diagnostic:**
```python
# Check if class exists in storage
meta = Container.pcm._get_remote_latest_meta("MyFilter")
print(f"Remote metadata: {meta}")

# Try explicit pull
if meta:
    try:
        cls = Container.pcm.pull("MyFilter", meta['hash'])
        print("✓ Manual pull successful")
    except Exception as e:
        print(f"✗ Pull failed: {e}")
```

### Issue: Version not found
```python
# List all available versions (if LocalStorage)
import os
storage_path = Container.storage.storage_path
class_dir = f"{storage_path}/plugins/MyFilter"

if os.path.exists(class_dir):
    files = os.listdir(class_dir)
    pkl_files = [f for f in files if f.endswith('.pkl')]
    print(f"Available versions: {pkl_files}")
```

---

## See Also

- [PetClassManager](pet_class_manager.md) - Low-level storage operations
- [Remote Injection Guide](../../remote_injection/index.md) - Complete deployment workflow
- [Storage Backends](../../api/index.md#storage) - Available storage options
- [Container](../../api/index.md#container--dependency-injection) - Dependency injection system
- [BaseFactory](../../api/index.md#base-classes) - Parent class documentation

---

## API Reference

::: labchain.container.persistent.pet_factory.PetFactory
    options:
        show_root_heading: false
        show_source: true
        heading_level: 3
        members: true
        show_if_no_docstring: true
