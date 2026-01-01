---
icon: material/content-save
---

# Caching Guide

This guide covers everything you need to know about LabChain's caching system: from local caching for individual researchers to distributed caching for team collaboration.

## Why Caching Matters

Machine learning experiments often involve expensive computations that get repeated unnecessarily:

- **Deep learning embeddings** over large corpora (hours)
- **Feature extraction** from images or text (hours)
- **Data preprocessing** at scale (minutes to hours)
- **Model training** with expensive operations (varies)

When testing multiple classifiers on the same preprocessed data, traditional workflows recompute everything. LabChain's hash-based caching eliminates this waste.

## How Caching Works

LabChain uses **content-addressable storage** with cryptographic hashing:

1. **Hash Generation**: Each filter computes a unique hash from:

    - Class name (e.g., `StandardScalerPlugin`)
    - **Public attributes** (constructor parameters)
    - Input data hash (for trainable filters)

2. **Cache Lookup**: Before executing, LabChain checks if the hash exists in storage

3. **Cache Hit**: Download and use cached result (seconds)

4. **Cache Miss**: Compute, store result, continue (normal execution time)

### Critical: Public Attributes Define Identity

**Only public attributes (no underscore prefix) are included in the hash and must be constructor parameters.**

```python
from labchain import BaseFilter, Container

@Container.bind()
class MyFilter(BaseFilter):
    def __init__(self, scale: float = 1.0, offset: float = 0.0):
        super().__init__(scale=scale, offset=offset)
        # ✅ Public attributes - included in hash, must be in constructor
        self.scale = scale
        self.offset = offset

        # ✅ Private attributes - excluded from hash, internal state
        self._fitted = False
        self._mean = None

    def fit(self, x: XYData, y: XYData | None):
        self._mean = x.value.mean()  # Internal state
        self._fitted = True

    def predict(self, x: XYData) -> XYData:
        # Uses public parameters and private state
        result = (x.value - self._mean) * self.scale + self.offset
        return XYData.mock(result)
```

**Hash formula for trainable filters:**
```
filter_hash = hash(class_name, public_attributes, training_data_hash)
```

**Hash formula for non-trainable filters:**
```
filter_hash = hash(class_name, public_attributes)
```

**Why this matters:**

- Changing `scale` or `offset` → **New hash** → Cache miss (correct!)
- Changing `_mean` (private) → **Same hash** → Cache hit (correct!)
- Adding new public attribute without updating constructor → **Error**

## Local Caching

Perfect for individual researchers iterating on experiments.

### Setup

```python
from labchain import Container
from labchain.plugins.storage import LocalStorage

# Configure local filesystem storage
Container.storage = LocalStorage(
    storage_path='./cache'  # Where to store cached results
)
```

### Basic Usage

```python
from labchain import F3Pipeline
from labchain.plugins.filters import Cached, StandardScalerPlugin, KnnFilter
from labchain.plugins.metrics import F1

pipeline = F3Pipeline(
    filters=[
        Cached(
            filter=StandardScalerPlugin(),
            cache_data=True,      # Cache transformed data
            cache_filter=True,    # Cache fitted model
            overwrite=False       # Reuse existing cache
        ),
        KnnFilter(n_neighbors=5)
    ],
    metrics=[F1()]
)

# First run: computes everything, stores in ./cache
pipeline.fit(x_train, y_train)

# Second run: loads from cache (much faster!)
pipeline.fit(x_train, y_train)
```

### Cache Control Parameters

```python
Cached(
    filter=MyExpensiveFilter(),

    cache_data=True,     # Store processed output data
    cache_filter=True,   # Store fitted filter state
    overwrite=False,     # Force recomputation if True
    storage=None         # Use custom storage (None = Container.storage)
)
```

**`cache_data`**: Cache the filter's output

- `True`: Store `predict()` results
- `False`: Always recompute output (but may still cache filter state)

**`cache_filter`**: Cache the fitted filter itself

- `True`: Store filter after `fit()` for later reuse
- `False`: Always retrain filter

**`overwrite`**: Force cache invalidation

- `True`: Ignore existing cache, recompute and overwrite
- `False`: Use cache if available

**`storage`**: Override global storage

- `None`: Use `Container.storage`
- Custom instance: Use specific storage for this filter

### Practical Example

```python
from labchain.plugins.filters import Cached

# Expensive embedding computation
embeddings_filter = Cached(
    filter=BERTEmbeddings(model='bert-base-uncased'),
    cache_data=True,      # Cache embeddings (expensive to compute)
    cache_filter=False,   # Don't cache model (deterministic, no training)
    overwrite=False
)

# Cheap normalization
scaler_filter = Cached(
    filter=StandardScalerPlugin(),
    cache_data=True,      # Cache scaled data
    cache_filter=True,    # Cache mean/std parameters
    overwrite=False
)

pipeline = F3Pipeline(
    filters=[embeddings_filter, scaler_filter, KnnFilter()],
    metrics=[F1()]
)

# First run: ~3 hours (embeddings)
pipeline.fit(x_train, y_train)

# Second run with different classifier: ~10 seconds
pipeline2 = F3Pipeline(
    filters=[embeddings_filter, scaler_filter, SVMFilter()],
    metrics=[F1()]
)
pipeline2.fit(x_train, y_train)  # Embeddings and scaling: cache hit!
```

## Distributed Caching

Share cached results across team members or institutions using cloud storage.

### Setup with S3

```python
from labchain import Container
from labchain.plugins.storage import S3Storage

# Configure S3 storage
Container.storage = S3Storage(
    bucket_name='my-team-ml-cache',
    prefix='experiments/',           # Optional: organize by project
    region='us-east-1',               # AWS region
    access_key='YOUR_ACCESS_KEY',    # Use env vars in production!
    secret_key='YOUR_SECRET_KEY'
)
```

**Production recommendation**: Use environment variables or IAM roles:

```python
import os

Container.storage = S3Storage(
    bucket_name=os.getenv('LABCHAIN_BUCKET'),
    prefix=os.getenv('LABCHAIN_PREFIX', 'experiments/'),
    region=os.getenv('AWS_REGION', 'us-east-1'),
    # access_key and secret_key from AWS credentials chain
)
```

### Team Collaboration Workflow

**Researcher A** (computes expensive embeddings):

```python
from labchain import Container
from labchain.plugins.storage import S3Storage
from labchain.plugins.filters import Cached

# Configure shared storage
Container.storage = S3Storage(
    bucket_name='team-cache',
    prefix='mental-health-detection/'
)

# Define pipeline with expensive operation
pipeline_a = F3Pipeline(
    filters=[
        Cached(
            filter=BERTEmbeddings(model='bert-large', max_length=512),
            cache_data=True,
            cache_filter=False
        ),
        SVMFilter()
    ]
)

# This takes 3 hours, uploads embeddings to S3
pipeline_a.fit(x_train, y_train)
```

**Researcher B** (reuses embeddings, tests different model):

```python
from labchain import Container
from labchain.plugins.storage import S3Storage
from labchain.plugins.filters import Cached

# Same storage configuration
Container.storage = S3Storage(
    bucket_name='team-cache',
    prefix='mental-health-detection/'
)

# Different pipeline, SAME embeddings filter
pipeline_b = F3Pipeline(
    filters=[
        Cached(
            filter=BERTEmbeddings(model='bert-large', max_length=512),  # Identical!
            cache_data=True,
            cache_filter=False
        ),
        RandomForestFilter()  # Different classifier
    ]
)

# This takes ~30 seconds: downloads embeddings from S3, trains RF
pipeline_b.fit(x_train, y_train)
```

**Key insight**: As long as the filter configuration and input data are identical, the hash matches and cache sharing works automatically.

### Cache Key Requirements for Sharing

For cache hits across researchers:

1. **Identical filter class**: Same Python class
2. **Identical public parameters**: Same constructor arguments
3. **Identical input data**: Same data hash (or user-provided hash)

```python
# ✅ These produce the SAME hash (cache hit)
filter_a = BERTEmbeddings(model='bert-base', max_length=128)
filter_b = BERTEmbeddings(model='bert-base', max_length=128)

# ❌ These produce DIFFERENT hashes (cache miss)
filter_c = BERTEmbeddings(model='bert-base', max_length=256)  # Different param
filter_d = BERTEmbeddings(model='bert-large', max_length=128) # Different param
```

## Advanced: TTL and Heartbeat for Race-Free Distributed Caching

When multiple processes or machines attempt to cache the same computation simultaneously, race conditions can occur. LabChain provides `CachedWithLocking` with TTL-based expiration and heartbeat-based crash detection to coordinate distributed processes.

### Why Locking Matters

**Problem**: Without locks, multiple processes can start training the same model simultaneously:

```
Process A: Starts training model_abc123 (3 hours)
Process B: Starts training model_abc123 (3 hours) ← Redundant!
Process C: Starts training model_abc123 (3 hours) ← Redundant!
Result: 9 hours wasted, 3x CO₂ emissions
```

**Solution**: With locking, only one process trains:

```
Process A: 🔒 Acquires lock → Trains (3 hours) → Uploads → 🔓 Releases
Process B: ⏳ Waits for lock → Downloads trained model (30s)
Process C: ⏳ Waits for lock → Downloads trained model (30s)
Result: 3 hours total, cache reused
```

### LockingLocalStorage for Single-Machine Parallelism

Perfect for multiprocessing on one machine (local development, CI/CD, single-node training):

```python
from labchain import Container
from labchain.plugins.storage import LockingLocalStorage
from labchain.plugins.filters import CachedWithLocking

# Configure locking storage
Container.storage = LockingLocalStorage(storage_path='./cache')

# Wrap filter with locking cache
cached_filter = CachedWithLocking(
    filter=MyExpensiveFilter(),
    cache_data=True,
    cache_filter=True,
    lock_ttl=3600,           # Lock valid for 1 hour
    lock_timeout=7200,       # Wait up to 2 hours for other processes
    heartbeat_interval=30,   # Send heartbeat every 30 seconds
    auto_heartbeat=True      # Automatic heartbeat during long operations
)

# Use in parallel workflows
cached_filter.fit(x_train, y_train)  # Only one process computes
```

**How it works:**

- Uses atomic file operations (`O_CREAT | O_EXCL`) for kernel-level locking
- Safe across multiple processes on the same filesystem (including NFS)
- Lock files stored in `cache/locks/` directory

### LockingS3Storage for Multi-Machine Distribution

Perfect for cloud deployments (EC2, Kubernetes, multi-datacenter):

```python
from labchain.plugins.storage import LockingS3Storage
from labchain.plugins.filters import CachedWithLocking

# Configure S3 locking storage
Container.storage = LockingS3Storage(
    bucket_name='my-team-ml-cache',
    region='us-east-1',
    access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
)

# Same API, now distributed across machines
cached_filter = CachedWithLocking(
    filter=BERTEmbeddings(model='bert-large'),
    cache_data=True,
    lock_ttl=7200,          # 2 hour training time
    lock_timeout=10800,     # 3 hour max wait
    heartbeat_interval=60   # Heartbeat every minute
)

# EC2 Instance A: Trains model
cached_filter.fit(x_train, y_train)  # 🔒 Locks, trains, uploads

# EC2 Instance B (simultaneously): Waits and downloads
cached_filter.fit(x_train, y_train)  # ⏳ Waits, downloads result
```

**How it works:**

- Uses S3 PUT operations (atomic at object level)
- Works with any S3-compatible service (AWS S3, MinIO, DigitalOcean Spaces)
- Lock metadata stored as S3 objects in `locks/` prefix

### TTL (Time-To-Live) Configuration

TTL defines how long a lock is considered valid before becoming stale:

```python
cached_filter = CachedWithLocking(
    filter=MyFilter(),
    lock_ttl=3600,  # Lock expires after 1 hour
)
```

**Lock metadata structure:**
```json
{
    "owner": "hostname.local",
    "pid": 12345,
    "created_at": 1735562400.5,
    "ttl": 3600
}
```

**What happens when TTL expires:**

1. Lock becomes "stale"
2. Other processes can "steal" the lock
3. Original process (if still running) loses exclusivity
4. Useful for crash recovery

**Choosing TTL values:**
```python
# Quick experiments (preprocessing in minutes)
lock_ttl=600  # 10 minutes

# Standard training (hours)
lock_ttl=3600  # 1 hour

# Deep learning (many hours)
lock_ttl=14400  # 4 hours

# Rule of thumb: Set TTL to 2x expected operation time
```

### Heartbeat-Based Crash Detection

Heartbeat detects if a process crashes before TTL expires:

```python
cached_filter = CachedWithLocking(
    filter=MyFilter(),
    lock_ttl=3600,              # 1 hour TTL
    heartbeat_interval=30,      # Update every 30 seconds
    auto_heartbeat=True         # Automatic updates during fit/predict
)
```

**How heartbeat works:**

1. **During operation**: Background thread updates `last_heartbeat` timestamp every `heartbeat_interval` seconds
2. **Health check**: Other processes check if heartbeat is stale (>3x interval)
3. **Crash detection**: If heartbeat stops, lock is considered "dead" before TTL expires
4. **Lock stealing**: Dead locks can be acquired by other processes

**Heartbeat metadata:**
```json
{
    "owner": "worker-node-5",
    "pid": 9876,
    "created_at": 1735562400.0,
    "ttl": 3600,
    "last_heartbeat": 1735562850.5,  // ← Updated every 30s
    "heartbeat_interval": 30
}
```

**Death detection logic:**
```python
# A lock is considered "dead" if:
heartbeat_age = current_time - last_heartbeat
if heartbeat_age > (heartbeat_interval * 3):
    # Process likely crashed, steal the lock
```

**Example: Long training with crash recovery:**

```python
from multiprocessing import Pool

def train_model(model_id):
    storage = LockingLocalStorage('./cache')

    model = HeavyModel()
    cached = CachedWithLocking(
        filter=model,
        storage=storage,
        lock_ttl=7200,          # 2 hour TTL
        heartbeat_interval=60,  # 1 minute heartbeat
        auto_heartbeat=True
    )

    try:
        # If this process crashes, heartbeat stops
        # Other processes detect death after ~3 minutes
        cached.fit(x_train, y_train)
    except Exception as e:
        print(f"Training failed: {e}")

# Run on 4 workers
with Pool(4) as p:
    p.map(train_model, ['model_1'] * 4)
# Only 1 worker trains, others wait
# If worker crashes, another takes over
```

### Manual Heartbeat Updates

For custom long-running operations, update heartbeat manually:

```python
cached = CachedWithLocking(
    filter=MyFilter(),
    storage=storage,
    lock_ttl=3600,
    heartbeat_interval=60,
    auto_heartbeat=False  # Disable automatic heartbeat
)

# Acquire lock manually
lock_name = f"model_{cached.filter._m_hash}"
if storage.try_acquire_lock(lock_name, ttl=3600, heartbeat_interval=60):
    try:
        for epoch in range(100):
            train_epoch()

            # Update heartbeat every 10 epochs
            if epoch % 10 == 0:
                storage.update_heartbeat(lock_name)
    finally:
        storage.release_lock(lock_name)
```

### Console Output

`CachedWithLocking` provides detailed logging when verbose mode is enabled:

**Process that acquires lock:**
```
🔒 Lock acquired: model_abc12345 (TTL: 3600s, HB: 30s)
💓 Heartbeat started for model training (interval: 30s)
🔨 Training model abc12345...
💾 Caching model abc12345...
✅ Model abc12345 cached
💓 Heartbeat stopped for model training
🔓 Lock released: model_abc12345
```

**Process that waits:**
```
⏳ Waiting for model abc12345...
📥 Loading cached model abc12345...
```

**Stale lock detection:**
```
⏰ Lock expired (age: 3750s > TTL: 3600s)
🔒 Lock acquired: model_abc12345 (stealing stale lock)
```

**Dead process detection:**
```
💀 Lock appears dead (heartbeat age: 195s)
🔒 Lock acquired: model_abc12345 (stealing dead lock)
```

### Comparison: Regular vs Locking Cache

| Feature | `Cached` | `CachedWithLocking` |
|---------|----------|---------------------|
| **Race conditions** | ❌ Possible | ✅ Prevented |
| **Parallel safety** | ❌ May duplicate work | ✅ Single computation |
| **Crash recovery** | ❌ Manual | ✅ Automatic (TTL + heartbeat) |
| **Overhead** | Minimal | Small (lock coordination) |
| **Storage requirement** | Any | Must be `LockingLocalStorage` or `LockingS3Storage` |
| **Use case** | Single process | Multiple processes/machines |

### Best Practices

**1. Choose appropriate TTL:**
```python
# Short operations (minutes)
lock_ttl = 2 * expected_duration_seconds

# Long operations (hours)
lock_ttl = 1.5 * expected_duration_seconds

# Very long operations (>6 hours)
lock_ttl = expected_duration_seconds + 3600  # +1 hour buffer
```

**2. Set heartbeat interval:**
```python
# General rule: 1/20 to 1/10 of TTL
lock_ttl = 3600
heartbeat_interval = 180  # 3 minutes (1/20 of TTL)

# For crash-sensitive workloads: shorter interval
heartbeat_interval = 30  # 30 seconds (faster detection)
```

**3. Configure timeout appropriately:**
```python
# Timeout should exceed TTL
lock_timeout = lock_ttl * 1.5

# For production: add extra buffer
lock_timeout = lock_ttl * 2
```

**4. Use verbose mode during development:**
```python
cached = CachedWithLocking(filter=model, ...)
cached.verbose(True)  # See all lock activity
```

**5. Handle timeouts gracefully:**
```python
try:
    cached.fit(x_train, y_train)
except TimeoutError as e:
    print(f"Training timeout: {e}")
    # Fallback: train without cache
    model.fit(x_train, y_train)
```

## Cache Storage Structure

LabChain organizes cache using this structure:

```
storage_path/
├── FilterClassName/
│   ├── filter_hash_1/
│   │   ├── model               # Fitted filter (if cache_filter=True)
│   │   ├── data_hash_a         # Cached output for input A
│   │   ├── data_hash_b         # Cached output for input B
│   │   └── ...
│   ├── filter_hash_2/
│   │   └── ...
│   └── ...
└── AnotherFilter/
    └── ...
```

**Example**:
```
./cache/
├── BERTEmbeddings/
│   ├── a3f5d8e1.../            # Hash of BERT config
│   │   ├── model               # (empty if cache_filter=False)
│   │   ├── 7b2c1f...           # Embeddings for dataset A
│   │   └── 9e4a3d...           # Embeddings for dataset B
│   └── b7e2c9f4.../            # Different BERT config
│       └── ...
└── StandardScalerPlugin/
    └── c1d8f3a2.../
        ├── model               # Fitted scaler (mean, std)
        └── 5f9b2e...           # Scaled data
```

## Debugging Cache Issues

### Check if Cache Exists

```python
from labchain import Container

# After configuring storage
storage = Container.storage

# Check specific hash
exists = storage.check_if_exists(
    hashcode='your_data_hash',
    context='FilterClassName/filter_hash'
)
print(f"Cache exists: {exists}")
```

### List Cached Files

```python
# List all cached results for a filter
files = storage.list_stored_files(
    context='BERTEmbeddings/a3f5d8e1...'
)
print(files)
```

### Enable Verbose Logging

```python
pipeline = F3Pipeline(
    filters=[
        Cached(filter=MyFilter(), cache_data=True)
    ],
    metrics=[F1()]
)

# Enable verbose mode to see cache hits/misses
pipeline.verbose(True)

pipeline.fit(x_train, y_train)
# Output will show:
# - Cache lookups
# - Cache hits/misses
# - Data downloads/uploads
```

### Common Issues

**Cache miss when expecting hit:**

1. **Different public parameters**: Check constructor arguments match exactly
2. **Different input data**: Verify data hashes are identical
3. **Code changes**: Modifying filter code doesn't invalidate cache by hash alone
4. **Storage misconfiguration**: Verify storage path or S3 bucket is correct

**Debugging checklist:**

```python
# 1. Check filter hash
filter = MyFilter(param1=value1)
print(f"Filter hash: {filter._m_hash}")

# 2. Check input data hash
print(f"Input hash: {x_train._hash}")

# 3. Enable verbose mode
pipeline.verbose(True)

# 4. Check storage configuration
print(f"Storage path: {Container.storage.get_root_path()}")

# 5. Force cache refresh to test
cached = Cached(filter=filter, overwrite=True)
```

## Best Practices

### 1. Cache Expensive Operations Only

```python
# ✅ DO: Cache slow operations
Cached(filter=BERTEmbeddings())      # Minutes to hours
Cached(filter=LargeDataTransform())  # Hours

# ❌ DON'T: Cache trivial operations
Cached(filter=StandardScalerPlugin())  # Milliseconds (minimal benefit)
```

### 2. Use Appropriate Storage

```python
# Individual researcher
Container.storage = LocalStorage('./cache')

# Small team, shared filesystem
Container.storage = LocalStorage('/shared/nfs/cache')

# Distributed team
Container.storage = S3Storage(bucket='team-cache')
```

### 3. Organize Cache by Project

```python
# Use prefixes to keep experiments organized
Container.storage = S3Storage(
    bucket='ml-research',
    prefix='project-depression-detection/'
)
```

### 4. Version Control Filter Definitions

Changing filter code without changing public parameters can cause issues:

```python
# v1
class MyFilter(BaseFilter):
    def predict(self, x):
        return x * 2  # Bug!

# v2
class MyFilter(BaseFilter):
    def predict(self, x):
        return x * 3  # Fixed!
```

**Problem**: Same hash, different behavior!

**Solution**: Add version as public parameter:

```python
class MyFilter(BaseFilter):
    def __init__(self, multiplier: int = 2, version: str = "v1"):
        super().__init__(multiplier=multiplier, version=version)
        self.multiplier = multiplier
        self.version = version  # Forces new hash
```

### 5. Document Cache Dependencies

```python
# At the top of your experiment script
"""
Cache dependencies:
- BERTEmbeddings: Requires model 'bert-base-uncased' in cache
- DataPreprocessor v2: Uses updated tokenization (incompatible with v1)
"""
```

## Performance Metrics

From our mental health detection case study:

| Scenario | Without Caching | With Caching | Savings |
|----------|----------------|--------------|---------|
| 5 classifiers on BERT embeddings | 16h 20min | 4h 20min | **12 hours** |
| CO₂ emissions (conservative) | 7.86 kg | 2.26 kg | **5.6 kg** |
| CO₂ emissions (high estimate) | 35.4 kg | 10.1 kg | **25.3 kg** |

## Summary

- **Hash-based caching** automatically reuses identical computations
- **Public attributes** define filter identity (must be constructor parameters)
- **Private attributes** are excluded from hash (internal state)
- **Local caching**: Single researcher, fast iteration
- **Distributed caching**: Team collaboration, cloud storage
- **Cache control**: `cache_data`, `cache_filter`, `overwrite`
- **Debugging**: Verbose mode, check hashes, verify storage

**Next steps:**

- [Quick Start](../quick_start/) — Basic installation and usage
- [Architecture](../architecture/) — How LabChain works internally
- [Best Practices](../best_practices/) — Production-ready patterns
