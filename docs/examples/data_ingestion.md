# Data Ingestion and Storage in labchain

This tutorial demonstrates how to use different storage backends in labchain for storing and retrieving data, including local storage and Amazon S3.

## Prerequisites

Before running this example, make sure you have:

1. labchain installed
2. Necessary libraries imported:
   ```python
   from labchain.container import Container
   from labchain.plugins.storage import S3Storage
   import pandas as pd
   import numpy as np
   import os
   ```
3. For S3 storage: You need an S3-compatible service provider. Options include:
    - Cloud providers: Amazon AWS, IDrive e2
    - Self-hosted solutions: MinIO, Ceph
    - Local development: LocalStack (for simulating S3 locally)

   Ensure you have the necessary credentials and endpoint information for your chosen S3 service.

## 1. Creating Sample Data

Let's start by creating some sample data that we'll use throughout this tutorial:

```python
# Create sample data
df = pd.DataFrame({
    'A': np.random.rand(100),
    'B': np.random.randint(0, 100, 100),
    'C': ['cat', 'dog', 'bird'] * 33 + ['cat']
})
```

## 2. Local Storage

By default, labchain uses local storage. Here's how to use it:

### Storing Data Locally

```python
# Store the DataFrame locally
Container.ds.save('sample_data_local', df)
print("Data stored successfully locally")
```
```bash
	 * Saving in local path: cache/datasets/sample_data_local
	 * Saved !
Data stored successfully locally
```

### Listing Local Data

```python
local_files = Container.ds.list()
print("Files in local storage:", local_files)
```
```bash
Files in local storage: ['sample_data', 'sample_data_local']
```

### Retrieving Local Data

```python
# Retrieve the stored DataFrame from local storage
retrieved_df = Container.ds.load('sample_data_local')
print("Data retrieved successfully from local storage")
print(retrieved_df.value.head())
```
```bash
Data retrieved successfully from local storage
	 * Downloading: <_io.BufferedReader name='cache/datasets/sample_data_local'>
          A   B     C
0  0.858184  40   cat
1  0.467917  62   dog
2  0.810327  86  bird
3  0.194756  89   cat
4  0.313579  61   dog
```


### Updating Local Data

```python
# Update the DataFrame
df['D'] = np.random.choice(['X', 'Y', 'Z'], 100)

# Store the updated DataFrame locally
Container.ds.update('sample_data_local', df)
print("Updated data stored successfully locally")

# Retrieve and display the updated DataFrame
updated_df = Container.ds.load('sample_data_local')
print(updated_df.value.head())
```
```bash
	 * Saving in local path: cache/datasets/sample_data_local
	 * Saved !
Updated data stored successfully locally
	 * Downloading: <_io.BufferedReader name='cache/datasets/sample_data_local'>
          A   B     C  D
0  0.858184  40   cat  Z
1  0.467917  62   dog  Y
2  0.810327  86  bird  X
3  0.194756  89   cat  X
4  0.313579  61   dog  X
```

### Deleting Local Data

```python
# Delete the stored data from local storage
Container.ds.delete('sample_data_local')
print("Data deleted successfully from local storage")
```
```bash
Data deleted successfully from local storage
```


## 3. S3 Storage

Now, let's see how to use S3 storage for the same operations:

### Configuring S3 Storage

First, we need to configure the S3 storage backend:

```python
# Configure S3 storage
s3_storage = S3Storage(
    bucket=os.environ.get('TEST_BUCKET_NAME'), # type: ignore
    region_name=os.environ.get('REGION_NAME'), # type: ignore
    access_key=os.environ.get('ACCESS_KEY'), # type: ignore
    access_key_id=os.environ.get('ACCESS_KEY_ID'), # type: ignore
    endpoint_url=os.environ.get('ENDPOINT_URL'),
)

# Set S3 storage as the default storage backend
Container.storage = s3_storage
```

### Storing Data in S3

```python
# Store the DataFrame in S3
Container.ds.save('sample_data_s3', df)
print("Data stored successfully in S3")
```
```bash
- Binary prepared!
- Stream ready!
 	 * Object size 8e-08 GBs
Upload Complete!
Data stored successfully in S3
```

### Listing Data in S3

```python
s3_files = Container.ds.list()
print("Files in S3 bucket:", s3_files)
```
```bash
Files in S3 bucket: ['test-bucket/datasets/sample_data_s3']
```

### Retrieving Data from S3

```python
# Retrieve the stored DataFrame from S3
retrieved_df = Container.ds.load('sample_data_s3')
print("Data retrieved successfully from S3")
print(retrieved_df.value.head())
```
```bash
Data retrieved successfully from S3
          A   B     C  D
0  0.301524  95   cat  Y
1  0.101139  20   dog  X
2  0.852597  49  bird  X
3  0.049054  59   cat  Z
4  0.463926  59   dog  X
```

### Updating Data in S3

```python
# Update the DataFrame
df['E'] = np.random.choice(['P', 'Q', 'R'], 100)

# Store the updated DataFrame in S3
Container.ds.update('sample_data_s3', df)
print("Updated data stored successfully in S3")

# Retrieve and display the updated DataFrame
updated_df = Container.ds.load('sample_data_s3')
print(updated_df.value.head())
```
```bash
- Binary prepared!
- Stream ready!
 	 * Object size 8e-08 GBs
Upload Complete!
Updated data stored successfully in S3
          A   B     C  D  E
0  0.735935  60   cat  Y  P
1  0.772428  23   dog  Y  Q
2  0.509925   6  bird  X  Q
3  0.775553   7   cat  Z  R
4  0.395329  81   dog  X  P
```

### Deleting Data from S3

```python
# Delete the stored data from S3
Container.ds.delete('sample_data_s3')
print("Data deleted successfully from S3")
```
```bash
Deleted!
Data deleted successfully from S3
```
## Conclusion

This tutorial demonstrated how to use both local storage and S3 storage in labchain to:

1. Store data
2. List stored data
3. Retrieve data
4. Update stored data
5. Delete data

The `Container.ds` interface provides a consistent way to interact with different storage backends, making it easy to switch between local and S3 storage as needed.
