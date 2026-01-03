# Clustering Filters

::: labchain.plugins.filters.clustering

## Overview

The Clustering Filters module in labchain provides a collection of unsupervised learning algorithms for clustering data. These filters are designed to work seamlessly within the labchain ecosystem, offering a consistent interface and enhanced functionality for various clustering tasks.

## Available Clustering Algorithms

### K-Means Clustering

The K-Means clustering algorithm is implemented in the `KMeansFilter`. This popular clustering method aims to partition n observations into k clusters, where each observation belongs to the cluster with the nearest mean (cluster centroid).

#### Usage

```python
from labchain.plugins.filters.clustering.kmeans import KMeansFilter

kmeans_clusterer = KMeansFilter(n_clusters=3, init='k-means++', n_init=10, max_iter=300)
```

#### Parameters

- `n_clusters` (int): The number of clusters to form and the number of centroids to generate.
- `init` (str): Method for initialization of centroids. Options include 'k-means++' and 'random'.
- `n_init` (int): Number of times the k-means algorithm will be run with different centroid seeds.
- `max_iter` (int): Maximum number of iterations for a single run.

### DBSCAN Clustering

The Density-Based Spatial Clustering of Applications with Noise (DBSCAN) algorithm is implemented in the `DBSCANFilter`. This algorithm is particularly effective for datasets with clusters of arbitrary shape.

#### Usage

```python
from labchain.plugins.filters.clustering.dbscan import DBSCANFilter

dbscan_clusterer = DBSCANFilter(eps=0.5, min_samples=5)
```

#### Parameters

- `eps` (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
- `min_samples` (int): The number of samples in a neighborhood for a point to be considered as a core point.

## Comprehensive Example: Clustering with Synthetic Data

In this example, we'll demonstrate how to use the Clustering Filters with synthetic data, showcasing both K-Means and DBSCAN algorithms, as well as integration with GridSearchCV for parameter tuning.

```python
from labchain.plugins.pipelines.gs_cv_pipeline import GridSearchCVPipeline
from labchain.plugins.filters.clustering.kmeans import KMeansFilter
from labchain.plugins.filters.clustering.dbscan import DBSCANFilter
from labchain.base.base_types import XYData
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import silhouette_score
import numpy as np

# Generate synthetic datasets
X_blobs, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
X_moons, _ = make_moons(n_samples=300, noise=0.05, random_state=0)

# Create XYData objects
X_blobs_data = XYData(_hash='X_blobs', _path='/tmp', _value=X_blobs)
X_moons_data = XYData(_hash='X_moons', _path='/tmp', _value=X_moons)

# K-Means Clustering
kmeans_pipeline = GridSearchCVPipeline(
    filterx=[KMeansFilter],
    param_grid=KMeansFilter.item_grid(n_clusters=[2, 3, 4, 5], init=['k-means++', 'random']),
    scoring='silhouette',
    cv=5
)

# Fit K-Means on blobs dataset
kmeans_pipeline.fit(X_blobs_data)

# Make predictions
kmeans_labels = kmeans_pipeline.predict(X_blobs_data)
print("K-Means Cluster Labels:", kmeans_labels.value)

# DBSCAN Clustering
dbscan_pipeline = GridSearchCVPipeline(
    filterx=[DBSCANFilter],
    param_grid=DBSCANFilter.item_grid(eps=[0.1, 0.2, 0.3], min_samples=[3, 5, 7]),
    scoring='silhouette',
    cv=5
)

# Fit DBSCAN on moons dataset
dbscan_pipeline.fit(X_moons_data)

# Make predictions
dbscan_labels = dbscan_pipeline.predict(X_moons_data)
print("DBSCAN Cluster Labels:", dbscan_labels.value)

# Evaluate the models
kmeans_silhouette = silhouette_score(X_blobs, kmeans_labels.value)
dbscan_silhouette = silhouette_score(X_moons, dbscan_labels.value)

print("K-Means Silhouette Score:", kmeans_silhouette)
print("DBSCAN Silhouette Score:", dbscan_silhouette)
```

This example demonstrates how to:

1. Generate synthetic datasets suitable for different clustering algorithms
2. Create XYData objects for use with labchain
3. Set up GridSearchCV pipelines for both K-Means and DBSCAN clustering
4. Fit the models and make predictions
5. Evaluate the models using silhouette scores

## Best Practices

1. **Data Preprocessing**: Ensure your data is properly preprocessed before applying clustering filters. This may include scaling, normalization, or handling missing values.

2. **Algorithm Selection**: Choose the appropriate clustering algorithm based on the characteristics of your data and the specific requirements of your problem.

3. **Parameter Tuning**: Use `GridSearchCVPipeline` to find the optimal parameters for your chosen clustering algorithm, as demonstrated in the example.

4. **Cluster Evaluation**: Always evaluate your clustering results using appropriate metrics such as silhouette score, Calinski-Harabasz index, or Davies-Bouldin index.

5. **Visualization**: Visualize your clustering results to gain insights into the structure of your data and the performance of the clustering algorithm.

6. **Ensemble Methods**: Consider using ensemble clustering techniques to improve the robustness and stability of your clustering results.

## Conclusion

The Clustering Filters module in labchain provides a powerful set of tools for unsupervised learning tasks. By leveraging these filters in combination with other labchain components, you can build efficient and effective clustering pipelines. The example with synthetic data demonstrates how easy it is to use these clustering algorithms and integrate them with GridSearchCV for parameter tuning.
