# Clustering Metrics

::: labchain.plugins.metrics.clustering

## Overview

The Clustering Metrics module in LabChain provides a set of evaluation metrics specifically designed for assessing the performance of clustering algorithms. These metrics help in understanding various aspects of a clustering model's performance, such as cluster homogeneity, completeness, and overall quality.

## Available Clustering Metrics

### Normalized Mutual Information (NMI)

The Normalized Mutual Information score is implemented in the `NMI` class. It measures the mutual information between the true labels and the predicted clusters, normalized by the arithmetic mean of the labels' and clusters' entropy.

#### Usage

```python
from labchain.plugins.metrics.clustering import NMI
from labchain.base.base_types import XYData

nmi_metric = NMI()
score = nmi_metric.evaluate(x_data, y_true, y_pred)
```

### Adjusted Rand Index (ARI)

The Adjusted Rand Index is implemented in the `ARI` class. It measures the similarity between two clusterings, adjusted for chance. It has a value close to 0 for random labeling and 1 for perfect clustering.

#### Usage

```python
from labchain.plugins.metrics.clustering import ARI
from labchain.base.base_types import XYData

ari_metric = ARI()
score = ari_metric.evaluate(x_data, y_true, y_pred)
```

### Silhouette Score

The Silhouette Score is implemented in the `Silhouette` class. It measures how similar an object is to its own cluster compared to other clusters. The silhouette value ranges from -1 to 1, where a high value indicates that the object is well matched to its own cluster and poorly matched to neighboring clusters.

#### Usage

```python
from labchain.plugins.metrics.clustering import Silhouette
from labchain.base.base_types import XYData

silhouette_metric = Silhouette()
score = silhouette_metric.evaluate(x_data, y_true, y_pred)
```

### Calinski-Harabasz Index

The Calinski-Harabasz Index is implemented in the `CalinskiHarabasz` class. It's also known as the Variance Ratio Criterion. The score is defined as the ratio of the sum of between-clusters dispersion and of inter-cluster dispersion for all clusters. A higher Calinski-Harabasz score relates to a model with better defined clusters.

#### Usage

```python
from labchain.plugins.metrics.clustering import CalinskiHarabasz
from labchain.base.base_types import XYData

ch_metric = CalinskiHarabasz()
score = ch_metric.evaluate(x_data, y_true, y_pred)
```

### Homogeneity Score

The Homogeneity Score is implemented in the `Homogeneity` class. It measures whether all of its clusters contain only data points which are members of a single class. The score is bounded below by 0 and above by 1. A higher value indicates better homogeneity.

#### Usage

```python
from labchain.plugins.metrics.clustering import Homogeneity
from labchain.base.base_types import XYData

homogeneity_metric = Homogeneity()
score = homogeneity_metric.evaluate(x_data, y_true, y_pred)
```

### Completeness Score

The Completeness Score is implemented in the `Completeness` class. It measures whether all members of a given class are assigned to the same cluster. The score is bounded below by 0 and above by 1. A higher value indicates better completeness.

#### Usage

```python
from labchain.plugins.metrics.clustering import Completeness
from labchain.base.base_types import XYData

completeness_metric = Completeness()
score = completeness_metric.evaluate(x_data, y_true, y_pred)
```

## Comprehensive Example: Evaluating a Clustering Model

In this example, we'll demonstrate how to use the Clustering Metrics to evaluate the performance of a clustering model.

```python
from labchain.plugins.filters.clustering.kmeans import KMeansPlugin
from labchain.plugins.metrics.clustering import NMI, ARI, Silhouette, CalinskiHarabasz, Homogeneity, Completeness
from labchain.base.base_types import XYData
from sklearn.datasets import make_blobs
import numpy as np

# Generate sample data
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Create XYData object
X_data = XYData(_hash='X_data', _path='/tmp', _value=X)

# Create and fit the clustering model
kmeans = KMeansPlugin(n_clusters=4, random_state=0)
kmeans.fit(X_data)

# Get cluster predictions
y_pred = kmeans.predict(X_data)

# Initialize metrics
nmi_metric = NMI()
ari_metric = ARI()
silhouette_metric = Silhouette()
ch_metric = CalinskiHarabasz()
homogeneity_metric = Homogeneity()
completeness_metric = Completeness()

# Compute metrics
nmi_score = nmi_metric.evaluate(X_data, y_true, y_pred.value)
ari_score = ari_metric.evaluate(X_data, y_true, y_pred.value)
silhouette_score = silhouette_metric.evaluate(X_data, y_true, y_pred.value)
ch_score = ch_metric.evaluate(X_data, y_true, y_pred.value)
homogeneity_score = homogeneity_metric.evaluate(X_data, y_true, y_pred.value)
completeness_score = completeness_metric.evaluate(X_data, y_true, y_pred.value)

# Print results
print(f"Normalized Mutual Information: {nmi_score}")
print(f"Adjusted Rand Index: {ari_score}")
print(f"Silhouette Score: {silhouette_score}")
print(f"Calinski-Harabasz Index: {ch_score}")
print(f"Homogeneity Score: {homogeneity_score}")
print(f"Completeness Score: {completeness_score}")
```

This example demonstrates how to:

1. Generate sample clustering data
2. Create XYData objects for use with LabChain
3. Train a KMeans clustering model
4. Make predictions on the dataset
5. Initialize and compute various clustering metrics
6. Print the evaluation results

## Best Practices

1. **Multiple Metrics**: Use multiple metrics to get a comprehensive view of your clustering model's performance. Different metrics capture different aspects of clustering quality.

2. **Ground Truth**: When available, use metrics that compare against ground truth labels (like NMI, ARI) for a more robust evaluation.

3. **Internal Metrics**: When ground truth is not available, rely on internal metrics like Silhouette Score and Calinski-Harabasz Index.

4. **Interpretation**: Remember that the interpretation of these metrics can depend on the specific characteristics of your dataset and the clustering algorithm used.

5. **Visualization**: Complement these metrics with visualization techniques to get a better understanding of your clustering results.

6. **Parameter Tuning**: Use these metrics to guide the tuning of your clustering algorithm's parameters (e.g., number of clusters).

7. **Stability**: Consider evaluating the stability of your clustering results by running the algorithm multiple times with different initializations.

8. **Domain Knowledge**: Always interpret these metrics in the context of your domain knowledge and the specific goals of your clustering task.

## Conclusion

The Clustering Metrics module in LabChain provides essential tools for evaluating the performance of clustering models. By using these metrics in combination with other LabChain components, you can gain valuable insights into your model's strengths and weaknesses. The example demonstrates how easy it is to compute and interpret these metrics within the LabChain ecosystem, enabling you to make informed decisions about your clustering models.
