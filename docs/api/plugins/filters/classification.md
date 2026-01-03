
# Classification Filters

::: labchain.plugins.filters.classification

## Overview

The Classification Filters module in labchain provides a collection of powerful classification algorithms that can be easily integrated into your machine learning pipelines. These filters are designed to work seamlessly with the labchain ecosystem, providing a consistent interface and enhanced functionality.

## Available Classifiers

### SVM Classifier

The Support Vector Machine (SVM) classifier is implemented in the `ClassifierSVMPlugin`. This versatile classifier is effective for both linear and non-linear classification tasks.

#### Usage
```python

from labchain.plugins.filters.classification.svm import ClassifierSVMPlugin

svm_classifier = ClassifierSVMPlugin(C=1.0, kernel='rbf', gamma='scale')
```

#### Parameters

- `C` (float): Regularization parameter. The strength of the regularization is inversely proportional to C.
- `kernel` (str): The kernel type to be used in the algorithm. Options include 'linear', 'poly', 'rbf', and 'sigmoid'.
- `gamma` (str or float): Kernel coefficient for 'rbf', 'poly' and 'sigmoid' kernels.

### K-Nearest Neighbors Classifier

The K-Nearest Neighbors (KNN) classifier is implemented in the `KnnFilter`. This simple yet effective classifier is based on the principle of finding the K nearest neighbors to make predictions.

#### Usage

```python
from labchain.plugins.filters.classification.knn import KnnFilter

knn_classifier = KnnFilter(n_neighbors=5, weights='uniform')
```

#### Parameters

- `n_neighbors` (int): Number of neighbors to use for kneighbors queries.
- `weights` (str): Weight function used in prediction. Options are 'uniform' (all points in each neighborhood are weighted equally) or 'distance' (weight points by the inverse of their distance).

## Comprehensive Example: Iris Dataset Classification

In this example, we'll demonstrate how to use the Classification Filters with the Iris dataset, showcasing both SVM and KNN classifiers, as well as integration with GridSearchCV.

```python
from labchain.plugins.pipelines.gs_cv_pipeline import GridSearchCVPipeline
from labchain.plugins.filters.classification.svm import ClassifierSVMPlugin
from labchain.plugins.filters.classification.knn import KnnFilter
from labchain.base.base_types import XYData
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create XYData objects
X_train_data = XYData(_hash='X_train', _path='/tmp', _value=X_train)
y_train_data = XYData(_hash='y_train', _path='/tmp', _value=y_train)
X_test_data = XYData(_hash='X_test', _path='/tmp', _value=X_test)
y_test_data = XYData(_hash='y_test', _path='/tmp', _value=y_test)

# Create a pipeline with SVM classifier
svm_pipeline = GridSearchCVPipeline(
    filterx=[ClassifierSVMPlugin],
    param_grid=ClassifierSVMPlugin.item_grid(C=[0.1, 1, 10], kernel=['linear', 'rbf']),
    scoring='accuracy',
    cv=5
)

# Fit the SVM pipeline
svm_pipeline.fit(X_train_data, y_train_data)

# Make predictions with SVM
svm_predictions = svm_pipeline.predict(X_test_data)
print("SVM Predictions:", svm_predictions.value)

# Create a pipeline with KNN classifier
knn_pipeline = GridSearchCVPipeline(
    filterx=[KnnFilter],
    param_grid=KnnFilter.item_grid(n_neighbors=[3, 5, 7], weights=['uniform', 'distance']),
    scoring='accuracy',
    cv=5
)

# Fit the KNN pipeline
knn_pipeline.fit(X_train_data, y_train_data)

# Make predictions with KNN
knn_predictions = knn_pipeline.predict(X_test_data)
print("KNN Predictions:", knn_predictions.value)

# Evaluate the models
from sklearn.metrics import accuracy_score

svm_accuracy = accuracy_score(y_test, svm_predictions.value)
knn_accuracy = accuracy_score(y_test, knn_predictions.value)

print("SVM Accuracy:", svm_accuracy)
print("KNN Accuracy:", knn_accuracy)
```

This example demonstrates how to:

1. Load and prepare the Iris dataset
2. Create XYData objects for use with labchain
3. Set up GridSearchCV pipelines for both SVM and KNN classifiers
4. Fit the models and make predictions
5. Evaluate the models using accuracy scores

## Best Practices

1. **Data Preprocessing**: Ensure your data is properly preprocessed before applying classification filters. This may include scaling, normalization, or handling missing values.

2. **Hyperparameter Tuning**: Use `GridSearchCVPipeline` to find the optimal hyperparameters for your chosen classifier, as demonstrated in the example.

3. **Model Evaluation**: Always evaluate your model's performance using appropriate metrics and cross-validation techniques. In the example, we used accuracy, but consider other metrics like precision, recall, or F1-score depending on your specific problem.

4. **Feature Selection**: Consider applying feature selection techniques to improve model performance and reduce overfitting, especially when dealing with high-dimensional datasets.

5. **Ensemble Methods**: Experiment with combining multiple classifiers to create ensemble models, which can often lead to improved performance.

## Conclusion

The Classification Filters module in labchain provides a robust set of tools for tackling various classification tasks. By leveraging these filters in combination with other labchain components, you can build powerful and efficient machine learning pipelines. The example with the Iris dataset demonstrates how easy it is to use these classifiers and integrate them with GridSearchCV for hyperparameter tuning.
