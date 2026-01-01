# Classification Metrics

::: labchain.plugins.metrics.classification

## Overview

The Classification Metrics module in LabChain provides a set of evaluation metrics specifically designed for assessing the performance of classification models. These metrics help in understanding various aspects of a classifier's performance, such as accuracy, precision, recall, and F1-score.

## Available Classification Metrics

### Accuracy Score

The Accuracy Score is implemented in the `AccuracyScoreMetric`. It computes the accuracy of a classification model by comparing the predicted labels with the true labels.

#### Usage

```python
from labchain.plugins.metrics.classification.accuracy_score import AccuracyScoreMetric

accuracy_metric = AccuracyScoreMetric()
score = accuracy_metric.compute(y_true, y_pred)
```

### Precision Score

The Precision Score is implemented in the `PrecisionScoreMetric`. It computes the precision of a classification model, which is the ratio of true positive predictions to the total number of positive predictions.

#### Usage

```python
from labchain.plugins.metrics.classification.precision_score import PrecisionScoreMetric

precision_metric = PrecisionScoreMetric(average='weighted')
score = precision_metric.compute(y_true, y_pred)
```

#### Parameters

- `average` (str): The averaging method. Options include 'micro', 'macro', 'weighted', 'samples', and None.

### Recall Score

The Recall Score is implemented in the `RecallScoreMetric`. It computes the recall of a classification model, which is the ratio of true positive predictions to the total number of actual positive instances.

#### Usage

```python
from labchain.plugins.metrics.classification.recall_score import RecallScoreMetric

recall_metric = RecallScoreMetric(average='weighted')
score = recall_metric.compute(y_true, y_pred)
```

#### Parameters

- `average` (str): The averaging method. Options include 'micro', 'macro', 'weighted', 'samples', and None.

### F1 Score

The F1 Score is implemented in the `F1ScoreMetric`. It computes the F1 score, which is the harmonic mean of precision and recall.

#### Usage

```python
from labchain.plugins.metrics.classification.f1_score import F1ScoreMetric

f1_metric = F1ScoreMetric(average='weighted')
score = f1_metric.compute(y_true, y_pred)
```

#### Parameters

- `average` (str): The averaging method. Options include 'micro', 'macro', 'weighted', 'samples', and None.

## Comprehensive Example: Evaluating a Classification Model

In this example, we'll demonstrate how to use the Classification Metrics to evaluate the performance of a classification model.

```python
from labchain.plugins.filters.classification.svm import ClassifierSVMPlugin
from labchain.plugins.metrics.classification.accuracy_score import AccuracyScoreMetric
from labchain.plugins.metrics.classification.precision_score import PrecisionScoreMetric
from labchain.plugins.metrics.classification.recall_score import RecallScoreMetric
from labchain.plugins.metrics.classification.f1_score import F1ScoreMetric
from labchain.base.base_types import XYData
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

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

# Create and train the classifier
classifier = ClassifierSVMPlugin(kernel='rbf')
classifier.fit(X_train_data, y_train_data)

# Make predictions
predictions = classifier.predict(X_test_data)

# Initialize metrics
accuracy_metric = AccuracyScoreMetric()
precision_metric = PrecisionScoreMetric(average='weighted')
recall_metric = RecallScoreMetric(average='weighted')
f1_metric = F1ScoreMetric(average='weighted')

# Compute metrics
accuracy = accuracy_metric.compute(y_test_data, predictions)
precision = precision_metric.compute(y_test_data, predictions)
recall = recall_metric.compute(y_test_data, predictions)
f1 = f1_metric.compute(y_test_data, predictions)

# Print results
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
```

This example demonstrates how to:

1. Load and prepare the Iris dataset
2. Create XYData objects for use with LabChain
3. Train an SVM classifier
4. Make predictions on the test set
5. Initialize and compute various classification metrics
6. Print the evaluation results

## Best Practices

1. **Multiple Metrics**: Use multiple metrics to get a comprehensive view of your model's performance. Different metrics capture different aspects of classification performance.

2. **Class Imbalance**: Be aware of class imbalance in your dataset. In such cases, accuracy alone might not be a good metric. Consider using precision, recall, and F1-score.

3. **Averaging Methods**: When dealing with multi-class classification, pay attention to the averaging method used in metrics like precision, recall, and F1-score. 'Weighted' average is often a good choice for imbalanced datasets.

4. **Cross-Validation**: Use cross-validation to get a more robust estimate of your model's performance, especially with smaller datasets.

5. **Confusion Matrix**: Consider using a confusion matrix in addition to these metrics for a more detailed view of your model's performance across different classes.

6. **ROC AUC**: For binary classification problems, consider using the ROC AUC score as an additional metric.

7. **Threshold Adjustment**: Remember that metrics like precision and recall can be affected by adjusting the classification threshold. Consider exploring different thresholds if needed.

8. **Domain-Specific Metrics**: Depending on your specific problem, you might need to implement custom metrics that are more relevant to your domain.

## Conclusion

The Classification Metrics module in LabChain provides essential tools for evaluating the performance of classification models. By using these metrics in combination with other LabChain components, you can gain valuable insights into your model's strengths and weaknesses. The example demonstrates how easy it is to compute and interpret these metrics within the LabChain ecosystem, enabling you to make informed decisions about your classification models.
