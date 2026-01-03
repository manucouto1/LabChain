# Tutorial: F3Pipeline with Cache

This tutorial will guide you through a simple example of using F3Pipeline, including the Cache functionality.

## 1. Imports

First, we import the required classes and functions:

```python
from labchain import F3Pipeline
from labchain.plugins.filters import StandardScalerPlugin
from labchain.plugins.filters import ClassifierSVMPlugin
from labchain.plugins.metrics import F1
from labchain.base import XYData
from labchain.plugins.filters import Cached
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from rich import print
```

## 2. Data Preparation

We load the Iris dataset and split it into training and test sets:

```python
# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear objetos XYData
x_train = XYData('Iris dataset X train', 'dataset', X_train)
y_train = XYData('Iris dataset Y train', 'dataset', y_train)
x_test = XYData('Iris dataset X test', 'dataset', X_test)
y_test = XYData('Iris dataset Y test', 'dataset', y_test)
```

## 3. Creating the Pipeline with Cache

We create a pipeline that includes a cached `StandardScalerPlugin` and a `ClassifierSVMPlugin`:

```python
pipeline = F3Pipeline(
    filters=[
        Cached(
            filter=StandardScalerPlugin(),
            cache_data=True,
            cache_filter=True,
            overwrite=False
        ),
        ClassifierSVMPlugin()
    ],
    metrics=[F1()]
)
print(pipeline)
```
```bash
F3Pipeline(
    filters=[
        Cached(filter=StandardScalerPlugin(), cache_data=True, cache_filter=True, overwrite=False, storage=None),
        ClassifierSVMPlugin(C=1.0, kernel='linear', gamma='scale')
    ],
    metrics=[F1(average='weighted')],
    overwrite=False,
    store=False,
    log=False
)
```

## 4. Training the Model

We train the model with the training data:

```python
pipeline.fit(x_train, y_train)
```
```bash
____________________________________________________________________________________________________
Fitting pipeline...
****************************************************************************************************
* Cached({'filter': StandardScalerPlugin({}), 'cache_data': True, 'cache_filter': True, 'overwrite': False, 'storage': None}):
         - El filtro StandardScalerPlugin({}) con hash 4f0150e0e11419085ce0f08ab077b7e5891f817b No existe, se va a entrenar.
         - El filtro StandardScalerPlugin({}) Se cachea.
	 * Saving in local path: cache/StandardScalerPlugin/4f0150e0e11419085ce0f08ab077b7e5891f817b/model
	 * Saved !
         - El dato XYData(_hash='f77f9d95466939988cdd6a13f0cb91260b94c99d', _path='StandardScalerPlugin/4f0150e0e11419085ce0f08ab077b7e5891f817b') No existe, se va a crear.
         - El dato XYData(_hash='f77f9d95466939988cdd6a13f0cb91260b94c99d', _path='StandardScalerPlugin/4f0150e0e11419085ce0f08ab077b7e5891f817b') Se cachea.
	 * Saving in local path: cache/StandardScalerPlugin/4f0150e0e11419085ce0f08ab077b7e5891f817b/f77f9d95466939988cdd6a13f0cb91260b94c99d
	 * Saved !
* ClassifierSVMPlugin({'C': 1.0, 'kernel': 'linear', 'gamma': 'scale'}):
```

## 5. Prediction and Evaluation

We make predictions and evaluate the model:

```python
# Make predictions
predictions = pipeline.predict(x_test)

# Evaluate the model
evaluation = pipeline.evaluate(x_test, y_test, predictions)
print("Evaluation results:", evaluation)
```
```bash
Predicting pipeline...
****************************************************************************************************
* Cached({'filter': StandardScalerPlugin({}), 'cache_data': True, 'cache_filter': True, 'overwrite': False, 'storage': None})
         - El dato XYData(_hash='0403ab1f29eb3d1d59f857aa03f7af153d7ff357', _path='StandardScalerPlugin/4f0150e0e11419085ce0f08ab077b7e5891f817b') No existe, se va a crear.
         - El dato XYData(_hash='0403ab1f29eb3d1d59f857aa03f7af153d7ff357', _path='StandardScalerPlugin/4f0150e0e11419085ce0f08ab077b7e5891f817b') Se cachea.
	 * Saving in local path: cache/StandardScalerPlugin/4f0150e0e11419085ce0f08ab077b7e5891f817b/0403ab1f29eb3d1d59f857aa03f7af153d7ff357
	 * Saved !
* ClassifierSVMPlugin({'C': 1.0, 'kernel': 'linear', 'gamma': 'scale'})
____________________________________________________________________________________________________
Evaluating pipeline...
____________________________________________________________________________________________________
Resultados de la evaluación:
{'F1': 0.9664109121909632}
```

## 6. Demonstrating Cache Usage

We run the pipeline again to demonstrate the use of cache:

```python
print("Segunda ejecución (debería usar datos en caché):")
# Crear el pipeline con Cache
pipeline = F3Pipeline(
    filters=[
        Cached(
            filter=StandardScalerPlugin(),
            cache_data=True,
            cache_filter=True,
            overwrite=False,
        ),
        KnnFilter(),
    ],
    metrics=[F1()],
)
print(pipeline)

pipeline.fit(x_train, y_train)
predictions = pipeline.predict(x_test)
evaluation = pipeline.evaluate(x_test, y_test, predictions)

print("Resultados de la evaluación:", evaluation)
```
```bash
Segunda ejecución (debería usar datos en caché):
F3Pipeline(
    filters=[
        Cached(filter=StandardScalerPlugin(), cache_data=True, cache_filter=True, overwrite=False, storage=None),
        KnnFilter(
            n_neighbors=5,
            weights='uniform',
            algorithm='auto',
            leaf_size=30,
            p=2,
            metric='minkowski',
            metric_params=None,
            n_jobs=None
        )
    ],
    metrics=[F1(average='weighted')],
    overwrite=False,
    store=False,
    log=False
)
____________________________________________________________________________________________________
Fitting pipeline...
****************************************************************************************************
* Cached({'filter': StandardScalerPlugin({}), 'cache_data': True, 'cache_filter': True, 'overwrite': False,
'storage': None}):
         - El filtro StandardScalerPlugin({}) Existe, se crea lambda.
         - El dato XYData(_hash='f77f9d95466939988cdd6a13f0cb91260b94c99d', _path='StandardScalerPlugin/4f0150e0e11419085ce0f08ab077b7e5891f817b') Existe, se crea lambda.
* KnnFilter({'n_neighbors': 5, 'weights': 'uniform', 'algorithm': 'auto', 'leaf_size': 30, 'p': 2, 'metric': 'minkowski', 'metric_params': None, 'n_jobs': None}):
	 * Downloading: <_io.BufferedReader name='cache/StandardScalerPlugin/4f0150e0e11419085ce0f08ab077b7e5891f817b/f77f9d95466939988cdd6a13f0cb91260b94c99d'>
____________________________________________________________________________________________________
Predicting pipeline...
****************************************************************************************************
* Cached({'filter': StandardScalerPlugin({}), 'cache_data': True, 'cache_filter': True, 'overwrite': False, 'storage': None})
         - El dato XYData(_hash='a2131ee7caeb76fd704b11b77d0456223b7e0437', _path='StandardScalerPlugin/4f0150e0e11419085ce0f08ab077b7e5891f817b') No existe, se va a crear.
         - Existe un Lambda por lo que se recupera el filtro del storage.
	 * Downloading: <_io.BufferedReader name='cache/StandardScalerPlugin/4f0150e0e11419085ce0f08ab077b7e5891f817b/model'>
         - El dato XYData(_hash='a2131ee7caeb76fd704b11b77d0456223b7e0437', _path='StandardScalerPlugin/4f0150e0e11419085ce0f08ab077b7e5891f817b') Se cachea.
	 * Saving in local path: cache/StandardScalerPlugin/4f0150e0e11419085ce0f08ab077b7e5891f817b/a2131ee7caeb76fd704b11b77d0456223b7e0437
	 * Saved !
* KnnFilter({'n_neighbors': 5, 'weights': 'uniform', 'algorithm': 'auto', 'leaf_size': 30, 'p': 2, 'metric': 'minkowski', 'metric_params': None, 'n_jobs': None})
____________________________________________________________________________________________________
Evaluating pipeline...
____________________________________________________________________________________________________
Resultados de la evaluación:
{'F1': 1.0}

```

In this second execution, you should notice that the `StandardScalerPlugin` uses the cached data, which may result in faster execution time.

## Conclusion

This tutorial has demonstrated how to use F3Pipeline with the Cache functionality in labchain. We've seen how to create a pipeline that includes a cached filter, how to train the model, make predictions, and evaluate performance. We've also shown how caching can improve performance in subsequent runs.
