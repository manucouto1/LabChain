from labchain.base import BaseMetric
from labchain.base.base_clases import BaseFilter
from labchain.container.container import Container
from labchain.base.base_types import XYData
from sklearn.neighbors import KNeighborsClassifier
from typing import List, Literal, Optional, Dict, Any

__all__ = ["KnnFilter"]


@Container.bind()
class KnnFilter(BaseFilter):
    """
    A wrapper for scikit-learn's KNeighborsClassifier using the framework3 BaseFilter interface.

    This filter implements the K-Nearest Neighbors algorithm for classification within the framework3 ecosystem.

    Key Features:
        - Integrates scikit-learn's KNeighborsClassifier with framework3
        - Supports various KNN parameters like number of neighbors, weights, and distance metrics
        - Provides methods for fitting the model and making predictions
        - Includes a static method for generating parameter grids for hyperparameter tuning

    Usage:
        The KnnFilter can be used to perform K-Nearest Neighbors classification on your data:

        ```python
        from framework3.plugins.filters.classification.knn import KnnFilter
        from framework3.base.base_types import XYData
        import numpy as np

        # Create sample data
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        y = np.array([0, 0, 1, 1])
        X_data = XYData(_hash='X_data', _path='/tmp', _value=X)
        y_data = XYData(_hash='y_data', _path='/tmp', _value=y)

        # Create and fit the KNN filter
        knn = KnnFilter(n_neighbors=3, weights='uniform')
        knn.fit(X_data, y_data)

        # Make predictions
        X_test = XYData(_hash='X_test', _path='/tmp', _value=np.array([[2.5, 3.5]]))
        predictions = knn.predict(X_test)
        print(predictions.value)
        ```

    Attributes:
        _clf (KNeighborsClassifier): The underlying scikit-learn KNN classifier.

    Methods:
        fit(x: XYData, y: Optional[XYData], evaluator: BaseMetric | None = None) -> Optional[float]:
            Fit the KNN model to the given data.
        predict(x: XYData) -> XYData:
            Make predictions using the fitted KNN model.
        item_grid(**kwargs) -> tuple[type[BaseFilter], Dict[str, List[Any]]]:
            Generate a parameter grid for hyperparameter tuning.

    Note:
        This filter uses scikit-learn's implementation of KNN, which may have its own dependencies and requirements.
        Ensure that scikit-learn is properly installed and compatible with your environment.
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        weights: Literal["uniform", "distance"] = "uniform",
        algorithm: Literal["auto", "ball_tree", "kd_tree", "brute"] = "auto",
        leaf_size: int = 30,
        p: int = 2,
        metric: str = "minkowski",
        metric_params: Optional[Dict[str, Any]] = None,
        n_jobs: Optional[int] = None,
    ):
        """
        Initialize a new KnnFilter instance.

        This constructor sets up the KnnFilter with the specified parameters and
        initializes the underlying scikit-learn KNeighborsClassifier.

        Args:
            n_neighbors (int): Number of neighbors to use for knn. Defaults to 5.
            weights (Literal["uniform", "distance"]): Weight function used in prediction. Defaults to "uniform".
            algorithm (Literal["auto", "ball_tree", "kd_tree", "brute"]): Algorithm used to compute nearest neighbors. Defaults to "auto".
            leaf_size (int): Leaf size passed to BallTree or KDTree. Defaults to 30.
            p (int): Power parameter for the Minkowski metric. Defaults to 2 (Euclidean distance).
            metric (str): The distance metric to use for the tree. Defaults to "minkowski".
            metric_params (Optional[Dict[str, Any]]): Additional keyword arguments for the metric function. Defaults to None.
            n_jobs (Optional[int]): The number of parallel jobs to run for neighbors search. Defaults to None.

        Note:
            The parameters are passed directly to scikit-learn's KNeighborsClassifier.
            Refer to scikit-learn's documentation for detailed information on these parameters.
        """
        super().__init__(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p,
            metric=metric,
            metric_params=metric_params,
            n_jobs=n_jobs,
        )
        self._clf = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p,
            metric=metric,
            metric_params=metric_params,
            n_jobs=n_jobs,
        )

    def fit(
        self, x: XYData, y: Optional[XYData], evaluator: BaseMetric | None = None
    ) -> Optional[float]:
        """
        Fit the KNN model to the given data.

        This method trains the KNN classifier on the provided input features and target values.

        Args:
            x (XYData): The input features for training.
            y (Optional[XYData]): The target values for training.
            evaluator (BaseMetric | None): An optional evaluator for the model. Not used in this method.

        Returns:
            Optional[float]: The score of the fitted model on the training data.

        Note:
            This method uses scikit-learn's fit method internally.
            The score is calculated using scikit-learn's score method, which computes the mean accuracy.
        """
        self._clf.fit(x.value, y.value)  # type: ignore
        return self._clf.score(x.value, y.value)  # type: ignore

    def predict(self, x: XYData) -> XYData:
        """
        Make predictions using the fitted KNN model.

        This method uses the trained KNN classifier to make predictions on new input data.

        Args:
            x (XYData): The input features to predict.

        Returns:
            XYData: The predicted values wrapped in an XYData object.

        Note:
            This method uses scikit-learn's predict method internally.
            The predictions are wrapped in an XYData object for consistency with the framework.
        """
        predictions = self._clf.predict(x.value)
        return XYData.mock(predictions)

    @staticmethod
    def item_grid(
        **kwargs: Dict[str, List[Any]],
    ) -> tuple[type[BaseFilter], Dict[str, List[Any]]]:
        """
        Generate a parameter grid for hyperparameter tuning.

        This static method provides a way to generate a grid of parameters for use in
        hyperparameter optimization techniques like grid search.

        Args:
            **kwargs (Dict[str, List[Any]]): Keyword arguments to override default parameter ranges.

        Returns:
            tuple[type[BaseFilter], Dict[str, List[Any]]]: A tuple containing the KnnFilter class
            and a dictionary of parameter names and their possible values.

        Note:
            The returned dictionary can be used directly with hyperparameter tuning tools
            that accept parameter grids, such as scikit-learn's GridSearchCV.
        """

        return KnnFilter, kwargs  # type: ignore
