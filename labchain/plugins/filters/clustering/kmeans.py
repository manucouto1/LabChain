from labchain.base import BaseMetric
from labchain.base.base_clases import BaseFilter
from labchain.container.container import Container
from labchain.base.base_types import XYData
from sklearn.cluster import KMeans
from typing import Literal, Optional, Dict, Any

__all__ = ["KMeansFilter"]


@Container.bind()
class KMeansFilter(BaseFilter):
    """
    A wrapper for scikit-learn's KMeans clustering algorithm using the framework3 BaseFilter interface.

    This filter implements the K-Means clustering algorithm within the framework3 ecosystem.

    Key Features:
        - Integrates scikit-learn's KMeans with framework3
        - Supports various KMeans parameters like number of clusters, initialization method, and algorithm
        - Provides methods for fitting the model, making predictions, and transforming data
        - Includes a static method for generating parameter grids for hyperparameter tuning

    Usage:
        The KMeansFilter can be used to perform K-Means clustering on your data:

        ```python
        from framework3.plugins.filters.clustering.kmeans import KMeansFilter
        from framework3.base.base_types import XYData
        import numpy as np

        # Create sample data
        X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
        X_data = XYData(_hash='X_data', _path='/tmp', _value=X)

        # Create and fit the KMeans filter
        kmeans = KMeansFilter(n_clusters=2, random_state=42)
        kmeans.fit(X_data)

        # Make predictions
        X_test = XYData(_hash='X_test', _path='/tmp', _value=np.array([[0, 0], [4, 4]]))
        predictions = kmeans.predict(X_test)
        print(predictions.value)
        ```

    Attributes:
        _clf (KMeans): The underlying scikit-learn KMeans clustering model.

    Methods:
        fit(x: XYData, y: Optional[XYData], evaluator: BaseMetric | None = None) -> Optional[float]:
            Fit the KMeans model to the given data.
        predict(x: XYData) -> XYData:
            Predict the closest cluster for each sample in X.
        transform(x: XYData) -> XYData:
            Transform X to a cluster-distance space.
        item_grid(**kwargs) -> Dict[str, Any]:
            Generate a parameter grid for hyperparameter tuning.

    Note:
        This filter uses scikit-learn's implementation of KMeans, which may have its own dependencies and requirements.
        Ensure that scikit-learn is properly installed and compatible with your environment.
    """

    def __init__(
        self,
        n_clusters: int = 8,
        init: Literal["k-means++", "random"] = "k-means++",
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
        algorithm: Literal["lloyd", "elkan"] = "lloyd",
    ):
        """
        Initialize a new KMeansFilter instance.

        This constructor sets up the KMeansFilter with the specified parameters and
        initializes the underlying scikit-learn KMeans model.

        Args:
            n_clusters (int): The number of clusters to form. Defaults to 8.
            init (Literal["k-means++", "random"]): Method for initialization. Defaults to 'k-means++'.
            n_init (int): Number of times the k-means algorithm will be run with different centroid seeds. Defaults to 10.
            max_iter (int): Maximum number of iterations of the k-means algorithm for a single run. Defaults to 300.
            tol (float): Relative tolerance with regards to Frobenius norm of the difference
                         in the cluster centers of two consecutive iterations to declare convergence. Defaults to 1e-4.
            random_state (Optional[int]): Determines random number generation for centroid initialization. Defaults to None.
            algorithm (Literal["lloyd", "elkan"]): K-means algorithm to use. Defaults to 'lloyd'.

        Note:
            The parameters are passed directly to scikit-learn's KMeans.
            Refer to scikit-learn's documentation for detailed information on these parameters.
        """
        super().__init__(
            n_clusters=n_clusters,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
            algorithm=algorithm,
        )
        self._clf = KMeans(
            n_clusters=n_clusters,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
            algorithm=algorithm,
        )

    def fit(
        self, x: XYData, y: Optional[XYData], evaluator: BaseMetric | None = None
    ) -> Optional[float]:
        """
        Fit the KMeans model to the given data.

        This method trains the KMeans model on the provided input features.

        Args:
            x (XYData): The input features for training.
            y (Optional[XYData]): Not used, present for API consistency.
            evaluator (BaseMetric | None): An optional evaluator for the model. Not used in this method.

        Returns:
            Optional[float]: The inertia (within-cluster sum-of-squares) of the fitted model.

        Note:
            This method uses scikit-learn's fit method internally.
            The inertia is returned as a measure of how well the model fits the data.
        """
        self._clf.fit(x.value)
        return self._clf.inertia_  # type: ignore

    def predict(self, x: XYData) -> XYData:
        """
        Predict the closest cluster for each sample in X.

        This method uses the trained KMeans model to predict cluster labels for new input data.

        Args:
            x (XYData): The input features to predict.

        Returns:
            XYData: The predicted cluster labels wrapped in an XYData object.

        Note:
            This method uses scikit-learn's predict method internally.
            The predictions are wrapped in an XYData object for consistency with the framework.
        """
        predictions = self._clf.predict(x.value)
        return XYData.mock(predictions)

    def transform(self, x: XYData) -> XYData:
        """
        Transform X to a cluster-distance space.

        This method computes the distance between each sample in X and the cluster centers.

        Args:
            x (XYData): The input features to transform.

        Returns:
            XYData: The transformed data wrapped in an XYData object.

        Note:
            This method uses scikit-learn's transform method internally.
            The transformed data is wrapped in an XYData object for consistency with the framework.
        """
        transformed = self._clf.transform(x.value)
        return XYData.mock(transformed)

    @staticmethod
    def item_grid(**kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a parameter grid for hyperparameter tuning.

        This static method provides a way to generate a grid of parameters for use in
        hyperparameter optimization techniques like grid search.

        Args:
            **kwargs (Dict[str, Any]): Keyword arguments representing the parameter names and their possible values.

        Returns:
            Dict[str, Any]: A dictionary of parameter names and their possible values.

        Note:
            The returned dictionary can be used directly with hyperparameter tuning tools
            that accept parameter grids, such as scikit-learn's GridSearchCV.
            The parameter names are prefixed with "KMeansFilter__" for compatibility with nested estimators.
        """

        return dict(map(lambda x: (f"KMeansFilter__{x[0]}", x[1]), kwargs.items()))
