from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score,
    silhouette_score,
    calinski_harabasz_score,
    homogeneity_score,
    completeness_score,
)
from labchain.base.base_types import Float
from labchain.base.base_types import XYData
from labchain.base.base_clases import BaseMetric
from labchain.container.container import Container
from typing import Any, Dict, Unpack

import numpy as np

from labchain.plugins.metrics.utils.types import NMIClustKwargs, SilhouetteKwargs

__all__ = [
    "NMI",
    "ARI",
    "Silhouette",
    "CalinskiHarabasz",
    "Homogeneity",
    "Completeness",
]


@Container.bind()
class NMI(BaseMetric):
    """
    Normalized Mutual Information (NMI) metric for clustering evaluation.

    This class calculates the NMI score, which is a normalization of the Mutual Information (MI) score
    to scale the results between 0 (no mutual information) and 1 (perfect correlation).

    Key Features:
        - Measures the agreement of the true labels and predicted clusters, ignoring permutations
        - Normalized to output values between 0 and 1
        - Suitable for comparing clusterings of different sizes

    Usage:
        The NMI metric can be used to evaluate clustering algorithms:

        ```python
        from framework3.plugins.metrics.clustering import NMI
        from framework3.base.base_types import XYData
        import numpy as np

        # Create sample data
        x_data = XYData(value=np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]]))
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 1, 1, 1])

        # Create and use the NMI metric
        nmi_metric = NMI()
        score = nmi_metric.evaluate(x_data, y_true, y_pred)
        print(f"NMI Score: {score}")
        ```

    Methods:
        evaluate(x_data: XYData, y_true: Any, y_pred: Any, **kwargs) -> Float | np.ndarray:
            Calculate the Normalized Mutual Information score.

    Note:
        This metric uses scikit-learn's normalized_mutual_info_score function internally. Ensure that scikit-learn
        is properly installed and compatible with your environment.
    """

    def evaluate(
        self, x_data: XYData, y_true: Any, y_pred: Any, **kwargs: Unpack[NMIClustKwargs]
    ) -> Float | np.ndarray:
        """
        Calculate the Normalized Mutual Information score.

        This method computes the NMI score between two clusterings.

        Args:
            x_data (XYData): The input data (not used in this metric, but required by the interface).
            y_true (Any): The ground truth labels.
            y_pred (Any): The predicted cluster labels.
            **kwargs (Unpack[NMIClustKwargs]): Additional keyword arguments passed to sklearn's normalized_mutual_info_score.

        Returns:
            Float | np.ndarray: The NMI score.

        Note:
            This method uses scikit-learn's normalized_mutual_info_score function internally.
        """
        return normalized_mutual_info_score(y_true, y_pred, **kwargs)


@Container.bind()
class ARI(BaseMetric):
    """
    Adjusted Rand Index (ARI) metric for clustering evaluation.

    This class calculates the ARI score, which is the corrected-for-chance version of the Rand Index.
    It measures similarity between two clusterings, adjusted for chance.

    Key Features:
        - Measures the similarity of the cluster assignments, ignoring permutations and with chance normalization
        - Suitable for comparing clusterings of different sizes
        - Symmetric: switching argument order will not change the score

    Usage:
        The ARI metric can be used to evaluate clustering algorithms:

        ```python
        from framework3.plugins.metrics.clustering import ARI
        from framework3.base.base_types import XYData
        import numpy as np

        # Create sample data
        x_data = XYData(value=np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]]))
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 1, 1, 1])

        # Create and use the ARI metric
        ari_metric = ARI()
        score = ari_metric.evaluate(x_data, y_true, y_pred)
        print(f"ARI Score: {score}")
        ```

    Methods:
        evaluate(x_data: XYData, y_true: Any, y_pred: Any, **kwargs) -> Float | np.ndarray:
            Calculate the Adjusted Rand Index score.

    Note:
        This metric uses scikit-learn's adjusted_rand_score function internally. Ensure that scikit-learn
        is properly installed and compatible with your environment.
    """

    def evaluate(
        self, x_data: XYData, y_true: Any, y_pred: Any, **kwargs: Dict[str, Any]
    ) -> Float | np.ndarray:
        """
        Calculate the Adjusted Rand Index score.

        This method computes the ARI score between two clusterings.

        Args:
            x_data (XYData): The input data (not used in this metric, but required by the interface).
            y_true (Any): The ground truth labels.
            y_pred (Any): The predicted cluster labels.
            **kwargs (Dict[str,Any]): Additional keyword arguments passed to sklearn's adjusted_rand_score.

        Returns:
            Float | np.ndarray: The ARI score.

        Note:
            This method uses scikit-learn's adjusted_rand_score function internally.
        """
        return adjusted_rand_score(y_true, y_pred)


@Container.bind()
class Silhouette(BaseMetric):
    """
    Silhouette Coefficient metric for clustering evaluation.

    This class calculates the Silhouette Coefficient, which is calculated using the mean
    intra-cluster distance and the mean nearest-cluster distance for each sample.

    Key Features:
        - Measures how similar an object is to its own cluster compared to other clusters
        - Ranges from -1 to 1, where higher values indicate better-defined clusters
        - Can be used to determine the optimal number of clusters

    Usage:
        The Silhouette metric can be used to evaluate clustering algorithms:

        ```python
        from framework3.plugins.metrics.clustering import Silhouette
        from framework3.base.base_types import XYData
        import numpy as np

        # Create sample data
        x_data = XYData(value=np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]]))
        y_pred = np.array([0, 0, 0, 1, 1, 1])

        # Create and use the Silhouette metric
        silhouette_metric = Silhouette()
        score = silhouette_metric.evaluate(x_data, None, y_pred)
        print(f"Silhouette Score: {score}")
        ```

    Methods:
        evaluate(x_data: XYData, y_true: Any, y_pred: Any, **kwargs) -> Float | np.ndarray:
            Calculate the Silhouette Coefficient.

    Note:
        This metric uses scikit-learn's silhouette_score function internally. Ensure that scikit-learn
        is properly installed and compatible with your environment.
    """

    def evaluate(
        self,
        x_data: XYData,
        y_true: Any,
        y_pred: Any,
        **kwargs: Unpack[SilhouetteKwargs],
    ) -> Float | np.ndarray:
        """
        Calculate the Silhouette Coefficient.

        This method computes the Silhouette Coefficient for each sample.

        Args:
            x_data (XYData): The input data.
            y_true (Any): Not used for this metric, but required by the interface.
            y_pred (Any): The predicted cluster labels.
            **kwargs (Unpack[SilhouetteKwargs]): Additional keyword arguments passed to sklearn's silhouette_score.

        Returns:
            Float | np.ndarray: The Silhouette Coefficient.

        Note:
            This method uses scikit-learn's silhouette_score function internally.
        """
        return silhouette_score(x_data.value, y_pred, **kwargs)


@Container.bind()
class CalinskiHarabasz(BaseMetric):
    """
    Calinski-Harabasz Index metric for clustering evaluation.

    This class calculates the Calinski-Harabasz Index, which is the ratio of the sum of
    between-clusters dispersion and of inter-cluster dispersion for all clusters.

    Key Features:
        - Measures the ratio of between-cluster variance to within-cluster variance
        - Higher values indicate better-defined clusters
        - Can be used to determine the optimal number of clusters

    Usage:
        The Calinski-Harabasz metric can be used to evaluate clustering algorithms:

        ```python
        from framework3.plugins.metrics.clustering import CalinskiHarabasz
        from framework3.base.base_types import XYData
        import numpy as np

        # Create sample data
        x_data = XYData(value=np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]]))
        y_pred = np.array([0, 0, 0, 1, 1, 1])

        # Create and use the Calinski-Harabasz metric
        ch_metric = CalinskiHarabasz()
        score = ch_metric.evaluate(x_data, None, y_pred)
        print(f"Calinski-Harabasz Score: {score}")
        ```

    Methods:
        evaluate(x_data: XYData, y_true: Any, y_pred: Any, **kwargs) -> Float | np.ndarray:
            Calculate the Calinski-Harabasz Index.

    Note:
        This metric uses scikit-learn's calinski_harabasz_score function internally. Ensure that scikit-learn
        is properly installed and compatible with your environment.
    """

    def evaluate(
        self, x_data: XYData, y_true: Any, y_pred: Any, **kwargs: Dict[str, Any]
    ) -> Float | np.ndarray:
        """
        Calculate the Calinski-Harabasz Index.

        This method computes the Calinski-Harabasz Index for the clustering.

        Args:
            x_data (XYData): The input data.
            y_true (Any): Not used for this metric, but required by the interface.
            y_pred (Any): The predicted cluster labels.
            **kwargs (Dict[str,Any]): Additional keyword arguments passed to sklearn's calinski_harabasz_score.

        Returns:
            Float | np.ndarray: The Calinski-Harabasz Index.

        Note:
            This method uses scikit-learn's calinski_harabasz_score function internally.
        """
        return calinski_harabasz_score(x_data.value, y_pred)


@Container.bind()
class Homogeneity(BaseMetric):
    """
    Homogeneity metric for clustering evaluation.

    This class calculates the Homogeneity score, which measures whether all of its clusters
    contain only data points which are members of a single class.

    Key Features:
        - Measures the extent to which each cluster contains only members of a single class
        - Ranges from 0 to 1, where 1 indicates perfectly homogeneous clustering
        - Invariant to label switching

    Usage:
        The Homogeneity metric can be used to evaluate clustering algorithms:

        ```python
        from framework3.plugins.metrics.clustering import Homogeneity
        from framework3.base.base_types import XYData
        import numpy as np

        # Create sample data
        x_data = XYData(value=np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]]))
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 1, 1, 1])

        # Create and use the Homogeneity metric
        homogeneity_metric = Homogeneity()
        score = homogeneity_metric.evaluate(x_data, y_true, y_pred)
        print(f"Homogeneity Score: {score}")
        ```

    Methods:
        evaluate(x_data: XYData, y_true: Any, y_pred: Any, **kwargs) -> Float | np.ndarray:
            Calculate the Homogeneity score.

    Note:
        This metric uses scikit-learn's homogeneity_score function internally. Ensure that scikit-learn
        is properly installed and compatible with your environment.
    """

    def evaluate(
        self, x_data: XYData, y_true: Any, y_pred: Any, **kwargs: Dict[str, Any]
    ) -> Float | np.ndarray:
        """
        Calculate the Homogeneity score.

        This method computes the Homogeneity score for the clustering.

        Args:
            x_data (XYData): The input data (not used in this metric, but required by the interface).
            y_true (Any): The ground truth labels.
            y_pred (Any): The predicted cluster labels.
            **kwargs (Dict[str,Any]): Additional keyword arguments passed to sklearn's homogeneity_score.

        Returns:
            Float | np.ndarray: The Homogeneity score.

        Note:
            This method uses scikit-learn's homogeneity_score function internally.
        """
        return homogeneity_score(y_true, y_pred)


@Container.bind()
class Completeness(BaseMetric):
    """
    Completeness metric for clustering evaluation.

    This class calculates the Completeness score, which measures whether all members of
    a given class are assigned to the same cluster.

    Key Features:
        - Measures the extent to which all members of a given class are assigned to the same cluster
        - Ranges from 0 to 1, where 1 indicates perfectly complete clustering
        - Invariant to label switching

    Usage:
        The Completeness metric can be used to evaluate clustering algorithms:

        ```python
        from framework3.plugins.metrics.clustering import Completeness
        from framework3.base.base_types import XYData
        import numpy as np

        # Create sample data
        x_data = XYData(value=np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]]))
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 1, 1, 1])

        # Create and use the Completeness metric
        completeness_metric = Completeness()
        score = completeness_metric.evaluate(x_data, y_true, y_pred)
        print(f"Completeness Score: {score}")
        ```

    Methods:
        evaluate (x_data: XYData, y_true: Any, y_pred: Any, **kwargs) -> Float | np.ndarray:
            Calculate the Completeness score.

    Note:
        This metric uses scikit-learn's completeness_score function internally. Ensure that scikit-learn
        is properly installed and compatible with your environment.
    """

    def evaluate(
        self, x_data: XYData, y_true: Any, y_pred: Any, **kwargs: Dict[str, Any]
    ) -> Float | np.ndarray:
        """
        Calculate the Completeness score.

        This method computes the Completeness score for the clustering.

        Args:
            x_data (XYData): The input data (not used in this metric, but required by the interface).
            y_true (Any): The ground truth labels.
            y_pred (Any): The predicted cluster labels.
            **kwargs (Dict[str,Any]): Additional keyword arguments passed to sklearn's completeness_score.

        Returns:
            Float | np.ndarray: The Completeness score.

        Note:
            This method uses scikit-learn's completeness_score function internally.
        """
        return completeness_score(y_true, y_pred)
