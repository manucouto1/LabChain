from typing import Any, Dict, List, Optional
from labchain.base import BaseMetric
from labchain.base.base_types import XYData
from labchain.base.base_clases import BaseFilter
from labchain.container.container import Container
from sklearn.decomposition import PCA

__all__ = ["PCAPlugin"]


@Container.bind()
class PCAPlugin(BaseFilter):
    """
    A plugin for performing Principal Component Analysis (PCA) on input data.

    This plugin integrates scikit-learn's PCA implementation into the framework3 ecosystem,
    allowing for easy dimensionality reduction within pipelines.

    Key Features:
        - Utilizes scikit-learn's PCA for dimensionality reduction
        - Supports customization of the number of components to keep
        - Provides methods for fitting the PCA model and transforming data
        - Integrates seamlessly with framework3's BaseFilter interface
        - Includes a static method for generating parameter grids for hyperparameter tuning

    Usage:
        The PCAPlugin can be used to perform dimensionality reduction on your data:

        ```python
        from framework3.plugins.filters.transformation.pca import PCAPlugin
        from framework3.base.base_types import XYData
        import numpy as np

        # Create a PCAPlugin instance
        pca_plugin = PCAPlugin(n_components=2)

        # Create some sample data
        X = XYData.mock(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
        y = None  # PCA doesn't use y for fitting

        # Fit the PCA model
        pca_plugin.fit(X, y)

        # Transform new data
        new_data = XYData.mock(np.array([[2, 3, 4], [5, 6, 7]]))
        transformed_data = pca_plugin.predict(new_data)
        print(transformed_data.value)  # This will be a 2x2 array
        ```

    Attributes:
        _pca (PCA): The underlying scikit-learn PCA object used for dimensionality reduction.

    Methods:
        fit(x: XYData, y: Optional[XYData], evaluator: BaseMetric | None = None) -> Optional[float]:
            Fit the PCA model to the given data.
        predict(x: XYData) -> XYData:
            Apply dimensionality reduction to the input data.
        item_grid(n_components: List[int]) -> Dict[str, Any]:
            Generate a parameter grid for hyperparameter tuning.

    Note:
        This plugin uses scikit-learn's implementation of PCA, which may have its own
        dependencies and requirements. Ensure that scikit-learn is properly installed
        and compatible with your environment.
    """

    def __init__(self, n_components: int = 2):
        """
        Initialize a new PCAPlugin instance.

        This constructor sets up the PCAPlugin with the specified number of components
        and initializes the underlying scikit-learn PCA object.

        Args:
            n_components (int): The number of components to keep after dimensionality reduction.
                                Defaults to 2.

        Note:
            The n_components parameter is passed directly to scikit-learn's PCA.
            Refer to scikit-learn's documentation for detailed information on this parameter.
        """
        super().__init__(
            n_components=n_components
        )  # Initialize the BaseFilter and BasePlugin parent classes.
        self._pca = PCA(n_components=n_components)

    def fit(
        self, x: XYData, y: Optional[XYData], evaluator: BaseMetric | None = None
    ) -> Optional[float]:
        """
        Fit the PCA model to the given data.

        This method trains the PCA model on the provided input features.

        Args:
            x (XYData): The input features to fit the PCA model.
            y (Optional[XYData]): Not used in PCA, but required by the BaseFilter interface.
            evaluator (BaseMetric | None): An optional evaluator for the model. Not used in this method.

        Returns:
            Optional[float]: Always returns None as PCA doesn't have a standard evaluation metric.

        Note:
            This method uses scikit-learn's fit method internally.
            The y parameter is ignored as PCA is an unsupervised method.
        """
        self._pca.fit(x.value)
        return None

    def predict(self, x: XYData) -> XYData:
        """
        Apply dimensionality reduction to the input data.

        This method uses the trained PCA model to transform new input data,
        reducing its dimensionality.

        Args:
            x (XYData): The input features to transform.

        Returns:
            XYData: The transformed data with reduced dimensionality, wrapped in an XYData object.

        Note:
            This method uses scikit-learn's transform method internally.
            The transformed data is wrapped in an XYData object for consistency with the framework.
        """
        return XYData.mock(self._pca.transform(x.value))

    @staticmethod
    def item_grid(n_components: List[int]) -> Dict[str, Any]:
        """
        Generate a parameter grid for hyperparameter tuning.

        This static method creates a dictionary that can be used for grid search
        over different numbers of components in PCA.

        Args:
            n_components (List[int]): A list of integers representing different numbers
                                      of components to try in the grid search.

        Returns:
            Dict[str, Any]: A dictionary with the parameter name as key and the list of
                            values to try as value.

        Note:
            This method is typically used in conjunction with hyperparameter tuning
            techniques like GridSearchCV.
        """
        return {"PCAPlugin__n_components": n_components}
