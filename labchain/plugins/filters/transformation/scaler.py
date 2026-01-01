from typing import Optional
from sklearn.preprocessing import StandardScaler
from labchain.base import BaseMetric
from labchain.base.base_types import XYData
from labchain.base.base_clases import BaseFilter
from labchain.container.container import Container


__all__ = ["StandardScalerPlugin"]


@Container.bind()
class StandardScalerPlugin(BaseFilter):
    """
    A plugin for standardizing features by removing the mean and scaling to unit variance.

    This plugin integrates scikit-learn's StandardScaler into the framework3 ecosystem,
    allowing for easy feature standardization within pipelines.

    Key Features:
        - Utilizes scikit-learn's StandardScaler for feature standardization
        - Removes the mean and scales features to unit variance
        - Provides methods for fitting the scaler and transforming data
        - Integrates seamlessly with framework3's BaseFilter interface

    Usage:
        The StandardScalerPlugin can be used to standardize features in your data:

        ```python
        from framework3.plugins.filters.transformation.scaler import StandardScalerPlugin
        from framework3.base.base_types import XYData
        import numpy as np

        # Create a StandardScalerPlugin instance
        scaler_plugin = StandardScalerPlugin()

        # Create some sample data
        X = XYData.mock(np.array([[0, 0], [0, 0], [1, 1], [1, 1]]))
        y = None  # StandardScaler doesn't use y for fitting

        # Fit the StandardScaler
        scaler_plugin.fit(X, y)

        # Transform new data
        new_data = XYData.mock(np.array([[2, 2], [-1, -1]]))
        scaled_data = scaler_plugin.predict(new_data)
        print(scaled_data.value)
        # Output will be standardized, with mean 0 and unit variance
        # For example: [[ 1.41421356  1.41421356]
        #               [-1.41421356 -1.41421356]]
        ```

    Attributes:
        _scaler (StandardScaler): The underlying scikit-learn StandardScaler object used for standardization.

    Methods:
        fit(x: XYData, y: Optional[XYData], evaluator: BaseMetric | None = None) -> Optional[float]:
            Fit the StandardScaler to the given data.
        predict(x: XYData) -> XYData:
            Perform standardization on the input data.

    Note:
        This plugin uses scikit-learn's implementation of StandardScaler, which may have its own
        dependencies and requirements. Ensure that scikit-learn is properly installed and compatible
        with your environment.
    """

    def __init__(self):
        """
        Initialize a new StandardScalerPlugin instance.

        This constructor sets up the StandardScalerPlugin and initializes the underlying
        scikit-learn StandardScaler object.

        Note:
            No parameters are required for initialization as StandardScaler uses default settings.
            For customized scaling, consider extending this class and modifying the StandardScaler initialization.
        """
        super().__init__()  # Call the BaseFilter constructor to initialize the plugin's parameters
        self._scaler = StandardScaler()

    def fit(
        self, x: XYData, y: Optional[XYData], evaluator: BaseMetric | None = None
    ) -> Optional[float]:
        """
        Fit the StandardScaler to the given data.

        This method computes the mean and standard deviation of the input features,
        which will be used for subsequent scaling operations.

        Args:
            x (XYData): The input features to fit the StandardScaler.
            y (Optional[XYData]): Not used in StandardScaler, but required by the BaseFilter interface.
            evaluator (BaseMetric | None): An optional evaluator for the model. Not used in this method.

        Returns:
            Optional[float]: Always returns None as StandardScaler doesn't have a standard evaluation metric.

        Note:
            This method uses scikit-learn's fit method internally.
            The y parameter is ignored as StandardScaler is an unsupervised method.
        """
        self._scaler.fit(x.value)
        return None  # StandardScaler doesn't use y for fitting

    def predict(self, x: XYData) -> XYData:
        """
        Perform standardization on the input data.

        This method applies the standardization transformation to new input data,
        centering and scaling the features based on the computed mean and standard deviation.

        Args:
            x (XYData): The input features to standardize.

        Returns:
            XYData: The standardized version of the input data, wrapped in an XYData object.

        Note:
            This method uses scikit-learn's transform method internally.
            The transformed data is wrapped in an XYData object for consistency with the framework.
        """
        return XYData.mock(self._scaler.transform(x.value))
