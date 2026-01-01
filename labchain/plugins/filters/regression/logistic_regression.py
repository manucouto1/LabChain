from typing import Optional
from labchain.base import BaseFilter, BaseMetric, BasePlugin, XYData
from labchain.container.container import Container
from sklearn.linear_model import LogisticRegression

__all__ = ["LogistiRegressionlugin"]

Container.bind()


class LogistiRegressionlugin(BaseFilter, BasePlugin):
    """
    A plugin that implements logistic regression using scikit-learn's LogisticRegression.

    This plugin wraps the LogisticRegression model from scikit-learn and adapts it
    to work within the framework3 ecosystem, providing a seamless integration for
    logistic regression tasks.

    Key Features:
        - Utilizes scikit-learn's LogisticRegression implementation
        - Supports customization of maximum iterations and tolerance
        - Provides methods for fitting the model and making predictions
        - Integrates with framework3's BaseFilter and BasePlugin interfaces

    Usage:
        The LogistiRegressionlugin can be used to perform logistic regression on your data:

        ```python
        import numpy as np
        from framework3.base import XYData
        from framework3.plugins.filters.regression.logistic_regression import LogistiRegressionlugin

        # Create sample data
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        y = np.array([0, 0, 1, 1])
        X_data = XYData(_hash='X_data', _path='/tmp', _value=X)
        y_data = XYData(_hash='y_data', _path='/tmp', _value=y)

        # Create and fit the LogistiRegressionlugin
        log_reg = LogistiRegressionlugin(max_ite=100, tol=1e-4)
        log_reg.fit(X_data, y_data)

        # Make predictions
        X_test = XYData(_hash='X_test', _path='/tmp', _value=np.array([[2.5, 3.5]]))
        predictions = log_reg.predict(X_test)
        print(predictions.value)

        # Access the underlying scikit-learn model
        print(log_reg._logistic.coef_)
        ```

    Attributes:
        _logistic (LogisticRegression): The underlying scikit-learn LogisticRegression model.

    Methods:
        fit(x: XYData, y: Optional[XYData], evaluator: BaseMetric | None = None) -> Optional[float]:
            Fit the logistic regression model to the given data.
        predict(x: XYData) -> XYData:
            Make predictions using the fitted logistic regression model.

    Note:
        This plugin uses scikit-learn's implementation of LogisticRegression, which may have its own
        dependencies and requirements. Ensure that scikit-learn is properly installed and compatible
        with your environment.
    """

    def __init__(self, max_ite: int, tol: float):
        """
        Initialize a new LogistiRegressionlugin instance.

        This constructor sets up the LogistiRegressionlugin with the specified parameters
        and initializes the underlying scikit-learn LogisticRegression model.

        Args:
            max_ite (int): Maximum number of iterations for the solver to converge.
            tol (float): Tolerance for stopping criteria.

        Note:
            The parameters are passed directly to scikit-learn's LogisticRegression.
            Refer to scikit-learn's documentation for detailed information on these parameters.
        """
        self._logistic = LogisticRegression(max_iter=max_ite, tol=tol)

    def fit(
        self, x: XYData, y: Optional[XYData], evaluator: BaseMetric | None = None
    ) -> Optional[float]:
        """
        Fit the logistic regression model to the given data.

        This method trains the logistic regression model on the provided input features and target values.

        Args:
            x (XYData): The input features.
            y (Optional[XYData]): The target values.
            evaluator (BaseMetric | None): An optional evaluator for the model. Not used in this method.

        Returns:
            Optional[float]: The mean accuracy on the given test data and labels.

        Raises:
            ValueError: If y is None.

        Note:
            This method uses scikit-learn's fit method internally.
            The score (mean accuracy) is returned as a measure of how well the model fits the data.
        """
        if y is None:
            raise ValueError(
                "Target values (y) cannot be None for logistic regression."
            )
        self._logistic.fit(x._value, y._value)  # type: ignore
        return self._logistic.score(x._value, y._value)  # type: ignore

    def predict(self, x: XYData) -> XYData:
        """
        Make predictions using the fitted logistic regression model.

        This method uses the trained logistic regression model to predict class labels for new input data.

        Args:
            x (XYData): The input features to predict.

        Returns:
            XYData: The predicted class labels wrapped in an XYData object.

        Note:
            This method uses scikit-learn's predict method internally.
            The predictions are wrapped in an XYData object for consistency with the framework.
        """
        return XYData.mock(self._logistic.predict(x.value))
