from typing import Any, Dict, List, Literal, Optional
from labchain.base import BaseMetric
from labchain.base.base_types import XYData
from labchain.base.base_clases import BaseFilter, BasePlugin
from labchain.container.container import Container
from sklearn.svm import SVC

L = Literal["linear", "poly", "rbf", "sigmoid"]
__all__ = ["ClassifierSVMPlugin"]


@Container.bind()
class ClassifierSVMPlugin(BaseFilter, BasePlugin):
    """
    A plugin for Support Vector Machine (SVM) classification using scikit-learn's SVC.

    This plugin integrates the SVC (Support Vector Classification) implementation from scikit-learn
    into the framework3 ecosystem, allowing for seamless use of SVM classification in pipelines
    and supporting hyperparameter tuning through grid search.

    Key Features:
        - Wraps scikit-learn's SVC for use within framework3
        - Supports various kernel types: linear, polynomial, RBF, and sigmoid
        - Allows customization of regularization parameter (C) and kernel coefficient (gamma)
        - Provides methods for fitting the model, making predictions, and generating parameter grids

    Usage:
        The ClassifierSVMPlugin can be used to perform SVM classification on your data:

        ```python
        from framework3.base.base_types import XYData
        import numpy as np

        # Create sample data
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        y = np.array([0, 0, 1, 1])
        X_data = XYData(_hash='X_data', _path='/tmp', _value=X)
        y_data = XYData(_hash='y_data', _path='/tmp', _value=y)

        # Create and fit the SVM classifier
        svm_plugin = ClassifierSVMPlugin(C=1.0, kernel='rbf', gamma='scale')
        svm_plugin.fit(X_data, y_data)

        # Make predictions
        X_test = XYData(_hash='X_test', _path='/tmp', _value=np.array([[2.5, 3.5]]))
        predictions = svm_plugin.predict(X_test)
        print(predictions.value)

        # Generate parameter grid for hyperparameter tuning
        grid_params = ClassifierSVMPlugin.item_grid(C=[0.1, 1, 10], kernel=['linear', 'rbf'], gamma=['scale', 'auto'])
        print(grid_params)
        ```

    Attributes:
        _model (SVC): The underlying scikit-learn SVC model.

    Methods:
        fit(x: XYData, y: Optional[XYData], evaluator: BaseMetric | None = None) -> Optional[float]:
            Fit the SVM model to the given data.
        predict(x: XYData) -> XYData:
            Make predictions using the fitted SVM model.
        item_grid(C: List[float], kernel: List[L], gamma: List[float | Literal['scale', 'auto']]) -> Dict[str, List[Any]]:
            Generate a parameter grid for hyperparameter tuning.

    Note:
        This plugin uses scikit-learn's implementation of SVM, which may have its own dependencies and requirements.
        Ensure that scikit-learn is properly installed and compatible with your environment.
    """

    def __init__(
        self,
        C: float = 1.0,
        gamma: float | Literal["scale", "auto"] = "scale",
        kernel: L = "linear",
    ) -> None:
        """
        Initialize a new ClassifierSVMPlugin instance.

        This constructor sets up the ClassifierSVMPlugin with the specified parameters and
        initializes the underlying scikit-learn SVC model.

        Args:
            C (float): Regularization parameter. Defaults to 1.0.
            gamma (float | Literal["scale", "auto"]): Kernel coefficient. Defaults to "scale".
            kernel (L): Specifies the kernel type to be used in the algorithm.
                        Can be 'linear', 'poly', 'rbf', or 'sigmoid'. Defaults to "linear".

        Note:
            The parameters are passed directly to scikit-learn's SVC.
            Refer to scikit-learn's documentation for detailed information on these parameters.
        """
        super().__init__(C=C, kernel=kernel, gamma=gamma)
        self._model = SVC(C=C, kernel=kernel, gamma=gamma)

    def fit(
        self, x: XYData, y: Optional[XYData], evaluator: BaseMetric | None = None
    ) -> Optional[float]:
        """
        Fit the SVM model to the given data.

        This method trains the SVM classifier on the provided input features and target values.

        Args:
            x (XYData): The input features for training.
            y (Optional[XYData]): The target values for training.
            evaluator (BaseMetric | None): An optional evaluator for the model. Not used in this method.

        Returns:
            Optional[float]: The score of the fitted model on the training data, or None if y is None.

        Note:
            This method uses scikit-learn's fit method internally.
            The score is calculated using scikit-learn's score method, which computes the mean accuracy.
        """
        if y is not None:
            self._model.fit(x.value, y.value)  # type: ignore
            return self._model.score(x.value, y.value)  # type: ignore
        return None

    def predict(self, x: XYData) -> XYData:
        """
        Make predictions using the fitted SVM model.

        This method uses the trained SVM classifier to make predictions on new input data.

        Args:
            x (XYData): The input features to predict.

        Returns:
            (XYData): The predicted values wrapped in an XYData object.

        Note:
            This method uses scikit-learn's predict method internally.
            The predictions are wrapped in an XYData object for consistency with the framework.
        """
        return XYData.mock(self._model.predict(x.value))

    @staticmethod
    def item_grid(
        C: List[float],
        kernel: List[L],
        gamma: List[float] | List[Literal["scale", "auto"]] = ["scale"],  # type: ignore[assignment]
    ) -> Dict[str, List[Any]]:
        """
        Generate a parameter grid for hyperparameter tuning.

        This static method provides a way to generate a grid of parameters for use in
        hyperparameter optimization techniques like grid search.

        Args:
            C (List[float]): List of regularization parameter values to try.
            kernel (List[L]): List of kernel types to try.
            gamma (List[float] | List[Literal['scale', 'auto']]): List of gamma values to try. Defaults to ["scale"].

        Returns:
            Dict[str, List[Any]]: A dictionary of parameter names and their possible values.

        Note:
            The returned dictionary can be used directly with hyperparameter tuning tools
            that accept parameter grids, such as scikit-learn's GridSearchCV.
        """
        return {
            "ClassifierSVMPlugin__C": C,
            "ClassifierSVMPlugin__kernel": kernel,
            "ClassifierSVMPlugin__gamma": gamma,
        }
