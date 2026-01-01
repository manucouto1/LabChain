from typing import Any, Dict, List, Type, cast
from labchain.base.base_clases import BaseFilter
from sklearn.base import BaseEstimator

from labchain.base.base_types import XYData
from labchain.base.exceptions import NotTrainableFilterError


class SkWrapper(BaseEstimator):
    """
    A wrapper class for BaseFilter that implements scikit-learn's BaseEstimator interface.

    This class allows BaseFilter objects to be used with scikit-learn's GridSearchCV and other
    scikit-learn compatible tools.

    Key Features:
        - Wraps any BaseFilter subclass to make it compatible with scikit-learn
        - Implements fit, predict, and transform methods
        - Supports getting and setting parameters
        - Handles NotTrainableFilterError for filters that don't require training

    Usage:
        ```python
        from framework3.plugins.filters.classification.svm import ClassifierSVMPlugin
        from framework3.utils.skestimator import SkWrapper
        import numpy as np
        from sklearn.model_selection import GridSearchCV

        # Create a sample BaseFilter
        class SampleFilter(ClassifierSVMPlugin):
            pass

        # Create an instance of SkWrapper
        wrapper = SkWrapper(SampleFilter, C=1.0, kernel='rbf')

        # Use the wrapper with sklearn's GridSearchCV
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        y = np.array([0, 0, 1, 1])
        param_grid = {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']}
        grid_search = GridSearchCV(wrapper, param_grid, cv=3)
        grid_search.fit(X, y)

        # Make predictions
        print(grid_search.predict([[2.5, 3.5]]))
        ```

    Attributes:
        _z_clazz (Type[BaseFilter]): The BaseFilter class to be wrapped.
        _model (BaseFilter): An instance of the wrapped BaseFilter class.
        kwargs (Dict[str, Any]): Keyword arguments passed to the wrapped BaseFilter class.

    Methods:
        get_zclazz() -> str: Get the name of the wrapped BaseFilter class.
        fit(x: Any, y: Any, *args, **kwargs) -> 'SkWrapper': Fit the wrapped model to the given data.
        predict(x: Any) -> Any: Make predictions using the wrapped model.
        transform(x: Any) -> Any: Transform the input data using the wrapped model.
        get_params(deep: bool = True) -> Dict[str, Any]: Get the parameters of the estimator.
        set_params(**parameters) -> 'SkWrapper': Set the parameters of the estimator.
    """

    def __init__(self, z_clazz: type[BaseFilter], **kwargs: Any):
        """
        Initialize the SkWrapper.

        Args:
            z_clazz (Type[BaseFilter]): The BaseFilter class to be wrapped.
            **kwargs: Keyword arguments to be passed to the wrapped BaseFilter class.
        """
        self._z_clazz: type[BaseFilter] = z_clazz
        self._model: BaseFilter = self._z_clazz(**kwargs)  # type: ignore
        self.kwargs = kwargs

    def get_zclazz(self) -> str:
        """
        Get the name of the wrapped BaseFilter class.

        Returns:
            str: The name of the wrapped BaseFilter class.
        """
        return self._z_clazz.__name__

    def fit(self, x, y, *args: List[Any], **kwargs: Dict[str, Any]) -> "SkWrapper":
        """
        Fit the wrapped model to the given data.

        Args:
            x (Any): The input features.
            y (Any): The target values.
            *args (List[Any]): Additional positional arguments.
            **kwargs (Dict[str, Any]): Additional keyword arguments.

        Returns:
            SkWrapper: The fitted estimator.
        """
        try:
            self._model.fit(XYData.mock(x), XYData.mock(y))
        except NotTrainableFilterError:
            self._model.init()

        return self

    def predict(self, x) -> Any:
        """
        Make predictions using the wrapped model.

        Args:
            x (Any): The input features.

        Returns:
            Any: The predicted values.
        """
        return self._model.predict(XYData.mock(x)).value

    def transform(self, x) -> Any:
        """
        Transform the input data using the wrapped model.

        Args:
            x (Any): The input features.

        Returns:
            Any: The transformed data.
        """

        return self._model.predict(XYData.mock(x)).value

    def get_params(self, deep=True) -> Dict[str, Any]:
        """
        Get the parameters of the estimator.

        Args:
            deep (bool): If True, will return the parameters for this estimator and
                         contained subobjects that are estimators.

        Returns:
            Dict[str, Any]: Parameter names mapped to their values.
        """
        return self.kwargs | {"z_clazz": self._z_clazz}

    def set_params(self, **parameters: Any) -> "SkWrapper":
        """
        Set the parameters of the estimator.

        Args:
            **parameters (dict): Estimator parameters.

        Returns:
            (SkWrapper): Estimator instance.
        """
        for param, value in parameters.items():
            if param == "z_clazz":
                if type(value) is Type[BaseFilter]:
                    self._z_clazz = value
                else:
                    raise ValueError("z_clazz must be a subclass of BaseFilter")
            else:
                self.kwargs[param] = value
        self._model = cast(BaseFilter, self._z_clazz(**self.kwargs))
        return self
