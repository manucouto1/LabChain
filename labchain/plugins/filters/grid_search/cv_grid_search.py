from typing import Any, Dict, List, Type

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from labchain.base.base_types import TxyData, XYData
from labchain.base.base_clases import BaseFilter
from labchain.container.container import Container
from sklearn.model_selection import GridSearchCV

from labchain.utils.skestimator import SkWrapper

__all__ = ["GridSearchCVPlugin"]


class SkFilterWrapper(BaseEstimator):
    """
    A wrapper class for BaseFilter that implements scikit-learn's BaseEstimator interface.

    This class enables BaseFilter objects to be used with scikit-learn's GridSearchCV,
    bridging the gap between framework3's filters and scikit-learn's estimator interface.

    Key Features:
        - Wraps BaseFilter objects to comply with scikit-learn's BaseEstimator interface
        - Allows use of framework3 filters in scikit-learn's GridSearchCV
        - Provides methods for fitting, predicting, and parameter management

    Usage:
        The SkFilterWrapper can be used to wrap a BaseFilter for use with GridSearchCV:

        ```python
        from framework3.plugins.filters.clasification.svm import ClassifierSVMPlugin
        import numpy as np

        # Set the class to be wrapped
        SkFilterWrapper.z_clazz = ClassifierSVMPlugin

        # Create an instance of SkFilterWrapper
        wrapper = SkFilterWrapper(C=1.0, kernel='rbf')

        # Use the wrapper with sklearn's GridSearchCV
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        y = np.array([0, 0, 1, 1])
        wrapper.fit(X, y)
        print(wrapper.predict([[2.5, 3.5]]))
        ```

    Attributes:
        z_clazz (Type[BaseFilter]): The BaseFilter class to be wrapped.
        _model (BaseFilter): The instance of the wrapped BaseFilter.
        kwargs (Dict[str, Any]): The keyword arguments used to initialize the wrapped BaseFilter.

    Methods:
        fit(x, y, *args, **kwargs): Fit the wrapped model to the given data.
        predict(x): Make predictions using the wrapped model.
        get_params(deep=True): Get the parameters of the estimator.
        set_params(**parameters): Set the parameters of the estimator.

    Note:
        This wrapper is specifically designed to work with framework3's BaseFilter and
        scikit-learn's GridSearchCV. Ensure that the wrapped BaseFilter is compatible
        with the data and operations you intend to perform.
    """

    z_clazz: Type[BaseFilter]

    def __init__(self, clazz, **kwargs: Dict[str, Any]):
        """
        Initialize a new SkFilterWrapper instance.

        This constructor creates an instance of the specified BaseFilter class
        with the given parameters.

        Args:
            clazz (Type[BaseFilter]): The BaseFilter class to be instantiated.
            **kwargs (Dict[str, Any]): Keyword arguments to be passed to the BaseFilter constructor.

        Note:
            The initialized BaseFilter instance is stored in self._model, and
            the kwargs are stored for later use in get_params and set_params.
        """
        self._model = clazz(**kwargs)
        self.kwargs = kwargs

    def fit(
        self, x: TxyData, y: TxyData, *args: List[Any], **kwargs: Dict[str, Any]
    ) -> "SkFilterWrapper":
        """
        Fit the wrapped model to the given data.

        This method wraps the input data in XYData objects and calls the fit method
        of the wrapped BaseFilter.

        Args:
            x (TxyData): The input features.
            y (TxyData): The target values.
            *args (List[Any]): Additional positional arguments (not used).
            **kwargs (Dict[str,Any]): Additional keyword arguments (not used).

        Returns:
            self (SkFilterWrapper): The fitted estimator.

        Note:
            This method modifies the internal state of the wrapped model.
        """
        self._model.fit(XYData.mock(x), XYData.mock(y))
        return self

    def predict(self, x: TxyData) -> TxyData:
        """
        Make predictions using the wrapped model.

        This method wraps the input data in an XYData object, calls the predict method
        of the wrapped BaseFilter, and returns the raw value from the resulting XYData.

        Args:
            x (TxyData): The input features.

        Returns:
            (TxyData): The predicted values.

        Note:
            The return value is the raw prediction, not wrapped in an XYData object.
        """
        return self._model.predict(XYData.mock(x)).value

    def get_params(self, deep=True) -> Dict[str, Any]:
        """
        Get the parameters of the estimator.

        This method returns the kwargs used to initialize the wrapped BaseFilter.

        Args:
            deep (bool): If True, will return the parameters for this estimator and
                         contained subobjects that are estimators. Not used in this implementation.

        Returns:
            (Dict[str, Any]): Parameter names mapped to their values.

        Note:
            The 'deep' parameter is included for compatibility with scikit-learn,
            but it doesn't affect the output in this implementation.
        """
        return {**self.kwargs}

    def set_params(self, **parameters: Dict[str, Any]) -> "SkFilterWrapper":
        """
        Set the parameters of the estimator.

        This method creates a new instance of the wrapped BaseFilter with the specified parameters.

        Args:
            **parameters (Dict[str,Any]): Estimator parameters.

        Returns:
            self (SkFilterWrapper): Estimator instance.

        Note:
            This method replaces the existing wrapped model with a new instance.
        """
        self._model = SkFilterWrapper.z_clazz(**parameters)  # type: ignore
        return self


@Container.bind()
class GridSearchCVPlugin(BaseFilter):
    """
    A plugin for performing hyperparameter tuning on BaseFilter objects using scikit-learn's GridSearchCV.

    This plugin automates the process of finding optimal hyperparameters for a given BaseFilter
    by evaluating different combinations of parameters through cross-validation.

    Key Features:
        - Integrates scikit-learn's GridSearchCV with framework3's BaseFilter
        - Supports hyperparameter tuning for any BaseFilter
        - Allows specification of parameter grid, scoring metric, and cross-validation strategy
        - Provides methods for fitting the model and making predictions with the best found parameters

    Usage:
        The GridSearchCVPlugin can be used to perform hyperparameter tuning on a BaseFilter:

        ```python
        from framework3.plugins.filters.clasification.svm import ClassifierSVMPlugin
        from framework3.base.base_types import XYData
        import numpy as np

        # Create sample data
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        y = np.array([0, 0, 1, 1])
        X_data = XYData(_hash='X_data', _path='/tmp', _value=X)
        y_data = XYData(_hash='y_data', _path='/tmp', _value=y)

        # Define the parameter grid
        param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        }

        # Create the GridSearchCVPlugin
        grid_search = GridSearchCVPlugin(
            filterx=ClassifierSVMPlugin,
            param_grid=param_grid,
            scoring='accuracy',
            cv=3
        )

        # Fit the grid search
        grid_search.fit(X_data, y_data)

        # Make predictions
        X_test = XYData(_hash='X_test', _path='/tmp', _value=np.array([[2.5, 3.5]]))
        predictions = grid_search.predict(X_test)
        print(predictions.value)

        # Access the best parameters
        print(grid_search._clf.best_params_)
        ```

    Attributes:
        _clf (GridSearchCV): The GridSearchCV object used for hyperparameter tuning.

    Methods:
        fit(x: XYData, y: XYData): Fit the GridSearchCV object to the given data.
        predict(x: XYData) -> XYData: Make predictions using the best estimator found by GridSearchCV.

    Note:
        This plugin uses scikit-learn's GridSearchCV, which may have its own dependencies and requirements.
        Ensure that scikit-learn is properly installed and compatible with your environment.
    """

    def __init__(
        self,
        filterx: Type[BaseFilter],
        param_grid: Dict[str, Any],
        scoring: str,
        cv: int = 2,
    ):
        """
        Initialize a new GridSearchCVPlugin instance.

        This constructor sets up the GridSearchCVPlugin with the specified BaseFilter,
        parameter grid, scoring metric, and cross-validation strategy.

        Args:
            filterx (Type[BaseFilter]): The BaseFilter class to be tuned.
            param_grid (Dict[str, Any]): Dictionary with parameters names as keys and lists of parameter settings to try as values.
            scoring (str): Strategy to evaluate the performance of the cross-validated model on the test set.
            cv (int): Determines the cross-validation splitting strategy. Defaults to 2.

        Note:
            The GridSearchCV object is initialized with a Pipeline containing the specified BaseFilter
            wrapped in an SkWrapper to ensure compatibility with scikit-learn's API.
        """
        super().__init__(filterx=filterx, param_grid=param_grid, scoring=scoring, cv=cv)

        self._clf: GridSearchCV = GridSearchCV(
            estimator=Pipeline(steps=[(filterx.__name__, SkWrapper(filterx))]),
            param_grid=param_grid,
            scoring=scoring,
            cv=cv,
            refit=True,
            verbose=0,
        )

    def fit(self, x, y):
        """
        Fit the GridSearchCV object to the given data.

        This method performs the grid search over the specified parameter grid,
        fitting the model with different parameter combinations and selecting the best one.

        Args:
            x (XYData): The input features.
            y (XYData): The target values.

        Note:
            This method modifies the internal state of the GridSearchCV object,
            storing the best parameters and the corresponding fitted model.
        """
        self._clf.fit(x.value, y.value)  # type: ignore

    def predict(self, x) -> XYData:
        """
        Make predictions using the best estimator found by GridSearchCV.

        This method uses the best model found during the grid search to make predictions
        on new data.

        Args:
            x (XYData): The input features.

        Returns:
            (XYData): The predicted values wrapped in an XYData object.

        Note:
            The predictions are wrapped in an XYData object for consistency with the framework.
        """
        return XYData.mock(self._clf.predict(x.value))  # type: ignore
