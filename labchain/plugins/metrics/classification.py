from typing import Literal, Unpack
from sklearn.metrics import f1_score, precision_score, recall_score
from labchain.base.base_types import Float
from labchain.base.base_types import XYData
from labchain.base.base_clases import BaseMetric
from labchain.container.container import Container

import numpy as np

from labchain.plugins.metrics.utils.types import PrecissionKwargs

__all__ = ["F1", "Precission", "Recall"]


@Container.bind()
class F1(BaseMetric):
    """
    F1 score metric for classification tasks.

    This class calculates the F1 score, which is the harmonic mean of precision and recall.
    It's particularly useful when you need a balance between precision and recall.

    Key Features:
        - Calculates F1 score for binary and multiclass classification
        - Supports different averaging methods (micro, macro, weighted, etc.)
        - Integrates with framework3's BaseMetric interface

    Usage:
        The F1 metric can be used to evaluate classification models:

        ```python
        from framework3.plugins.metrics.classification import F1
        from framework3.base.base_types import XYData
        import numpy as np

        # Create sample data
        y_true = XYData(value=np.array([0, 1, 2, 0, 1, 2]))
        y_pred = XYData(value=np.array([0, 2, 1, 0, 0, 1]))
        x_data = XYData(value=np.array([1, 2, 3, 4, 5, 6]))

        # Create and use the F1 metric
        f1_metric = F1(average='macro')
        score = f1_metric.evaluate(x_data, y_true, y_pred)
        print(f"F1 Score: {score}")
        ```

    Attributes:
        average (str): The type of averaging performed on the data. Default is 'weighted'.

    Methods:
        evaluate(x_data: XYData, y_true: XYData | None, y_pred: XYData, **kwargs) -> Float | np.ndarray:
            Calculate the F1 score for the given predictions and true values.

    Note:
        This metric uses scikit-learn's f1_score function internally. Ensure that scikit-learn
        is properly installed and compatible with your environment.
    """

    def __init__(
        self,
        average: Literal["micro", "macro", "samples", "weighted", "binary"] = "binary",
    ):
        """
        Initialize a new F1 metric instance.

        This constructor sets up the F1 metric with the specified averaging method.

        Args:
            average (Literal['micro', 'macro', 'samples', 'weighted', 'binary']): The type of averaging performed on the data. Default is 'weighted'.
                           Other options include 'micro', 'macro', 'samples', 'binary', or None.

        Note:
            The 'average' parameter is passed directly to scikit-learn's f1_score function.
            Refer to scikit-learn's documentation for detailed information on averaging methods.
        """
        self.average = average

    def evaluate(
        self,
        x_data: XYData,
        y_true: XYData | None,
        y_pred: XYData,
        **kwargs: Unpack[PrecissionKwargs],
    ) -> Float | np.ndarray:
        """
        Calculate the F1 score for the given predictions and true values.

        This method computes the F1 score, which is the harmonic mean of precision and recall.

        Args:
            x_data (XYData): The input data (not used in this metric, but required by the interface).
            y_true (XYData | None): The ground truth (correct) target values.
            y_pred (XYData): The estimated targets as returned by a classifier.
            **kwargs (Unpack[PrecissionKwargs]): Additional keyword arguments passed to sklearn's f1_score function.

        Returns:
            Float | np.ndarray: The F1 score or array of F1 scores if average is None.

        Raises:
            ValueError: If y_true is None.

        Note:
            This method uses scikit-learn's f1_score function internally with zero_division=0.
        """
        if y_true is None:
            raise ValueError("Ground truth (y_true) must be provided.")

        return f1_score(
            y_true.value,
            y_pred.value,
            average=self.average,
            **kwargs,  # type: ignore
        )  # type: ignore


@Container.bind()
class Precission(BaseMetric):
    """
    Precision metric for classification tasks.

    This class calculates the precision score, which is the ratio tp / (tp + fp) where tp is
    the number of true positives and fp the number of false positives.

    Key Features:
        - Calculates precision score for binary and multiclass classification
        - Supports different averaging methods (micro, macro, weighted, etc.)
        - Integrates with framework3's BaseMetric interface

    Usage:
        The Precission metric can be used to evaluate classification models:

        ```python
        from framework3.plugins.metrics.classification import Precission
        from framework3.base.base_types import XYData
        import numpy as np

        # Create sample data
        y_true = XYData(value=np.array([0, 1, 2, 0, 1, 2]))
        y_pred = XYData(value=np.array([0, 2, 1, 0, 0, 1]))
        x_data = XYData(value=np.array([1, 2, 3, 4, 5, 6]))

        # Create and use the Precission metric
        precision_metric = Precission(average='macro')
        score = precision_metric.evaluate(x_data, y_true, y_pred)
        print(f"Precision Score: {score}")
        ```

    Attributes:
        average (Literal["micro", "macro", "samples", "weighted", "binary"]|None): The type of averaging performed on the data. Default is 'weighted'.

    Methods:
        evaluate (x_data: XYData, y_true: XYData | None, y_pred: XYData, **kwargs) -> Float | np.ndarray:
            Calculate the precision score for the given predictions and true values.

    Note:
        This metric uses scikit-learn's precision_score function internally. Ensure that scikit-learn
        is properly installed and compatible with your environment.
    """

    def __init__(
        self,
        average: Literal["micro", "macro", "samples", "weighted", "binary"]
        | None = "binary",
    ):
        """
        Initialize a new Precission metric instance.

        This constructor sets up the Precission metric with the specified averaging method.

        Args:
            average (Literal["micro", "macro", "samples", "weighted", "binary"]|None): The type of averaging performed on the data. Default is 'weighted'.
                                  Options are 'micro', 'macro', 'samples', 'weighted', 'binary', or None.

        Note:
            The 'average' parameter is passed directly to scikit-learn's precision_score function.
            Refer to scikit-learn's documentation for detailed information on averaging methods.
        """
        super().__init__(average=average)

    def evaluate(
        self,
        x_data: XYData,
        y_true: XYData | None,
        y_pred: XYData,
        **kwargs: Unpack[PrecissionKwargs],
    ) -> Float | np.ndarray:
        """
        Calculate the precision score for the given predictions and true values.

        This method computes the precision score, which is the ratio of true positives to the
        sum of true and false positives.

        Args:
            x_data (XYData): The input data (not used in this metric, but required by the interface).
            y_true (XYData | None): The ground truth (correct) target values.
            y_pred (XYData): The estimated targets as returned by a classifier.
            **kwargs (Unpack[PrecissionKwargs]): Additional keyword arguments passed to sklearn's precision_score function.

        Returns:
            Float | np.ndarray: The precision score or array of precision scores if average is None.

        Raises:
            ValueError: If y_true is None.

        Note:
            This method uses scikit-learn's precision_score function internally with zero_division=0.
        """
        if y_true is None:
            raise ValueError("Ground truth (y_true) must be provided.")
        return precision_score(
            y_true.value,
            y_pred.value,
            average=self.average,
            **kwargs,  # type: ignore
        )  # type: ignore


@Container.bind()
class Recall(BaseMetric):
    """
    Recall metric for classification tasks.

    This class calculates the recall score, which is the ratio tp / (tp + fn) where tp is
    the number of true positives and fn the number of false negatives.

    Key Features:
        - Calculates recall score for binary and multiclass classification
        - Supports different averaging methods (micro, macro, weighted, etc.)
        - Integrates with framework3's BaseMetric interface

    Usage:
        The Recall metric can be used to evaluate classification models:

        ```python
        from framework3.plugins.metrics.classification import Recall
        from framework3.base.base_types import XYData
        import numpy as np

        # Create sample data
        y_true = XYData(value=np.array([0, 1, 2, 0, 1, 2]))
        y_pred = XYData(value=np.array([0, 2, 1, 0, 0, 1]))
        x_data = XYData(value=np.array([1, 2, 3, 4, 5, 6]))

        # Create and use the Recall metric
        recall_metric = Recall(average='macro')
        score = recall_metric.evaluate(x_data, y_true, y_pred)
        print(f"Recall Score: {score}")
        ```

    Attributes:
        average (str | None): The type of averaging performed on the data. Default is 'weighted'.

    Methods:
        evaluate(x_data: XYData, y_true: XYData | None, y_pred: XYData, **kwargs) -> Float | np.ndarray:
            Calculate the recall score for the given predictions and true values.

    Note:
        This metric uses scikit-learn's recall_score function internally. Ensure that scikit-learn
        is properly installed and compatible with your environment.
    """

    def __init__(
        self,
        average: Literal["micro", "macro", "samples", "weighted", "binary"]
        | None = "binary",
    ):
        """
        Initialize a new Recall metric instance.

        This constructor sets up the Recall metric with the specified averaging method.

        Args:
            average (str | None): The type of averaging performed on the data. Default is 'weighted'.
                                  Options are 'micro', 'macro', 'samples', 'weighted', 'binary', or None.

        Note:
            The 'average' parameter is passed directly to scikit-learn's recall_score function.
            Refer to scikit-learn's documentation for detailed information on averaging methods.
        """
        super().__init__(average=average)

    def evaluate(
        self,
        x_data: XYData,
        y_true: XYData | None,
        y_pred: XYData,
        **kwargs: Unpack[PrecissionKwargs],
    ) -> Float | np.ndarray:
        """
        Calculate the recall score for the given predictions and true values.

        This method computes the recall score, which is the ratio of true positives to the
        sum of true positives and false negatives.

        Args:
            x_data (XYData): The input data (not used in this metric, but required by the interface).
            y_true (XYData | None): The ground truth (correct) target values.
            y_pred (XYData): The estimated targets as returned by a classifier.
            **kwargs (Unpack[PrecissionKwargs]): Additional keyword arguments passed to sklearn's recall_score function.

        Returns:
            Float | np.ndarray: The recall score or array of recall scores if average is None.

        Raises:
            ValueError: If y_true is None.

        Note:
            This method uses scikit-learn's recall_score function internally with zero_division=0.
        """
        if y_true is None:
            raise ValueError("Ground truth (y_true) must be provided.")
        return recall_score(
            y_true.value,
            y_pred.value,
            average=self.average,
            **kwargs,  # type: ignore
        )  # type: ignore
