from sklearn.metrics import confusion_matrix
from labchain import BaseMetric, Container, XYData

from numpy import exp
import numpy as np

__all__ = ["ERDE_5", "ERDE_50"]


class ERDE(BaseMetric):
    def __init__(self, k: int = 5):
        self.k = k

    def evaluate(
        self, x_data: XYData, y_true: XYData | None, y_pred: XYData
    ) -> float | np.ndarray:
        if y_true is None:
            raise ValueError("y_true must be provided for evaluation")

        all_erde = []
        _, _, _, tp = confusion_matrix(y_true.value, y_pred.value).ravel()
        for expected, result, count in list(
            zip(y_true.value, y_pred.value, x_data.value.n_texts.values.tolist())
        ):
            if result == 1 and expected == 0:
                all_erde.append(float(tp) / len(y_true.value))
            elif result == 0 and expected == 1:
                all_erde.append(1.0)
            elif result == 1 and expected == 1:
                all_erde.append(1.0 - (1.0 / (1.0 + exp(count - self.k))))
            elif result == 0 and expected == 0:
                all_erde.append(0.0)
        return float(np.mean(all_erde) * 100)


@Container.bind()
class ERDE_5(BaseMetric):
    def __init__(self):
        self._erde = ERDE(k=5)

    def evaluate(
        self, x_data: XYData, y_true: XYData | None, y_pred: XYData
    ) -> float | np.ndarray:
        if y_true is None:
            raise ValueError("y_true must be provided for evaluation")

        return self._erde.evaluate(x_data, y_true, y_pred)


@Container.bind()
class ERDE_50(BaseMetric):
    def __init__(self):
        self._erde = ERDE(k=50)

    def evaluate(
        self, x_data: XYData, y_true: XYData | None, y_pred: XYData
    ) -> float | np.ndarray:
        if y_true is None:
            raise ValueError("y_true must be provided for evaluation")

        return self._erde.evaluate(x_data, y_true, y_pred)
