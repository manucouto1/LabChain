from typing import Any, Callable, Literal, Mapping
from sklearn.svm import SVC

from labchain import Container, XYData
from labchain.base import BaseFilter


@Container.bind()
class ClassifierSVM(BaseFilter):
    def __init__(
        self,
        C: float = 1,
        kernel: Callable
        | Literal["linear", "poly", "rbf", "sigmoid", "precomputed"] = "rbf",
        gamma: float | Literal["scale", "auto"] = "scale",
        coef0: float = 0.0,
        tol: float = 0.001,
        decision_function_shape: Literal["ovo", "ovr"] = "ovr",
        class_weight_1: Mapping[Any, Any] | str | None = None,
        probability: bool = False,
    ):
        super().__init__()
        self.proba = probability
        self._model = SVC(
            C=C,
            kernel=kernel,
            gamma=gamma,
            coef0=coef0,
            tol=tol,
            decision_function_shape=decision_function_shape,
            class_weight=class_weight_1,
            probability=probability,
            random_state=43,
        )

    def fit(self, x: XYData, y: XYData | None):
        if y is None:
            raise ValueError("y must be provided for training")
        self._model.fit(x.value, y.value)

    def predict(self, x: XYData) -> XYData:
        if self.proba:
            result = list(map(lambda i: i[1], self._model.predict_proba(x.value)))
            return XYData.mock(result)
        else:
            result = self._model.predict(x.value)
            return XYData.mock(result)
