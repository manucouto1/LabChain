from typing import Literal
from sklearn.ensemble import RandomForestClassifier

from labchain import Container, XYData
from labchain.base import BaseFilter


@Container.bind()
class GaussianNaiveBayes(BaseFilter):
    def __init__(
        self,
        n_estimators=100,
        criterion: Literal["gini", "entropy", "log_loss"] = "gini",
        max_depth=2,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features: float | Literal["sqrt", "log2"] = "sqrt",
        class_weight=None,
        proba=False,
    ):
        super().__init__()
        self.proba = proba
        self._model = RandomForestClassifier(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            class_weight=class_weight,
            random_state=0,
        )

    def fit(self, x: XYData, y: XYData | None):
        if y is None:
            raise ValueError("y must be provided for training")
        self._model.fit(x.value, y.value)

    def predict(self, x: XYData) -> XYData:
        if self.proba:
            result = list(map(lambda i: i[1], self._model.predict_proba(x.value)))
        else:
            result = self._model.predict(x.value)
        return XYData.mock(result)
