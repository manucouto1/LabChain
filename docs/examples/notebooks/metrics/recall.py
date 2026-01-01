from labchain import BaseMetric, Container, XYData, Recall

import numpy as np

__all__ = ["RecallScore"]


@Container.bind()
class RecallScore(BaseMetric):
    def evaluate(
        self, x_data: XYData, y_true: XYData | None, y_pred: XYData
    ) -> float | np.ndarray:
        if y_true is None:
            raise ValueError("y_true must be provided for evaluation")

        x_data.value["_y"] = y_pred.value

        aux = x_data.value.groupby(["user"]).agg(
            {"label": "first", "_y": lambda x: 1 if any(list(x)) else 0}
        )

        return float(
            Recall().evaluate(
                x_data, XYData.mock(aux.label.tolist()), XYData.mock(aux._y.tolist())
            )
        )
