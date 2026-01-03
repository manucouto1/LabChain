from typing import Any, Callable, Literal, TypedDict

from numpy.typing import ArrayLike


class CoherenceEvaluateKwargs(TypedDict, total=False):
    f_vocab: list[str]
    topk: int
    processes: int


class NMIClustKwargs(TypedDict, total=False):
    average_method: str


class SilhouetteKwargs(TypedDict, total=False):
    metric: str | Callable[..., Any]
    sample_size: int | None
    random_state: int | None


class PrecissionKwargs(TypedDict, total=False):
    labels: ArrayLike | None
    pos_label: str | int
    average: Literal["micro", "macro", "samples", "weighted", "binary"] | None
    sample_weight: ArrayLike | None
    zero_division: int | Literal["warn"]
