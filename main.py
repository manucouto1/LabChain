from typing import cast
from labchain.base import BasePipeline, BasePlugin, XYData
from labchain.container import Container

from labchain.plugins.filters.classification.knn import KnnFilter
from labchain.plugins.filters.transformation.pca import PCAPlugin
from labchain.plugins.metrics import F1, Precission, Recall
from labchain.plugins.pipelines import F3Pipeline

from labchain.plugins.filters.cache.cached_filter import Cached

from rich import print

from sklearn import datasets


# gs_pipeline = GridSearchCVPipeline(
#         filterx=[
#             ClassifierSVMPlugin,
#             KnnFilter
#         ],
#         param_grid=ClassifierSVMPlugin.item_grid(C=[1.0, 10], kernel=['rbf']),
#         scoring='f1_weighted',
#         cv=2,
#         metrics=[]
# )
# cached_pipeline = Cached(
#     filter=gs_pipeline,
#     cache_data=True,
#     cache_filter=True,
#     overwrite=True,
# )


def main():
    pipeline = F3Pipeline(
        filters=[
            Cached(
                filter=PCAPlugin(n_components=1),
                cache_data=False,
                cache_filter=True,
            ),
            Cached(
                filter=KnnFilter(),
                cache_data=False,
                cache_filter=True,
            ),
        ],
        metrics=[F1(), Precission(), Recall()],
    )

    print(pipeline)

    dumped_pipeline = pipeline.item_dump()

    print(dumped_pipeline)

    reconstructed_pipeline: BasePipeline = cast(
        BasePipeline, BasePlugin.build_from_dump(dumped_pipeline, Container.pif)
    )

    print(reconstructed_pipeline)

    F3Pipeline(
        filters=[
            pipeline,
        ],
        metrics=[],
    )

    # Test data
    print(pipeline)

    iris = datasets.load_iris()

    X = XYData(
        _hash="Iris X data",
        _path="/datasets",
        _value=iris.data,  # type: ignore
    )
    y = XYData(
        _hash="Iris y data",
        _path="/datasets",
        _value=iris.target,  # type: ignore
    )
    print(X)
    print(y)

    reconstructed_pipeline.fit(X, y)

    X = XYData(
        _hash="Iris X data changed",
        _path="/datasets",
        _value=iris.data,  # type: ignore
    )
    prediction = reconstructed_pipeline.predict(x=X)

    y_pred = XYData.mock(prediction.value)

    evaluate = reconstructed_pipeline.evaluate(X, y, y_pred=y_pred)

    print(reconstructed_pipeline)

    print(evaluate)


if __name__ == "__main__":
    main()
