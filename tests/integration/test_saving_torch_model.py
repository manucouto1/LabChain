import pytest
import torch
from labchain.base.base_types import XYData
from labchain.base.base_clases import BaseFilter
from labchain.container.container import Container
from labchain.plugins.filters.cache.cached_filter import Cached
from labchain.plugins.storage.local_storage import LocalStorage
from transformers import BertTokenizer, BertModel


@Container.bind()
class TopicEmbedder(BaseFilter):
    def __init__(self, model_path: str):
        super().__init__(model_path=model_path)  # type: ignore[assignment]  # Suppress unused variable error
        # self.model_path = model_path  # type: ignore[assignment]  # Suppress unused variable error
        self._tokenizer = BertTokenizer.from_pretrained(model_path)
        self._model = BertModel.from_pretrained(model_path)

    def fit(self, x: XYData, y: XYData | None) -> None:
        pass

    def predict(self, x: XYData) -> XYData:
        all_m = []
        for topic in x.value:
            topic_m = []
            for word in topic:
                encoded_input = self._tokenizer(str(word), return_tensors="pt")
                out = self._model(**encoded_input)
                mean_embeddings = torch.mean(out[0].detach().cpu(), axis=1)  # type: ignore
                topic_m.append(mean_embeddings)
            all_m.append(torch.stack(topic_m))
        all_stack = torch.stack(all_m)
        return XYData.mock(all_stack.squeeze(2))


@pytest.fixture
def topic_embedder():
    return TopicEmbedder("bert-base-uncased")


def test_topic_embedder_serialization(topic_embedder, tmp_path):
    # Crear un almacenamiento local temporal
    storage = LocalStorage(str(tmp_path))

    # Crear un filtro en caché con el TopicEmbedder
    cached_filter = Cached(topic_embedder, storage=storage)

    # Datos de prueba
    x = XYData.mock([["hello", "world"], ["test", "serialization"]])

    # Ajustar y predecir con el filtro en caché
    cached_filter.fit(x, None)
    original_prediction = cached_filter.predict(x)

    # Forzar la serialización y deserialización
    cached_filter.__setstate__(cached_filter.__getstate__())

    # Predecir de nuevo con el filtro deserializado
    deserialized_prediction = cached_filter.predict(x)

    # Verificar que las predicciones son iguales
    assert torch.allclose(original_prediction.value, deserialized_prediction.value)

    # Verificar que los atributos privados se han restaurado correctamente
    assert hasattr(cached_filter.filter, "_tokenizer")
    assert hasattr(cached_filter.filter, "_model")
    assert isinstance(cached_filter.filter._tokenizer, BertTokenizer)
    assert isinstance(cached_filter.filter._model, BertModel)

    # Verificar que el tokenizer y el modelo funcionan correctamente
    test_input = "Test sentence"
    encoded_input = cached_filter.filter._tokenizer(test_input, return_tensors="pt")
    output = cached_filter.filter._model(**encoded_input)

    assert output[0].shape[1] == len(cached_filter.filter._tokenizer.encode(test_input))


def test_topic_embedder_caching(topic_embedder, tmp_path):
    # Crear un almacenamiento local temporal
    storage = LocalStorage(str(tmp_path))

    # Crear un filtro en caché con el TopicEmbedder
    cached_filter = Cached(
        topic_embedder, storage=storage, cache_filter=True, cache_data=True
    )

    # Datos de prueba
    x = XYData.mock([["hello", "world"], ["test", "serialization"]])

    # Ajustar y predecir con el filtro en caché
    cached_filter.fit(x, None)
    original_prediction = cached_filter.predict(x)

    # Crear un nuevo filtro en caché con el mismo almacenamiento
    new_cached_filter = Cached(
        TopicEmbedder("bert-base-uncased"),
        storage=storage,
        cache_filter=True,
        cache_data=True,
    )

    assert topic_embedder == TopicEmbedder("bert-base-uncased")

    new_cached_filter.fit(x, None)

    # Predecir con el nuevo filtro (debería cargar el modelo y los datos del caché)
    cached_prediction = new_cached_filter.predict(x)

    # Verificar que las predicciones son iguales
    assert torch.allclose(original_prediction.value, cached_prediction.value)

    # Verificar que los atributos privados se han restaurado correctamente
    assert hasattr(new_cached_filter.filter, "_tokenizer")
    assert hasattr(new_cached_filter.filter, "_model")
    assert isinstance(new_cached_filter.filter._tokenizer, BertTokenizer)
    assert isinstance(new_cached_filter.filter._model, BertModel)

    # Verificar que el tokenizer y el modelo funcionan correctamente
    test_input = "Test sentence"
    encoded_input = new_cached_filter.filter._tokenizer(test_input, return_tensors="pt")
    output = new_cached_filter.filter._model(**encoded_input)

    assert output[0].shape[1] == len(
        new_cached_filter.filter._tokenizer.encode(test_input)
    )

    # Verificar que se está utilizando el caché
    assert new_cached_filter._lambda_filter is not None
