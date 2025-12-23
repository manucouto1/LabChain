import pytest
from framework3.base.base_types import XYData
from framework3.base.base_clases import BaseFilter
from typing import Optional

from framework3.base.exceptions import NotTrainableFilterError


class TrainableFilter(BaseFilter):
    def __init__(self):
        super().__init__()
        self.is_fitted = False

    def fit(self, x: XYData, y: Optional[XYData]) -> None:
        self.is_fitted = True

    def predict(self, x: XYData) -> XYData:
        if not self.is_fitted:
            raise ValueError("This filter needs to be fitted before prediction")
        return x


class NonTrainableFilter(BaseFilter):
    def __init__(self):
        super().__init__()

    def predict(self, x: XYData) -> XYData:
        return x


def test_trainable_filter():
    filter = TrainableFilter()
    x = XYData.mock([1, 2, 3])

    # Verificar que el filtro está inicializado con valores por defecto
    # basados en atributos públicos y nombre de clase
    assert hasattr(filter, "_m_hash")
    assert hasattr(filter, "_m_str")
    assert hasattr(filter, "_m_path")

    # Guardar hash inicial (basado solo en clase y atributos públicos)
    initial_hash = filter._m_hash

    # Intentar predecir sin entrenar debería fallar
    with pytest.raises(ValueError):
        filter.predict(x)

    # Entrenar el filtro
    filter.fit(x, None)

    # Verificar que el hash ha cambiado después del entrenamiento
    # (ahora incluye el hash de los datos de entrada)
    assert filter._m_hash != initial_hash
    assert hasattr(filter, "_m_hash")
    assert hasattr(filter, "_m_str")
    assert hasattr(filter, "_m_path")

    # La predicción ahora debería funcionar
    result = filter.predict(x)
    assert isinstance(result, XYData)


def test_non_trainable_filter():
    filter = NonTrainableFilter()
    x = XYData.mock([1, 2, 3])

    # Verificar que el filtro está inicializado desde el principio
    # con valores por defecto basados en clase y atributos públicos
    assert hasattr(filter, "_m_hash")
    assert hasattr(filter, "_m_str")
    assert hasattr(filter, "_m_path")

    # Guardar hash inicial
    initial_hash = filter._m_hash

    # La predicción debería funcionar sin necesidad de entrenamiento
    result = filter.predict(x)
    assert isinstance(result, XYData)

    # Intentar llamar a fit debería lanzar una excepción
    with pytest.raises(NotTrainableFilterError):
        filter.fit(x, None)

    # El hash no debería haber cambiado (el fit falló)
    assert filter._m_hash == initial_hash


def test_trainable_filter_hash_includes_data():
    """Verifica que el hash de un filtro trainable incluye información de los datos"""
    filter1 = TrainableFilter()
    filter2 = TrainableFilter()

    x1 = XYData.mock([1, 2, 3])
    x2 = XYData.mock([4, 5, 6])

    # Ambos filtros deberían tener el mismo hash inicial (misma clase, mismos atributos)
    assert filter1._m_hash == filter2._m_hash

    # Después de entrenar con datos diferentes, los hashes deberían ser distintos
    filter1.fit(x1, None)
    filter2.fit(x2, None)

    assert filter1._m_hash != filter2._m_hash
