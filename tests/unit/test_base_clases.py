from abc import abstractmethod
import inspect
from typing import Dict, Optional, get_type_hints
import pytest
import typeguard
import numpy as np
from labchain.base import BasePlugin, BaseFilter, BaseMetric, BasePipeline
from labchain.base.base_types import XYData


class ConcreteFilter(BaseFilter):
    def fit(self, x: XYData, y: Optional[XYData]) -> None:
        pass

    def predict(self, x: XYData) -> XYData:
        return x


def test_type_checking_init():
    class TestPlugin(BasePlugin):
        def __init__(self, a: int, b: str):
            super().__init__(a=a, b=b)

    # Should pass
    TestPlugin(a=1, b="test")

    # Should raise TypeError
    with pytest.raises(typeguard.TypeCheckError):
        TestPlugin(a="not an int", b=123)  # type: ignore


def test_inherit_type_annotations():
    class AbstractBase(BasePlugin):
        @abstractmethod
        def abstract_method(self, x: int, y: str) -> float:
            pass

    class ConcreteClass(AbstractBase):
        def abstract_method(self, x, y):
            return float(x)

    concrete_instance = ConcreteClass()
    method_annotations = get_type_hints(concrete_instance.abstract_method)

    assert "x" in method_annotations
    assert method_annotations["x"] is int
    assert "y" in method_annotations
    assert method_annotations["y"] is str
    assert "return" in method_annotations
    assert method_annotations["return"] is float


def test_type_checking_concrete_methods():
    class TestPlugin(BasePlugin):
        def __init__(self):
            super().__init__()

        def concrete_method(self, x: int) -> str:
            return str(x)

    test_instance = TestPlugin()

    # Should pass
    result = test_instance.concrete_method(5)
    assert result == "5"

    # Should raise TypeError
    with pytest.raises(typeguard.TypeCheckError):
        test_instance.concrete_method("not an int")  # type: ignore


def test_multiple_inheritance_annotation_inheritance():
    class BaseA(BasePlugin):
        @abstractmethod
        def method_a(self, x: int) -> str:
            pass

    class BaseB(BasePlugin):
        @abstractmethod
        def method_b(self, y: float) -> bool:
            pass

    class ConcreteClass(BaseA, BaseB):
        def method_a(self, x):
            return str(x)

        def method_b(self, y):
            return y > 0

    concrete_instance = ConcreteClass()

    method_a_annotations = get_type_hints(concrete_instance.method_a)
    method_b_annotations = get_type_hints(concrete_instance.method_b)

    assert "x" in method_a_annotations
    assert method_a_annotations["x"] is int
    assert "return" in method_a_annotations
    assert method_a_annotations["return"] is str

    assert "y" in method_b_annotations
    assert method_b_annotations["y"] is float
    assert "return" in method_b_annotations
    assert method_b_annotations["return"] is bool


def test_base_plugin_allows_extra_params():
    class TestPlugin(BasePlugin):
        def __init__(self, param1: int, param2: str, extra_param: str):
            super().__init__(param1=param1, param2=param2)
            self.extra_param = extra_param

    # Create an instance with extra parameters
    plugin = TestPlugin(param1=1, param2="test", extra_param="extra")

    # Check if the extra parameter is stored
    assert hasattr(plugin, "extra_param")
    assert plugin.extra_param == "extra"

    # Verify that the original parameters are also present
    assert plugin.param1 == 1  # type: ignore
    assert plugin.param2 == "test"  # type: ignore

    # Check if the extra parameter is included in the model dump
    dump = plugin.model_dump()
    assert "extra_param" in dump
    assert dump["extra_param"] == "extra"


def test_base_plugin_item_dump():
    class TestPlugin(BasePlugin):
        def __init__(self, param1: int, param2: str, **kwargs):
            super().__init__(param1=param1, param2=param2)
            self._param3 = kwargs["param3"]

    plugin = TestPlugin(param1=1, param2="test", param3=3)

    # Test default behavior
    dump = plugin.item_dump()

    assert isinstance(dump, dict)
    assert "param1" in dump["params"]
    assert dump["params"]["param1"] == 1
    assert "param2" in dump["params"]
    assert dump["params"]["param2"] == "test"

    # Test with exclude
    dump = plugin.item_dump(exclude={"param2"})
    assert "param1" in dump["params"]
    assert "param2" not in dump["params"]

    # Test with include
    dump = plugin.item_dump(include={"_param3"})
    assert "param1" in dump["params"]
    assert "param2" in dump["params"]
    assert "_param3" in dump

    # Test with mode='json'
    extra = plugin.get_extra()
    assert "_param3" in extra

    assert isinstance(dump, dict)
    assert all(isinstance(key, str) for key in dump.keys())


def test_base_filter_subclass_initialization():
    class ConcreteFilter(BaseFilter):
        def fit(self, x: XYData, y: Optional[XYData]) -> None:
            pass

        def predict(self, x: XYData) -> XYData:
            return XYData.mock(x.value)

    concrete_filter = ConcreteFilter()

    assert hasattr(concrete_filter, "fit")
    assert hasattr(concrete_filter, "predict")
    assert callable(concrete_filter.fit)
    assert callable(concrete_filter.predict)

    # Test fit method
    x_data = XYData.mock(np.array([[1, 2], [3, 4]]))
    y_data = XYData.mock(np.array([0, 1]))

    concrete_filter.fit(x_data, y_data)

    # Test predict method
    x_test = XYData.mock(np.array([[5, 6]]))
    result = concrete_filter.predict(x_test)
    assert isinstance(result.value, np.ndarray)
    assert result.value.shape == x_test.value.shape


def test_basepipeline_abstract_methods():
    class IncompleteBasePipeline(BasePipeline):
        def fit(self, x: XYData, y: Optional[XYData]) -> None:
            pass

        def predict(self, x: XYData) -> XYData:
            return x

    with pytest.raises(TypeError) as excinfo:
        IncompleteBasePipeline()  # type: ignore

    assert "Can't instantiate abstract class IncompleteBasePipeline" in str(
        excinfo.value
    )
    assert "abstract methods" in str(excinfo.value)
    assert "start" in str(excinfo.value)

    class CompleteBasePipeline(BasePipeline):
        def fit(self, x: XYData, y: Optional[XYData]) -> None:
            pass

        def predict(self, x: XYData) -> XYData:
            return x

        def start(
            self, x: XYData, y: Optional[XYData], X_: Optional[XYData]
        ) -> Optional[XYData]:
            return None

        def evaluate(
            self, x_data: XYData, y_true: XYData | None, y_pred: XYData
        ) -> Dict[str, float]:
            return {}

    # Should not raise any exception
    CompleteBasePipeline()


def test_base_metric_evaluate_implementation():
    class ConcreteMetric(BaseMetric):
        def evaluate(
            self, x_data: XYData, y_true: XYData | None, y_pred: XYData
        ) -> float:
            return 0.5

    class InvalidMetric(BaseMetric):
        pass

    # Valid implementation
    concrete_metric = ConcreteMetric()
    result = concrete_metric.evaluate(
        XYData.mock(np.array([1, 2, 3])),
        XYData.mock(np.array([1, 2, 3])),
        XYData.mock(np.array([1, 2, 3])),
    )
    assert isinstance(result, (float, np.ndarray))

    # Invalid implementation (missing evaluate method)
    with pytest.raises(TypeError):
        InvalidMetric()  # type: ignore

    # Check if the evaluate method has the correct signature
    evaluate_signature = inspect.signature(ConcreteMetric.evaluate)
    expected_params = ["self", "x_data", "y_true", "y_pred"]
    assert list(evaluate_signature.parameters.keys()) == expected_params
    assert evaluate_signature.return_annotation is float


def test_no_init_method_defined():
    class NoInitPlugin(BasePlugin):
        def some_method(self):
            pass

    # Should not raise any exceptions
    plugin = NoInitPlugin()

    # Check if the class is properly created
    assert isinstance(plugin, NoInitPlugin)
    assert isinstance(plugin, BasePlugin)

    # Ensure that the method is present and callable
    assert hasattr(plugin, "some_method")
    assert callable(plugin.some_method)

    # Verify that type checking is still applied to other methods
    with pytest.raises(TypeError):
        plugin.some_method("invalid argument")  # type: ignore


def test_base_filter_equality():
    class ExampleFilter(BaseFilter):
        def fit(self, x: XYData, y: Optional[XYData]) -> None:
            pass

        def predict(self, x: XYData) -> XYData:
            return x

    # Test equality
    filter1 = ExampleFilter(bert_uncased="bert-base-uncased")
    filter2 = ExampleFilter(bert_uncased="bert-base-uncased")
    assert filter1 == filter2 == ExampleFilter(bert_uncased="bert-base-uncased")

    # Test inequality
    filter3 = ExampleFilter(bert_uncased="bert-large-uncased")
    assert filter1 != filter3

    # Test inequality with different types
    assert filter1 != "not a filter"

    # Test hash consistency
    assert hash(filter1) == hash(filter2)
    assert hash(filter1) != hash(filter3)


def test_base_filter_repr():
    class ExampleFilter(BaseFilter):
        def fit(self, x: XYData, y: Optional[XYData]) -> None:
            pass

        def predict(self, x: XYData) -> XYData:
            return x

    filter = ExampleFilter(bert_uncased="bert-base-uncased")
    expected_repr = "ExampleFilter({'bert_uncased': 'bert-base-uncased'})"
    assert repr(filter) == expected_repr


def test_base_filter_with_multiple_params():
    class ComplexFilter(BaseFilter):
        def __init__(self, param1: int, param2: str):
            ...
            # super().__init__(param1=param1, param2=param2)

        def fit(self, x: XYData, y: Optional[XYData]) -> None:
            pass

        def predict(self, x: XYData) -> XYData:
            return x

    filter1 = ComplexFilter(param1=1, param2="test")
    filter2 = ComplexFilter(param1=1, param2="test")
    filter3 = ComplexFilter(param1=1, param2="different")

    assert filter1 == filter2
    assert filter1 != filter3
    assert hash(filter1) == hash(filter2)
    assert hash(filter1) != hash(filter3)
