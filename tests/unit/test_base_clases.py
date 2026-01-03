# tests/unit/test_base_clases_attribute_tracking.py

import pytest
from typing import Any, Dict, Optional
from labchain.base.base_clases import BasePlugin, BaseFilter
from labchain.base.base_types import XYData


class TestBasePluginAttributeTracking:
    """Test suite for BasePlugin attribute tracking rules."""

    def test_public_attributes_passed_to_super_init(self):
        """Test that public attributes passed to super().__init__() are tracked."""

        class MyPlugin(BasePlugin):
            def __init__(self, param1: int, param2: str):
                self.param1 = param1
                self.param2 = param2

        plugin = MyPlugin(param1=42, param2="test")

        assert plugin._public_attributes == {"param1": 42, "param2": "test"}
        assert plugin.param1 == 42
        assert plugin.param2 == "test"

    def test_public_attribute_set_after_init_without_kwargs_raises_error(self):
        """Test that setting public attributes after init (without **kwargs) raises an error."""

        class MyPlugin(BasePlugin):
            def __init__(self, param1: int):
                super().__init__(param1=param1)
                # This should raise an error
                self.param2 = "not allowed"

        with pytest.raises(AttributeError) as exc_info:
            MyPlugin(param1=42)

        assert "param2" in str(exc_info.value)
        assert "not declared in the __init__ signature" in str(exc_info.value).lower()

    def test_public_attributes_with_kwargs_allowed(self):
        """Test that public attributes can be set freely when **kwargs is in signature."""

        class MyPlugin(BasePlugin):
            def __init__(self, param1: int, **kwargs):
                super().__init__(param1=param1, **kwargs)
                self.param2 = "allowed with kwargs"
                self.param3 = 123

        plugin = MyPlugin(param1=42, extra="value")

        assert plugin._public_attributes == {
            "param1": 42,
            "extra": "value",
            "param2": "allowed with kwargs",
            "param3": 123,
        }

    def test_private_attributes_always_allowed(self):
        """Test that private attributes can always be set without restrictions."""

        class MyPlugin(BasePlugin):
            def __init__(self, param1: int):
                super().__init__(param1=param1)
                self._internal_state = "allowed"
                self._cache: Dict[str, Any] = {}

        plugin = MyPlugin(param1=42)

        assert plugin._public_attributes == {"param1": 42}
        assert plugin._internal_state == "allowed"
        assert plugin._cache == {}
        # Private attributes are not tracked in _public_attributes
        assert "_internal_state" not in plugin._public_attributes
        assert "_cache" not in plugin._public_attributes

    def test_methods_never_tracked(self):
        """Test that methods are never added to _public_attributes."""

        class MyPlugin(BasePlugin):
            def __init__(self, param1: int):
                super().__init__(param1=param1)
                self.my_method = lambda x: x * 2  # Even if we assign a callable

        plugin = MyPlugin(param1=42)

        assert "my_method" not in plugin._public_attributes
        assert plugin._public_attributes == {"param1": 42}

    def test_private_attributes_in_kwargs_tracked(self):
        """Test that private attributes passed via constructor are tracked separately."""

        class MyPlugin(BasePlugin):
            def __init__(self, param1: int, _internal: str):
                super().__init__(param1=param1, _internal=_internal)

        plugin = MyPlugin(param1=42, _internal="secret")

        assert plugin._public_attributes == {"param1": 42}
        assert plugin._private_attributes == {"_internal": "secret"}
        assert plugin._internal == "secret"

    def test_helpful_error_message(self):
        """Test that the error message is helpful and suggests solutions."""

        class MyPlugin(BasePlugin):
            def __init__(self, param1: int):
                super().__init__(param1=param1)
                self.param2 = "not allowed"

        with pytest.raises(AttributeError) as exc_info:
            MyPlugin(param1=42)

        error_message = str(exc_info.value)

        # Check that error message contains helpful guidance
        assert "param2" in error_message
        assert "not declared in the __init__ signature" in error_message.lower()

        print(error_message.lower())
        # Should suggest solutions
        assert any(
            phrase in error_message.lower()
            for phrase in [
                "add 'param2' to __init__ signature and",
                "add **kwargs to __init__ to accept dynamic",
                "make it private (internal state) by renaming to '_param2'",
            ]
        )

    def test_init_with_default_parameters(self):
        """Test that attributes with default values work correctly."""

        class MyPlugin(BasePlugin):
            def __init__(self, param1: str = "default"):
                self.param1 = param1

        # Without passing the parameter (uses default)
        plugin1 = MyPlugin()
        assert plugin1._public_attributes == {"param1": "default"}
        assert plugin1.param1 == "default"

        # Passing the parameter explicitly
        plugin2 = MyPlugin(param1="custom")
        assert plugin2._public_attributes == {"param1": "custom"}
        assert plugin2.param1 == "custom"

    def test_filter_with_default_parameters(self):
        """Test that filters with default parameters work correctly."""

        class MyFilter(BaseFilter):
            def __init__(self, n_components: int = 5):
                self.n_components = n_components
                self._model = None

            def predict(self, x: XYData) -> XYData:
                return x

        # Without passing the parameter
        filter1 = MyFilter()
        assert filter1._public_attributes == {"n_components": 5}
        assert filter1.n_components == 5

        # Passing the parameter
        filter2 = MyFilter(n_components=10)
        assert filter2._public_attributes == {"n_components": 10}
        assert filter2.n_components == 10


class TestBaseFilterAttributeTracking:
    """Test suite for BaseFilter attribute tracking rules."""

    def test_filter_with_explicit_params(self):
        """Test filter with explicit parameters in constructor."""

        class MyFilter(BaseFilter):
            def __init__(self, n_components: int, learning_rate: float):
                super().__init__(n_components=n_components, learning_rate=learning_rate)

            def predict(self, x: XYData) -> XYData:
                return x

        filter = MyFilter(n_components=5, learning_rate=0.01)

        assert filter._public_attributes == {"n_components": 5, "learning_rate": 0.01}
        assert filter.n_components == 5
        assert filter.learning_rate == 0.01

    def test_filter_setting_invalid_public_attribute_raises_error(self):
        """Test that setting an undeclared public attribute in filter raises error."""

        class MyFilter(BaseFilter):
            def __init__(self, n_components: int):
                super().__init__(n_components=n_components)
                self.model = "sklearn model"  # Should raise error

            def predict(self, x: XYData) -> XYData:
                return x

        with pytest.raises(AttributeError) as exc_info:
            MyFilter(n_components=5)

        assert "model" in str(exc_info.value)

    def test_filter_with_kwargs_allows_flexibility(self):
        """Test filter with **kwargs allows flexible attribute setting."""

        class MyFilter(BaseFilter):
            def __init__(self, n_components: int, **kwargs):
                super().__init__(n_components=n_components, **kwargs)
                self.model = None  # Allowed because of **kwargs

            def predict(self, x: XYData) -> XYData:
                return x

        filter = MyFilter(n_components=5, extra_param="value")

        assert "n_components" in filter._public_attributes
        assert "extra_param" in filter._public_attributes
        assert "model" in filter._public_attributes

    def test_filter_internal_state_always_allowed(self):
        """Test that filters can set internal state (_attributes) freely."""

        class MyFilter(BaseFilter):
            def __init__(self, n_components: int):
                super().__init__(n_components=n_components)
                self._model: str | None = None
                self._fitted = False
                self._cache: Dict[str, Any] = {}

            def fit(self, x: XYData, y: Optional[XYData]) -> None:
                self._fitted = True
                self._model = "trained model"

            def predict(self, x: XYData) -> XYData:
                return x

        filter = MyFilter(n_components=5)

        # Public attributes should only contain constructor params
        assert filter._public_attributes == {"n_components": 5}

        # Private attributes should be accessible
        assert filter._model is None
        assert filter._fitted is False
        assert filter._cache == {}

        # After fit, internal state can be updated
        filter.fit(XYData.mock([1, 2, 3]), None)
        assert filter._fitted is True
        assert filter._model == "trained model"

    def test_filter_equality_based_on_public_attributes(self):
        """Test that filter equality is based on public attributes only."""

        class MyFilter(BaseFilter):
            def __init__(self, n_components: int):
                super().__init__(n_components=n_components)
                self._internal_id = id(self)  # Different for each instance

            def predict(self, x: XYData) -> XYData:
                return x

        filter1 = MyFilter(n_components=5)
        filter2 = MyFilter(n_components=5)
        filter3 = MyFilter(n_components=10)

        # Same public attributes -> equal
        assert filter1 == filter2
        assert hash(filter1) == hash(filter2)

        # Different public attributes -> not equal
        assert filter1 != filter3
        assert hash(filter1) != hash(filter3)

        # Internal state doesn't affect equality
        assert filter1._internal_id != filter2._internal_id
        assert filter1 == filter2  # Still equal

    def test_filter_repr_shows_only_public_attributes(self):
        """Test that filter repr shows only public attributes."""

        class MyFilter(BaseFilter):
            def __init__(self, n_components: int):
                super().__init__(n_components=n_components)
                self._internal = "secret"

            def predict(self, x: XYData) -> XYData:
                return x

        filter = MyFilter(n_components=5)

        repr_str = repr(filter)
        assert "n_components" in repr_str
        assert "5" in repr_str
        assert "_internal" not in repr_str
        assert "secret" not in repr_str

    def test_filter_inheritance_with_super_init(self):
        """Test that filters can call super().__init__() and still set attributes."""

        class MyFilter(BaseFilter):
            def __init__(self, training_time: float = 2.0):
                super().__init__()  # Call BaseFilter.__init__()
                self.training_time = training_time

            def predict(self, x: XYData) -> XYData:
                return x

        filter = MyFilter(training_time=5.0)

        assert filter._public_attributes == {"training_time": 5.0}
        assert filter.training_time == 5.0

    def test_nested_inheritance_chain(self):
        """Test a deeper inheritance chain with multiple super().__init__() calls."""

        class MiddlePlugin(BasePlugin):
            def __init__(self, param1: int):
                super().__init__()
                self.param1 = param1

        class LeafPlugin(MiddlePlugin):
            def __init__(self, param1: int, param2: str):
                super().__init__(param1=param1)
                self.param2 = param2

        plugin = LeafPlugin(param1=42, param2="test")

        assert plugin._public_attributes == {"param1": 42, "param2": "test"}


class TestRealWorldScenarios:
    """Test real-world usage patterns."""

    def test_sklearn_like_filter(self):
        """Test a scikit-learn style filter with fit/predict."""

        class StandardScaler(BaseFilter):
            def __init__(self, with_mean: bool = True, with_std: bool = True):
                super().__init__(with_mean=with_mean, with_std=with_std)
                self._mean = None
                self._std = None

            def fit(self, x: XYData, y: Optional[XYData] = None) -> None:
                import numpy as np

                self._mean = np.mean(x.value, axis=0)
                self._std = np.std(x.value, axis=0)

            def predict(self, x: XYData) -> XYData:
                result = (x.value - self._mean) / self._std
                return XYData.mock(result)

        scaler = StandardScaler(with_mean=True, with_std=False)

        assert scaler._public_attributes == {"with_mean": True, "with_std": False}
        assert scaler._mean is None
        assert scaler._std is None

    def test_neural_network_filter_with_kwargs(self):
        """Test a neural network filter that accepts flexible kwargs."""

        class NeuralNetFilter(BaseFilter):
            def __init__(self, layers: list, **kwargs):
                super().__init__(layers=layers, **kwargs)
                self._model = None
                self.optimizer = kwargs.get("optimizer", "adam")
                self.learning_rate = kwargs.get("learning_rate", 0.001)

            def predict(self, x: XYData) -> XYData:
                return x

        nn = NeuralNetFilter(
            layers=[128, 64, 10], optimizer="sgd", learning_rate=0.01, batch_size=32
        )

        assert "layers" in nn._public_attributes
        assert "optimizer" in nn._public_attributes
        assert "learning_rate" in nn._public_attributes
        assert "batch_size" in nn._public_attributes
        assert nn._model is None

    def test_cached_filter_wrapper(self):
        """Test a wrapper filter (like Cached) that wraps another filter."""

        class SimpleFilter(BaseFilter):
            def __init__(self, param: int):
                super().__init__(param=param)

            def predict(self, x: XYData) -> XYData:
                return x

        class WrapperFilter(BaseFilter):
            def __init__(self, filter: BaseFilter, cache: bool = True):
                super().__init__(filter=filter, cache=cache)

            def predict(self, x: XYData) -> XYData:
                return self.filter.predict(x)

        inner = SimpleFilter(param=42)
        wrapper = WrapperFilter(filter=inner, cache=True)

        assert "filter" in wrapper._public_attributes
        assert "cache" in wrapper._public_attributes
        assert wrapper.filter == inner
        assert wrapper.cache is True


class TestErrorMessages:
    """Test that error messages are clear and actionable."""

    def test_error_message_content(self):
        """Test the exact content of error messages."""

        class MyPlugin(BasePlugin):
            def __init__(self, param1: int):
                super().__init__(param1=param1)
                self.invalid_attr = "value"

        with pytest.raises(AttributeError) as exc_info:
            MyPlugin(param1=42)

        error_msg = str(exc_info.value)

        # Must mention the attribute name
        assert "invalid_attr" in error_msg

        # Must mention it's not in the signature
        assert "__init__" in error_msg or "signature" in error_msg.lower()

        # Must provide at least one solution
        solutions = [
            "add 'invalid_attr' to __init__ signature and",
            "add **kwargs to __init__ to accept dynamic",
            "make it private (internal state) by renaming to '_invalid_attr'",
        ]

        assert any(solution.lower() in error_msg.lower() for solution in solutions)

    def test_error_shows_available_params(self):
        """Test that error message shows what parameters are available."""

        class MyPlugin(BasePlugin):
            def __init__(self, param1: int, param2: str):
                super().__init__(param1=param1, param2=param2)
                self.param3 = "invalid"

        with pytest.raises(AttributeError) as exc_info:
            MyPlugin(param1=42, param2="test")

        error_msg = str(exc_info.value)

        # Should mention available parameters
        assert "param1" in error_msg or "param2" in error_msg


# from abc import abstractmethod
# import inspect
# from typing import Dict, Optional, get_type_hints
# import pytest
# import typeguard
# import numpy as np
# from labchain.base import BasePlugin, BaseFilter, BaseMetric, BasePipeline
# from labchain.base.base_types import XYData


# class ConcreteFilter(BaseFilter):
#     def fit(self, x: XYData, y: Optional[XYData]) -> None:
#         pass

#     def predict(self, x: XYData) -> XYData:
#         return x


# def test_type_checking_init():
#     class TestPlugin(BasePlugin):
#         def __init__(self, a: int, b: str):
#             super().__init__(a=a, b=b)

#     # Should pass
#     TestPlugin(a=1, b="test")

#     # Should raise TypeError
#     with pytest.raises(typeguard.TypeCheckError):
#         TestPlugin(a="not an int", b=123)  # type: ignore


# def test_inherit_type_annotations():
#     class AbstractBase(BasePlugin):
#         @abstractmethod
#         def abstract_method(self, x: int, y: str) -> float:
#             pass

#     class ConcreteClass(AbstractBase):
#         def abstract_method(self, x, y):
#             return float(x)

#     concrete_instance = ConcreteClass()
#     method_annotations = get_type_hints(concrete_instance.abstract_method)

#     assert "x" in method_annotations
#     assert method_annotations["x"] is int
#     assert "y" in method_annotations
#     assert method_annotations["y"] is str
#     assert "return" in method_annotations
#     assert method_annotations["return"] is float


# def test_type_checking_concrete_methods():
#     class TestPlugin(BasePlugin):
#         def __init__(self):
#             super().__init__()

#         def concrete_method(self, x: int) -> str:
#             return str(x)

#     test_instance = TestPlugin()

#     # Should pass
#     result = test_instance.concrete_method(5)
#     assert result == "5"

#     # Should raise TypeError
#     with pytest.raises(typeguard.TypeCheckError):
#         test_instance.concrete_method("not an int")  # type: ignore


# def test_multiple_inheritance_annotation_inheritance():
#     class BaseA(BasePlugin):
#         @abstractmethod
#         def method_a(self, x: int) -> str:
#             pass

#     class BaseB(BasePlugin):
#         @abstractmethod
#         def method_b(self, y: float) -> bool:
#             pass

#     class ConcreteClass(BaseA, BaseB):
#         def method_a(self, x):
#             return str(x)

#         def method_b(self, y):
#             return y > 0

#     concrete_instance = ConcreteClass()

#     method_a_annotations = get_type_hints(concrete_instance.method_a)
#     method_b_annotations = get_type_hints(concrete_instance.method_b)

#     assert "x" in method_a_annotations
#     assert method_a_annotations["x"] is int
#     assert "return" in method_a_annotations
#     assert method_a_annotations["return"] is str

#     assert "y" in method_b_annotations
#     assert method_b_annotations["y"] is float
#     assert "return" in method_b_annotations
#     assert method_b_annotations["return"] is bool


# def test_base_plugin_allows_extra_params():
#     class TestPlugin(BasePlugin):
#         def __init__(self, param1: int, param2: str, extra_param: str):
#             super().__init__(param1=param1, param2=param2)
#             self.extra_param = extra_param

#     # Create an instance with extra parameters
#     plugin = TestPlugin(param1=1, param2="test", extra_param="extra")

#     # Check if the extra parameter is stored
#     assert hasattr(plugin, "extra_param")
#     assert plugin.extra_param == "extra"

#     # Verify that the original parameters are also present
#     assert plugin.param1 == 1  # type: ignore
#     assert plugin.param2 == "test"  # type: ignore

#     # Check if the extra parameter is included in the model dump
#     dump = plugin.model_dump()
#     assert "extra_param" in dump
#     assert dump["extra_param"] == "extra"


# def test_base_plugin_item_dump():
#     class TestPlugin(BasePlugin):
#         def __init__(self, param1: int, param2: str, **kwargs):
#             super().__init__(param1=param1, param2=param2)
#             self._param3 = kwargs["param3"]

#     plugin = TestPlugin(param1=1, param2="test", param3=3)

#     # Test default behavior
#     dump = plugin.item_dump()

#     assert isinstance(dump, dict)
#     assert "param1" in dump["params"]
#     assert dump["params"]["param1"] == 1
#     assert "param2" in dump["params"]
#     assert dump["params"]["param2"] == "test"

#     # Test with exclude
#     dump = plugin.item_dump(exclude={"param2"})
#     assert "param1" in dump["params"]
#     assert "param2" not in dump["params"]

#     # Test with include
#     dump = plugin.item_dump(include={"_param3"})
#     assert "param1" in dump["params"]
#     assert "param2" in dump["params"]
#     assert "_param3" in dump

#     # Test with mode='json'
#     extra = plugin.get_extra()
#     assert "_param3" in extra

#     assert isinstance(dump, dict)
#     assert all(isinstance(key, str) for key in dump.keys())


# def test_base_filter_subclass_initialization():
#     class ConcreteFilter(BaseFilter):
#         def fit(self, x: XYData, y: Optional[XYData]) -> None:
#             pass

#         def predict(self, x: XYData) -> XYData:
#             return XYData.mock(x.value)

#     concrete_filter = ConcreteFilter()

#     assert hasattr(concrete_filter, "fit")
#     assert hasattr(concrete_filter, "predict")
#     assert callable(concrete_filter.fit)
#     assert callable(concrete_filter.predict)

#     # Test fit method
#     x_data = XYData.mock(np.array([[1, 2], [3, 4]]))
#     y_data = XYData.mock(np.array([0, 1]))

#     concrete_filter.fit(x_data, y_data)

#     # Test predict method
#     x_test = XYData.mock(np.array([[5, 6]]))
#     result = concrete_filter.predict(x_test)
#     assert isinstance(result.value, np.ndarray)
#     assert result.value.shape == x_test.value.shape


# def test_basepipeline_abstract_methods():
#     class IncompleteBasePipeline(BasePipeline):
#         def fit(self, x: XYData, y: Optional[XYData]) -> None:
#             pass

#         def predict(self, x: XYData) -> XYData:
#             return x

#     with pytest.raises(TypeError) as excinfo:
#         IncompleteBasePipeline()  # type: ignore

#     assert "Can't instantiate abstract class IncompleteBasePipeline" in str(
#         excinfo.value
#     )
#     assert "abstract methods" in str(excinfo.value)
#     assert "start" in str(excinfo.value)

#     class CompleteBasePipeline(BasePipeline):
#         def fit(self, x: XYData, y: Optional[XYData]) -> None:
#             pass

#         def predict(self, x: XYData) -> XYData:
#             return x

#         def start(
#             self, x: XYData, y: Optional[XYData], X_: Optional[XYData]
#         ) -> Optional[XYData]:
#             return None

#         def evaluate(
#             self, x_data: XYData, y_true: XYData | None, y_pred: XYData
#         ) -> Dict[str, float]:
#             return {}

#     # Should not raise any exception
#     CompleteBasePipeline()


# def test_base_metric_evaluate_implementation():
#     class ConcreteMetric(BaseMetric):
#         def evaluate(
#             self, x_data: XYData, y_true: XYData | None, y_pred: XYData
#         ) -> float:
#             return 0.5

#     class InvalidMetric(BaseMetric):
#         pass

#     # Valid implementation
#     concrete_metric = ConcreteMetric()
#     result = concrete_metric.evaluate(
#         XYData.mock(np.array([1, 2, 3])),
#         XYData.mock(np.array([1, 2, 3])),
#         XYData.mock(np.array([1, 2, 3])),
#     )
#     assert isinstance(result, (float, np.ndarray))

#     # Invalid implementation (missing evaluate method)
#     with pytest.raises(TypeError):
#         InvalidMetric()  # type: ignore

#     # Check if the evaluate method has the correct signature
#     evaluate_signature = inspect.signature(ConcreteMetric.evaluate)
#     expected_params = ["self", "x_data", "y_true", "y_pred"]
#     assert list(evaluate_signature.parameters.keys()) == expected_params
#     assert evaluate_signature.return_annotation is float


# def test_no_init_method_defined():
#     class NoInitPlugin(BasePlugin):
#         def some_method(self):
#             pass

#     # Should not raise any exceptions
#     plugin = NoInitPlugin()

#     # Check if the class is properly created
#     assert isinstance(plugin, NoInitPlugin)
#     assert isinstance(plugin, BasePlugin)

#     # Ensure that the method is present and callable
#     assert hasattr(plugin, "some_method")
#     assert callable(plugin.some_method)

#     # Verify that type checking is still applied to other methods
#     with pytest.raises(TypeError):
#         plugin.some_method("invalid argument")  # type: ignore


# def test_base_filter_equality():
#     class ExampleFilter(BaseFilter):
#         def fit(self, x: XYData, y: Optional[XYData]) -> None:
#             pass

#         def predict(self, x: XYData) -> XYData:
#             return x

#     # Test equality
#     filter1 = ExampleFilter(bert_uncased="bert-base-uncased")
#     filter2 = ExampleFilter(bert_uncased="bert-base-uncased")
#     assert filter1 == filter2 == ExampleFilter(bert_uncased="bert-base-uncased")

#     # Test inequality
#     filter3 = ExampleFilter(bert_uncased="bert-large-uncased")
#     assert filter1 != filter3

#     # Test inequality with different types
#     assert filter1 != "not a filter"

#     # Test hash consistency
#     assert hash(filter1) == hash(filter2)
#     assert hash(filter1) != hash(filter3)


# def test_base_filter_repr():
#     class ExampleFilter(BaseFilter):
#         def fit(self, x: XYData, y: Optional[XYData]) -> None:
#             pass

#         def predict(self, x: XYData) -> XYData:
#             return x

#     filter = ExampleFilter(bert_uncased="bert-base-uncased")
#     expected_repr = "ExampleFilter({'bert_uncased': 'bert-base-uncased'})"
#     assert repr(filter) == expected_repr


# def test_base_filter_with_multiple_params():
#     class ComplexFilter(BaseFilter):
#         def __init__(self, param1: int, param2: str):
#             ...
#             # super().__init__(param1=param1, param2=param2)

#         def fit(self, x: XYData, y: Optional[XYData]) -> None:
#             pass

#         def predict(self, x: XYData) -> XYData:
#             return x

#     filter1 = ComplexFilter(param1=1, param2="test")
#     filter2 = ComplexFilter(param1=1, param2="test")
#     filter3 = ComplexFilter(param1=1, param2="different")

#     assert filter1 == filter2
#     assert filter1 != filter3
#     assert hash(filter1) == hash(filter2)
#     assert hash(filter1) != hash(filter3)
