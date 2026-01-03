# labchain/base/base_clases.py

from __future__ import annotations  # noqa: D100
import hashlib
import inspect
from abc import ABC, abstractmethod
from types import NotImplementedType

from labchain.base.exceptions import NotTrainableFilterError
from typing import (
    Any,
    Dict,
    Generator,
    List,
    Optional,
    Self,
    Tuple,
    Type,
    TypeVar,
    get_type_hints,
)

import numpy as np
from fastapi.encoders import jsonable_encoder
from typeguard import typechecked

from labchain.base.base_factory import BaseFactory
from labchain.base.base_types import Float, XYData

from rich import print as rprint

__all__ = ["BasePlugin", "BaseFilter", "BaseMetric"]

T = TypeVar("T")
B = TypeVar("B", bound="BasePlugin")


class BasePlugin(ABC):
    """
    Base class for all plugins in the framework.

    This abstract class provides core functionality for attribute management,
    serialization, and type checking. It serves as the foundation for all plugin
    types in the framework, ensuring consistent behavior and interfaces.

    Key Features:
        - Automatic separation of public and private attributes
        - Type checking for methods using typeguard
        - Inheritance of type annotations from abstract methods
        - JSON serialization and deserialization
        - Rich representation for debugging
        - Strict attribute tracking rules for hash consistency

    Attribute Rules:
        1. Public attributes must be declared in __init__ signature and passed to super().__init__()
        2. Alternatively, use **kwargs in __init__ to allow dynamic public attributes
        3. Private attributes (_*) can be set freely as internal state
        4. Methods and special attributes (__*) are never tracked

    Usage:
        To create a new plugin type, inherit from this class and implement
        the required methods. For example:
        ```python
        class MyCustomPlugin(BasePlugin):
            def __init__(self, param1: int, param2: str):
                super().__init__(param1=param1, param2=param2)
                self._internal_state = None  # Private: allowed

            def my_method(self):
                # Custom implementation
                pass
        ```

    Attributes:
        _public_attributes (dict): A dictionary containing all public attributes of the plugin.
        _private_attributes (dict): A dictionary containing all private attributes of the plugin.

    Methods:
        __new__(cls, *args, **kwargs):
            Creates a new instance of the plugin and applies type checking.

        __init__(**kwargs):
            Initializes the plugin instance, separating public and private attributes.

        model_dump(**kwargs) -> dict:
            Returns a copy of the public attributes.

        dict(**kwargs) -> dict:
            Alias for model_dump.

        json(**kwargs) -> dict:
            Returns a JSON-encodable representation of the public attributes.

        item_dump(include=[], **kwargs) -> Dict[str, Any]:
            Returns a dictionary representation of the plugin, including its class name and parameters.

        get_extra() -> Dict[str, Any]:
            Returns a copy of the private attributes.

        model_validate(obj) -> BasePlugin:
            Validates and creates an instance from a dictionary.

    Note:
        This class uses the ABC (Abstract Base Class) to define an interface
        that all plugins must adhere to. It also leverages Python's type hinting
        and the typeguard library for runtime type checking.
    """

    _hash: str | None = None

    def __new__(cls: type[Self], *args: Any, **kwargs: Any) -> Self:
        """
        Create a new instance of the BasePlugin class.

        This method applies type checking to the __init__ method and all other methods,
        and inherits type annotations from abstract methods in parent classes.

        Args:
            *args Any: Variable length argument list.
            **kwargs Any: Arbitrary keyword arguments.

        Returns:
            BasePlugin: A new instance of the BasePlugin class.
        """
        instance = super().__new__(cls)

        # Get the signature of the __init__ method
        init_signature = inspect.signature(cls.__init__)

        # Check if __init__ has **kwargs parameter
        has_var_keyword = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in init_signature.parameters.values()
        )

        # Initialize attribute dictionaries and store init signature
        instance.__dict__["_init_signature"] = init_signature
        instance.__dict__["_has_var_keyword"] = has_var_keyword

        # If **kwargs is present, ALL kwargs go to public/private attributes
        # Otherwise, only those in the signature
        if has_var_keyword:
            instance.__dict__["_public_attributes"] = {
                k: v for k, v in kwargs.items() if not k.startswith("_")
            }
            instance.__dict__["_private_attributes"] = {
                k: v for k, v in kwargs.items() if k.startswith("_")
            }
        else:
            instance.__dict__["_public_attributes"] = {
                k: v
                for k, v in kwargs.items()
                if not k.startswith("_") and k in init_signature.parameters
            }
            instance.__dict__["_private_attributes"] = {
                k: v
                for k, v in kwargs.items()
                if k.startswith("_") and k in init_signature.parameters
            }

        instance.__dict__["_initialization_complete"] = False

        # Store the original __init__ if not already wrapped
        if not hasattr(cls, "_original_init_stored"):
            original_init = cls.__init__

            # Apply typechecked FIRST to the original init (before wrapping)
            if original_init is not object.__init__:
                original_init = typechecked(original_init)

            def wrapped_init(self, *init_args, **init_kwargs):
                # Call the type-checked original __init__
                result = original_init(self, *init_args, **init_kwargs)
                # Automatically mark initialization as complete
                if type(self).__name__ == cls.__name__:
                    self.__dict__["_initialization_complete"] = True
                return result

            # Preserve metadata
            wrapped_init.__name__ = getattr(original_init, "__name__", "__init__")
            wrapped_init.__doc__ = getattr(original_init, "__doc__", None)

            # Replace __init__ with wrapped version
            cls.__init__ = wrapped_init  # type: ignore[method-assign]
            cls._original_init_stored = True

        # Inherit type annotations from abstract methods
        cls.__inherit_annotations()

        # Apply typechecked to all OTHER methods (not __init__, already done)
        for attr_name, attr_value in cls.__dict__.items():
            if inspect.isfunction(attr_value) and attr_name not in (
                "__init__",
                "__new__",
            ):
                setattr(cls, attr_name, typechecked(attr_value))

        return instance

    @classmethod
    def __inherit_annotations(cls):
        """
        Inherit type annotations from abstract methods in parent classes.

        This method is responsible for combining type annotations from abstract methods
        in parent classes with those in the concrete methods of the current class.
        This ensures that type hints are properly inherited and can be used for
        type checking and documentation purposes.

        Args:
            cls (type): The class on which this method is called.

        Note:
            This method modifies the `__annotations__` attribute of concrete methods
            in the class, combining them with annotations from corresponding abstract
            methods in parent classes.
        """
        for base in cls.__bases__:
            for name, method in base.__dict__.items():
                if getattr(method, "__isabstractmethod__", False):
                    if hasattr(cls, name):
                        concrete_method = getattr(cls, name)
                        abstract_annotations = get_type_hints(method)
                        concrete_annotations = get_type_hints(concrete_method)
                        combined_annotations = {
                            **abstract_annotations,
                            **concrete_annotations,
                        }
                        setattr(
                            concrete_method, "__annotations__", combined_annotations
                        )

    def __init__(self, **kwargs: Any):
        """
        Initialize the BasePlugin instance.

        This method is called after __new__ has set up the attribute dictionaries.
        It marks initialization as complete to enable strict attribute checking.

        Args:
            **kwargs (Any): Arbitrary keyword arguments that will be stored as attributes.
        """
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __getattr__(self, name: str) -> Any:
        """
        Custom attribute getter that checks both public and private attribute dictionaries.

        Args:
            name (str): The name of the attribute to retrieve.

        Returns:
            Any: The value of the requested attribute.

        Raises:
            AttributeError: If the attribute is not found in either public or private dictionaries.
        """
        if name in self.__dict__.get("_public_attributes", {}):
            return self.__dict__["_public_attributes"][name]
        elif name in self.__dict__.get("_private_attributes", {}):
            return self.__dict__["_private_attributes"][name]
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    def __setattr__(self, name: str, value: Any):
        """
        Custom attribute setter that separates public and private attributes.

        This method enforces the framework's attribute tracking rules:
        1. Private attributes (_*) can be set freely (internal state)
        2. Callables and special attributes (__*) are never tracked
        3. Public attributes must be declared in __init__ signature OR
           __init__ must have **kwargs to allow dynamic attributes
        4. Violation of rule 3 raises AttributeError with helpful message

        Args:
            name (str): The name of the attribute to set.
            value (Any): The value to assign to the attribute.

        Raises:
            AttributeError: If attempting to set an undeclared public attribute
                           without **kwargs in the constructor.
        """
        if "_public_attributes" not in self.__dict__:
            # During __new__ phase, use default behavior
            super().__setattr__(name, value)
            return

        # Skip callables and special attributes
        if callable(value) or name.startswith("__"):
            super().__setattr__(name, value)
            return

        initialization_complete = self.__dict__.get("_initialization_complete", False)

        # Private attributes are always allowed
        if name.startswith("_"):
            self.__dict__["_private_attributes"][name] = value
            super().__setattr__(name, value)
            return

        # For public attributes, check if allowed
        init_signature = self.__dict__.get("_init_signature")
        has_kwargs = self.__dict__.get("_has_var_keyword", False)

        is_init_param = init_signature and name in init_signature.parameters

        if not initialization_complete:
            # During initialization: only track if it's an init parameter OR we have **kwargs
            if is_init_param or has_kwargs:
                self.__dict__["_public_attributes"][name] = value
                super().__setattr__(name, value)
            else:
                # Trying to set a public attribute that's not in the signature and no **kwargs
                available_params = (
                    list(init_signature.parameters.keys()) if init_signature else []
                )
                available_params = [
                    p for p in available_params if p not in ("self", "args", "kwargs")
                ]

                raise AttributeError(
                    f"Cannot set public attribute '{name}' on {self.__class__.__name__}. "
                    f"This attribute is not declared in the __init__ signature.\n\n"
                    f"Available options:\n"
                    f"1. Add '{name}' to __init__ signature and pass to constructor:\n"
                    f"   def __init__(self, {', '.join(available_params + [name])}):\n"
                    f"       # Then instantiate with: {self.__class__.__name__}({', '.join(f'{p}=...' for p in available_params + [name])})\n\n"
                    f"2. Add **kwargs to __init__ to accept dynamic attributes:\n"
                    f"   def __init__(self, {', '.join(available_params)}, **kwargs):\n\n"
                    f"3. Make it private (internal state) by renaming to '_{name}'\n\n"
                    f"Current __init__ parameters: {available_params}"
                )
        else:
            # After initialization: can only update existing attributes or if we have **kwargs
            if has_kwargs or name in self.__dict__["_public_attributes"]:
                self.__dict__["_public_attributes"][name] = value
                super().__setattr__(name, value)
            else:
                available_params = (
                    list(init_signature.parameters.keys()) if init_signature else []
                )
                available_params = [
                    p for p in available_params if p not in ("self", "args", "kwargs")
                ]

                raise AttributeError(
                    f"Cannot set public attribute '{name}' on {self.__class__.__name__}. "
                    f"This attribute is not declared in the __init__ signature.\n\n"
                    f"Available options:\n"
                    f"1. Declare '{name}' in __init__ and pass to constructor\n"
                    f"2. Add **kwargs to __init__ for dynamic attributes\n"
                    f"3. Make it private by renaming to '_{name}'\n\n"
                    f"Current __init__ parameters: {available_params}"
                )

    def __repr__(self) -> str:
        """
        String representation of the plugin, showing its class name and public attributes.

        Returns:
            str: A string representation of the plugin.
        """
        return f"{self.__class__.__name__}({self._public_attributes})"

    def model_dump(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Return a copy of the public attributes.

        Args:
            **kwargs (Any): Additional keyword arguments (not used in this method).

        Returns:
            Dict[str, Any]: A copy of the public attributes.
        """
        return self._public_attributes.copy()

    def dict(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Alias for model_dump.

        Args:
            **kwargs (Any): Additional keyword arguments passed to model_dump.

        Returns:
            Dict[str, Any]: A copy of the public attributes.
        """
        return self.model_dump(**kwargs)

    def json(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Return a JSON-encodable representation of the public attributes.

        Args:
            **kwargs (Any): Additional keyword arguments passed to jsonable_encoder.

        Returns:
            Dict[str, Any]: A JSON-encodable representation of the public attributes.
        """
        return jsonable_encoder(self._public_attributes, **kwargs)

    def item_dump(self, include=[], **kwargs: Any) -> Dict[str, Any]:
        """
        Return a dictionary representation of the plugin, including its class name and parameters.

        Args:
            include (list): A list of private attributes to include in the dump.
            **kwargs (Any): Additional keyword arguments passed to jsonable_encoder.

        Returns:
            Dict[str, Any]: A dictionary representation of the plugin.
        """
        included = {k: v for k, v in self._private_attributes.items() if k in include}
        dump = {
            "clazz": self.__class__.__name__,
            "version_hash": self._hash,
            "params": jsonable_encoder(
                self._public_attributes,
                custom_encoder={
                    BasePlugin: lambda v: v.item_dump(include=include, **kwargs),
                    type: lambda v: {
                        "clazz": v.__name__,
                        "version_hash": v._hash,
                    },
                    np.integer: lambda x: int(x),
                    np.floating: lambda x: float(x),
                },
                **kwargs,
            ),
        }
        if include != []:
            dump.update(
                **jsonable_encoder(
                    included,
                    custom_encoder={
                        BasePlugin: lambda v: v.item_dump(include=include, **kwargs),
                        type: lambda v: {
                            "clazz": v.__name__,
                            "version_hash": v._hash,
                        },
                        np.integer: lambda x: int(x),
                        np.floating: lambda x: float(x),
                    },
                    **kwargs,
                )
            )

        return dump

    def get_extra(self) -> Dict[str, Any]:
        """
        Return a copy of the private attributes.

        Returns:
            Dict[str, Any]: A copy of the private attributes.
        """
        return self._private_attributes.copy()

    def __getstate__(self) -> Dict[str, Any]:
        """
        Prepare the object for pickling.

        Returns:
            Dict[str, Any]: A copy of the object's __dict__.
        """
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state: Dict[str, Any]):
        """
        Restore the object from its pickled state.

        Args:
            state (Dict[str, Any]): The pickled state of the object.
        """
        self.__dict__.update(state)

    @classmethod
    def model_validate(cls, obj: object) -> BasePlugin:
        """
        Validate and create an instance from a dictionary.

        Args:
            obj (Object): The object to validate and create an instance from.

        Returns:
            BasePlugin: An instance of the class.

        Raises:
            ValueError: If the input is not a dictionary.
        """
        if isinstance(obj, dict):
            return cls(**obj)
        raise ValueError(f"Cannot validate {type(obj)}")

    def __rich_repr__(self) -> Generator[Any, Any, Any]:
        """
        Rich representation of the plugin, used by the rich library.

        Yields:
            Generator[Any, Any, Any]: Key-value pairs of public attributes.
        """
        for key, value in self._public_attributes.items():
            yield key, value

    @staticmethod
    def build_from_dump(
        dump_dict: Dict[str, Any], factory: BaseFactory[BasePlugin]
    ) -> BasePlugin | Type[BasePlugin]:
        """
        Reconstruct a plugin instance from a dumped dictionary representation.

        This method handles nested plugin structures and uses a factory to create instances.

        Args:
            dump_dict (Dict[str, Any]): The dumped dictionary representation of the plugin.
            factory (BaseFactory[BasePlugin]): A factory for creating plugin instances.

        Returns:
            BasePlugin | Type[BasePlugin]: The reconstructed plugin instance or class.
        """

        clazz_name: str = dump_dict["clazz"]
        v_hash: str | None = dump_dict.get("version_hash")

        level_clazz: Type[BasePlugin] | None
        if v_hash and hasattr(factory, "get_version"):
            level_clazz = factory.get_version(clazz_name, v_hash)  # type: ignore
        else:
            level_clazz = factory.get(clazz_name)

        if level_clazz is None:
            raise ValueError(f"No class found for {clazz_name}")  # type: ignore

        if "params" in dump_dict:
            level_params: Dict[str, Any] = {}
            for k, v in dump_dict["params"].items():
                if isinstance(v, dict):
                    if "clazz" in v:
                        level_params[k] = BasePlugin.build_from_dump(v, factory)
                    else:
                        level_params[k] = v
                elif isinstance(v, list):
                    items: List[Any] = []
                    for i in v:
                        if isinstance(i, dict):
                            if "clazz" in i:
                                items.append(BasePlugin.build_from_dump(i, factory))
                            else:
                                items.append(i)
                        else:
                            items.append(i)

                    level_params[k] = items
                else:
                    level_params[k] = v
            return level_clazz(**level_params)
        else:
            return level_clazz


class BaseFilter(BasePlugin):
    """
    Base class for filter components in the framework.

    This abstract class extends BasePlugin and provides a structure for implementing
    filter operations, including fit and predict methods. It serves as the foundation
    for all filter types in the framework, ensuring consistent behavior and interfaces
    for machine learning operations.

    Key Features:
        - Implements fit and predict methods for machine learning operations
        - Provides caching mechanisms for model and data storage
        - Supports verbose output for debugging and monitoring
        - Implements equality and hashing methods for filter comparison
        - Supports serialization and deserialization of filter instances

    Usage:
        To create a new filter type, inherit from this class and implement
        the required methods. For example:
        ```python
        class MyCustomFilter(BaseFilter):
            def __init__(self, n_components: int = 2):
                super().__init__(n_components=n_components)
                self._model = None  # Private: internal state

            def fit(self, x: XYData, y: Optional[XYData] = None) -> None:
                self._print_acction("Fitting MyCustomFilter")
                # Implement fitting logic here
                data = x.value
                self._model = np.linalg.svd(data - np.mean(data, axis=0), full_matrices=False)

            def predict(self, x: XYData) -> XYData:
                self._print_acction("Predicting with MyCustomFilter")
                if self._model is None:
                    raise ValueError("Model not fitted yet.")
                # Implement prediction logic here
                data = x.value
                U, s, Vt = self._model
                transformed = np.dot(data - np.mean(data, axis=0), Vt.T[:, :self.n_components])
                return XYData(_value=transformed, _hash=x._hash, _path=self._m_path)
        ```

    Attributes:
        _verbose (bool): Controls the verbosity of output.
        _m_hash (str): Hash of the current model.
        _m_str (str): String representation of the current model.
        _m_path (str): Path to the current model.
        _original_fit (method): Reference to the original fit method.
        _original_predict (method): Reference to the original predict method.

    Methods:
        __init__(verbose=True, *args, **kwargs):
            Initializes the filter instance, setting up attributes and method wrappers.

        fit(x: XYData, y: Optional[XYData]) -> Optional[float]:
            Fits the filter to the input data.

        predict(x: XYData) -> XYData:
            Makes predictions using the fitted filter.

        verbose(value: bool) -> None:
            Sets the verbosity level for output.

        init() -> None:
            Initializes filter-specific attributes.

        _get_model_key(data_hash: str) -> Tuple[str, str]:
            Generates a unique key for the model.

        _get_data_key(model_str: str, data_hash: str) -> Tuple[str, str]:
            Generates a unique key for the data.

        grid(grid: Dict[str, List[Any] | Tuple[Any, Any]]) -> BaseFilter:
            Sets up grid search parameters.

        unwrap() -> BaseFilter:
            Returns the base filter without any wrappers.

    Note:
        This is an abstract base class. Concrete implementations should override
        the fit and predict methods to provide specific functionality.
    """

    def _print_acction(self, action: str) -> None:
        """
        Print an action message with formatting.

        This method is used for verbose output to indicate the current action being performed.

        Args:
            action (str): The action message to be printed.

        Returns:
            None
        """
        s_str = "_" * 100
        s_str += f"\n{action}...\n"
        s_str += "*" * 100

        if self._verbose:
            rprint(s_str)

    def verbose(self, value: bool) -> None:
        """
        Set the verbosity of the filter.

        Args:
            value (bool): If True, enables verbose output; if False, disables it.

        Returns:
            None
        """
        self._verbose = value

    def __init__(self, verbose=True, *args: Any, **kwargs: Any):
        """
        Initialize the BaseFilter instance.

        This method sets up attributes for storing model-related information and wraps
        the fit and predict methods with pre-processing steps.

        Args:
            verbose (bool, optional): If True, enables verbose output. Defaults to True.
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.
        """
        self._verbose = verbose
        self._original_fit = self.fit
        self._original_predict = self.predict

        # Replace fit and predict methods - use __dict__ directly to avoid __setattr__
        if hasattr(self, "fit"):
            self.__dict__["fit"] = self._pre_fit_wrapp
        if hasattr(self, "predict"):
            self.__dict__["predict"] = self._pre_predict_wrapp

        super().__init__(*args, **kwargs)

        m_hash, m_str = self._get_model_key(data_hash=" , ")

        self._m_hash: str = m_hash
        self._m_str: str = m_str
        self._m_path: str = f"{self._get_model_name()}/{m_hash}"

    def __eq__(self, other: object) -> bool | NotImplementedType:
        """
        Check equality between this filter and another object.

        Two filters are considered equal if they are of the same type and have the same public attributes.

        Args:
            other (object): The object to compare with this filter.

        Returns:
            bool: True if the objects are equal, False otherwise.
        """
        if not isinstance(other, BaseFilter):
            return NotImplemented
        return (
            type(self) is type(other)
            and self._public_attributes == other._public_attributes
        )

    def __hash__(self) -> int:
        """
        Generate a hash value for this filter.

        The hash is based on the filter's type and its public attributes.

        Returns:
            int: The hash value of the filter.
        """
        return hash((type(self), frozenset(self._public_attributes.items())))

    def _pre_fit(self, x: XYData, y: Optional[XYData] = None) -> Tuple[str, str, str]:
        """
        Perform pre-processing steps before fitting the model.

        This method generates and sets the model hash, path, and string representation.

        Args:
            x (XYData): The input data.
            y (Optional[XYData]): The target data, if applicable.

        Returns:
            Tuple[str, str, str]: A tuple containing the model hash, path, and string representation.
        """
        m_hash, m_str = self._get_model_key(
            data_hash=f'{x._hash}, {y._hash if y is not None else ""}'
        )
        m_path = f"{self._get_model_name()}/{m_hash}"

        print(f"Calling prefit on {self.__class__.__name__}")

        self._m_hash = m_hash
        self._m_path = m_path
        self._m_str = m_str
        return m_hash, m_path, m_str

    def _pre_predict(self, x: XYData) -> XYData:
        """
        Perform pre-processing steps before making predictions.

        This method generates a new XYData object with updated hash and path.

        Args:
            x (XYData): The input data for prediction.

        Returns:
            XYData: A new XYData object with updated hash and path.

        Raises:
            ValueError: If the model has not been trained or loaded.
        """
        try:
            d_hash, _ = self._get_data_key(self._m_str, x._hash)

            new_x = XYData(
                _hash=d_hash,
                _value=x._value,
                _path=f"{self._get_model_name()}/{self._m_hash}",
            )

            return new_x

        except Exception:
            raise ValueError("Trainable filter model not trained or loaded")

    def _pre_fit_wrapp(
        self, x: XYData, y: Optional[XYData] = None
    ) -> Optional[float | dict]:
        """
        Wrapper method for the fit function.

        This method performs pre-processing steps before calling the original fit method.

        Args:
            x (XYData): The input data.
            y (Optional[XYData]): The target data, if applicable.

        Returns:
            Optional[float]: The result of the original fit method.
        """
        m_hash = self._m_hash
        m_path = self._m_path
        m_str = self._m_str
        try:
            self._pre_fit(x, y)
            res = self._original_fit(x, y)
        except Exception as e:
            self._m_hash = m_hash
            self._m_path = m_path
            self._m_str = m_str
            raise e
        return res

    def _pre_predict_wrapp(self, x: XYData) -> XYData:
        """
        Wrapper method for the predict function.

        This method performs pre-processing steps before calling the original predict method.

        Args:
            x (XYData): The input data for prediction.

        Returns:
            XYData: The prediction results with updated hash and path.
        """
        new_x = self._pre_predict(x)
        return XYData(
            _hash=new_x._hash,
            _path=new_x._path,
            _value=self._original_predict(x)._value,
        )

    def __getstate__(self) -> Dict[str, Any]:
        """
        Prepare the object for pickling.

        This method ensures that the original fit and predict methods are stored for serialization.

        Returns:
            Dict[str, Any]: The object's state dictionary.
        """
        state = super().__getstate__()
        # Ensure we're storing the original methods for serialization
        state["fit"] = self._original_fit
        state["predict"] = self._original_predict
        return state

    def __setstate__(self, state: Dict[str, Any]):
        """
        Restore the object from its pickled state.

        This method restores the wrapper methods after deserialization.

        Args:
            state (Dict[str, Any]): The pickled state of the object.
        """
        super().__setstate__(state)
        # Restore the wrapper methods after deserialization
        self.__dict__["fit"] = self._pre_fit_wrapp
        self.__dict__["predict"] = self._pre_predict_wrapp

    def fit(self, x: XYData, y: Optional[XYData]) -> Optional[float | dict]:
        """
        Method for fitting the filter to the data.

        This method should be overridden by subclasses to implement specific fitting logic.

        Args:
            x (XYData): The input data.
            y (Optional[XYData]): The target data, if applicable.

        Returns:
            Optional[float]: An optional float value, typically used for metrics or loss.

        Raises:
            NotTrainableFilterError: If the filter does not support fitting.
        """
        raise NotTrainableFilterError("This filter does not support fitting.")

    @abstractmethod
    def predict(self, x: XYData) -> XYData:
        """
        Abstract method for making predictions using the filter.

        This method must be implemented by subclasses to provide specific prediction logic.

        Args:
            x (XYData): The input data.

        Returns:
            XYData: The prediction results.
        """
        ...

    def _get_model_name(self) -> str:
        """
        Get the name of the model.

        Returns:
            str: The name of the model (class name).
        """
        return self.__class__.__name__

    def _get_model_key(self, data_hash: str) -> Tuple[str, str]:
        """
        Generate a unique key for the model based on its parameters and input data.

        Args:
            data_hash (str): A hash representing the input data.

        Returns:
            Tuple[str, str]: A tuple containing the model hash and a string representation.
        """
        model_str = f"<{self.item_dump(exclude=set('extra_params'))}>({data_hash})"
        model_hashcode = hashlib.sha1(model_str.encode("utf-8")).hexdigest()
        return model_hashcode, model_str

    def _get_data_key(self, model_str: str, data_hash: str) -> Tuple[str, str]:
        """
        Generate a unique key for the data based on the model and input data.

        Args:
            model_str (str): A string representation of the model.
            data_hash (str): A hash representing the input data.

        Returns:
            Tuple[str, str]: A tuple containing the data hash and a string representation.
        """
        data_str = f"{model_str}.predict({data_hash})"
        data_hashcode = hashlib.sha1(data_str.encode("utf-8")).hexdigest()
        return data_hashcode, data_str

    def grid(self, grid: Dict[str, List[Any] | Tuple[Any, Any] | dict]) -> BaseFilter:
        """
        Set up grid search parameters for the filter.

        This method allows defining a grid of hyperparameters for optimization.

        Args:
            grid (Dict[str, List[Any] | Tuple[Any, Any]]): A dictionary where keys are parameter names
                and values are lists or tuples of possible values.

        Returns:
            BaseFilter: The filter instance with grid search parameters set.
        """
        self._grid = grid
        return self

    def unwrap(self) -> BaseFilter:
        """
        Return the base filter without any wrappers.

        This method is useful when you need to access the original filter without any
        additional layers or modifications added by wrappers.

        Returns:
            BaseFilter: The unwrapped base filter.
        """
        return self

    @staticmethod
    def clear_memory():
        import gc

        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except ImportError:
            pass


class BaseMetric(BasePlugin):
    """
    Base class for implementing metric calculations in the framework.

    This abstract class defines the interface for metric evaluation and provides
    a structure for implementing various performance metrics. It extends BasePlugin
    to inherit core functionality for attribute management and serialization.

    Key Features:
        - Abstract evaluate method for implementing specific metric calculations
        - higher_better attribute to indicate if higher metric values are better
        - Inherits BasePlugin functionality for attribute management and serialization

    Usage:
        To create a new metric, inherit from this class and implement the evaluate method:
        ```python
        from framework3.base.base_clases import BaseMetric
        from framework3.base.base_types import XYData
        import numpy as np

        class MeanSquaredError(BaseMetric):
            higher_better = False

            def evaluate(self, x_data: XYData, y_true: XYData, y_pred: XYData) -> float:
                return np.mean((y_true.value - y_pred.value) ** 2)
        ```

    Attributes:
        higher_better (bool): Indicates whether higher values of the metric are better.
                              Defaults to True.

    Methods:
        evaluate(x_data: XYData, y_true: XYData | None, y_pred: XYData) -> Float | np.ndarray:
            Abstract method to be implemented by subclasses for specific metric calculations.
    """

    higher_better: bool = True

    @abstractmethod
    def evaluate(
        self, x_data: XYData, y_true: XYData | None, y_pred: XYData
    ) -> Float | np.ndarray:
        """
        Evaluate the metric based on the provided data.

        This abstract method should be implemented by subclasses to calculate
        the specific metric. It provides a standardized interface for all metrics
        in the framework.

        Args:
            x_data (XYData): The input data used for the prediction.
            y_true (XYData | None): The ground truth or actual values. Can be None for some metrics.
            y_pred (XYData): The predicted values.

        Returns:
            Float | np.ndarray: The calculated metric value. This can be a single float
                                or a numpy array, depending on the specific metric implementation.

        Raises:
            NotImplementedError: If the subclass does not implement this method.

        Note:
            Subclasses must override this method to provide the specific metric calculation logic.
        """

        ...
