from functools import singledispatch, update_wrapper
from typing import Any, Callable, Protocol, TypeVar, cast

T = TypeVar("T")
R = TypeVar("R")  # Return type of registered functions


class DispatchableMethod(Protocol[R]):
    """
    Protocol for a dispatchable method.

    This protocol defines the interface for a method that can be dispatched
    based on the type of its arguments and can register new implementations.

    Key Features:
        - Defines a callable interface
        - Provides a register method for new implementations

    Usage:
        This protocol is typically used as a type hint for methods that support
        dynamic dispatch based on argument types.

    Methods:
        __call__(*args: Any, **kwargs: Any) -> R:
            Call the method with the given arguments.
        register(cls: type[T], func: Callable[..., R]) -> Callable[..., R]:
            Register a new implementation for the given class.

    Note:
        This is a Protocol class and is not meant to be instantiated directly.
        It serves as a structural subtyping tool for static type checking.
    """

    def __call__(self, *args: Any, **kwargs: Any) -> R:
        """
        Call the method with the given arguments.

        This method represents the main functionality of the dispatchable method.
        It will be called when the method is invoked and will dispatch to the
        appropriate implementation based on the types of the arguments.

        Args:
            *args (Any): Positional arguments passed to the method.
            **kwargs (Any): Keyword arguments passed to the method.

        Returns:
            R: The result of calling the appropriate implementation of the method.
        """
        ...

    def register(self, cls: type[T], func: Callable[..., R]) -> Callable[..., R]:
        """
        Register a new implementation for the given class.

        This method allows registering new implementations for specific types.
        When the dispatchable method is called with an instance of the registered
        class as its argument, the corresponding implementation will be used.

        Args:
            cls (type[T]): The class for which to register the implementation.
            func (Callable[..., R]): The function to be used as the implementation
                for the given class.

        Returns:
            Callable[..., R]: The registered function, allowing for decorator-style usage.

        Example:
            ```python
            @my_method.register(int)
            def _(self, arg: int):
                return f"Integer implementation: {arg}"
            ```
        """
        ...


class SingleDispatch(Protocol[R]):
    """
    Protocol for a single dispatch function.

    This protocol defines the interface for a function that can be dispatched
    based on the type of its first argument and can register new implementations.

    Key Features:
        - Defines a callable interface
        - Provides methods for registering and dispatching implementations

    Usage:
        This protocol is typically used as a type hint for functions that support
        single dispatch based on the type of the first argument.

    Methods:
        __call__(*args: Any, **kwargs: Any) -> R:
            Call the function with the given arguments.
        register(cls: type, func: Callable[..., R]) -> Callable[..., R]:
            Register a new implementation for the given class.
        dispatch(cls: type) -> Callable[..., R]:
            Return the implementation for the given class.

    Note:
        This is a Protocol class and is not meant to be instantiated directly.
        It serves as a structural subtyping tool for static type checking.
    """

    def __call__(self, *args: Any, **kwargs: Any) -> R:
        """
        Call the function with the given arguments.

        This method represents the main functionality of the single dispatch function.
        It will be called when the function is invoked and will dispatch to the
        appropriate implementation based on the type of the first argument.

        Args:
            *args (Any): Positional arguments passed to the function.
            **kwargs (Any): Keyword arguments passed to the function.

        Returns:
            R: The result of calling the appropriate implementation of the function.
        """
        ...

    def register(self, cls: type, func: Callable[..., R]) -> Callable[..., R]:
        """
        Register a new implementation for the given class.

        This method allows registering new implementations for specific types.
        When the single dispatch function is called with an instance of the registered
        class as its first argument, the corresponding implementation will be used.

        Args:
            cls (type): The class for which to register the implementation.
            func (Callable[..., R]): The function to be used as the implementation
                for the given class.

        Returns:
            Callable[..., R]: The registered function, allowing for decorator-style usage.

        Example:
            ```python
            @process.register(int)
            def _(arg: int):
                return f"Integer implementation: {arg}"
            ```
        """
        ...

    def dispatch(self, cls: type) -> Callable[..., R]:
        """
        Return the implementation for the given class.

        This method is used internally by the single dispatch mechanism to retrieve
        the appropriate implementation for a given type.

        Args:
            cls (type): The class for which to retrieve the implementation.

        Returns:
            Callable[..., R]: The implementation function registered for the given class.

        Note:
            This method is typically used internally by the dispatch mechanism and
            not called directly by users of the single dispatch function.
        """
        ...


def methdispatch(func: Callable[..., R]) -> DispatchableMethod[R]:
    """
    Decorator for creating a method dispatch.

    This decorator creates a wrapper around the given function that dispatches
    based on the type of the second argument (typically 'self' in method calls).

    Key Features:
        - Creates a dispatchable method
        - Dispatches based on the type of the second argument
        - Allows registration of new implementations

    Usage:
        Use this decorator on methods that need to dispatch based on the type
        of their second argument (typically the first argument after 'self').

        ```python
        class MyClass:
            @methdispatch
            def my_method(self, arg):
                return "Default implementation"

            @my_method.register(int)
            def _(self, arg: int):
                return f"Integer implementation: {arg}"
        ```

    Args:
        func (Callable[..., R]): The function to be wrapped.

    Returns:
        DispatchableMethod[R]: A wrapper function with dispatch capabilities.

    Note:
        The wrapped method will dispatch based on the type of the second argument,
        which is typically the first argument after 'self' in method calls.
    """
    dispatcher = singledispatch(func)

    def wrapper(*args, **kw):
        return dispatcher.dispatch(args[1].__class__)(*args, **kw)

    wrapper = cast(DispatchableMethod[R], wrapper)
    setattr(wrapper, "register", dispatcher.register)
    update_wrapper(wrapper, func)
    return wrapper


def fundispatch(func: SingleDispatch[R]) -> SingleDispatch[R]:
    """
    Decorator for creating a function dispatch.

    This decorator creates a wrapper around the given function that dispatches
    based on the type of the first argument.

    Key Features:
        - Creates a dispatchable function
        - Dispatches based on the type of the first argument
        - Allows registration of new implementations

    Usage:
        Use this decorator on functions that need to dispatch based on the type
        of their first argument.

        ```python
        @fundispatch
        def process(arg):
            return "Default implementation"

        @process.register(int)
        def _(arg: int):
            return f"Integer implementation: {arg}"

        @process.register(str)
        def _(arg: str):
            return f"String implementation: {arg}"
        ```

    Args:
        func (SingleDispatch[R]): The function to be wrapped.

    Returns:
        SingleDispatch[R]: A wrapper function with dispatch capabilities.

    Note:
        The wrapped function will dispatch based on the type of the first argument.
        If the first argument is a type object, it will dispatch on that type directly.
        Otherwise, it will dispatch on the type of the first argument.
    """
    dispatcher = singledispatch(func)

    def wrapper(*args: Any, **kwargs: Any) -> R:
        arg_type = args[0] if isinstance(args[0], type) else type(args[0])
        return dispatcher.dispatch(arg_type)(*args, **kwargs)

    wrapper = cast(SingleDispatch[R], wrapper)
    setattr(wrapper, "register", dispatcher.register)
    setattr(wrapper, "dispatch", dispatcher.dispatch)
    update_wrapper(wrapper, func)
    return wrapper
