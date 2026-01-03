from typing import Dict, Iterator, Tuple, Type, Generic
from labchain.base.base_types import TypePlugable
from rich import print as rprint

__all__ = ["BaseFactory"]


class BaseFactory(Generic[TypePlugable]):
    """
    A generic factory class for managing and creating pluggable components.

    This class provides a flexible way to register, retrieve, and manage
    different types of components (plugins) in the framework.

    Key Features:
        - Dynamic registration and retrieval of components
        - Support for attribute-style and dictionary-style access
        - Iteration over registered components
        - Rich printing of available components

    Usage:
        To create a new factory for a specific type of component, inherit from this class
        and specify the type of components it will manage. For example:

        ```python
        from framework3.base.base_factory import BaseFactory
        from framework3.base.base_plugin import BasePlugin

        class MyComponentFactory(BaseFactory[BasePlugin]):
            pass

        factory = MyComponentFactory()
        factory['ComponentA'] = ComponentA
        factory['ComponentB'] = ComponentB

        component_a = factory['ComponentA']()
        component_b = factory['ComponentB']()
        ```

    Attributes:
        _foundry (Dict[str, Type[TypePlugable]]): Internal dictionary to store registered components.

    Methods:
        __getattr__(name: str) -> Type[TypePlugable]:
            Retrieve a component by attribute access.

        __setattr__(name: str, value: Type[TypePlugable]) -> None:
            Set a component by attribute assignment.

        __setitem__(name: str, value: Type[TypePlugable]) -> None:
            Set a component using dictionary-like syntax.

        __getitem__(name: str, default: Type[TypePlugable] | None = None) -> Type[TypePlugable]:
            Retrieve a component using dictionary-like syntax.

        __iter__() -> Iterator[Tuple[str, Type[TypePlugable]]]:
            Provide an iterator over the registered components.

        __contains__(item: str) -> bool:
            Check if a component is registered in the factory.

        get(name: str, default: Type[TypePlugable] | None = None) -> Type[TypePlugable]:
            Retrieve a component by name.

        print_available_components() -> None:
            Print a list of all available components in the factory.

    Note:
        This class uses Generic[TypePlugable] to allow type hinting for the specific
        type of components managed by the factory.
    """

    def __init__(self):
        """
        Initialize the BaseFactory with an empty dictionary to store components.
        """
        self._foundry: Dict[str, Type[TypePlugable]] = {}

    def __getattr__(self, name: str) -> Type[TypePlugable]:
        """
        Retrieve a component by attribute access.

        This method allows components to be accessed as if they were attributes
        of the factory instance.

        Args:
            name (str): The name of the component to retrieve.

        Returns:
            Type[TypePlugable]: The requested component class.

        Raises:
            AttributeError: If the component is not found in the factory.

        Example:
            ```python
            factory = MyComponentFactory()
            component_class = factory.ComponentA
            ```
        """
        if name in self._foundry:
            return self._foundry[name]
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    def __setattr__(self, name: str, value: Type[TypePlugable]) -> None:
        """
        Set a component by attribute assignment.

        This method allows components to be registered as if they were attributes
        of the factory instance.

        Args:
            name (str): The name to assign to the component.
            value (Type[TypePlugable]): The component class to register.

        Example:
            ```python
            factory = MyComponentFactory()
            factory.ComponentA = ComponentA
            ```
        """
        if name.startswith("_") or name in ("manager",):
            object.__setattr__(self, name, value)
        else:
            self._foundry[name] = value

    def __setitem__(self, name: str, value: Type[TypePlugable]) -> None:
        """
        Set a component using dictionary-like syntax.

        This method allows components to be registered using dictionary-style
        item assignment.

        Args:
            name (str): The name to assign to the component.
            value (Type[TypePlugable]): The component class to register.

        Example:
            ```python
            factory = MyComponentFactory()
            factory['ComponentA'] = ComponentA
            ```
        """
        if name == "_foundry":
            super().__setattr__(name, value)
        else:
            self._foundry[name] = value

    def __getitem__(
        self, name: str, default: Type[TypePlugable] | None = None
    ) -> Type[TypePlugable]:
        """
        Retrieve a component using dictionary-like syntax.

        This method allows components to be accessed using dictionary-style
        item retrieval.

        Args:
            name (str): The name of the component to retrieve.
            default (Type[TypePlugable] | None, optional): Default value if component is not found.

        Returns:
            Type[TypePlugable]: The requested component class or the default value.

        Raises:
            AttributeError: If the component is not found and no default is provided.

        Example:
            ```python
            factory = MyComponentFactory()
            component_class = factory['ComponentA']
            ```
        """
        if name in self._foundry:
            return self._foundry[name]
        else:
            if default is None:
                raise AttributeError(
                    f"'{self.__class__.__name__}' object has no attribute '{name}'"
                )
            return default

    def __iter__(self) -> Iterator[Tuple[str, Type[TypePlugable]]]:
        """
        Provide an iterator over the registered components.

        This method allows iteration over the (name, component) pairs in the factory.

        Returns:
            Iterator[Tuple[str, Type[TypePlugable]]]: An iterator of (name, component) pairs.

        Example:
            ```python
            factory = MyComponentFactory()
            for name, component_class in factory:
                print(f"{name}: {component_class}")
            ```
        """
        return iter(self._foundry.items())

    def __contains__(self, item: str) -> bool:
        """
        Check if a component is registered in the factory.

        This method allows the use of the 'in' operator to check for component existence.

        Args:
            item (str): The name of the component to check.

        Returns:
            bool: True if the component is registered, False otherwise.

        Example:
            ```python
            factory = MyComponentFactory()
            if 'ComponentA' in factory:
                print("ComponentA is available")
            ```
        """
        return item in self._foundry

    def get(
        self, name: str, default: Type[TypePlugable] | None = None
    ) -> Type[TypePlugable] | None:
        """
        Retrieve a component by name.

        This method provides a way to safely retrieve components with an optional default value.

        Args:
            name (str): The name of the component to retrieve.
            default (Type[TypePlugable] | None, optional): Default value if component is not found.

        Returns:
            Type[TypePlugable]: The requested component class or the default value.

        Raises:
            AttributeError: If the component is not found and no default is provided.

        Example:
            ```python
            factory = MyComponentFactory()
            component_class = factory.get('ComponentA', DefaultComponent)
            ```
        """
        if name in self._foundry:
            return self._foundry[name]
        else:
            return default

    def print_available_components(self):
        """
        Print a list of all available components in the factory.

        This method uses rich formatting to display the components in a visually appealing way.

        Example:
            ```python
            factory = MyComponentFactory()
            factory.print_available_components()
            ```
        """
        rprint(f"[bold]Available {self.__class__.__name__[:-7]}s:[/bold]")
        for name, binding in self._foundry.items():
            rprint(f"  - [green]{name}[/green]: {binding}")
