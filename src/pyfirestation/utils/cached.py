"""Includes class for caching properties with @cached_property with automatic recalculation on attribute setting.

"""

from functools import cached_property
from typing import Any


class Cached:
    """Base class for caching properties with @cached_property with automatic recalculation on attribute setting.

    Simple base class for constructing classes with cached properties using @cached_property decorators. The setting of
    a class attribute triggers the re-calculation of the cached properties, as to attempt to maintain its integrity.
    This implementation should only be used in simple cases and the user should always guarantee that it is compatible
    with the specific implementation intended. Additionally, when performance is critical, one should opt for a
    different approach, such as creating an attribute/property dependency graph and re-calculating only the required
    fields, as opposed to this class where every cached property is re-calculated upon attribute setting.
    """

    def __setattr__(self, attr: Any, value: Any) -> None:
        """Deletes cached @cached_property attributes/properties upon attribute setting."""
        # Force raising exception upon trying to set cached property
        cls = self.__class__
        if isinstance(getattr(cls, attr, cls), cached_property):
            raise AttributeError(f"can't set attribute {attr}")

        # Set attr and delete cached properties
        super().__setattr__(attr, value)
        self._reload()

    def _reload(self) -> None:
        """Deletes cached @cached_property attributes/properties to force re-calculation.

        Upon attribute modification, the value of cached properties may be inconsistent, so their value should be
        re-calculated. The properties decorated with @cached_property can be forced to be recalculated by deleting
        the respective attribute. This method programmatically iterates over all  the classes attributes and properties
        and deletes the cached properties to force re-calculation.
        Based on: https://stackoverflow.com/questions/73129762/iterate-through-all-cached-property-attributes
        """
        # Get cached properties
        cls = self.__class__
        attrs = [attr for attr in dir(self) if isinstance(getattr(cls, attr, cls), cached_property)]

        # Delete cached attrs if initialized
        for attr in attrs:
            # If the attr/property has not been calculated, delattr will raise an exception
            if attr in self.__dict__:
                delattr(self, attr)

