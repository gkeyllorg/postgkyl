"""The object-model layer: the verb-less ``GDataState`` container."""

from .gdatastate import GDataState
from .collection import flatten_datasets, group_blocks
from .gdatastategroup import GDataStateGroup

__all__ = ["GDataState", "flatten_datasets", "group_blocks", "GDataStateGroup"]
