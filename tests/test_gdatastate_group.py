"""Tests for postgkyl.gdatastate.gdatastategroup.GDataStateGroup -- the verb-less container.

Ported from tests_bak/test_group.py: only the state-concerned tests survive
(construction, flattening, indexing, iteration, combining, repr). Tests that
exercised broadcasting (``__getattr__`` dispatch to member verbs) or terminal
verbs (``plot``/``info``/``animate``/``plotly_animate``/``collect``/``evaluate``) are
dropped here -- those methods are deferred to the layer-10 fluent group; see
that layer's worklist.
"""

from __future__ import annotations

import numpy as np
import pytest

from postgkyl.gdatastate.gdatastategroup import GDataStateGroup
from postgkyl.gdatastate.gdatastate import GDataState


def _line(tag: str = "default", offset: float = 0.0) -> GDataState:
  d = GDataState(tag=tag)
  d.push([np.linspace(0.0, 1.0, 9)], (np.arange(8.0) + offset)[:, None])
  return d


class _SubGData(GDataState):
  """Stand-in for the fluent ``GData`` subclass (layer 10 adds the real one)."""


class TestConstruction:

  def test_from_list(self):
    g = GDataStateGroup([_line("a"), _line("b")])
    assert len(g) == 2

  def test_flattens_nested(self):
    g = GDataStateGroup([_line("a"), [_line("b"), _line("c")]])
    assert len(g) == 3

  def test_flattens_nested_group(self):
    inner = GDataStateGroup([_line("b"), _line("c")])
    g = GDataStateGroup([_line("a"), inner])
    assert len(g) == 3
    assert all(isinstance(d, GDataState) for d in g)

  def test_iter_and_index(self):
    a, b = _line("a"), _line("b")
    g = GDataStateGroup([a, b])
    assert list(g) == [a, b]
    assert g[0] is a

  def test_slice_returns_group(self):
    g = GDataStateGroup([_line("a"), _line("b"), _line("c")])
    assert isinstance(g[:2], GDataStateGroup)
    assert len(g[:2]) == 2

  def test_rejects_non_gdata(self):
    with pytest.raises(TypeError):
      GDataStateGroup([1, 2, 3])

  def test_empty_group_default(self):
    g = GDataStateGroup()
    assert len(g) == 0
    assert list(g) == []

  def test_empty_group_from_empty_list(self):
    g = GDataStateGroup([])
    assert len(g) == 0

  def test_group_of_one(self):
    a = _line("a")
    g = GDataStateGroup([a])
    assert len(g) == 1
    assert g[0] is a

  def test_heterogeneous_member_types(self):
    a = _line("a")
    b = _SubGData(tag="b")
    b.push([np.linspace(0.0, 1.0, 5)], np.arange(4.0)[:, None])
    g = GDataStateGroup([a, b])
    assert len(g) == 2
    assert type(g[0]) is GDataState
    assert isinstance(g[1], _SubGData)


class TestCombining:

  def test_with_appends(self):
    g = GDataStateGroup([_line("a")]).with_(_line("b"), _line("c"))
    assert len(g) == 3

  def test_with_accepts_group(self):
    g = GDataStateGroup([_line("a")]).with_(GDataStateGroup([_line("b")]))
    assert len(g) == 2

  def test_and_operator(self):
    g = GDataStateGroup([_line("a")]) & GDataStateGroup([_line("b")])
    assert len(g) == 2

  def test_with_does_not_mutate(self):
    g = GDataStateGroup([_line("a")])
    g.with_(_line("b"))
    assert len(g) == 1


class TestSequenceAndRepr:

  def test_datasets_is_defensive_copy(self):
    a, b = _line("a"), _line("b")
    g = GDataStateGroup([a, b])
    members = g.datasets
    members.append(_line("c"))
    assert len(g) == 2

  def test_repr_shows_count(self):
    g = GDataStateGroup([_line("a"), _line("b")])
    assert repr(g) == "<GDataStateGroup [2 datasets]>"

  def test_repr_empty(self):
    assert repr(GDataStateGroup()) == "<GDataStateGroup [0 datasets]>"
