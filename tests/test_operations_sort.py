"""Tests for the ``sort`` verb and its ``numerics.natural_sort_key`` helper."""

from __future__ import annotations

from postgkyl import numerics, operations
from postgkyl.gdatastate.gdatastate import GDataState


def _named(file_name):
  d = GDataState()
  d._file_name = file_name
  return d


def test_natural_sort_key_orders_embedded_numbers_numerically():
  names = ["field_10.gkyl", "field_1.gkyl", "field_2.gkyl", "field_20.gkyl"]
  assert sorted(names, key=numerics.natural_sort_key) == [
      "field_1.gkyl", "field_2.gkyl", "field_10.gkyl", "field_20.gkyl"
  ]


def test_natural_sort_key_beats_plain_lexicographic_sort():
  # The whole point of natural sort: a plain string sort gets this wrong.
  names = ["field_0.gkyl", "field_1.gkyl", "field_10.gkyl", "field_2.gkyl"]
  assert sorted(names) != sorted(names, key=numerics.natural_sort_key)
  assert sorted(names, key=numerics.natural_sort_key) == [
      "field_0.gkyl", "field_1.gkyl", "field_2.gkyl", "field_10.gkyl"
  ]


def test_sort_reorders_datasets_by_filename():
  a = _named("field_10.gkyl")
  b = _named("field_2.gkyl")
  out = operations.sort(a, b)
  assert [d.file_name for d in out] == ["field_2.gkyl", "field_10.gkyl"]


def test_sort_accepts_a_list_argument():
  frames = [
      _named("field_10.gkyl"),
      _named("field_2.gkyl"),
      _named("field_1.gkyl")
  ]
  out = operations.sort(frames)
  assert [d.file_name
          for d in out] == ["field_1.gkyl", "field_2.gkyl", "field_10.gkyl"]


def test_sort_reverse():
  a = _named("field_1.gkyl")
  b = _named("field_2.gkyl")
  out = operations.sort(a, b, reverse=True)
  assert [d.file_name for d in out] == ["field_2.gkyl", "field_1.gkyl"]


def test_sort_does_not_mutate_or_copy_datasets():
  a = _named("field_2.gkyl")
  b = _named("field_1.gkyl")
  out = operations.sort(a, b)
  assert out == [b, a]
  assert out[0] is b and out[1] is a


def test_sort_empty_returns_empty():
  assert operations.sort() == []
