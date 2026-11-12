"""Evaluate time and space normalization scales from metadata expressions.

Accept numeric literals, metadata names, and basic arithmetic. Resolve the
expression for every input frame and require one finite, positive scale shared
by all frames.
"""

from __future__ import annotations

import ast
import operator
from numbers import Real

import numpy as np

_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _expression(node, metadata, name):
  if isinstance(node, ast.Constant) and type(node.value) in (int, float):
    return float(node.value)
  if isinstance(node, ast.Name):
    value = metadata.get(node.id)
    if isinstance(value, bool) or not isinstance(value, Real):
      raise ValueError(f"missing numeric {name} metadata '{node.id}'")
    return float(value)
  if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
    left = _expression(node.left, metadata, name)
    right = _expression(node.right, metadata, name)
    try:
      return _BINARY[type(node.op)](left, right)
    except ZeroDivisionError as err:
      raise ValueError(f"{name} expression divides by zero") from err
  if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
    return _UNARY[type(node.op)](_expression(node.operand, metadata, name))
  raise ValueError(
      f"{name} supports only numbers, metadata names, +, -, *, / and parentheses"
  )


def scale(frames, value, name):
  """Return one positive scale shared by all frames."""
  if value is None:
    return 1.0
  if isinstance(value, str):
    try:
      numbers = [float(value)]
    except ValueError:
      if any(value in frame.ctx for frame in frames):
        nodes = [ast.Name(id=value, ctx=ast.Load())]
      else:
        try:
          nodes = [ast.parse(value, mode="eval").body]
        except SyntaxError as err:
          raise ValueError(f"invalid {name} expression '{value}'") from err
      numbers = [_expression(nodes[0], frame.ctx, name) for frame in frames]
  else:
    numbers = [float(value)]
  if not all(np.isfinite(number) and number > 0 for number in numbers):
    raise ValueError(f"{name} must be finite and positive")
  if not np.allclose(numbers, numbers[0], rtol=1e-12, atol=0):
    raise ValueError(f"{name} differs across frames")
  return numbers[0]
