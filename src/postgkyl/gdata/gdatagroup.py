"""``GDataGroup`` -- the fluent group container over
``gdatastate.GDataStateGroup``.

Mirrors how :class:`~postgkyl.gdata.gdata.GData` adds the fluent verb methods
on top of the verb-less :class:`~postgkyl.gdatastate.gdatastate.GDataState`: this class
adds *broadcasting* verbs on top of the verb-less
:class:`~postgkyl.gdatastate.gdatastategroup.GDataStateGroup`, without duplicating a single
verb body.

Contract
--------
Any attribute name that is not defined on this class itself (and does not
start with ``_``) is resolved by :meth:`__getattr__`, by looking it up on
every member, in order (**broadcasting**):

- If the attribute is a *verb method* on every member, calling the broadcast
  invokes that method on each member with the same arguments. If every
  member's result is a ``GDataState`` (or subclass), the results are wrapped
  in a *new* group of the caller's own concrete class, so chains stay fluent:
  ``group.interpolate().select(z0=0.0)``. Otherwise -- a terminal verb whose result is
  not a dataset (``.plot()`` -> one Figure per member, ``.write()`` -> one
  path per member, ``.integrate()`` -> one float per member,
  ``.extract_input()`` -> one string per member, ...) -- a plain ``list`` of
  the per-member results is returned, in member order. Note this means
  ``group.plot()`` renders one figure *per member* (broadcast), not one
  shared overlaid figure: there is no multi-dataset plot verb at the ``operations``
  layer to delegate to, and ``api`` does not import ``render`` directly (see
  ``tests/test_postgkyl.py``'s ``_ALLOWED`` map).
- If the attribute is a *non-callable* value on every member (a property such
  as ``num_dims`` or ``backend``), it resolves immediately to a plain
  ``list`` of the per-member values, in member order -- no closure, no
  call needed.
- Attribute names starting with ``_`` are never broadcast (raises
  ``AttributeError``), so private/dunder probes and pickling machinery are
  unaffected. An attribute missing from any member also raises
  ``AttributeError`` immediately, at access time.

Six verbs are **not** broadcast because they combine or reorder the members
into a single result rather than acting on each independently; these are
defined explicitly below, delegating to the matching multi-dataset function
in ``operations``/``api.verbs``: ``sort`` (reorder the group), ``info`` (one
combined summary), ``collect`` (stack into one dataset), ``evaluate``
(evaluate an RPN expression over the members), ``animate`` (one animation,
one frame per member), ``plotly_animate`` (one Plotly animation, one frame per
member) -- matching the deferred worklist
from layer 05's report (the old ``src_bak`` class's non-broadcast methods
were exactly ``__getattr__`` broadcasting, ``plot``, ``info``, ``animate``,
``plotly_animate``, ``collect``, ``evaluate``). ``plot`` is still not in the
new ``operations`` verb inventory as a multi-dataset verb, so it stays a
broadcast (one figure per member, see above); ``plotly`` (single-dataset) is
now a plain ``GData`` method, so ``group.plotly()`` is already covered by the
broadcast rule too -- only ``info``/``collect``/``evaluate``/``animate``/
``plotly_animate`` and ``sort`` need the explicit treatment here.

``operations.grid`` has no fluent spelling anywhere (not on ``GData``, so not
broadcast here either) -- see ``api/gdata.py`` for why.

``load`` is also explicit rather than broadcast: it is the group's lifecycle
method, appending newly loaded member(s) to this group and returning ``self``
so an initially empty group can be assembled fluently.
"""

from __future__ import annotations

from postgkyl import operations
from postgkyl.gdatastate.gdatastategroup import GDataStateGroup
from postgkyl.gdatastate.gdatastate import GDataState

from . import verbs


class GDataGroup(GDataStateGroup):
  """A group whose members' fluent verbs broadcast over the whole group."""

  # ------------------------------------------------------- data lifecycle
  def load(self, file_name: str, *, tag: str = "default", label: str = "",
      ctx: dict | None = None, value_form: str | None = None,
      basis_type: str | None = None, poly_order: int | None = None,
      **read_kwargs) -> "GDataGroup":
    """Load and append one file (or a glob) and return this group.

    Successive calls accumulate members, enabling chains such as::

        group = GDataGroup()
        group.load(frame0).load(frame1).local_poly().collect().plot()

    Loading completes before this group is changed, so a failed read leaves
    its existing members untouched.  A glob appends every match in natural
    filename order, with the same options and errors as :func:`postgkyl.load`.
    """
    # Same-layer import at call time avoids the construction-time cycle:
    # gdata.load builds GDataGroup results, while GData imports this class for
    # fluent group-returning verbs.
    from .load import load as load_data

    loaded = load_data(file_name, tag=tag, label=label, ctx=ctx,
        value_form=value_form, basis_type=basis_type,
        poly_order=poly_order, **read_kwargs)
    if isinstance(loaded, GDataStateGroup):
      additions = loaded.datasets
    # end
    else:
      additions = [loaded]
    # end
    self._datasets.extend(additions)
    return self
  # end

  def __getattr__(self, name: str):
    if name.startswith("_"):
      raise AttributeError(name)
    # end
    values = [getattr(member, name) for member in self._datasets]
    if values and not all(callable(v) for v in values):
      return values
    # end

    def broadcast(*args, **kwargs):
      results = [v(*args, **kwargs) for v in values]
      if results and all(isinstance(r, GDataState) for r in results):
        return type(self)(results)
      # end
      return results
    # end
    return broadcast
  # end

  # ------------------------------------------------------- combining (typed)
  # Overridden (not inherited) so the result stays the caller's concrete
  # subclass, mirroring GDataState._result's ``type(self)`` trick.
  def with_(self, *others) -> "GDataGroup":
    """Return a new group (same concrete class) with ``others`` appended."""
    return type(self)(self._datasets + list(others))
  # end

  __and__ = with_

  def sort(self, *, reverse: bool = False) -> "GDataGroup":
    """Return a naturally filename-sorted group (see ``operations.sort``)."""
    return type(self)(verbs.sort(*self._datasets, reverse=reverse))
  # end

  def __getitem__(self, index):
    """Index or slice; a slice returns a group of the same concrete class."""
    result = self._datasets[index]
    return type(self)(result) if isinstance(index, slice) else result
  # end

  # ------------------------------------------------------- terminal (typed)
  def info(self, *, header: bool = True) -> list:
    """Summarize every member (see ``operations.info``); returns a list of strings."""
    return operations.info(*self._datasets, header=header)
  # end

  def collect(self, *, sumdata: bool = False, period: float | None = None,
      offset: float = 0.0, chunk: int | None = None, tag: str | None = None,
      label: str | None = None):
    """Combine the members into one dataset along a time axis (see
    ``api.verbs.collect``). Returns a list of datasets when ``chunk`` is given."""
    return verbs.collect(*self._datasets, sumdata=sumdata, period=period,
        offset=offset, chunk=chunk, tag=tag, label=label)
  # end

  def evaluate(self, chain: str, *, tag: str | None = None, label: str | None = None):
    """Evaluate an RPN expression over the members (see ``api.verbs.evaluate``)."""
    return verbs.evaluate(chain, *self._datasets, tag=tag, label=label)
  # end

  def animate(self, **kwargs):
    """Animate the members, one frame each (see ``api.verbs.animate``)."""
    return verbs.animate(*self._datasets, **kwargs)
  # end

  def plotly_animate(self, **kwargs):
    """Animate the members with Plotly, one frame each (see ``api.verbs.plotly_animate``)."""
    return verbs.plotly_animate(*self._datasets, **kwargs)
  # end
# end
