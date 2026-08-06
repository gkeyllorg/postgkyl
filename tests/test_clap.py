"""Tests for the high-level scriptable pgkyl API (``postgkyl.clap``)."""
import os

import postgkyl as pg
from postgkyl.clap import _clap_gen
from postgkyl.clap import PgkylSession


class TestApiGeneration:
  """The generated ``clap.py`` must stay in sync with the click commands."""

  def test_clap_in_sync(self):
    path = _clap_gen._target_path()
    with open(path, "r") as fh:
      current = fh.read()
    assert current == _clap_gen.render(), (
        "postgkyl/clap/clap.py is stale; run 'python src/postgkyl/clap/_clap_gen.py'.")


class TestApiSession:
  """The session methods mirror the command chain on a shared stack."""
  dir_path = f"{os.path.dirname(__file__)}/test_data"

  def test_load_interp_chain(self):
    session = PgkylSession(batch_mode=True)
    session.load(f"{self.dir_path:s}/shock-f-ser-p1.gkyl")
    session.interpolate(basis_type="ms", poly_order=1)
    assert session.data.get_num_datasets() == 1

  def test_reconstructed_command_orders_options_before_arguments(self):
    """Options must precede positional arguments in the reconstructed CLI.

    pgkyl chains commands, so any token following a positional argument is
    parsed as the next command. The fragment for ``ev`` must therefore read
    ``ev --tag t '<chain>'`` (not ``ev '<chain>' --tag t``, which would treat
    ``--tag`` as a new command) and the ``chain`` metavar must never leak in as
    a literal token.
    """
    from postgkyl.pgkyl import cli

    session = PgkylSession(batch_mode=True)
    fragment = session._format_command(
        cli.commands["ev"],
        {"chain": "f0 f1 +", "tag": "result", "label": None, "all": False},
    )
    assert fragment == "ev --tag result 'f0 f1 +'"

  def test_set_globals_applies_to_all_subsequent_loads(self):
    """``set_globals`` mirrors a group-level pgkyl option (``--c2p-vel``).

    The global applies to every file loaded afterwards, so two datasets combined
    with ``ev`` share a consistent grid type (a per-file ``load`` option would
    set it on only one, and ``ev`` would drop the differing key). It must also
    reconstruct ahead of the data, matching ``pgkyl --c2p-vel map f1 f2 ...``.
    """
    session = PgkylSession(batch_mode=True)
    cmap = f"{self.dir_path:s}/bimaxwellian-mapc2p-vel.gkyl"
    session.set_globals(c2p_vel=cmap)
    session.load(f"{self.dir_path:s}/bimaxwellian-elc.gkyl", tag="jf")
    session.load(f"{self.dir_path:s}/bimaxwellian-jacobvel.gkyl", tag="jac")
    grid_types = {d.ctx.get("grid_type") for d in session.data.iterator()}
    assert grid_types == {"c2p_vel"}

    session.ev("jf jac /", tag="df")
    session.activate(tag="df")
    session.interpolate(basis_type="gkhyb", poly_order=1)

    assert session.get_cmd().index("--c2p-vel") < session.get_cmd().index("bimaxwellian-elc")
