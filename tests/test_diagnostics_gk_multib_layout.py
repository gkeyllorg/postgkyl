"""Tests for ``postgkyl.diagnostics.gk.multib_layout``.

Two synthetic geometries on circles about (R0, 0), each block an annular
sector ``r in [r0, r1]``, ``theta in [t0, t1]`` (rows along r, columns along
theta), so every midplane crossing and radius is known exactly:

- *single null*: a core ring (r 0.2-0.5, closing on itself) under a SOL block
  (r 0.5-0.7) running from theta = -pi/3 through the outboard (theta = 0)
  and inboard (theta = pi) midplanes to 4 pi/3, both ends lower targets;
- *half domain*: a core block (theta 0-pi/2) and a PF block (pi/2-3pi/4)
  under an outboard SOL block and its leg; theta = 0 lies on Z = 0 (the
  reflecting midplane), theta = 3 pi/4 is an upper target.
"""

from __future__ import annotations

import struct

import msgpack
import importlib

import numpy as np
import pytest

from generate_test_data import write_gkyl_field
from postgkyl import io

ml = importlib.import_module("postgkyl.diagnostics.gk.multib_layout")

R0 = 1.5
PHYS, LO, UP = 5, 1, 3  # physical face; lower and upper face, positive


def _sector(r0, r1, t0, t1, nr, nt):
  r = np.linspace(r0, r1, nr + 1)[:, None]
  t = np.linspace(t0, t1, nt + 1)[None, :]
  return np.stack([R0 + r * np.cos(t), r * np.sin(t)], axis=-1)


def _conn(spec):
  """``{block: ((psi_lo, psi_hi), (theta_lo, theta_hi))}``, each face None
  (physical) or ``(neighbour, its face edge)``."""
  out = np.zeros((len(spec), 2, 2, 3), dtype=int)
  for b, dirs in spec.items():
    for d, faces in enumerate(dirs):
      for e, face in enumerate(faces):
        out[b, d, e] = (b, d, PHYS) if face is None else (face[0], d, face[1])
  return out


def single_null():
  nodes = {
      0: _sector(0.2, 0.5, -np.pi, np.pi, 3, 8),
      1: _sector(0.5, 0.7, -np.pi / 3, 4 * np.pi / 3, 2, 10),
  }
  conn = _conn({
      0: ((None, (1, LO)), ((0, UP), (0, LO))),
      1: (((0, UP), None), (None, None)),
  })
  return conn, nodes


def half_domain():
  nodes = {
      0: _sector(0.2, 0.5, 0.0, np.pi / 2, 3, 3),
      1: _sector(0.2, 0.5, np.pi / 2, 3 * np.pi / 4, 3, 2),
      2: _sector(0.5, 0.7, 0.0, np.pi / 2, 2, 3),
      3: _sector(0.5, 0.7, np.pi / 2, 3 * np.pi / 4, 2, 2),
  }
  conn = _conn({
      0: ((None, (2, LO)), (None, (1, LO))),
      1: ((None, (3, LO)), ((0, UP), None)),
      2: (((0, UP), None), (None, (3, LO))),
      3: (((1, UP), None), ((2, UP), None)),
  })
  return conn, nodes


def test_single_null_layout():
  conn, nodes = single_null()
  assert ml.psi_grows_outward(conn, nodes)  # the ring's physical face is r 0.2
  lay = ml.derive_layout(conn, nodes)
  assert lay["region"] == ("core", "sol")
  assert lay["under"] == (None, None)  # 8 vs 10 theta cells: not aligned
  assert not lay["half_domain"]
  assert [(c.blocks, c.ring, c.start, c.end, c.side, c.labels)
          for c in lay["chains"]] == [((0, ), True, "ring", "ring", "", ()),
                                      ((1, ), False, "target", "target", "SOL",
                                       ("main SOL", ))]
  t_out, t_in = lay["targets"]
  assert (t_out.name, t_out.leg, t_out.pf, t_out.theta_end,
          t_out.mid_block) == ("lower outer target", 1, None, 0, 1)
  assert (t_in.name, t_in.theta_end) == ("lower inner target", -1)
  out, inb = lay["stacks"]["outboard"], lay["stacks"]["inboard"]
  assert (out.core, out.sol, out.mirror, out.blocks) == (None, 1, False, (1, ))
  # Row 0 crosses Z = 0 at theta nodes 2 (theta = 0) and 8 (theta = pi).
  assert out.crossing == pytest.approx(2.0, abs=1e-12)
  assert inb.crossing == pytest.approx(8.0, abs=1e-12)
  np.testing.assert_allclose(ml.midplane_radii(nodes[1], out.crossing),
                             R0 + np.array([0.5, 0.6, 0.7]),
                             rtol=0,
                             atol=1e-14)
  np.testing.assert_allclose(ml.midplane_radii(nodes[1], inb.crossing),
                             R0 - np.array([0.5, 0.6, 0.7]),
                             rtol=0,
                             atol=1e-14)


def test_side_rule_is_the_axis_else_the_core_middle():
  conn, nodes = single_null()
  # The core ring spans R0 - 0.5 .. R0 + 0.5: its middle R is R0.
  assert ml.derive_layout(conn, nodes)["side_r"] == pytest.approx(R0, abs=1e-15)
  lay = ml.derive_layout(conn, nodes, r_axis=R0 + 0.05)
  assert lay["side_r"] == R0 + 0.05  # the magnetic axis, when given, decides
  assert set(lay["stacks"]) == {"outboard", "inboard"}


def test_midplane_weights_bracket_z0():
  _, nodes = single_null()
  # Cell centres either side of theta = 0 are mirror images: equal weights.
  for j0, j1, w in ml.midplane_weights(nodes[1], 2.0):
    assert (j0, j1) == (1, 2)
    assert w == pytest.approx(0.5, abs=1e-14)


def test_half_domain_layout():
  conn, nodes = half_domain()
  lay = ml.derive_layout(conn, nodes)
  assert lay["region"] == ("core", "pf", "sol", "sol")
  assert lay["under"] == (None, None, 0, 1)
  assert lay["half_domain"]
  sol = next(c for c in lay["chains"] if c.blocks == (2, 3))
  assert (sol.start, sol.end, sol.labels) == ("mirror", "target",
                                              ("outboard", "upper outer leg"))
  (tg, ) = lay["targets"]
  assert (tg.name, tg.leg, tg.pf, tg.theta_end, tg.mid_block,
          tg.mid_crossing) == ("upper outer target", 3, 1, -1, 2, None)
  (side, st), = lay["stacks"].items()
  assert (side, st.core, st.sol, st.mirror, st.blocks) == ("outboard", 0, 2,
                                                           True, (0, 2))
  # At the reflecting midplane each row's value is its first cell's.
  assert all(w == (0, 0, 0.0) for w in ml.midplane_weights(nodes[2]))


def test_flip_psi_round_trip():
  conn, _ = single_null()
  flipped = ml.flip_psi(conn)
  np.testing.assert_array_equal(flipped[0, 0, 1], [0, 0, PHYS])  # ring's r 0.2
  np.testing.assert_array_equal(flipped[0, 0, 0], [1, 0, UP])
  np.testing.assert_array_equal(ml.flip_psi(flipped), conn)


def test_contrary_block():
  conn, nodes = single_null()
  assert ml.contrary_blocks(conn, nodes) == ()
  turned = dict(nodes)
  turned[1] = nodes[1][::-1]  # its rows now run against the topology
  assert ml.contrary_blocks(conn, turned) == (1, )


def test_single_block_topology():
  conn, nodes = single_null()
  ring = ml.single_block_topology(nodes[0])
  np.testing.assert_array_equal(ring[0, 0, :, 2], [PHYS, PHYS])
  np.testing.assert_array_equal(ring[0, 1], [[0, 1, UP], [0, 1, LO]])
  assert np.all(ml.single_block_topology(nodes[1])[..., 2] == PHYS)


def _write_topology(path, conn):
  meta = msgpack.packb({
      "num_blocks": len(conn),
      "ndim": 2,
      "connections": conn.ravel().tolist()
  })
  with open(path, "wb") as f:
    f.write(b"gkyl0" + struct.pack("<qqq", 1, 4, len(meta)) + meta)


def _write_nodes(path, nd):
  values = np.concatenate([nd, np.zeros(nd.shape[:2] + (1, ))], axis=-1)
  write_gkyl_field(path,
                   list(nd.shape[:2]), [0.0, 0.0],
                   list(map(float, nd.shape[:2])),
                   values,
                   1,
                   "serendipity",
                   metadata={"geometry_type": 1})


def test_reads_a_run_whose_psi_grows_inward(tmp_path):
  # The files hold psi growing inward (as STEP's do): rows and psi faces flipped.
  conn, nodes = single_null()
  _write_topology(tmp_path / "sn-block_topo.gkyl", ml.flip_psi(conn))
  for b, nd in nodes.items():
    _write_nodes(tmp_path / f"sn_b{b}-geo_corn_nodes.gkyl", nd[::-1])
  assert io.read_metadata(str(tmp_path /
                              "sn-block_topo.gkyl"))["num_blocks"] == 2
  assert ml.block_prefixes("sn", str(tmp_path)) == (str(tmp_path / "sn_b0"),
                                                    str(tmp_path / "sn_b1"))
  lay = ml.multib_layout("sn", path=str(tmp_path))
  assert lay.reversed == (True, True)
  assert not lay.single_block
  for b in nodes:
    np.testing.assert_allclose(lay.nodes[b], nodes[b], rtol=0, atol=1e-15)
  np.testing.assert_array_equal(lay.connections, conn)
  expected = ml.derive_layout(conn, nodes)
  assert lay.region == expected["region"]
  assert lay.targets == expected["targets"]
  assert lay.stacks == expected["stacks"]
  assert lay.lcfs_row == (None, None)
  assert lay.file_row(1, 0) == 1  # the inner cell row is the file's last
  np.testing.assert_array_equal(lay.rows(1, np.arange(2)), [1, 0])
  assert "lower outer target: leg block 1" in str(lay)


def test_single_block_run(tmp_path):
  _, nodes = single_null()
  _write_nodes(tmp_path / "one-geo_corn_nodes.gkyl", nodes[0])
  assert ml.block_prefixes("one", str(tmp_path)) == (str(tmp_path / "one"), )
  lay = ml.multib_layout("one", path=str(tmp_path))
  assert lay.single_block
  assert lay.region == ("core", )
  assert lay.file_name(0, "field", 3) == str(tmp_path / "one-field_3.gkyl")
