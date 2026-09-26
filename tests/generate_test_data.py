"""Generate synthetic .gkyl test files for the postgkyl test suite.

Run directly to regenerate:
    python tests/generate_test_data.py

Called automatically by conftest.py at the start of each pytest session.

Field files encode polyOrder and basisType in their msgpack metadata block so
GData auto-populates ctx["poly_order"] and ctx["basis_type"] on load.
Every file produced by generate_all also records description, analytic_function,
and generation_method metadata. Inspect them with ``pgkyl FILE info --all`` or
``pg.load(FILE).info(all=True)``. Random coefficient fixtures explicitly state
that they have no prescribed analytic function.

C2P mapping files store modal DG coefficients for analytical coordinate
transformations.  The basis is inferred by GData from num_comps/ndim via
_get_basis_p().  Two mapping types are provided:
  - "stretch": linear map (comp domain [0,1]^n → physical domain phys_bounds)
  - "rotation": 2D rotation by angle α about the origin

Dynvector files (file_type=2, a bare time series with no spatial grid --
e.g. a field-energy history) are written by ``write_gkyl_dynvector``.

Analytic 4D--6D fields are projected using Gkeyll's own basis evaluations.
They are generated when the compiled library is available; their consumers
are native tests and skip when it is unavailable.
"""
import struct
from pathlib import Path

import msgpack
import numpy as np
from scipy import constants

_RNG = np.random.default_rng(42)
_SQRT3 = np.sqrt(3)
_RANDOM_METADATA = {
    "analytic_function":
    ("No prescribed analytic function. Each cell contains the DG polynomial "
     "sum_m c_m*phi_m, with independent normal modal coefficients; "
     "these are not samples of a smooth field."),
    "generation_method":
    ("NumPy default_rng(42).standard_normal, drawn sequentially from the "
     "module-level generator in generation order. Repeated generate_all "
     "calls in one process advance that stream.")
}

# Component counts per basis -- mirrors the tables in src/postgkyl/data/dg.py
# serendipity: indexed as [ndim-1][poly_order]   (p=0 → 1 component)
_COMPS_SER = [
    [1, 2, 3, 4, 5],  # 1D
    [1, 4, 8, 12, 17],  # 2D
    [1, 8, 20, 32, 50],  # 3D
]
# tensor: indexed as [ndim-1][poly_order-1]   (p starts at 1)
_COMPS_TEN = [
    [2, 3, 4, 5],  # 1D
    [4, 9, 16, 25],  # 2D
    [8, 27, 64, 125],  # 3D
]
# maximal-order: indexed as [ndim-1][poly_order-1]
_COMPS_MAX = [
    [2, 3, 4, 5],  # 1D
    [3, 6, 10, 15],  # 2D
    [4, 10, 20, 35],  # 3D
]

_COMPS = {
    "serendipity": (_COMPS_SER, lambda p: p),
    "tensor": (_COMPS_TEN, lambda p: p - 1),
    "maximal-order": (_COMPS_MAX, lambda p: p - 1),
}


def num_comps(basis: str, ndim: int, poly_order: int) -> int:
  table, idx_fn = _COMPS[basis]
  return table[ndim - 1][idx_fn(poly_order)]


def write_gkyl_field(
    path: Path,
    cells: list[int],
    lower: list[float],
    upper: list[float],
    values: np.ndarray,
    poly_order: int,
    basis_type: str,
    time: float = 0.0,
    frame: int = 0,
    metadata: dict | None = None,
    no_metadata: bool = False,
) -> None:
  """Write a .gkyl v1 field, optionally omitting its msgpack metadata."""
  ndim = len(cells)
  nc = values.shape[-1]

  meta = b"" if no_metadata else msgpack.packb({
      "polyOrder": poly_order,
      "basisType": basis_type,
      "time": time,
      "frame": frame,
      **(metadata or {}),
  })

  with open(path, "wb") as f:
    # --- version-1 header ---
    f.write(b"gkyl0")
    f.write(struct.pack("<q", 1))  # version
    f.write(struct.pack("<q", 1))  # file_type = 1 (field)
    f.write(struct.pack("<q", len(meta)))  # meta_size
    f.write(meta)

    # --- field body ---
    f.write(struct.pack("<q", 2))  # real_type = 2 → float64
    f.write(struct.pack("<q", ndim))
    for c in cells:
      f.write(struct.pack("<q", c))
    for lo in lower:
      f.write(struct.pack("<d", lo))
    for hi in upper:
      f.write(struct.pack("<d", hi))
    esznc = nc * 8  # element_size * num_comps (bytes)
    size = int(np.prod(cells))  # total number of cells
    f.write(struct.pack("<q", esznc))
    f.write(struct.pack("<q", size))
    f.write(values.astype("<f8").tobytes())  # C-order, little-endian float64


def write_gkyl_dynvector(path: Path,
                         time: np.ndarray,
                         values: np.ndarray,
                         metadata: dict | None = None) -> None:
  """Write a minimal valid .gkyl v1 binary dynvector (file_type=2) file.

    A dynvector has no spatial grid -- just a time series of ``num_comps``
    values per sample (e.g. a field-energy history). Layout per
    ``gkyl_reader.py``'s ``_read_t2_v1``: header, real_type, esznc, size,
    then all of TIME_DATA followed by all of DATA (C-order).
    """
  nc = values.shape[-1]
  size = len(time)
  meta = msgpack.packb(metadata) if metadata else b""

  with open(path, "wb") as f:
    f.write(b"gkyl0")
    f.write(struct.pack("<q", 1))  # version
    f.write(struct.pack("<q", 2))  # file_type = 2 (dynvec)
    f.write(struct.pack("<q", len(meta)))
    f.write(meta)
    f.write(struct.pack("<q", 2))  # real_type = 2 -> float64
    esznc = nc * 8  # element_size * num_comps (bytes)
    f.write(struct.pack("<q", esznc))
    f.write(struct.pack("<q", size))
    f.write(time.astype("<f8").tobytes())
    f.write(values.astype("<f8").tobytes())  # C-order, little-endian float64


# ---------------------------------------------------------------------------
# C2P mapping value generators
# ---------------------------------------------------------------------------


def _c2p_stretch_values(
    cells: list[int],
    phys_lo: list[float],
    phys_hi: list[float],
    num_modes: int,
) -> np.ndarray:
  """Modal DG coefficients for a linear stretch mapping (comp [0,1]^n → phys).

    For each cell the mapping is:
        coord_d(xi') = coord_mid_d + (dx_phys_d/2) * xi'_d
    Modal serendipity coefficients (any poly_order):
        c_0 = 2 * coord_mid       (constant mode, normalized by 1/2)
        c_{d+1} = dx_phys / sqrt(3) (linear mode in direction d)
        all higher modes = 0      (linear function has no quadratic terms)
    """
  ndim = len(cells)
  dx = [(phys_hi[d] - phys_lo[d]) / cells[d] for d in range(ndim)]
  values = np.zeros((*cells, ndim * num_modes))

  grids = np.meshgrid(*[np.arange(cells[d]) for d in range(ndim)],
                      indexing="ij")
  for d in range(ndim):
    mid = phys_lo[d] + dx[d] * (grids[d] + 0.5)
    off = d * num_modes
    values[..., off] = 2.0 * mid  # constant mode
    values[..., off + 1 + d] = dx[d] / _SQRT3  # linear mode in d-th direction
  return values


def _c2p_rotation_values(
    cells: list[int],
    comp_lo: list[float],
    comp_hi: list[float],
    angle: float,
    num_modes: int,
) -> np.ndarray:
  """Modal DG coefficients for a 2D rotation mapping by *angle* radians.

    The computational domain is [comp_lo[0], comp_hi[0]] x [comp_lo[1], comp_hi[1]].
    The mapping is x = xi*cos - eta*sin, y = xi*sin + eta*cos.
    Only valid for 2D serendipity; the linear rotation is exact at any poly_order.
    """
  assert len(cells) == 2, "rotation mapping only implemented for 2D"
  ca, sa = np.cos(angle), np.sin(angle)
  N_x, N_y = cells
  dxi = (comp_hi[0] - comp_lo[0]) / N_x
  deta = (comp_hi[1] - comp_lo[1]) / N_y

  ii, jj = np.mgrid[0:N_x, 0:N_y]
  xi_mid = comp_lo[0] + dxi * (ii + 0.5)
  eta_mid = comp_lo[1] + deta * (jj + 0.5)

  x_mid = xi_mid * ca - eta_mid * sa
  y_mid = xi_mid * sa + eta_mid * ca

  values = np.zeros((N_x, N_y, 2 * num_modes))
  # x-coordinate modal coefficients
  values[..., 0] = 2.0 * x_mid  # constant
  values[..., 1] = dxi * ca / _SQRT3  # xi'-mode (dx/dxi' * mapping factor)
  values[..., 2] = -deta * sa / _SQRT3  # eta'-mode
  # y-coordinate modal coefficients
  values[..., num_modes + 0] = 2.0 * y_mid
  values[..., num_modes + 1] = dxi * sa / _SQRT3
  values[..., num_modes + 2] = deta * ca / _SQRT3
  return values


# ---------------------------------------------------------------------------
# Analytic four-component 1-D profile
# ---------------------------------------------------------------------------


def _mirror_comparison_profiles(z: np.ndarray, alpha: float) -> np.ndarray:
  """Symmetric beam profiles used by the alpha-convergence example.

    Density and both temperatures are even in ``z``; parallel velocity is
    odd.  ``alpha`` introduces a small, pedestal-localized difference so the
    two generated datasets remain close enough to read as a convergence
    comparison.  Components are already in the plotting units used by the
    example: m^-3, m/s, keV, and keV.

    Let r=abs(z), s=log10(alpha/2e-5), P=exp(-((r-0.86)/0.16)^2).
    n0=1.01e13+2.45e19/(1+exp((r-0.92)/0.08))+4.5e18*exp(-(r/0.55)^4).
    u0=1.30e6*tanh(z/0.05)/(1+exp((0.82-r)/0.05)).
    Tpar0=0.101+11.8/(1+exp((r-0.90)/0.10))
          +0.4*exp(-((r-0.55)/0.15)^2).
    Tperp0=0.101+20.8/(1+exp((r-0.98)/0.13))
           +10*exp(-((r-0.86)/0.10)^2).
    Components: (n0*(1-0.025*s*P), u0*(1+0.020*s*P),
                 Tpar0*(1+0.035*s*P), Tperp0*(1+0.030*s*P)).
    """
  if alpha <= 0.0:
    raise ValueError("alpha must be positive")

  radius = np.abs(z)
  sensitivity = np.log10(alpha / 2.0e-5)
  pedestal = np.exp(-((radius - 0.86) / 0.16)**2)

  density = (1.01e13 + 2.45e19 / (1.0 + np.exp(
      (radius - 0.92) / 0.08)) + 4.5e18 * np.exp(-(radius / 0.55)**4))
  velocity = (1.30e6 * np.tanh(z / 0.05) / (1.0 + np.exp(
      (0.82 - radius) / 0.05)))
  t_parallel = (0.101 + 11.8 / (1.0 + np.exp(
      (radius - 0.90) / 0.10)) + 0.4 * np.exp(-((radius - 0.55) / 0.15)**2))
  t_perpendicular = (0.101 + 20.8 / (1.0 + np.exp(
      (radius - 0.98) / 0.13)) + 10.0 * np.exp(-((radius - 0.86) / 0.10)**2))

  return np.stack([
      density * (1.0 - 0.025 * sensitivity * pedestal),
      velocity * (1.0 + 0.020 * sensitivity * pedestal),
      t_parallel * (1.0 + 0.035 * sensitivity * pedestal),
      t_perpendicular * (1.0 + 0.030 * sensitivity * pedestal),
  ],
                  axis=-1)


def _project_1d_p1(fn, lower: float, upper: float, cells: int,
                   *args) -> np.ndarray:
  """Cellwise L2 projection of a vector-valued analytic function onto p1.

    The 1-D modal serendipity basis is ``(1/sqrt(2), sqrt(3/2)*xi)`` on
    ``[-1, 1]``.  Eight-point Gauss quadrature resolves the smooth analytic
    profiles well within each cell.  The result is field-blocked as Gkeyll
    expects: ``[f0_mode0, f0_mode1, f1_mode0, ...]``.
    """
  xi, weights = np.polynomial.legendre.leggauss(8)
  dz = (upper - lower) / cells
  centers = lower + (np.arange(cells) + 0.5) * dz
  z = centers[:, None] + 0.5 * dz * xi[None, :]
  samples = fn(z, *args)
  basis = np.stack([
      np.full_like(xi, 1.0 / np.sqrt(2.0)),
      np.sqrt(3.0 / 2.0) * xi,
  ],
                   axis=-1)
  coefficients = np.einsum("cqf,qb,q->cfb", samples, basis, weights)
  return coefficients.reshape(cells, -1)


# ---------------------------------------------------------------------------
# Analytic high-dimensional fields
# ---------------------------------------------------------------------------

# (stem, ndim, poly_order, basis_type). Gkeyll has no 6D gkhybrid or p2 basis.
ANALYTIC_CONFIGS = [
    (f"analytic_{ndim}d_{short}_p{order}", ndim, order, basis)
    for ndim in (4, 5, 6) for short, basis in (
        ("ms", "serendipity"),
        ("mt", "tensor"),
        ("gkhyb", "gkhybrid"),
    ) for order in ((1, ) if basis == "gkhybrid" or ndim == 6 else (1, 2))
    if not (ndim == 6 and basis == "gkhybrid")
]


def analytic_fields(points: np.ndarray, basis: str, order: int) -> np.ndarray:
  """Two exactly representable polynomials, with coordinates on the last axis.

  f = 2 + sum((d+1)*x_d) + x_0*x_last, plus x_q^2/2 when supported.
  g = product(1 + (d+1)*x_d/5), with an extra (1+x_q/3) for serendipity
  p2 and gkhybrid, or x_d^2/10 in every factor for tensor p2.

  Here q=ndim-2 is v_parallel for gkhybrid. These terms exercise mixed
  modes, the quadratic velocity modes, and tensor-only quadratic products.
  """
  ndim = points.shape[-1]
  q = ndim - 2
  quadratic = order == 2 or basis == "gkhybrid"
  f = (2.0 + points @ np.arange(1, ndim + 1) + points[..., 0] * points[..., -1])
  if quadratic:
    f = f + 0.5 * points[..., q]**2
  factors = 1.0 + points * np.arange(1, ndim + 1) / 5.0
  if basis == "tensor" and order == 2:
    factors = factors + points**2 / 10.0
  g = np.prod(factors, axis=-1)
  if quadratic and basis != "tensor":
    g = g * (1.0 + points[..., q] / 3.0)
  return np.stack([f, g], axis=-1)


def _generate_analytic_fields(out_dir: Path) -> None:
  """Project the polynomials by exact Gauss integration, field-blocked."""
  from postgkyl import gpython

  if not gpython.available():
    return
  for stem, ndim, order, basis in ANALYTIC_CONFIGS:
    # Unequal counts, origins and widths expose axis/cell ordering mistakes.
    cells = [2, 3] + [1] * (ndim - 3) + [2]
    lower = -0.5 + np.arange(ndim) / 10.0
    upper = lower + 1.0 + np.arange(ndim) / 5.0
    dx = (upper - lower) / cells
    centers = np.stack(np.meshgrid(
        *[lower[d] + (np.arange(cells[d]) + 0.5) * dx[d] for d in range(ndim)],
        indexing="ij"),
                       axis=-1).reshape(-1, ndim)
    nq = 3 if order == 2 or basis == "gkhybrid" else 2
    nodes, weights = gpython.basis.gauss_quad(ndim, nq)
    matrix = gpython.basis.eval_matrix(basis, ndim, order, nodes)
    samples = analytic_fields(centers[:, None, :] + nodes * dx / 2, basis,
                              order)
    coefficients = np.einsum("cqf,qb,q->cfb",
                             samples,
                             matrix,
                             weights,
                             optimize=True)
    write_gkyl_field(out_dir / f"{stem}.gkyl",
                     cells,
                     lower,
                     upper,
                     coefficients.reshape(*cells, -1),
                     order,
                     basis,
                     metadata={
                         "description":
                         ("Two polynomial scalar fields (f, g) for testing "
                          "high-dimensional DG mixed and velocity modes. "
                          "Coordinates x_d use zero-based indices; "
                          "q=ndim-2 is v_parallel for gkhybrid."),
                         "analytic_function":
                         analytic_fields.__doc__,
                         "generation_method":
                         (f"Cellwise L2 projection with {nq}-point Gauss "
                          "quadrature per axis and Gkeyll basis evaluations; "
                          "exact up to roundoff. Coefficients are grouped "
                          "by field, f then g.")
                     })


# ---------------------------------------------------------------------------
# Configuration tables
# ---------------------------------------------------------------------------

#  (stem, ndim, cells, poly_order, basis_type)
_FIELD_CONFIGS: list[tuple] = [
    ("1d_ms_p1", 1, [8], 1, "serendipity"),
    ("1d_ms_p2", 1, [8], 2, "serendipity"),
    ("2d_ms_p1", 2, [8, 8], 1, "serendipity"),
    ("2d_ms_p2", 2, [8, 8], 2, "serendipity"),
    ("2d_mt_p1", 2, [8, 8], 1, "tensor"),
    ("2d_mt_p2", 2, [8, 8], 2, "tensor"),
    ("2d_mo_p1", 2, [8, 8], 1, "maximal-order"),
    ("2d_mo_p2", 2, [8, 8], 2, "maximal-order"),
    ("3d_ms_p1", 3, [4, 4, 4], 1, "serendipity"),
]

# C2P mapping files.
# (stem, kind, cells, poly_order, basis_type, extra...)
#   kind="stretch":  extra = (phys_lo, phys_hi)          -- comp domain [0,1]^n
#   kind="rotation": extra = (angle,)                     -- comp domain [0,1]^2
_C2P_CONFIGS: list[tuple] = [
    # Linear stretch: physical x∈[0,2], y∈[0,3]; paired with 2d_ms_p1.gkyl
    ("2d_c2p_stretch_ms_p1", "stretch", [8, 8], 1, "serendipity", [0.0, 0.0],
     [2.0, 3.0]),
    # Same stretch for p=2; paired with 2d_ms_p2.gkyl
    ("2d_c2p_stretch_ms_p2", "stretch", [8, 8], 2, "serendipity", [0.0, 0.0],
     [2.0, 3.0]),
    # Rotation by 45°; paired with 2d_ms_p1.gkyl (comp domain [0,1]^2)
    ("2d_c2p_rot45_ms_p1", "rotation", [8, 8], 1, "serendipity", np.pi / 4),
]


def generate_all(out_dir: Path | str) -> None:
  """Write all synthetic test files to *out_dir*."""
  out_dir = Path(out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  _generate_analytic_fields(out_dir)

  # Constant f=4 on [-1,1], with orthonormal p1 modal coefficients.
  write_gkyl_field(out_dir / "fsimple.gkyl", [1], [-1.], [1.],
                   np.array([[4 * np.sqrt(2), 0.]]),
                   1,
                   "serendipity",
                   metadata={
                       "description": "Constant scalar field for DG checks.",
                       "analytic_function": "f(x) = 4 on [-1, 1].",
                       "generation_method":
                       "Exact orthonormal modal coefficients."
                   })

  # The highest 1x1v hybrid mode vanishes at a 2x2 Gauss rule. Native
  # quadrature must retain it using Gkeyll's six-node rule.
  hybrid = np.zeros((1, 1, 6))
  hybrid[..., 5] = 1.0
  write_gkyl_field(out_dir / "fsimple_hyb.gkyl", [1, 1], [-1., -1.], [1., 1.],
                   hybrid,
                   1,
                   "hybrid",
                   metadata={
                       "description":
                       ("Highest 1x1v hybrid basis mode, testing quadrature "
                        "that resolves a quadratic velocity dependence."),
                       "analytic_function":
                       ("f(x,v) = sqrt(15)/4 * x * (3*v^2 - 1) "
                        "on [-1,1]^2."),
                       "generation_method":
                       ("Exact modal coefficients: mode 5 is 1, all others 0.")
                   })

  # Named GK sources for the load_quantity example, in SI units. Piecewise
  # constant profiles in a p1 basis give exact cellwise nonlinear quantities.
  x = (np.arange(64) + 0.5) / 64
  temperature = constants.elementary_charge * (20.0 + 10.0 * x)
  maxwellian = np.stack([
      np.full_like(x, 1e19),
      np.zeros_like(x), temperature / constants.electron_mass
  ],
                        axis=-1)
  for suffix, samples, metadata in (
      ("elc_MaxwellianMoments_0", maxwellian, {
          "mass":
          constants.electron_mass,
          "charge":
          -constants.elementary_charge,
          "description":
          "Electron Maxwellian moments (density, u_parallel, T/m).",
          "analytic_function":
          ("n(x)=1e19 m^-3; u_parallel(x)=0 m/s; "
           "T(x)/m_e=e*(20+10*x)/m_e m^2/s^2, "
           "where T is in joules and e is the elementary charge.")
      }),
      ("field_0", (5.0 * np.sin(2 * np.pi * x))[:, None], {
          "description": "Electrostatic potential for gyrokinetic diagnostics.",
          "analytic_function": "phi(x)=5*sin(2*pi*x) V."
      }),
      ("geo_int_bmag", (1.0 + 0.5 * x)[:, None], {
          "description":
          "Magnetic-field magnitude for gyrokinetic diagnostics.",
          "analytic_function": "B(x)=1+0.5*x T."
      }),
  ):
    coefficients = np.zeros((64, samples.shape[-1], 2))
    coefficients[..., 0] = np.sqrt(2.0) * samples
    write_gkyl_field(out_dir / f"gk_quantity_1d_p1-{suffix}.gkyl", [64], [0.0],
                     [1.0],
                     coefficients.reshape(64, -1),
                     poly_order=1,
                     basis_type="serendipity",
                     metadata={
                         **metadata, "generation_method":
                         ("Sample at 64 cell centers x=(i+1/2)/64 on [0,1]; "
                          "store piecewise constants in a p1 modal basis "
                          "with zero slopes, rather than projecting the "
                          "continuous function.")
                     })

  # Packed GK moments with nonzero slopes and jumps between unit-width cells.
  # Each field is a + b*xi in its cell, xi in [-1, 1].
  gk_mean = np.array([[3., 1., 6., 4.], [5., 2., 8., 7.]])
  gk_slope = np.array([[0.7, -0.4, 1.2, 0.5], [-0.6, 0.3, -0.8, 1.1]])
  gk_coeffs = np.stack([np.sqrt(2.) * gk_mean,
                        np.sqrt(2. / 3.) * gk_slope],
                       axis=-1).reshape(2, 8)
  write_gkyl_field(out_dir / "gk_moments_p1.gkyl", [2], [0.], [2.],
                   gk_coeffs,
                   1,
                   "serendipity",
                   metadata={
                       "mass":
                       2.,
                       "charge":
                       3.,
                       "description":
                       ("Packed gyrokinetic moments (M0, M1, M2par, M2perp) "
                        "with slopes and jumps; synthetic mass=2, charge=3."),
                       "analytic_function":
                       ("F_i(x)=a_i+b_i*xi, xi=2*(x-i)-1, cell i=0,1. "
                        f"a={gk_mean.tolist()}; b={gk_slope.tolist()}."),
                       "generation_method":
                       ("Exact p1 modal coefficients (sqrt(2)*a, "
                        "sqrt(2/3)*b), grouped by moment.")
                   })

  # Local derivatives see affine slopes, not jumps in the cell means.
  for ndim in (1, 2, 3):
    cells = (2, ) * ndim
    nb = 2**ndim
    drift = np.zeros((*cells, 5, nb))
    norm = np.sqrt(nb)
    drift[..., 0, 0] = norm * np.arange(2**ndim).reshape(cells)
    drift[..., 0, 1:ndim + 1] = norm / np.sqrt(3.) * np.arange(1, ndim + 1)
    drift[..., 1:, 0] = norm * np.array([2., 3., 4., 5.])
    write_gkyl_field(out_dir / f"gk_drift_{ndim}d_p1.gkyl",
                     list(cells), [0.] * ndim, [2.] * ndim,
                     drift.reshape(*cells, 5 * nb),
                     1,
                     "serendipity",
                     metadata={
                         "description":
                         ("Five scalar fields for local gyrokinetic drift "
                          "derivatives: one discontinuous affine field and "
                          "four constant auxiliary fields."),
                         "analytic_function":
                         ("F0(x)=j+sum_{d=0}^{ndim-1}(d+1)*xi_d, "
                          "xi_d=2*(x_d-i_d)-1 in cell i; j is the "
                          "zero-based C-order flat cell index. "
                          "(F1,F2,F3,F4)=(2,3,4,5)."),
                         "generation_method":
                         "Exact orthonormal p1 modal coefficients."
                     })

  # Selection tests need unit-width cells and two complete p2 tensor fields.
  selection_nc = 2 * num_comps("tensor", 2, 2)
  selection_values = np.arange(4 * 3 * selection_nc,
                               dtype=float).reshape(4, 3, selection_nc)
  write_gkyl_field(out_dir / "select_2d_tensor_p2.gkyl", [4, 3], [0.0, 0.0],
                   [4.0, 3.0],
                   selection_values,
                   poly_order=2,
                   basis_type="tensor",
                   metadata={
                       "description":
                       ("Two scalar tensor-p2 fields with distinguishable "
                        "coefficients for testing cell and field selection."),
                       "analytic_function":
                       ("No prescribed continuous analytic function. "
                        "In cell (i,j), F_k=sum_{m=0}^8 c_{ijkm}*phi_m, "
                        "c_{ijkm}=18*(3*i+j)+9*k+m; k=0,1."),
                       "generation_method":
                       ("Consecutive integers reshaped in C order to "
                        "(4,3,18) modal coefficients, grouped by field.")
                   })

  # Stationary shock-tube initial state: p0 modal coefficients are sqrt(2)
  # times the physical values in the orthonormal 1D basis.
  x = (np.arange(100) + 0.5) / 100
  rho = np.where(x < 0.5, 1.0, 0.125)
  pressure = np.where(x < 0.5, 1.0, 0.1)
  moments = np.stack([
      rho,
      np.zeros_like(x),
      np.zeros_like(x),
      np.zeros_like(x), pressure / (5.0 / 3.0 - 1.0)
  ],
                     axis=-1)
  write_gkyl_field(out_dir / "shock_tube_1d_p0.gkyl", [100], [0.0], [1.0],
                   np.sqrt(2.0) * moments,
                   poly_order=0,
                   basis_type="serendipity",
                   metadata={
                       "description":
                       ("Stationary shock-tube initial state: density, three "
                        "momentum densities, and total energy density."),
                       "analytic_function":
                       ("rho=1 and p=1 for x<0.5; rho=0.125 and p=0.1 "
                        "for x>=0.5. Momentum=(0,0,0); E=p/(gamma-1), "
                        "gamma=5/3. Domain [0,1]."),
                       "generation_method":
                       ("Exact piecewise-constant p0 modal coefficients; "
                        "the discontinuity coincides with a cell edge.")
                   })
  growth_time = np.linspace(0.0, 10.0, 101)
  write_gkyl_dynvector(out_dir / "exponential_energy.gkyl",
                       growth_time, (1e-6 * np.exp(0.4 * growth_time))[:, None],
                       metadata={
                           "description":
                           ("Exponential energy history for growth-rate "
                            "fitting; amplitude growth rate is 0.2."),
                           "analytic_function":
                           "E(t)=1e-6*exp(0.4*t).",
                           "generation_method":
                           ("101 equally spaced time samples on [0,10], "
                            "including both endpoints; one component.")
                       })

  # Time-resolved, positive travelling waves, encoded as p0 modal fields.
  # Zero-padded frame names preserve time order under shell glob expansion.
  wave_x = (np.arange(64) + 0.5) * (2 * np.pi / 64)
  for frame, time in enumerate(np.linspace(0.0, 2 * np.pi, 16, endpoint=False)):
    wave = 1.0 + 0.6 * np.cos(wave_x - time)
    write_gkyl_field(out_dir / f"travelling_wave_{frame:03d}.gkyl", [64], [0.0],
                     [2 * np.pi],
                     np.sqrt(2) * wave[:, None],
                     poly_order=0,
                     basis_type="serendipity",
                     time=float(time),
                     frame=frame,
                     metadata={
                         "description":
                         "Positive 1D scalar wave travelling at unit speed.",
                         "analytic_function":
                         "f(x,t)=1+0.6*cos(x-t).",
                         "generation_method":
                         ("Cell-center samples on [0,2*pi], stored as p0 "
                          "modal constants. 16 frames at t=2*pi*frame/16.")
                     })
  surface_axes = [(np.arange(32) + 0.5) * (2 * np.pi / 32)] * 2
  surface_x, surface_y = np.meshgrid(*surface_axes, indexing="ij")
  for frame, time in enumerate(np.linspace(0.0, 2 * np.pi, 12, endpoint=False)):
    wave = 1.0 + 0.6 * np.cos(surface_x - time) * np.cos(surface_y)
    write_gkyl_field(out_dir / f"wave_surface_{frame:03d}.gkyl", [32, 32],
                     [0.0, 0.0], [2 * np.pi, 2 * np.pi],
                     2 * wave[..., None],
                     poly_order=0,
                     basis_type="serendipity",
                     time=float(time),
                     frame=frame,
                     metadata={
                         "description":
                         "Positive 2D wave surface travelling in x.",
                         "analytic_function":
                         "f(x,y,t)=1+0.6*cos(x-t)*cos(y).",
                         "generation_method":
                         ("Cell-center samples on [0,2*pi]^2, stored as p0 "
                          "modal constants. 12 frames at t=2*pi*frame/12.")
                     })
  # An anisotropic Gaussian scalar field for volume and isosurface views.
  volume_axis = (np.arange(24) + 0.5) / 6 - 2
  vx, vy, vz = np.meshgrid(volume_axis, volume_axis, volume_axis, indexing="ij")
  blob = np.exp(-(vx**2 + 2 * vy**2 + 0.5 * vz**2))
  write_gkyl_field(
      out_dir / "gaussian_volume.gkyl", [24, 24, 24], [-2.0] * 3, [2.0] * 3,
      np.sqrt(8) * blob[..., None],
      poly_order=0,
      basis_type="serendipity",
      metadata={
          "description":
          "Anisotropic Gaussian scalar for volume and isosurface views.",
          "analytic_function":
          "f(x,y,z)=exp(-(x^2+2*y^2+0.5*z^2)).",
          "generation_method":
          ("Cell-center samples on a 24^3 grid over [-2,2]^3, "
           "stored as p0 modal constants.")
      })

  # --- field files (random DG coefficients) ---
  for stem, ndim, cells, poly_order, basis_type in _FIELD_CONFIGS:
    nc = num_comps(basis_type, ndim, poly_order)
    lower = [0.0] * ndim
    upper = [1.0] * ndim
    values = _RNG.standard_normal((*cells, nc))
    write_gkyl_field(
        out_dir / f"{stem}.gkyl",
        cells,
        lower,
        upper,
        values,
        poly_order=poly_order,
        basis_type=basis_type,
        metadata={
            **_RANDOM_METADATA, "description":
            ("Random scalar DG field for basis, reader, and operation "
             "coverage; no physical interpretation or positivity guarantee.")
        },
    )

  # --- c2p mapping files (analytical DG coordinate coefficients) ---
  for entry in _C2P_CONFIGS:
    stem, kind, cells, poly_order, basis_type, *extra = entry
    nc_per_dim = num_comps(basis_type, len(cells), poly_order)
    comp_lo = [0.0] * len(cells)
    comp_hi = [1.0] * len(cells)

    if kind == "stretch":
      phys_lo, phys_hi = extra
      values = _c2p_stretch_values(cells, phys_lo, phys_hi, nc_per_dim)
      formula = ("X_d=lo_d+(hi_d-lo_d)*xi_d on computational [0,1]^2; "
                 f"lo={phys_lo}, hi={phys_hi}.")
    elif kind == "rotation":
      angle = extra[0]
      values = _c2p_rotation_values(cells, comp_lo, comp_hi, angle, nc_per_dim)
      formula = ("X=xi*cos(a)-eta*sin(a); Y=xi*sin(a)+eta*cos(a); "
                 f"a={angle} radians, computational (xi,eta) in [0,1]^2.")
    else:
      raise ValueError(f"Unknown c2p kind: {kind!r}")

    write_gkyl_field(
        out_dir / f"{stem}.gkyl",
        cells,
        comp_lo,
        comp_hi,
        values,
        poly_order=poly_order,
        basis_type=basis_type,
        metadata={
            "description":
            f"Computational-to-physical coordinate map: {kind}.",
            "analytic_function":
            formula,
            "generation_method":
            ("Exact affine modal coefficients, grouped by physical "
             "coordinate; higher-order modes are zero.")
        },
    )

  # --- symmetric four-component beam profiles for a convergence plot ---
  profile_lower, profile_upper, profile_cells = -2.5, 2.5, 256
  for alpha, stem in (
      (2.0e-4, "mirror_comparison_2em4_1d_ms_p1"),
      (2.0e-5, "mirror_comparison_2em5_1d_ms_p1"),
  ):
    values = _project_1d_p1(
        _mirror_comparison_profiles,
        profile_lower,
        profile_upper,
        profile_cells,
        alpha,
    )
    write_gkyl_field(
        out_dir / f"{stem}.gkyl",
        [profile_cells],
        [profile_lower],
        [profile_upper],
        values,
        poly_order=1,
        basis_type="serendipity",
        metadata={
            "description":
            ("Synthetic symmetric mirror-beam profiles (density, parallel "
             "velocity, parallel and perpendicular temperatures) for an "
             "algorithm-sensitivity plot, not simulation output."),
            "analytic_function":
            _mirror_comparison_profiles.__doc__,
            "alpha":
            alpha,
            "generation_method":
            ("Cellwise L2 projection onto p1 using eight-point Gauss "
             "quadrature on 256 cells over [-2.5,2.5]; "
             "coefficients grouped by component.")
        },
    )

  # --- two-frame distribution-like family (shared grid, one file per frame) ---
  distf_cells = [64, 32]
  distf_nc = num_comps("serendipity", 2, 2)
  for frame in (0, 1):
    values = _RNG.standard_normal((*distf_cells, distf_nc))
    write_gkyl_field(
        out_dir / f"distf_p2_{frame}.gkyl",
        distf_cells,
        [0.0, 0.0],
        [1.0, 1.0],
        values,
        poly_order=2,
        basis_type="serendipity",
        time=0.1 * frame,
        frame=frame,
        metadata={
            **_RANDOM_METADATA, "description":
            ("Two-frame distribution-shaped scalar fixture for collection "
             "and selection; random coefficients, not a physical or "
             "necessarily positive distribution function.")
        },
    )

  # --- multiblock family: 3 blocks x 2 frames of one 2-D field ---
  # Gkeyll's multiblock naming is '<sim>_b<N>-<quantity>_<frame>.gkyl' (a
  # real example: rt_gk_multib_sheath_1x2v_p1_b2-geo_int_B3.gkyl). The
  # blocks tile the x axis into abutting, disjoint domains -- one field on
  # a decomposed domain, which is what postgkyl must draw as one picture.
  mb_cells = [8, 6]
  mb_nc = num_comps("serendipity", 2, 1)
  for block in range(3):
    for frame in (0, 1):
      values = _RNG.standard_normal((*mb_cells, mb_nc)) + block
      write_gkyl_field(
          out_dir / f"mb_sim_b{block}-elc_M0_{frame}.gkyl",
          mb_cells,
          [float(block), 0.0],
          [float(block + 1), 1.0],
          values,
          poly_order=1,
          basis_type="serendipity",
          time=0.1 * frame,
          frame=frame,
          metadata={
              **_RANDOM_METADATA, "description":
              ("Random scalar DG field over three adjoining x blocks and "
               "two frames for multiblock plotting; not physical density."),
              "generation_method":
              (_RANDOM_METADATA["generation_method"] +
               f" Add block index {block} to every modal coefficient "
               "(not a uniform shift of physical field values).")
          },
      )

  # --- equation-agnostic geometry: two blocks and frames, in 2-D and 3-D ---
  # The fields are constant, while the geometry is affine in computational
  # coordinates. Both values and physical grid locations have analytic oracles.
  for ndim in (2, 3):
    edges = np.linspace(0.0, 1.0, 5)
    centers = 0.5 * (edges[:-1] + edges[1:])
    gauss = (centers[:, None] + np.array([-1.0, 1.0])[None, :] /
             (8 * np.sqrt(3.0))).ravel()
    coords = np.meshgrid(*([gauss] * ndim), indexing="ij")
    for block in (0, 1):
      prefix = f"mapped_{ndim}d_b{block}"
      for frame in (0, 1):
        value = 7.0 + block + 2 * frame
        write_gkyl_field(out_dir / f"{prefix}-fluid_{frame}.gkyl", [4] * ndim,
                         [0.0] * ndim, [1.0] * ndim,
                         np.full((4, ) * ndim + (1, ), value * 2**(ndim / 2)),
                         poly_order=0,
                         basis_type="serendipity",
                         frame=frame,
                         metadata={
                             "description":
                             "Constant scalar on a mapped geometry block.",
                             "analytic_function":
                             (f"f=7+block+2*frame={value}; block={block}, "
                              f"frame={frame}."),
                             "generation_method":
                             "Exact p0 modal coefficients."
                         })
      # 2-D uses Cartesian X/Y/Z; 3-D uses R/Z/phi on a field-aligned grid.
      point_values = (np.stack(
          [2.0 + block + coords[0],
           np.zeros_like(coords[0]), coords[1]],
          axis=-1) if ndim == 2 else np.stack([
              2.0 + block + coords[0], coords[2], 2 * np.pi * coords[1] +
              0.2 * coords[2]
          ],
                                              axis=-1))
      mapping_formula = (
          f"(X,Y,Z)=(2+{block}+x,0,z) in computational (x,z)." if ndim == 2 else
          f"(R,Z,phi)=(2+{block}+x,z,2*pi*y+0.2*z) in computational (x,y,z).")
      write_gkyl_field(out_dir / f"{prefix}-geo_int_nodes.gkyl", [8] * ndim,
                       [0.0] * ndim, [1.0] * ndim,
                       point_values,
                       poly_order=0,
                       basis_type="serendipity",
                       metadata={
                           "value_form":
                           "nodal",
                           "geometry_type":
                           3 if ndim == 2 else 1,
                           "description":
                           ("Physical coordinates for equation-agnostic "
                            "geometry: Cartesian X/Y/Z in 2D, cylindrical "
                            "R/Z/phi in 3D, with phi in radians."),
                           "analytic_function":
                           mapping_formula,
                           "generation_method":
                           ("Point values at two Gauss nodes per axis in "
                            "each of four computational cells on [0,1]; "
                            "stored on an 8-per-axis grid as nodal geometry.")
                       })
      if ndim == 2:
        # Exact p1 coefficients for X=2+block+x, Y=0, Z=z.
        x, z = np.meshgrid(centers, centers, indexing="ij")
        modal = np.zeros((4, 4, 12))
        modal[..., 0] = 2 * (2.0 + block + x)
        modal[..., 1] = 0.25 / np.sqrt(3.0)
        modal[..., 8] = 2 * z
        modal[..., 10] = 0.25 / np.sqrt(3.0)
        write_gkyl_field(
            out_dir / f"{prefix}-geo_int_mapc2p.gkyl", [4, 4], [0.0, 0.0],
            [1.0, 1.0],
            modal,
            poly_order=1,
            basis_type="serendipity",
            metadata={
                "geometry_type":
                3,
                "description":
                "Cartesian computational-to-physical coordinate map.",
                "analytic_function":
                mapping_formula,
                "generation_method": ("Exact p1 modal coefficients, grouped by "
                                      "physical coordinate X, Y, Z.")
            })

  # --- dynvector (bare time series, e.g. a field-energy history) ---
  # Two components, each a smooth logistic growth-then-saturate curve (one
  # slower-rising than the other) -- long enough (>15700 points) to
  # exercise the CLI's fit/growth leading-window scan, strictly positive
  # throughout (exp2's log-linear auto-guess needs y > 0), and with a
  # second component so column-selecting verbs (e.g. val2coord) have
  # something to select.
  n_energy = 15714
  t = np.linspace(0.0, 100.0, n_energy)
  y0, y_plateau, t_mid, k = 1e-6, 1.0, 30.0, 0.5
  energy0 = y0 + (y_plateau - y0) / (1.0 + np.exp(-k * (t - t_mid)))
  energy1 = y0 + (y_plateau - y0) / (1.0 + np.exp(-0.5 * k * (t - 1.5 * t_mid)))
  write_gkyl_dynvector(out_dir / "energy_dynvec.gkyl",
                       t,
                       np.stack([energy0, energy1], axis=-1),
                       metadata={
                           "description":
                           ("Two positive synthetic energy histories with "
                            "logistic growth and saturation, for fitting "
                            "and time-series component selection."),
                           "analytic_function":
                           (f"E0(t)={y0}+({y_plateau}-{y0})/"
                            f"(1+exp(-{k}*(t-{t_mid}))); "
                            f"E1(t)={y0}+({y_plateau}-{y0})/"
                            f"(1+exp(-{0.5*k}*(t-{1.5*t_mid})))."),
                           "generation_method":
                           (f"{n_energy} equally spaced time samples on "
                            "[0,100], including both endpoints.")
                       })


if __name__ == "__main__":
  out = Path(__file__).parent / "test_data" / "generated"
  generate_all(out)
  files = sorted(out.glob("*.gkyl"))
  print(f"Generated {len(files)} files in {out}:")
  for f in files:
    print(f"  {f.name}")
