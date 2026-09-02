"""Generate synthetic .gkyl test files for the postgkyl test suite.

Run directly to regenerate:
    python tests/generate_test_data.py

Called automatically by conftest.py at the start of each pytest session.

Field files encode polyOrder and basisType in their msgpack metadata block so
GData auto-populates ctx["poly_order"] and ctx["basis_type"] on load.

C2P mapping files store modal DG coefficients for analytical coordinate
transformations.  The basis is inferred by GData from num_comps/ndim via
_get_basis_p().  Two mapping types are provided:
  - "stretch": linear map (comp domain [0,1]^n → physical domain phys_bounds)
  - "rotation": 2D rotation by angle α about the origin

Dynvector files (file_type=2, a bare time series with no spatial grid --
e.g. a field-energy history) are written by ``write_gkyl_dynvector``.
"""
import struct
from pathlib import Path

import msgpack
import numpy as np

_RNG = np.random.default_rng(42)
_SQRT3 = np.sqrt(3)

# Component counts per basis -- mirrors the tables in src/postgkyl/data/dg.py
# serendipity: indexed as [ndim-1][poly_order]   (p=0 → 1 component)
_COMPS_SER = [
    [1, 2, 3, 4, 5],    # 1D
    [1, 4, 8, 12, 17],  # 2D
    [1, 8, 20, 32, 50], # 3D
]
# tensor: indexed as [ndim-1][poly_order-1]   (p starts at 1)
_COMPS_TEN = [
    [2, 3, 4, 5],       # 1D
    [4, 9, 16, 25],     # 2D
    [8, 27, 64, 125],   # 3D
]
# maximal-order: indexed as [ndim-1][poly_order-1]
_COMPS_MAX = [
    [2, 3, 4, 5],       # 1D
    [3, 6, 10, 15],     # 2D
    [4, 10, 20, 35],    # 3D
]

_COMPS = {
    "serendipity":   (_COMPS_SER, lambda p: p),
    "tensor":        (_COMPS_TEN, lambda p: p - 1),
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
) -> None:
    """Write a minimal valid .gkyl v1 binary field file with msgpack metadata."""
    ndim = len(cells)
    nc = values.shape[-1]

    meta = msgpack.packb({
        "polyOrder": poly_order,
        "basisType": basis_type,
        "time": time,
        "frame": frame,
    })

    with open(path, "wb") as f:
        # --- version-1 header ---
        f.write(b"gkyl0")
        f.write(struct.pack("<q", 1))           # version
        f.write(struct.pack("<q", 1))           # file_type = 1 (field)
        f.write(struct.pack("<q", len(meta)))   # meta_size
        f.write(meta)

        # --- field body ---
        f.write(struct.pack("<q", 2))           # real_type = 2 → float64
        f.write(struct.pack("<q", ndim))
        for c in cells:
            f.write(struct.pack("<q", c))
        for lo in lower:
            f.write(struct.pack("<d", lo))
        for hi in upper:
            f.write(struct.pack("<d", hi))
        esznc = nc * 8                          # element_size * num_comps (bytes)
        size = int(np.prod(cells))              # total number of cells
        f.write(struct.pack("<q", esznc))
        f.write(struct.pack("<q", size))
        f.write(values.astype("<f8").tobytes()) # C-order, little-endian float64


def write_gkyl_dynvector(path: Path, time: np.ndarray, values: np.ndarray) -> None:
    """Write a minimal valid .gkyl v1 binary dynvector (file_type=2) file.

    A dynvector has no spatial grid -- just a time series of ``num_comps``
    values per sample (e.g. a field-energy history). Layout per
    ``gkyl_reader.py``'s ``_read_t2_v1``: header, real_type, esznc, size,
    then all of TIME_DATA followed by all of DATA (C-order).
    """
    nc = values.shape[-1]
    size = len(time)

    with open(path, "wb") as f:
        f.write(b"gkyl0")
        f.write(struct.pack("<q", 1))            # version
        f.write(struct.pack("<q", 2))            # file_type = 2 (dynvec)
        f.write(struct.pack("<q", 0))             # meta_size = 0 (no meta)
        f.write(struct.pack("<q", 2))             # real_type = 2 -> float64
        esznc = nc * 8                            # element_size * num_comps (bytes)
        f.write(struct.pack("<q", esznc))
        f.write(struct.pack("<q", size))
        f.write(time.astype("<f8").tobytes())
        f.write(values.astype("<f8").tobytes())   # C-order, little-endian float64


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

    grids = np.meshgrid(*[np.arange(cells[d]) for d in range(ndim)], indexing="ij")
    for d in range(ndim):
        mid = phys_lo[d] + dx[d] * (grids[d] + 0.5)
        off = d * num_modes
        values[..., off] = 2.0 * mid              # constant mode
        values[..., off + 1 + d] = dx[d] / _SQRT3 # linear mode in d-th direction
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
    dxi  = (comp_hi[0] - comp_lo[0]) / N_x
    deta = (comp_hi[1] - comp_lo[1]) / N_y

    ii, jj = np.mgrid[0:N_x, 0:N_y]
    xi_mid  = comp_lo[0] + dxi  * (ii + 0.5)
    eta_mid = comp_lo[1] + deta * (jj + 0.5)

    x_mid = xi_mid * ca - eta_mid * sa
    y_mid = xi_mid * sa + eta_mid * ca

    values = np.zeros((N_x, N_y, 2 * num_modes))
    # x-coordinate modal coefficients
    values[..., 0] = 2.0 * x_mid              # constant
    values[..., 1] = dxi  * ca / _SQRT3       # xi'-mode (dx/dxi' * mapping factor)
    values[..., 2] = -deta * sa / _SQRT3      # eta'-mode
    # y-coordinate modal coefficients
    values[..., num_modes + 0] = 2.0 * y_mid
    values[..., num_modes + 1] = dxi  * sa / _SQRT3
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
    """
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")

    radius = np.abs(z)
    sensitivity = np.log10(alpha / 2.0e-5)
    pedestal = np.exp(-((radius - 0.86) / 0.16) ** 2)

    density = (
        1.01e13
        + 2.45e19 / (1.0 + np.exp((radius - 0.92) / 0.08))
        + 4.5e18 * np.exp(-(radius / 0.55) ** 4)
    )
    velocity = (
        1.30e6 * np.tanh(z / 0.05)
        / (1.0 + np.exp((0.82 - radius) / 0.05))
    )
    t_parallel = (
        0.101
        + 11.8 / (1.0 + np.exp((radius - 0.90) / 0.10))
        + 0.4 * np.exp(-((radius - 0.55) / 0.15) ** 2)
    )
    t_perpendicular = (
        0.101
        + 20.8 / (1.0 + np.exp((radius - 0.98) / 0.13))
        + 10.0 * np.exp(-((radius - 0.86) / 0.10) ** 2)
    )

    return np.stack([
        density * (1.0 - 0.025 * sensitivity * pedestal),
        velocity * (1.0 + 0.020 * sensitivity * pedestal),
        t_parallel * (1.0 + 0.035 * sensitivity * pedestal),
        t_perpendicular * (1.0 + 0.030 * sensitivity * pedestal),
    ], axis=-1)


def _project_1d_p1(fn, lower: float, upper: float, cells: int, *args) -> np.ndarray:
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
    ], axis=-1)
    coefficients = np.einsum("cqf,qb,q->cfb", samples, basis, weights)
    return coefficients.reshape(cells, -1)


# ---------------------------------------------------------------------------
# Configuration tables
# ---------------------------------------------------------------------------

#  (stem, ndim, cells, poly_order, basis_type)
_FIELD_CONFIGS: list[tuple] = [
    ("1d_ms_p1", 1, [8],       1, "serendipity"),
    ("1d_ms_p2", 1, [8],       2, "serendipity"),
    ("2d_ms_p1", 2, [8, 8],    1, "serendipity"),
    ("2d_ms_p2", 2, [8, 8],    2, "serendipity"),
    ("2d_mt_p1", 2, [8, 8],    1, "tensor"),
    ("2d_mt_p2", 2, [8, 8],    2, "tensor"),
    ("2d_mo_p1", 2, [8, 8],    1, "maximal-order"),
    ("2d_mo_p2", 2, [8, 8],    2, "maximal-order"),
    ("3d_ms_p1", 3, [4, 4, 4], 1, "serendipity"),
]

# C2P mapping files.
# (stem, kind, cells, poly_order, basis_type, extra...)
#   kind="stretch":  extra = (phys_lo, phys_hi)          -- comp domain [0,1]^n
#   kind="rotation": extra = (angle,)                     -- comp domain [0,1]^2
_C2P_CONFIGS: list[tuple] = [
    # Linear stretch: physical x∈[0,2], y∈[0,3]; paired with 2d_ms_p1.gkyl
    ("2d_c2p_stretch_ms_p1", "stretch", [8, 8], 1, "serendipity",
     [0.0, 0.0], [2.0, 3.0]),
    # Same stretch for p=2; paired with 2d_ms_p2.gkyl
    ("2d_c2p_stretch_ms_p2", "stretch", [8, 8], 2, "serendipity",
     [0.0, 0.0], [2.0, 3.0]),
    # Rotation by 45°; paired with 2d_ms_p1.gkyl (comp domain [0,1]^2)
    ("2d_c2p_rot45_ms_p1", "rotation", [8, 8], 1, "serendipity",
     np.pi / 4),
]


def generate_all(out_dir: Path | str) -> None:
    """Write all synthetic test files to *out_dir*."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- field files (random DG coefficients) ---
    for stem, ndim, cells, poly_order, basis_type in _FIELD_CONFIGS:
        nc = num_comps(basis_type, ndim, poly_order)
        lower = [0.0] * ndim
        upper = [1.0] * ndim
        values = _RNG.standard_normal((*cells, nc))
        write_gkyl_field(
            out_dir / f"{stem}.gkyl",
            cells, lower, upper, values,
            poly_order=poly_order,
            basis_type=basis_type,
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
        elif kind == "rotation":
            angle = extra[0]
            values = _c2p_rotation_values(cells, comp_lo, comp_hi, angle, nc_per_dim)
        else:
            raise ValueError(f"Unknown c2p kind: {kind!r}")

        write_gkyl_field(
            out_dir / f"{stem}.gkyl",
            cells, comp_lo, comp_hi, values,
            poly_order=poly_order,
            basis_type=basis_type,
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
            [profile_cells], [profile_lower], [profile_upper], values,
            poly_order=1,
            basis_type="serendipity",
        )

    # --- two-frame distribution-like family (shared grid, one file per frame) ---
    distf_cells = [64, 32]
    distf_nc = num_comps("serendipity", 2, 2)
    for frame in (0, 1):
        values = _RNG.standard_normal((*distf_cells, distf_nc))
        write_gkyl_field(
            out_dir / f"distf_p2_{frame}.gkyl",
            distf_cells, [0.0, 0.0], [1.0, 1.0], values,
            poly_order=2,
            basis_type="serendipity",
            time=0.1 * frame,
            frame=frame,
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
                mb_cells, [float(block), 0.0], [float(block + 1), 1.0], values,
                poly_order=1,
                basis_type="serendipity",
                time=0.1 * frame,
                frame=frame,
            )

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
    write_gkyl_dynvector(
        out_dir / "energy_dynvec.gkyl", t, np.stack([energy0, energy1], axis=-1))


if __name__ == "__main__":
    out = Path(__file__).parent / "test_data" / "generated"
    generate_all(out)
    files = sorted(out.glob("*.gkyl"))
    print(f"Generated {len(files)} files in {out}:")
    for f in files:
        print(f"  {f.name}")
