"""Pure NumPy/SciPy helpers -- no internal imports (the leaf-most layer)."""

from .idx_parser import idx_parser
from .elementwise import grids_compatible, grid_is_prefix
from .calculus import integrate
from .mag_sq import mag_sq
from .rel_change import rel_change
from .rotation_matrix import rotation_matrix
from .fft import fft, init_polar, polar_isotropic
from .fit import (
    FIT_FUNCTIONS,
    FIT_NDIM,
    RPN_OPERATORS,
    RPN_FUNCTIONS,
    linear,
    quadratic,
    plane,
    quadratic2d,
    exp_plateau,
    gaussian,
    power,
    sinusoid,
    tanh_transition,
    exp2,
    rpn_param_names,
    rpn_ndim,
    fit_evaluate,
    fit,
    auto_guess,
    fit_best_window,
)
from .filters import fft_filtering, butter_filtering
from .ev_ops import cmds as ev_cmds
from .grid_centering import nodal_to_cell_centered_grid
from .downsample import downsample
from .natural_sort import natural_sort_key

__all__ = [
    "idx_parser",
    "grids_compatible",
    "grid_is_prefix",
    "integrate",
    "mag_sq",
    "rel_change",
    "rotation_matrix",
    "fft",
    "init_polar",
    "polar_isotropic",
    "FIT_FUNCTIONS",
    "FIT_NDIM",
    "RPN_OPERATORS",
    "RPN_FUNCTIONS",
    "linear",
    "quadratic",
    "plane",
    "quadratic2d",
    "exp_plateau",
    "gaussian",
    "power",
    "sinusoid",
    "tanh_transition",
    "exp2",
    "rpn_param_names",
    "rpn_ndim",
    "fit_evaluate",
    "fit",
    "auto_guess",
    "fit_best_window",
    "fft_filtering",
    "butter_filtering",
    "ev_cmds",
    "nodal_to_cell_centered_grid",
    "downsample",
    "natural_sort_key",
]
