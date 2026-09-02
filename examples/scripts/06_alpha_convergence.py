import postgkyl as pg
from _example_paths import TEST_DATA, prepare_output_dir

DATA = TEST_DATA / "generated"
OUTPUT_DIR = prepare_output_dir()

high_alpha = pg.load(DATA / "alpha_convergence_2em4_1d_ms_p1.gkyl").interpolate()
reference = pg.load(DATA / "alpha_convergence_2em5_1d_ms_p1.gkyl").interpolate()

figure_path = OUTPUT_DIR / "06_alpha_convergence.pdf"
pg.plot(
    high_alpha,
    reference,
    multiblock=True,
    color=["#D55E00", "#0072B2"],
    legend_labels=["2e-4", "2e-5"],
    linestyle=["-", "--"],
    legend=True,
    legend_subplot=0,
    legend_loc="best",
    title=r"Comparing $\alpha$ = 2e-4 to $\alpha$ = 2e-5",
    xlabel="z [m]",
    figsize="12,10",
    split_linear_log=True,
    split_point=0.0,
    split_log_side="right",
    split_width_ratios=(1.0, 1.0),
    split_gap=0.0,
    split_legend_side="log",
    split_log_nonpositive="mask",
    split_linear_ylim=[
        (0.0, 3e19),
        (-1.4e6, 1.4e6),
        (-2.0, 14.0),
        (0.0, 27.0),
    ],
    split_log_ylim=[
        (1e13, 4e19),
        (1e2, 2e6),
        (1e-1, 20.0),
        (1e-1, 50.0),
    ],
    subplot_ylabels=(
        r"Density [$\mathrm{m}^{-3}$],"
        r"$U_\parallel$ [m/s],"
        r"$T_\parallel$ [keV],"
        r"$T_\perp$ [keV]"
    ),
    saveas=figure_path,
    show=False,
)