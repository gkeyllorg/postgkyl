import postgkyl as pg
from _example_paths import TEST_DATA, prepare_output_dir

DATA = TEST_DATA / "generated"
OUTPUT_DIR = prepare_output_dir()
figure_path = OUTPUT_DIR / "mirror_comparison.pdf"

high_alpha = pg.load(DATA /
                     "mirror_comparison_2em4_1d_ms_p1.gkyl").interpolate()
reference = pg.load(DATA / "mirror_comparison_2em5_1d_ms_p1.gkyl").interpolate()

fig = pg.plot(
    high_alpha,
    reference,
    figure=0,
    legend_labels=["2e-4", "2e-5"],
    no_legend=False,
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
    subplot_ylabels=(r"Density [$\mathrm{m}^{-3}$],"
                     r"$U_\parallel$ [m/s],"
                     r"$T_\parallel$ [keV],"
                     r"$T_\perp$ [keV]"),
    saveas=figure_path,
    no_show=True,
)
fig.savefig(OUTPUT_DIR / "mirror_comparison.png", dpi=150)
