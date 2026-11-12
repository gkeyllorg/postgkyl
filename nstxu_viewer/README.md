# NSTX-U run viewer (2026-11-02)

A marimo notebook for looking through the NSTX-U runs frame by frame, in the DPP 2025 poster's theme (dark by
default, light on a switch): open it on a shot, a run and a frame, or play the run as a movie, probe any cell over
time, read the particle and energy budgets and the power to each target, and save the view, the figure or a movie.
Nothing here writes into a run directory; the outputs go to the data directory's `runs/` (Setup): the per-frame
cache `runs/cache/`, saved views `runs/states/`, exported figures and movies `runs/exports/`, Flux copies
`runs/flux/`, the server's log; the page's stylesheet is `runs/viewer.css` next to the notebook.

## Setup
This directory lives on postgkyl's branch `marimo-gui-jjz73`. The viewer does not import postgkyl (it reads the
`.gkyl` files itself, `gkyl_read.py`); the pgkyl commands it shows are postgkyl 2's.
```sh
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.lock.txt   # or: NSTXU_VIEWER_VENV=<a venv with these>
```
**The data directory** (`NSTXU_VIEWER_DATA`; default: this directory) holds what the viewer reads and writes:
- `viewer.json` (start from `viewer.example.json`): paths relative to `base`, itself relative to the data directory.
  - `base`: what the simulation directories (and the cache's file names) are relative to;
  - `sims`: `[label, simulation directory, default run, one-line description]` each; a simulation's runs are the run
    directories (a topology file, or a single-block run's files, and a frame) in its `runs/`, else in it;
  - `pgkyl` (optional): the pgkyl executable the commands name (else `pgkyl` from PATH);
  - `thomson_fallback` (optional): a directory of slim HDF5 shot files (the sim-init workflow's `data/h5_slim`);
  - `theme_files` (optional): the themes' PowerPoint files, which `check_viewer.py` compares the kept colours with.
- `flux.env` (from `flux.env.example`): Fetch from Flux's login and paths (`fetch_flux.sh`).
- `runs/`: everything the viewer writes.

A data directory can carry its own `view.sh` that sets both variables and runs this one, so that `./view.sh` there
opens the viewer on it.

```sh
./view.sh                                                 # app view, opened in the browser
./view.sh edit                                            # notebook view (every cell editable)
./view.sh --shot 203899 --frame 30 --qty T_elc            # open on a shot, a frame and a quantity
./view.sh --sim 'Sim 1' --frames 0:60 --probe 10,0,15 --tab Probe
./view.sh --state runs/states/my_view.json                # a view saved from the page
./view.sh --path DIR                                      # any run directory (multiblock or single-block)
./view.sh --sim 'Sim 2' --blocks 1-3 --cmp '2025 re-run/prod' # some blocks only, another run drawn over it
./view.sh --help                                          # every key
OPEN=0 ./view.sh                                          # print the URL only
```
Keys (`viewer_state.KEYS`): `shot`, `sim`, `run`, `frame` (negative: from the last), `qty`, `mode`
(value / change / step), `log`, `zoom`, `x`, `dark`, `tab`, `frames` (the movie's frames, `pgkyl load --frame`
syntax), `probe` (block,i,j), `blocks` (the blocks drawn: `1-3,10`), `cmp` (a run drawn over this one: `<simulation>/<run>` or a
directory), `path` (a run directory to open). A view opens from the defaults, then a saved view, then the command line, then the
page's address, each overriding the one before; the page writes the current view into its address, so a reload or
a bookmark comes back to it (the address carries the access token too: do not share it while the server runs). A
value that no longer fits (a run removed, a frame past the last) falls back to the default with a note at the top.
Save & export shows the `./view.sh` line for the current view.

In a VS Code terminal the page opens on your own machine (`$BROWSER` is VS Code's opener; it forwards the port), in
Chrome: a workspace's `.vscode/settings.json` can set `"workbench.externalBrowser": "chrome"` (it applies to every
external link opened from this window; delete the line to go back to the system default browser). If VS Code ignores
it as a workspace setting, put the same line in your user settings. Outside VS Code, view.sh prints the `ssh -L`
line to run on your machine. The access token stays on: anyone on a shared login node can reach 127.0.0.1. Ctrl-C
stops the server. Python: the venv of Setup (marimo 0.25.2, plotly 7.1.0, anywidget 0.11.0, imageio-ffmpeg 0.6.0,
h5py 3.16.0, numpy 1.26.4, matplotlib 3.11.0; full list in `requirements.lock.txt`).

## What it shows
- **Sidebar:** shot (searchable: type `203899`), run (every run directory of that simulation with a topology file
  and at least one frame; the production run by default), frame (slider + number box; it moves the movie too),
  movie frames (a frame range for the movie and the exported movie: `:`, `::2`, `-50:`, `0:60`), probe cell
  (block,i,j, or a click on the movie), quantity, value / change since the first frame / change since the previous
  frame (the last shows fronts moving), log scale, poloidal view (whole domain, lower/upper divertor,
  outboard/inboard midplane), **blocks drawn** (none chosen: every block; see below), **draw another run over it**
  (the overlay, see below), profile x axis (distance from the separatrix or R), **theme** (NSTX-U, PPPL or DIII-D;
  see Look) and **dark mode** (on by default; off = the theme's white look, page and figures together, no reload).
- **Header:** run, domain, cell count, frame span, key options; the full path of the simulation output (and the
  overlay's) with its file names; tiles for t, time step, λq and peak heat flux at each target, cells with n ≤ 0.
- **Movie** (`movie_widget.py`): poloidal plane (wall, separatrix dashed, x = n ≤ 0), every cell unrolled (the
  posters' strips), one flux surface of each strip against time (space-time: a disturbance running along the field
  is a slanted streak; the surface is outlined in the strips and chosen in the movie's bar), and the outboard /
  inboard midplane profiles. It plays in the browser: every frame it plays is sent to the page once (8-bit colour
  indices, 0.1-0.6 MB for these runs) and drawn on canvases at 2-60 frames/s, with no kernel round trips and so no
  fading. One colour scale over the frames it plays. Space: play / pause; arrows: step; loop; "interpolate" blends
  neighbouring frames for display only (labelled on the page). A click on a cell probes it (outlined in every
  panel); a click on the space-time strip probes that cell and moves to that time. The open tab's charts follow it
  as it plays (below); the notebook (tiles, slider, every tab redrawn) follows when playback pauses or on a scrub,
  step or click.
  Quantities: n, T, T_par, T_perp, u_par per species, ion Mach number, φ, q_par (total) and q_perp, the part of
  q_par carried by perpendicular energy.
- **Tabs:**
  - *Midplane profiles*: outboard and inboard (n, T, u_par, Mach, φ; shown frame against the first) and the selected
    quantity at the midplane over (radius, time).
  - *Targets*: q_par against midplane distance, λq and peak q_par over time, every target.
  - *Time traces*: dt, cells with n ≤ 0, particle content, field energy, the volume-mean n_e and T_e change in
    core / SOL / PF.
  - *Probe*: every quantity at the probed cell over time (both species), with its region's volume mean dotted; q_perp
    dashed on the q_par panel.
  - *Balance & power*: particles (each species) and energy, whole device: source, boundary losses, rate of change,
    residual; the power through each target's plate. See Definitions for the energy residual.
  - *Diffusivity*: the anomalous diffusivity D the run used, at the outboard and inboard midplane node by node
    (core, then SOL; log axis when it spans more than ×20) and in every cell on the poloidal plane and the strips,
    with where it comes from (below).
  - *Options*: what the run was started with, readable: the options it set at launch (its driver's section for each,
    what it is and the default, from the driver's own source), every other option the driver reads at its default,
    the files it was given, every option side by side with the overlay run's (orange: they differ, listed first;
    teal: the same), its jobs (restarts.log), and what the driver reported while setting the run up (below).
  - *Cells*: a per-cell table (searchable, downloadable). *Static figure*: the map as a matplotlib figure (its own
    colour scale: this frame or the whole run). *Run*: block layout, targets, options, command, restarts.log, slurm.
  - *Save & export*: the view (a state file, and its `./view.sh` line), the figure (png / pdf / svg, saved under
    `runs/exports` or downloaded to your machine), a movie of the map over the movie frames (MP4, H.264, one colour
    scale, the time on each frame; saved, shown and downloadable).

  The plotly tabs (*Midplane profiles* to *Balance & power*) follow the movie while it plays, in the page: the
  profiles and the target heat-flux curves take each frame's values, and the time cursors and λq lines move
  (`movie_widget.LivePlot`). While playing it rewrites plotly's own SVG through plotly's axis maps (~0.1 ms a chart
  frame, so the movie keeps its speed); when playback stops it hands plotly that frame, so hover and a resize agree.
  Their axes hold still. A y axis with moving curves spans them over the frames the movie plays (as its colour
  scale), leaving out frames whose extremes stand far from the rest (beyond 3 interquartile ranges past the
  quartiles: Sim 1's start-up to frame ~33, where q_par at the plates reaches 2400 MW/m² against ~130 later; Sim 2's
  heat burst, frames 6-10 in its targets tab), plus the frame shown and the dotted first frame. A frame that runs
  past the axes says so under the chart while it plays; paused there, the axes widen to show all of it. An axis a
  cursor would widen (at the first or last frame) spans every position. Where plotly's
  SVG is not laid out as expected (a plotly upgrade) it uses plotly's own updates instead, at most a third of the
  time. *Cells* and *Static figure* change when playback pauses. A tab's charts are drawn once scrolled into view.

Simulations offered: `viewer.json`'s `sims` (`gkview.SIMS`). Shot and equilibrium time come from each run's geqdsk header.

## Files
| file | what it is |
|---|---|
| `nstxu_viewer.py` | the marimo notebook (UI only) |
| `gkview.py` | data layer: run discovery (multiblock and single-block), readers (fields, dynvectors), block layout from the topology file (any layout, psi either way), derived quantities, profiles, target heat flux, per-frame cache, frame ranges, cell volumes and region means, particle / energy balance and target power, the diffusivity, the overlay's frame matching |
| `viewer_plots.py` | figures: map (matplotlib), profiles / targets / traces / probe / balance (plotly); the shared colour scale; figure and movie export; the poster theme's two looks; the page CSS |
| `movie_widget.py` | the browser-side movie (anywidget: the payload from the cache, and the canvas player in JavaScript), and LivePlot: the tab charts that follow it |
| `pgkyl_cmd.py` | the pgkyl command equivalent to each curve, cell and map the page draws (click it) |
| `viewer_state.py` | where the viewer opens: the view keys, state files, the `./view.sh` line, the blocks syntax |
| `run_options.py` | the Options tab: the driver's options read from its source, the run's values, its jobs, its start-up report |
| `thomson.py` | measured n_e: Thomson (MPTS) of the run's shot from `runs/flux/<shot>.h5`, the beam flattop, the averaging window, channel placement |
| `fetch_flux.sh` | everything about reaching PPPL Flux: copies a shot's Thomson / EFIT01 / CHERS / beam power to `runs/flux/`, through the laptop's forwarded socket (or from the laptop); the login `view.sh --flux` keeps |
| `gkyl_read.py` | the `.gkyl` reader (2D fields, topology files, their MessagePack metadata; numpy only) |
| `check_gkview.py` | data checks, against the poster and movie scripts (`sims_20261022/` under `base`; with a postgkyl venv: they need postgkyl and scipy) |
| `check_viewer.py` | viewer checks: headless, exports, pgkyl commands run, and with `--browser <chrome>` the page itself (with the viewer's venv) |
| `view.sh` | launcher: writes `runs/viewer.css` next to the notebook, starts marimo with the view keys, opens the page |
| `requirements.lock.txt` | the viewer venv's packages, pinned |
| `viewer.example.json`, `flux.env.example` | a data directory's settings, to copy |
| `pyproject.toml` | `[tool.marimo.display] theme = "dark"` (marimo reads the nearest pyproject.toml) |

## Look
The theme of the DPP 2025 poster (`aps_dpp_2025_poster_jjosephyzack.pptx`), read from the .pptx with
`viewer_plots.read_theme` (the same reader as `plot_suite_poster.py`) and kept in `viewer_plots.SCHEMES`, as are the
PPPL and DIII-D themes (`check_viewer.py` compares them with the files where `viewer.json` names them): Arial (rendered with its metric twin
Liberation Sans in the matplotlib maps, where Arial is not installed; an SVG export names Arial first), and its
colours in PowerPoint tone steps. Light = the posters (white, black ink, the progress posters' slate-blue ↔ crimson
map for changes); dark = the same theme on its slate (`dk2` darker 50 %, `#1B262B`), page included. Magnitudes in
inferno, as the DPP poster's n_e and T_e maps. Targets: orange = outer, teal = inner, dashed with squares = upper
(the theme has no four hues that stay apart under colour-vision deficiency, so the near-symmetric upper/lower pair
shares a hue; checked with the data-viz palette validator, see the docstring). Electrons teal, ions crimson; core
crimson, SOL gold, private flux teal; budget terms: source gray dashed, losses crimson, rate of change gold
dash-dot, residual ink dotted (clear of the targets' hues). Charts are SVG (no WebGL needed).

The dark/light switch sets `body[data-vscode-theme-kind]`, which marimo 0.25.2 reads before its own theme setting
(its VS Code hook), from a zero-height iframe: a marimo upgrade may need this checked. The same iframe puts the chosen
theme's page CSS (`viewer_plots.page_css`) into one `<style id="nstxu-theme">` of the page's head.

**Three themes** (`--theme`, the sidebar's radio; `viewer_plots.THEME_LOOKS`), each a PowerPoint theme read at
start-up, each light and dark, the same roles throughout (targets orange outer / blue-teal inner; electrons and PF the
inner hue; ions and core red; SOL a third hue), every role group checked with the data-viz palette validator (all
pairs under protan / deutan, normal vision, 3:1 against its own surface):
- *NSTX-U*: the DPP 2025 poster, as above.
- *PPPL*: `PPPL Powerpoint Slide Template.pptx` ('PPPL 2025', a dark scheme: white text on slate `dk2` #29404B,
  titles in its orange #F58025, Roboto Condensed titles). Dark charts sit on `dk2` darker 25 % (#1F3038; on the
  slides' own slate no marks inside the dark lightness band reach 3:1), cards on the slate; light = its 'White'
  slides. Diverging teal ↔ orange. Page headings in Roboto Condensed (a Google font; Arial Narrow / Arial where it
  cannot load) and its orange.
- *DIII-D*: `negD_CEDA_aug28.pptx` (Office 2007 scheme, Century Gothic, white slides under a navy title bar, body text
  `dk2` #1F497D), checked against `bernard_poster_aps25.pdf` (the same scheme; red / blue for the two cases compared,
  orange for emphasis). Dark charts on the title bar's navy #00217B darker 50 % (#00103E). Diverging blue ↔ red.
  Page text in Century Gothic (geometric look-alikes where it is not installed); a level-2 heading is the slides'
  navy title bar (its gradient as rendered: #172250 at the ends, #21409A in the middle).
Charts keep Arial in every theme (the PPPL template's own theme font; the label spacing is sized with its metrics).

## Definitions
Same as `sims_20261022/poster_progress.py` and `movie_density.py`: cell averages (c0 / 2); masses and charges from
the files' metadata; T = m (M2 − M1²/M0) / (3 M0) (M2perp = 2 p_perp / m); q_par = Σ_s m_s (M3par + M3perp) / 2;
q_perp = Σ_s m_s M3perp / 2, the part of q_par carried by perpendicular energy (Gkeyll's M3perp is the flux along B
of v_perp² = 2μB/m, the same factor as M2perp, checked in `gyrokinetic_mom_2x2v_ser_p1.c`); λq = the midplane width holding 63 % of |q_par| in the plate cells, each SOL surface at its Z = 0 radius (DPP 2025
definition). Midplane profiles: linear in θ between the two cells whose centres bracket Z = 0 (the cell next to a
reflecting midplane), at the row's Z = 0 radius. "Change since the previous frame" is against the previous
*written* frame, also when the movie plays a stride.

Region means (time traces, probe) are volume means: the cell volume is the Jacobian's cell average times the
logical cell size (`geo_int_jacobgeo`), the weighting postgkyl's `average --weight` uses. They differ from the
earlier medians where a few large cells carry a region: Sim 2's private-flux T_e during the 90 µs burst peaks at
+800 % as a volume mean, +150 % as a median.

Budgets (`gkview.balance`), from the integrated diagnostics: particles = source M0, the M0 of every boundary-flux
face (interior faces cancel pairwise), ∂f/∂t's M0; energy = Σ_s m_s M2 / 2 of the source (this driver integrates
the source's M0M1M2 moments, not its Hamiltonian; the source's q φ term cancels for a quasineutral source, measured
1e-11 W), the Hamiltonian of every boundary flux, ∂f/∂t's Hamiltonian minus the field-energy rate (as
`postgkyl.diagnostics.gk.energy_balance`; that diagnostic would take the source's M2 for its Hamiltonian, off by the
mass for this driver). Target power = the Hamiltonian boundary flux through the plate face of the leg and PF blocks,
both species. The diagnostics are per radian of toroidal angle: × 2π for one target's ring, × 2π × 2 for a half
domain's whole device (Sim 2: its source rate × 4π = its set particle rate, 1.8658e21 /s, to 0.03 %). At the last
frames: Sim 1 (7.12 ms) source 3.34 MW, outer targets 0.79 / 0.61 MW, inner 0.12 / 0.085 MW, 53 % of the boundary
energy loss on the plates; Sim 2 (300 µs) source 0.47 MW, lower outer target 0.070 MW, still heating (dE/dt 0.44 MW).

**The energy residual is not understood.** Particles close (Sim 1: 0.005 % electrons, 0.000 % ions; Sim 2 ions
0.00 %, electrons −5 %). Energy does not: +9.1 % of the source in Sim 1's steady state, −35 % in Sim 2's transient,
and large either way in the first 60 µs of the Sim 1 collisions on / off pair (so not the BGK operator alone). Read
the residual as an open question about the diagnostics' energy accounting, not as a physical loss.

The layout is derived from each run's `-block_topo.gkyl` and node files, never typed in: SOL = psi-upper face
physical, core/PF = psi-lower face physical (PF if a θ face is a target), a physical θ face on Z = 0 (span-relative)
is the reflecting midplane, SOL chains along θ give the strips, the chain's block reaching Z = 0 gives the midplane
radii and its psi-lower partner the core block below. Inner/outer from the separatrix radius at Z = 0 against the
core's mid-R; upper/lower from the target's Z.

## Blocks, the overlay, the diffusivity, the options (2026-11-10)
**The unrolled strips are reflected left to right** (the movie, its space-time panel, the static figure, the exported
movie): a strip's first cell is at its right. Turned a quarter turn, a strip then lies as its cells do in the poloidal
plane: Sim 1's outboard strip (upper outer leg at the left) clockwise gives upper leg up, core inside, SOL outside; its
inboard strip the same anticlockwise; a half domain's strip runs inner target, midplane, outer target, left to right.
The progress posters' strips (`movie_density.py`) keep the old direction. Only the drawing changed: the layout data,
the space-time panel's data and the probe's cells are as before.

**Blocks drawn.** A choice of blocks draws only those (the movie's plane and strips, the strips re-packed around the
chosen segments; the static figure, the exported movie, the cells table), on a colour scale over their cells; 'whole
domain' zooms to them. The midplane profiles show only the chosen blocks' rows, the Targets tab only the targets whose
leg is chosen, and the time traces add the chosen blocks' volume mean (dash-dot). None chosen = every block.

**Any run, any layout.** `--path DIR` opens any run directory (or a directory of runs) as one more simulation. The
layout is derived for more than NSTX-U's double null (`gkview.derive_layout`: the same rules where they applied
before, and the five NSTX-U layouts come out identical). A block with both psi faces physical (a single-block run), or
neither, is core when its theta chain closes (a ring, or mirror to mirror) and SOL otherwise. A SOL chain may cross
Z = 0 more than once (a single null's main SOL block crosses both midplanes; each target takes the crossing nearest
it). Allowed too: no block under a SOL block; a chain that never reaches Z = 0 (its target's surfaces then sit along
the plate, and it has no λq); blocks in no SOL chain (each chain gets a one-band strip). Where the files' psi index
grows inward (STEP's and TCV's grids), every block's rows are read inside out (`Run.rows`); the probe and the cells
table keep the files' own i. A **single-block run** (the gyrokinetic app's own files, `<name>-<what>.gkyl`, no
topology) opens with every face physical (theta periodic where the block closes on itself). Checked on the
device-agnostic driver's geometry of STEP (full double null, psi inward), ASDEX and TCV (lower single null), where the
old derivation refused STEP and TCV and named ASDEX's inner target 'outer'; and on Sim 2's block 2 linked under
single-block names (no real single-block run exists here). A 3D configuration space is refused by the reader.

**Overlay.** Another run (any simulation's run, or a directory) drawn over the shown one in long dashes, at its frame
nearest the shown time (none where it has no frame within half a frame spacing: before its start, after its end):
midplane profiles (at its own separatrix distance), target heat flux and its λq lines, λq and peak q_par over time,
time traces, the probe (its cell nearest the probed one, the distance said), the balances (its terms faint), the
diffusivity, and the options that differ (Options tab). The header says which of its frames is drawn, and the λq tiles
carry its values. Its curves follow the movie like the first run's (`gkview.Compared`: its rows at the matched frames).
Its time series are cut to the shown run's time span, so a longer overlay neither stretches the time axes nor sets
their y range. Not overlaid: the movie, the maps, the midplane (radius, time) map (different grids).

**The diffusivity** (`gkview.diffusivity`). Where the run wrote `<species>_anom_diff_0` (the library's D × gˣˣJ: the
driver's `D_profile` at the cells' corner nodes, multiplied by `geo_int_gxxj` with `gkyl_dg_mul_op`), it is divided
back per cell (the inverse weak product of the two p = 1 expansions), and the corner values are the driver's D at those
nodes again. In Sim 2, cells sharing a node agree to 8e-16, every psi row is constant to 1e-15 (the driver's D is a
function of psi only), and the values lie in the range the driver printed (0.487–0.9938 m²/s). Where the run wrote no
anom_diff (a uniform D: the streamlined driver writes it only with `D_profile_csv` or `D_edge`), the D its log printed
(`Physics preset/overrides ... D=`, the csv_init line × D_perp_mult, ...), else its options (`D_perp` × `D_perp_mult`,
`D_const`), else 'not recorded' (the driver's default applied, and the run's files do not say which). Sim 1 prod:
1 m²/s (its options); the 2025 re-run: 0.22 (its log); Sim 1's `ab_edgeD10_coll`: 10 on the core's inner rows, 0.22
elsewhere (its file). A driver setting `anomalous_diffusion.write_diagnostics = true` always would leave its D in
every run's files.

**Options** (`run_options.py`). The driver is the binary the run's command names; its source is found next to it
(`<binary>.c`, or the name without its `_<date>` suffix) with the headers it includes from there. Each
`opt_get_<type>_comma_kv(<opts>, "<key>", <target>)` call names an option and the field it sets. The field's comment in
its struct says what it is: a comment line between members heads a section, as the driver groups them, and a field
without a comment takes the one just above it when that one names it. The defaults come from the struct's designated
initializer where the driver has one (the streamlined driver: 70 options, 60 defaults; the 2025 driver: 42 options,
no initializer). It is the source as it is today: the page says when it is newer than the binary (its defaults may then
not be the run's; the values the run set are its own). The start-up report is the run's first job log, each line
once (every rank prints it), up to 'Gyrokinetic simulation initialized', without module messages, the per-rank block
lines and the library's tagged diagnostics (`TOK_GEO_...`, `GKYL_...`, counted). The streamlined driver files its
velocity-grid options under its 'gyroradius rules' section: that grouping is the source's.

## Labels, pgkyl commands, themes, the output path (2026-11-10, later)
**Label overlaps, audited and fixed.** No browser here, so: the matplotlib maps measured exactly (every drawn text's
box from Agg), the movie's canvases by a port of its text placement (Liberation Sans metrics, Chrome's baselines), the
plotly headers by plotly's placement rules (the legend pushes the top margin; an annotation does not and is clipped or
drawn over). Found: the profiles note 72 px above the plots with a 60 px margin (clipped) and in the legend's band;
the targets' colour note over the legend once four targets and an overlay fill it; the probe's cell line over the
legend, its first line clipped with an overlay; the diffusivity legend 11 px above the plots, over the subplot titles,
its note over 'inboard midplane'; the region end labels (time traces) running into the next column's y title at
~800 px and able to climb into the subplot title; the movie export's title touching the first strip title; zoomed
midplane views with 'Z (m)' off the figure and R ticks run together; the movie's poloidal-plane title past its canvas
with a block choice. Fixed by construction: nothing above the plots but subplot titles (13 px) and the legend,
`LEGEND_GAP` px above them; what a line style means is a legend entry (`legend_key`); the probe's cell line is on the
page above the chart; end labels get room from their measured width at a ~760 px chart and stay inside the curves'
span; the export title has its own band; the R-Z ticks are fewer and shorter; the movie fits its titles. After:
0 problems in all three audits, both looks' sizes, 3 simulations, with and without an overlay.

**pgkyl commands** (`pgkyl_cmd.py`, postgkyl 2's command line). Click a curve in any chart (every curve of the
clicked panel at the frame the chart shows; the overlay's at its matched frame; a λq or peak-q point at its own
frame), a cell, the space-time strip or a profile in the movie (the cell's value, its block's map, its history; the
profile), or a map figure (one window per block): a box under it gives the pgkyl command (copy button), `cd` into the
run and pgkyl (`viewer.json`'s, else the one on PATH) on the files the viewer read. They compute the numbers shown:
each file loaded under a tag saying what it holds, `select -c 0` (the coefficient behind a cell average: pgkyl 2 does
no arithmetic on modal coefficients), the viewer's T, u, Mach, q_par formulas in `evaluate` (its inputs named f0, f1,
... in load order), ÷ 2 and the page's units, changes against the same frame. evaluate takes and replaces the whole
working set: a curve per block (a midplane) evaluates each block's files after `activate`-ing them, then activates
the results for the plot; every frame (the Probe tab) collects each file's frames on its own first. pgkyl draws
blocks on their (psi, theta) grid (the strips), takes the theta cell nearest Z = 0 for a midplane (the viewer
interpolates), and the integrated diagnostics per radian; each command's `#` lines say so. Region means, the n ≤ 0
count, λq and Thomson have none, and say why. 2026-11-10 (pgkyl 1): every kind checked equal to gkview on all three
simulations (to 1e-16, 7e-8 for the float32 cache). 2026-11-12: ported to pgkyl 2.0.0, whose command line is new
(`load -f`, `select`, `evaluate` by position, `save`, `gk_*_balance -m -1`; no `write`, `ev` tags or
`--batch-mode`); see Checks.

**The output path**: the header gives the run's directory in full (and the overlay's), with the file name pattern.

## Measured profiles: Thomson n_e (2026-11-03)
The Midplane profiles tab overlays the shot's Thomson n_e (open circles, electron colour) on the density row, above
a strip of the shot's beam power and Thomson line density with the averaging window shaded. A switch hides it; a
range slider picks the frames (it snaps to Thomson frames in the beam flattop).

**Getting the data.** Perlmutter has no route to PPPL (`flux.pppl.gov` does not resolve here; `portal.pppl.gov:22`
does not answer); the laptop has one. Its ssh connection to Perlmutter forwards Flux's ssh port to a socket on the login
node it lands on (laptop `~/.ssh/config`, in the Perlmutter host entry, then reconnect; working since 2026-11-03):
```
RemoteForward /tmp/<your Perlmutter user name>-flux.sock flux.pppl.gov:22
```
- `./view.sh --flux` asks for the Flux password + Duo once, before the page starts, and keeps that login (an ssh master
  connection, its control socket in a private `/tmp/flux-cm.*` directory) until the viewer stops. Where a shot has no
  Flux copy, the Midplane profiles tab shows **Fetch from Flux**: it runs `fetch_flux.sh <shot>` through that login
  (never a prompt) and re-reads the file. Without `--flux`, or when the login has closed, the tab says how to get one.
- `./fetch_flux.sh 203655 203899` on that login node: the same fetch with its own login (asks for the password + Duo).
- On the laptop instead: `NSTXU_VIEWER_DATA=<data dir> bash <(ssh perlmutter cat <viewer dir>/fetch_flux.sh) 203655
  203899` (`FLUX=`, `PM=`: the laptop's ssh names; the data directory's `flux.env` is read over ssh).

`fetch_flux.sh` runs the sim-init workflow's `tools/extract_slim_h5.py` on Flux in a temporary directory, adding
`WF:\PNB`, and delivers `runs/flux/<shot>.h5` (~10 MB). Flux's login prints a banner and module lines, so the files
come back base64-encoded between marker lines. The data directory's `flux.env` sets `FLUX_HOST` (your Flux
user@host), `FLUX_PY` (a Flux python with h5py; python3 if it is not there) and `FLUX_TOOL` (where that tool is). The socket only exists on the login node the laptop's connection landed on: run `./view.sh --flux` in a
terminal of that connection. Without a fetched file, 203899 falls back to the workflow's slim copy (no beam power,
so no flattop); 203655 (Sim 1, 2025 re-run) has none.

Checked here with a stand-in for ssh that prints Flux's banner on stdout (same tool, the workflow's file as the
archive): through view.sh's login (BatchMode), through the socket, a closed login falling back to the socket, and no
route (exit 4 with the fix); each delivered file equals the source dataset for dataset; `./view.sh --flux` exports the
login to marimo and logs out and removes its directory when stopped; the tab shows the button, "restart" or "start
with --flux" in the three login states. The forward itself was tested by the user (password + Duo, `hostname` =
flux-login2). Not yet run: a real fetch from Flux.

**Definitions** (`thomson.py`, following `nstxu_sim_init_workflow` docs/02 and docs/08):
- MPTS branch TS3, then TS2, then BEST (the first with every node); per-channel fits `FIT_NE` ± `FIT_NE_ERR` on
  `FIT_RADII`, only where `VALID == 1`. Units asserted.
- Beam flattop: the elongation/triangularity scan's rule on `WF:\PNB` (within 0.2 MW of the maximum, steady over 15
  samples each side). Its smallest-|DRSEP| EFIT01 time must be the run's equilibrium time; the note says if not.
- Window, opening default: the later half of the flattop after the run's time, i.e. from the midpoint of (run's
  time, flattop end) to the flattop's last frame. Point = mean over the window's frames; bars = spread over the
  frames (fit error where a channel has one valid frame).
- Placement: each frame's channels measured from the run's own midplane separatrix, moved by EFIT01's separatrix
  shift (RMIDOUT / RMIDIN) from the run's time to that frame. EFIT01 is used only for the shift: at 325 ms it puts
  203899's separatrix +7.1 mm (outboard) and +2.5 mm (inboard) from the run's (h1a) one, a third of the run's
  outboard SOL. 203899 moves 2-3 cm outboard and ~4 cm inboard between 0.3 and 0.9 s, so a plain R overlay would
  be wrong; EFIT01's inboard separatrix also jitters by several mm between slices (averaging frames damps it).
- Thomson does not reach the inboard SOL (203899: innermost channel 67 mm inside the separatrix); there only core
  channels inside the run's span show.

Sanity check, Sim 2 at the frame nearest 325 ms: Thomson / the run's first-frame n_e = 0.90-1.08 on 6 of 7 channels
(0.68 at +12 mm in the far SOL, where the initial profile is the workflow's spline, extrapolated).

## Checks (2026-11-12: headless all pass, 28; the browser part not run)
2026-11-12, after the move into postgkyl's branch and the port to pgkyl 2.0.0 (run from the branch against a data
directory offering the three NSTX-U simulations; `check_gkview.py` with postgkyl 2.0.0 / NumPy 2.5, `check_viewer.py`
with the viewer venv): `check_gkview.py` all pass (unchanged by the move: the caches are found under their old names).
`check_viewer.py`: everything before, plus the kept theme colours == their three PowerPoint files; four pgkyl commands
run and equal the viewer (rtol 1e-12; in fact 2e-16 or bitwise): a cell's T_e at frame 20, the outboard midplane n_i
across core and SOL, the cell's T_e at every frame (31), the electrons' number summed over blocks; and every other kind
(block map, change map, midplane history, plate, dt, field energy, target power, both balances, D map, midplane D) runs
to a figure. By hand: no `viewer.json` and `--path` to a run opens it with every tab; neither gives a note on the page.

### 2026-11-10
Later on 11-10 (`check_viewer.py`, headless): the map figures' 535 texts (3 simulations; whole, zoomed, titled, blocks)
none over another or off the figure; 24 charts with only subplot titles and the legend above the plots, the legend
clear of them; 442 curves (every chart, overlays too) each with its pgkyl command and only known placeholders; two
commands run and equal the viewer (rtol 1e-12); `--theme PPPL --dark false` and `--theme diii-d` open those looks with
every tab built; each theme's page CSS; the side-by-side options tinted by row in either order. The full command and
audit checks (above) were scratchpad scripts. Not run: the page in a browser (no Chrome here; the plotly layout and
the click boxes are unverified in one).

2026-11-10 additions. `check_gkview.py`: the STEP / ASDEX / TCV layouts (psi direction, targets, both midplanes, every
block in a strip); a single-block run (Sim 2's block 2 under single-block names) opens, its fields equal block 2's, its
summary builds; Sim 2's D (node agreement, psi-row constancy, the driver's printed range), the edge-diffusion run's
10 m²/s, the uniform runs' D; the midplane D's rows; `match_frames` and `Compared`; a frame recomputed now equals its
cached row in all 75 entries (CACHE_VERSION stays 3: no cache was rebuilt); Sim 2's options (70 read, its 32 among
them, its job, its start-up report). `check_viewer.py` (headless): blocks with the overlay and every tab (Diffusivity
and Options included); blocks and an overlay that do not fit fall back with notes; `--path` on the single-block run with
Sim 2 over it; a block choice's figure, the D map and the movie's drawn cells; state files keep the new keys. Not run:
`--browser` (the movie's JavaScript: the reflected strips and space-time panel, the drawn cells, clicks), no Chrome here.

### Before (2026-11-02: all pass)
`check_gkview.py` (postgkyl venv):
- Derived strips == `movie_density` for `sim1_prod`, `sim2_prod`, `prod` (hand-written for Sim 1 and the 2025 run,
  `half_layout` for Sim 2); reflecting midplane found exactly when the files say `half_domain = 1`.
- q_par and λq at every poster target, frames 1 and the posters' frames: equal to `poster_progress.heat_flux` to
  3e-14 when both use the run's masses. The poster script hard-codes CODATA m_e and 2.014 amu; the runs used gkeyll's
  m_e (−1.7e-7 relative) and, in the 2025 re-run, m_i = 2.014 m_p (+0.73 %): its poster λq at 8 ms is 2.6351 mm,
  2.6333 mm with the run's own mass.
- Sim 2 outboard midplane n_e == the poster's profile (301 rows); T == the direct formula (4e-16); dt and integrated
  moments == postgkyl; the cached fields == the frames read directly.
- Frame ranges == `postgkyl.io.naming.select_frames`'s documented examples; malformed ones refused.
- Cell volumes > 0 and Σ n V == the integrated M0 within 0.17 % (Sim 1) / 0.12 % (Sim 2), every frame.
- Sim 1's particle balance closes; every target plate face is a physical θ face.

`check_viewer.py --browser <chrome>` (marimo venv; 25 checks, + 3 Thomson checks 11-03: reader == the workflow's
203899 channel export to 2e-16; beam flattop == the scan's own loop on 200 random traces, 154 with a flattop; Sim 2's
profiles carry Thomson points on both sides. 11-03 run without --browser: 14/14 pass):
- headless: every simulation opens on its default run with every tab built; a launch view opens where it says; one
  that does not fit falls back with a note per key; state files round-trip, a foreign file is refused; figures save
  as png / pdf / svg (the svg naming Arial first); a 10-frame movie is H.264 with 10 frames;
- in headless Chrome driving `./view.sh --sim 'Sim 1' --frame 20 --qty T_elc`: the page opens there and its address
  holds the view; dark by default (#1B262B) and the switch goes light and back; the movie at 30 frames/s draws 60
  times a second (90 frames in 3 s) with the notebook's header frame untouched while the open tab's two charts follow
  it (0.1 ms a frame); on pause the notebook takes the movie's frame and the movie stays there; the page draws a
  frame as plotly does (the notebook's redraw at the paused frame and at another: every curve vertex, marker and
  line within 0.05 px; measured 0.0); a click on the movie's strips probes that cell; Save figure writes its file;
  a reload comes back to the frame, probe and tab;
- a LivePlot alone in a page: a λq line hides at a frame where λq is undefined (no run has one yet), in the SVG path
  and in plotly's own drawing; a frame past the y axes (Sim 2's heat burst) says so under the chart, the frame shown
  fits; the plotly fallback (forced with spline curves) follows the frames while the page keeps ≥ 30 frames/s
  (measured 47-49).
- Not in the suite (run by hand 11-02, scratchpad): the same page-against-plotly comparison on every live tab, for
  Sim 1 and Sim 2, at the launch frame, the last frame and frame 0: within 0.005 px everywhere.

## Cost
One frame ~0.05 s (all blocks, 15 files each). The movie and the time traces need every frame once: Sim 1 357
frames 31 s, the 2025 run 401 frames 25 s, Sim 2 31 frames 5 s; later loads read only new or rewritten frames
(`reload frames` in the sidebar for a live run). The cache keeps every cell of every quantity (float32): 11.6 MB
for Sim 1, 8.4 MB for Sim 2, 7.0 MB for the 2025 run (`runs/cache`). Bump `gkview.CACHE_VERSION` when a summary
definition changes. A movie export renders ~0.3 s a frame (Sim 1's 357 frames: ~2 min).
