# Gkeyll run viewer (2026-11-02; until 2026-11-14 the NSTX-U run viewer)

A marimo notebook for looking through Gkeyll gyrokinetic runs frame by frame (multiblock tokamak runs of any
device, and single-block ones), in a slide theme (the shown simulation's device's: NSTX-U's DPP 2025 poster, DIII-D's
talks, TCV's, ASDEX Upgrade's and STEP's brands; light by default, dark on a switch): open it on a shot, a run and a frame, or play the run as a movie, probe any cell over
time, read the particle and energy budgets and the power to each target, and save the view, the figure or a movie.
Nothing here writes into a run directory; the outputs go to the data directory's `runs/` (Setup): the per-frame
cache `runs/cache/`, saved views `runs/states/`, exported figures and movies `runs/exports/`, Flux copies
`runs/flux/`, the server's log; the page's stylesheet is `runs/viewer.css` next to the notebook.

## Setup
This directory lives on postgkyl's branch `marimo-gui-jjz73`, and the viewer runs from it. The viewer does not
import postgkyl (it reads the `.gkyl` files itself, `gkyl_read.py`); the pgkyl commands it shows are postgkyl 2's.
```sh
cp viewer.example.json viewer.json     # then name your simulations in it (and flux.env from flux.env.example)
./view.sh                              # the first run makes .venv from requirements.lock.txt (Python 3.12+; ~30 s)
```
Without `viewer.json` (a fresh clone: `./view.sh` alone) the page opens anyway, every panel empty (A blank start,
below).
`.venv` can also be a link to an existing venv with these packages, or `GK_VIEWER_VENV` names one.
**The data directory** (`GK_VIEWER_DATA`; default: this directory, where git ignores `viewer.json`, `flux.env`,
`runs/` and `.venv`) holds what the viewer reads and writes:
- `viewer.json` (start from `viewer.example.json`): every path in it relative to `base` (`devices` too), itself
  relative to the data directory (or absolute). Read in one place, `gkview.load_config` (`gkview.reload()` reads it
  again: for a form that writes it); an entry that does not parse is left out with a note on the page.
  - `base`: what the simulation directories (and the cache's file names) are relative to;
  - `sims`: `[label, simulation directory, default run, one-line description]` each, optionally followed by an object
    of the simulation's own settings, `{"device": <name>, "geqdsk": <file>}` (both optional): `device` the device
    its runs simulate (else the top-level `device`); `geqdsk` the equilibrium file its runs read, used only where a
    run's own files (options_used.txt, command.txt) name none: it gives the run's shot and time and its wall as the
    run's own does (the header says which was used, and when the two disagree: the run's own wins). A simulation's
    runs are the run directories (a topology file, or a single-block run's files, and a frame) in its `runs/`, else
    in it;
  - `device` (optional): the device runs simulate unless their simulation names its own, named in a run's title
    before its shot (`NSTX-U #203899 @ 325 ms`; none: `#203899 @ 325 ms`);
  - `devices` (optional): per device (by the name `device` gives), `{"device_image": <as below>, "wall": <a
    geqdsk>}`: its picture (it wins over the top-level `device_image`), and a geqdsk whose limiter is drawn as its
    wall where a run's own geqdsk gives none (no geqdsk, or a limiter of fewer than 3 points: STEP's `step.geqdsk`
    has 1); the wall only, never the shot and time (`gkview.wall`). Paths relative to `base` like every other
    (`../pictures/...` reaches beside it);
  - `pgkyl` (optional): the pgkyl executable the commands name (else `pgkyl` from PATH);
  - `thomson_fallback` (optional): a directory of slim HDF5 shot files (the sim-init workflow's `data/h5_slim`);
  - `theme_files` (optional): the themes' source files, which `check_viewer.py` compares the kept colours and fonts
    with: PowerPoint files (NSTX-U, PPPL, DIII-D), or for a theme without a public template (TCV, ASDEX Upgrade,
    STEP) the JSON of its brand colours, each with its source (Device themes, below);
  - `device_image` (optional): a picture of the default device's poloidal cross-section (the top-level `device`'s:
    runs of another device get only their own, from `devices`), drawn behind the poloidal plane when the sidebar's
    **include device background** is on: `{"file": <relative to base>, "extent": [R_left, R_right, Z_bottom,
    Z_top], "opacity": 0.0-1.0}`; extent = where the picture's edges lie in metres (Device background, below);
  - `remote` (optional): `{"host": <ssh host name>, "base": <the path there of base>, "data": <the viewer's data
    directory there>}`, the machine that holds the runs, for a mirror here (Run it anywhere, below).
- `flux.env` (from `flux.env.example`): Fetch from Flux's login and paths (`fetch_flux.sh`).
- `runs/`: everything the viewer writes.

Another data directory (`GK_VIEWER_DATA=DIR ./view.sh`) keeps its own settings and outputs; a `view.sh` there can
set it and run this one. On Perlmutter (post-generals-2) this directory is the data directory since 2026-11-14:
`viewer.json` with `base` = `research/nstxu`, `flux.env`, `runs/` (the caches and the 203655 Flux copy, copied from
`research/nstxu/viewer_20261102/runs`), `.venv` a link to `tools/marimo/.venv`; that research directory's `view.sh`
forwards here.

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
(value / change / step), `log`, `zoom`, `x`, `dark` (left out: light), `theme` (left out: the shown simulation's
device's), `tab`, `frames` (the movie's frames, `pgkyl load --frame`
syntax), `probe` (block,i,j), `blocks` (the blocks drawn: `1-3,10`), `cmp` (a run drawn over this one: `<simulation>/<run>` or a
directory), `path` (a run directory to open), `background`, `opacity` (the device picture), `mirror` (a half domain
shown whole), `quantity`, `grid`, `separatrix`, `wall`, `outlines` (what the poloidal plane draws), `electrons`, `ions`,
`simulation`, `experiment` (the curves on the line charts). A view opens from
the defaults, then a saved view, then the command line, then the page's address, each overriding the one before; the
page writes the current view into its address, so a reload or a bookmark comes back to it (the address carries the
access token too: do not share it while the server runs). A
value that no longer fits (a run removed, a frame past the last) falls back to the default with a note at the top.
Save & export shows the `./view.sh` line for the current view. With no key the page opens light, on the default
device's first simulation (viewer.json's `device`; here Sim 1, NSTX-U), in that device's theme (2026-11-17).

In a VS Code terminal the page opens on your own machine (`$BROWSER` is VS Code's opener; it forwards the port), in
Chrome: a workspace's `.vscode/settings.json` can set `"workbench.externalBrowser": "chrome"` (it applies to every
external link opened from this window; delete the line to go back to the system default browser). If VS Code ignores
it as a workspace setting, put the same line in your user settings. Outside VS Code, view.sh prints the `ssh -L`
line to run on your machine. A terminal that VS Code revived after a window reload keeps the old window's socket
(`$VSCODE_IPC_HOOK_CLI`, gone): view.sh then opens the page through another live VS Code socket on this login node, or
prints the URL. The access token stays on: anyone on a shared login node can reach 127.0.0.1. Ctrl-C
stops the server. Python: the venv of Setup (marimo 0.25.2, plotly 7.1.0, anywidget 0.11.0, imageio-ffmpeg 0.6.0,
h5py 3.16.0, numpy 1.26.4, matplotlib 3.11.0; full list in `requirements.lock.txt`).

## A blank start (2026-11-16)
With nothing to show (no `viewer.json`, `"sims": []`, or simulations whose directories are missing or hold no run
that can be read) the page opens whole: the sidebar (simulation and run lists empty and idle, the frame slider idle,
every other control there), the header (its tiles '—'), the movie (its canvases, titles, axis titles and colour bar,
its controls idle, "no frames (no run)") and every tab, each chart with its panels, axis titles and legend and no data
in it; the tabs that describe a run's files (Options, Run) say what they show. A note at the top says that there is
no data yet and how to add some (`viewer.example.json`, or `./view.sh --path DIR`), and why each simulation named
was left out. For showing the page's layout before there are runs (or before `viewer.json` is filled in).

The page shows `gkview.empty_run()`: a `Run` with no directory, no block and no frame (electrons and ions, their
masses unknown; an outboard and an inboard midplane holding no block), threaded through the notebook as any run is,
so the cells and charts are the same code with nothing in them: no fake values (the time cursor, the separatrix line
in R and the axis values of the poloidal plane are left out rather than placed anywhere). Its summary has every
entry with no row; the shown frame is None.

## Device background (2026-11-14; per device 2026-11-16)
`viewer.json` puts a picture of a device's poloidal cross-section (a CAD drawing, a photo, any PNG or JPEG) behind
the poloidal plane, the shown run's device's (`Run.device`: its simulation's `device`, else the top-level one): its
`devices` entry's `device_image`, else the top-level `device_image` for the default device: in the movie and in every map figure (the Static figure tab, figure and
movie exports, the diffusivity maps), under the cells, the wall and the separatrix, clipped to the view. The sidebar's
**include device background** turns it on and off (launch key `--background true`; off by default; the switch is
there only when the shown run's device has a picture) and **background opacity** sets how strongly it shows, 0.05 to 1 (launch key
`--opacity 0.6`; it starts at `viewer.json`'s `opacity`): the movie redraws at once, without being made again. `extent` places it: the R of its left and right edges and the Z of its bottom
and top, in metres, so that its vessel lies on the run's wall (the geqdsk's limiter, drawn over it). The picture is
read once, at most 2000 px on its longest side (`viewer_plots.BACKGROUND_PX`), and sent to the page once
(`movie_widget.BackgroundLoader`). A picture that is not freely licensed belongs in the data directory, not in this
repo. NSTX-U: PPPL's section drawing `NSTX-SECTION-PLAN-VIEW-NO-PLASMA_020624.pdf`, in the "NSTX-U Drawings" folder
of the NSTX-U site's Presentations page (shared by link), rendered, cropped to R >= 0 and fitted to the limiter
(research/nstxu/viewer_20261102 on Perlmutter: the picture and how it was made).

## The whole machine (2026-11-17)
The poloidal view **whole machine** (launch key `--zoom 'whole machine'`) frames the whole device: the domain, the whole
wall (the geqdsk's limiter, which 'whole domain' cuts at the grid's height) and the device picture's `extent`, never
R < 0 (`viewer_plots.machine_points`, `zoom_boxes`). It follows 'whole domain' in the list wherever it shows more than
that view: every simulation here (NSTX-U: the section drawing whole, Z +-2.9 m; ASDEX Upgrade, TCV and STEP: their
pictures and walls; DIII-D: its wall). The movie, the map figures and the exports take it like every view. Checked
(`check_viewer.py` part_machine): on every simulation (a half domain shown whole too) the view frames the whole wall
and picture, and where it is not offered the whole domain already does; with neither it is not offered; the map's
axes hold them (ASDEX Upgrade, Sim 2 shown whole). The layout audit (`--browser`) opens it on every simulation.

## A half domain shown whole (2026-11-14)
A run on a half domain (a reflecting midplane: `layout.half`) can be shown whole on the poloidal plane: the
sidebar's **show the full domain (mirror the half domain)** (launch key `--mirror true`; off by default; there only for
a half domain) draws each cell again at its mirror image across Z = 0 with the same value, since the data are
up-down symmetric (`viewer_plots.z_signs`). Block outlines, the separatrix and the n <= 0 marks are mirrored too; the
wall is the whole vessel already. The whole-domain view then spans both halves, and the other half's divertor is
offered as a view of its own. A click on a mirrored cell probes its original. The unrolled strips, the profiles and
the time traces stay one half: the mirror image holds nothing new.

Checked (`check_viewer.py` part_poloidal): a picture red inside Sim 2's wall and blue outside it, sampled 5 px
either side of every wall segment: under the map (whole domain, lower divertor; a 3 cm shift of the picture fails
it) and, with `--browser`, under the movie (served on a data directory naming the picture, `--background true`);
eight arrow-key steps of the slider take it to 0.6 and the movie, not made again, shows the picture's blue blended
with the surface at 0.6 outside the wall; switched off it leaves no trace. The half domain shown whole: each cell's
colour at its mirror image, judged on cell insides only (both pixels inside one colour, in the drawn box), on the map
(whole domain and both midplane views: 167 of 167; without the switch 0 of 153) and on the page.

## The lines over the cells (2026-11-16)
The poloidal plane draws three lines over its cells, each on a sidebar switch of its own, on by default: **show the
separatrix** (dashed: the inner edge of each SOL block with an aligned block under it; launch key `--separatrix
false`), **show the wall (limiter)** (the geqdsk's limiter; `--wall false`) and **show the block boundaries** (every
block's outline, so the interfaces between blocks too; `--outlines false`; not `blocks`, which chooses the blocks
drawn). A switch is there only when the run has its line (a run with no separatrix in its layout or no limiter in its
geqdsk has none to show). They act in the movie, which redraws at once without being made again (`Movie.layers`,
read where the page draws the plane), and in every map figure (the Static figure tab, figure and movie exports, the
diffusivity maps: `viewer_plots.Layers`, one value passed to `map_figure`). The outlines and the separatrix follow the
cell edges as curves where the run writes them (The grid and the quantity, below). A half domain shown whole: the mirrored
separatrix and outlines follow their switches. The view keeps them (a saved view, the `./view.sh` line, the page's
address). The unrolled strips keep their lines whatever the switches: there the dashed line is the band edge their y
axis labels 'separatrix' and the vertical lines part the segments named above them, so without them the strips'
labels would point at nothing; the switches are the poloidal plane's, as the device picture and the mirror are.

Checked (`check_viewer.py` part_lines, on Sim 2): the map (matplotlib, at 150 dpi, the whole domain shown whole and
the lower divertor) and, with `--browser`, the movie (the whole domain shown whole over the device picture, the
sidebar's switches clicked through all eight states), judged on pixels: each switch alone, against all three off,
changes pixels along its own line (every 3 px along it, in each half of a mirrored view) and nowhere else; taken away
from all three, the same; all off, the map equals itself with its lines removed; the movie is not made again; the
address carries the switches. The Static figure and Diffusivity tabs (the notebook launched with two switches off) and
the movie export draw with the switches given. Run alone with `--browser`: 5 of 5 pass (share of the points along a
line that changed: 0.85-1.0 on the map, 0.69-1.0 on the page, where the block outlines are 0.4 px over the picture; 0
pixels changed elsewhere). Each wiring fault tried fails it: the separatrix and wall switches swapped in `map_figure`
(both map views: 0.004-0.07 along the lines, 1,858-3,767 pixels changed elsewhere); the mirror image's outlines drawn
whatever their switch (the map shown whole: 0.27 along the mirrored half, 2,997 pixels left with all off); the movie's
separatrix following the wall's switch (0.0 along it, 560 pixels elsewhere); the switches handed to the movie when it
is made (made again at every click: renders 1 to 8).

## The grid and the quantity (2026-11-16)
Two more sidebar switches for the poloidal plane: **show the grid (every cell edge)** (launch key `--grid true`; off by
default) and **show the plotted quantity** (`--quantity false`; on by default). Off, the quantity's cells go, and its x
marks; what stays is the grid, the lines and the device picture, on the same axes and view, so the grid can be seen
alone or over the picture. The colour bar stays where the strips use it (the page's layout and a full-width slide: the
strips keep their cells) and goes with the quantity where it is the plane's alone (a half-width slide). In the movie
both switches redraw it at once (`Movie.layers`); the grid's lines are built once per run (`movie_data`), one path,
stroked under the outlines, thin and in the theme's secondary ink at 45 %.

A Gkeyll cell edge is a curve: from a corner through the two Gauss-Legendre nodes of the edge (t = (1 ∓ 1/√3)/2) to the
next corner. `gkview.cell_edges` reads the nodes from each block's `geo_surf0_mc2p_nodal_fd` (constant-psi edges,
[npsi + 1, 2 ntheta]) and `geo_surf1_mc2p_nodal_fd` (constant-theta edges, [2 npsi, ntheta + 1]) files, the first three
of their 39 components (the rest are a finite-difference stencil), R = hypot(X, Y) for a mapc2p geometry as for the
corners, and draws each edge as the cubic through its corners and its two nodes. Every line along cell edges (grid,
outlines, separatrix) and every cell's fill, in the movie and the maps, follows these curves: on a coarse grid
(ASDEX Upgrade's 8 cells around, DIII-D's 12) a cell filled as its corners' quadrilateral fell well inside the curved
grid. Before trusting the files it checks that every Gauss node lies by its edge, within one chord length of the
chord's middle (a curved edge bends away by up to 0.88 of its chord, at STEP's X point); where a block lacks the files
or fails that, every edge of the run is drawn as its chord and the plane says so ("cell edges drawn as chords: ...").
Sim 1's and Sim 2's surface node files hold zeros (their builds did not fill them): chords there; the 2025 re-run and
the four device runs have them.

Checked (`check_viewer.py` part_grid): on ASDEX Upgrade's device run (by path, curved) and Sim 2 (shown whole,
chords), the map at 150 dpi: the grid alone, against the bare plane, changes pixels along every cell edge (each half
of a mirrored view) and nowhere else, and so does taking it away over the quantity; the quantity changes nearly every
cell's inside and nothing outside the cells; with both off and no line the plane is its bare surface; the grid passes
through the Gauss nodes (1308 of 1308) and not along the chords where they bend 4 px or more away (0 of 50 such
points); a half-width slide has its colour bar with the quantity only; Sim 2's plane carries the chords note. With
`--browser`, the movie of TCV's and DIII-D's runs at their first frame, the two switches clicked through their four
states: the same pixel tests, the movie never made again. Run alone with `--browser`: 5 of 5 pass. Wiring faults
tried fail it: the map's grid drawn along the chords (ASDEX: 40 of 48 chord points drawn); the movie filling its
cells whatever the quantity's switch (TCV and DIII-D: 0 of the cell insides changed).

## The curves shown (2026-11-17; a dropdown since 2026-11-19)
The sidebar's **curves on the line charts** is one dropdown of four choices, all chosen by default: **Electrons**,
**Ions**, **Simulation** and **Experiment** (until 2026-11-19 four switches; the launch keys and the view are the same:
`--electrons false`, `--ions false`, `--simulation false`, `--experiment false`; the view keeps them: a saved view, the
`./view.sh` line, the page's address). Below, a "switch" is one of these choices. A curve is drawn when every switch it
answers to is on (`viewer_plots.Curves`; each trace's tag, `curve_tag`; `shown_curves`):
- **Simulation**: every simulated curve, the shown run's and the overlay's. **Experiment**: every measured or analysis
  profile, Thomson n_e and T_e, CHERS T_i, TRANSP n_D and T_i (its predicted part dotted).
- **Electrons** / **Ions**: the curves of one species, told by the sign of its charge in the run's files (the overlay's
  species by the overlay's), never by its name: each species' midplane profiles, cells with n ≤ 0 and particle
  content, the time traces' region changes (the electrons' n and T), its probe history and region mean, its particle
  budget and conservation, its diffusivity (where the species' D differ). A measured profile is its species': Thomson's
  the electrons', CHERS's and TRANSP's the ions'. The ion Mach number is the ions'.
- **Neither species switch**, Simulation alone: a neutral species (charge 0, neither electrons nor ions; no run here
  has one), a field (φ, the field energy) and a sum over the species (q_par, λq and the peak heat
  flux at the targets, q_perp, the energy budget, the target power, a D shared by every species), and the time step.

Every line chart follows them, on the page and in its downloads (the toolbar's slide buttons and plotly's own: the
page's curves): Midplane profiles, Targets, Time traces, Probe, both charts of Balance & power, Diffusivity, the
overlay's curves in each. A label goes with its curves (the time traces' region names); a legend entry whose curve
goes moves to the next curve of its group. A panel left with no curve says which choices emptied it ('ions not
chosen'); a chart left with none is a note in its place ('Nothing to draw here: simulation not chosen among the
sidebar curves.'). Unchanged: the maps, each showing one chosen quantity (the movie's poloidal plane, strips and midplane
profiles; the midplane over time; the diffusivity map; the Static figure and Save figure), the shot's timeline (beam
power and Thomson line density, where the averaging window is chosen), and the notes under the charts.

Checked (`check_viewer.py` part_species): the 21 line charts of the three NSTX-U simulations, each with another drawn
over it and its shot's measured profiles (492 traces): all on, each is the chart drawn without the switches; each
switch alone off takes exactly the curves the chart itself says are its own, read from what the page shows of them
(legend names and groups, hover text, panel titles; not the tags): electrons 170, ions 158, simulation 464,
experiment 22; the rest unchanged, each legend group left with its entry, the region names gone with the electrons'
curves, a note in each panel it empties and in place of each chart it empties (Simulation: targets, time traces,
probe, both budgets, diffusivity); the midplane over time and the shot's timeline the same whatever the switches.
Sim 2's charges swapped, the two species switches swap (71 curves); its electrons given a charge of 0, their switch
takes nothing. The notebook launched with Ions and Experiment off keeps them in its view (the `./view.sh` line, a
state file) and hands the page 9 charts holding neither, their slide downloads too; launched with Simulation off, the
six simulated charts of Sim 2's tabs are the note. With `--browser`, Sim 2's page: the four switches on; Ions clicked
off, the profiles chart drawn again with 22 of its 44 traces (none an ion's; its slide figure the same), the address
saying `ions=false`; reloaded, still off. By hand (2026-11-17): the 60 charts of the three simulations (with and
without an overlay, both x axes) built by this code equal those of a4005d95, before the switches, but for the tags.

## gk-viewer --path DIR, and the directory field (2026-11-19)
Like postgkyl's `pgkyl-gui --path` (the marimo_gui branch): `gk-viewer --path DIR` opens a directory from anywhere.
`gk-viewer` is a link to `view.sh`, which finds its own directory through the link (`readlink -f`), so its
environment, `viewer.json` and `runs/` are the checkout's: `ln -s "$PWD/view.sh" ~/.local/bin/gk-viewer` (done here,
2026-11-19). Every `view.sh` key and option works through it (`gk-viewer --path DIR --flux`, `gk-viewer --help`). DIR
may be a run directory (its siblings offered beside it), a directory of runs (its last opened), or a simulation's
directory with `runs/` in it (`gkview.path_simulation`); it opens as a simulation of its own, first in the shot list,
named after the simulation (`sim2_dense_rho09 (--path)`, past a `runs/` directory); `viewer.json`'s simulations stay
offered. The sidebar's **directory (--path)** field, under the shot list, holds it: another directory typed there
(Enter, or leaving the field) replaces it and opens it; emptied, there is none. A path that is none of these says why
beside the field and leaves the page as it is, unless the page was on the directory before's simulation: that one
goes (as when the field is emptied), and the default simulation opens. The view keeps the path (the address, a saved view, the
`./view.sh` line). Checked: part_path (the link's help, the three kinds of directory and two wrong paths, the notebook
launched with one; --browser: a directory typed into the field opens it, the address saying so).

## What the poloidal plane draws: one dropdown (2026-11-19)
Six sidebar switches of what the poloidal plane draws are one dropdown, **drawn on the poloidal plane**, its choices
those of the switches, by the same view keys and launch keys: **the plotted quantity** (`quantity`, on by default; was
"show the plotted quantity"), **the grid (every cell edge)** (`grid`, off), **the separatrix** (`separatrix`, on),
**the wall (limiter)** (`wall`, on), **the block boundaries** (`outlines`, on) and **the device background**
(`background`, off; was "include device background"). A choice is offered only where the run has it, as each switch
was: a separatrix, a wall, a picture of its device; the view then keeps None for it. **show the full domain (mirror
the half domain)** stays a switch: showing a half domain whole remakes the movie, and marimo re-runs every cell that
reads the dropdown at each choice (in the dropdown, every choice remade the movie: the browser checks caught it). The
background opacity slider stays a slider; log scale and dark mode stay switches. Checked: the browser parts that
clicked the switches (lines, grid, poloidal) take the dropdown's choices on and off (`CDP.choose`, also the curves
dropdown's), each redrawing the movie without making it again.

## Flux coordinates and the sheath entrance (2026-11-19; after Churchill et al., NME 12 (2017) 978)
`gkview.flux_coords`: **psi_N** of every node and cell row, (psi - simag) / (sibry - simag) with the geqdsk's (EFIT's)
normalization, psi from the run's position map (`geo_corn_mc2nu_pos`, at the cells' corners); the separatrix row of
each SOL block sits at 0.99982 (Sim 1, the 2025 re-run), 1.00001 (Sim 2), 1.0001 (STEP), 0.9999 (TCV) and 1.0101
(ASDEX: the regression test's separatrix is not its geqdsk's boundary). **L_theta**, the poloidal length along each
flux surface (in R, Z) through its chain of blocks, from its target end (an outer target before an inner, a lower
before an upper, as the paper's runs from the outer divertor), along the cell-centre line of each cell row (the mean
of its node rows' edges: curves, or chords where the run lacks the surface nodes), with landmarks: the X-point, the
outboard and inboard midplanes (Z = 0, by R against the axis), the top (inside a SOL chain only). Sim 1's outboard
SOL runs 3.84-4.00 m from the lower to the upper outer target (X-points at 0.50 and 3.40 m, midplane 1.95 m); ASDEX's
over the top 4.14-4.66 m. The Cells table has both (columns psi_N, L_theta (m), L_theta from) and the probe note
gives them. No geqdsk (the DIII-D regression run): no psi_N, and says why.

`gkview.sheath_profiles` and the Targets tab's **The sheath entrance** (the paper's Figs. 5 and 6): at every target,
against psi_N, the electrons' sheath drop eΔΦ/T<sub>e</sub> the boundary condition applied (its cutoff, m<sub>e</sub>
v<sub>cut</sub><sup>2</sup>/2 from the `<species>_bc_z*_vcutsq` files: equal to φ at the plate face to 1e-15, the
wall at 0), with the textbook floating sheath, −½ ln[2π (m<sub>e</sub>/m<sub>i</sub>)(1 + T<sub>i</sub>/T<sub>e</sub>)],
dotted; and the ions' parallel Mach number toward the plate, with the Bohm criterion's 1. The plate-face average of
each moment (its p = 1 expansion at the face). At the latest frame at or before the one shown with the sheath files
(they come with the phase-space output: Sim 1's every 20th frame); the E×B term of Bohm–Chodura is not included. Found
(2026-11-19): Sim 1's drop 0.13-2.1 T<sub>e</sub> against the textbook 2.6-3.1, its ions 0.64-1.49 toward the plates;
Sim 2's outer plate 1.4 rising to 2.6 mid-SOL then falling, its ions supersonic (1.1-2.1). Checked: part_fluxsheath.

## Faster redraws and loads (2026-11-19)
Measured on Sim 1 (a quiet login node): a sidebar frame change on Time traces took ~4 s of Python, 2.6 s of it the
half-width slide specs (one per subplot); opening a run, ~5 s listing directories (9 calls to `runs_in`, Sim 1's run
directory holding ~250,000 files, globbed 3 times per call). Now: `viewer_plots.panel_figure` builds a panel from the
chart's layout JSON and its traces' (the same figures as before, 276 panels of 8 simulations compared), the chart's
template merged into plotly's default once per template (`_merged_template`: that merge was 0.23 s of a panel's
0.25 s): 61 s -> 3.6 s over those 276 panels. `gkview.dir_index`: one listing per directory, kept while its mtime is
unchanged, for the topology files, the single-block geometry, the frames, the species (M0 files) and the job logs
(`find_stem`, `is_run_dir`, `Run.frames`, `open_run`, `run_logs`). `read_dynvec` keeps each file's last read while its
size and mtime are unchanged (copies handed out); the device picture is encoded once per picture.

## What sets the time step (2026-11-19)
The Time traces tab ends with **what sets the time step**. Above: the step the run took (every step, its `-dt` file,
grey), the two limits the library takes it from, rebuilt at each frame, the **CFL limit** (cfl_frac over the largest
CFL rate in phase space) and the **ω<sub>H</sub> limit** (dashed); the step is the smaller. Each term of the CFL rate
alone is a coloured line: parallel streaming, E×B drift, magnetic drifts (curvature and ∇B, and their energy
exchange), E∥ acceleration, mirror force, collisions (BGK), diffusion. Below: each term's share of the CFL rate in the
phase-space cell where it peaks, the cell that sets the CFL limit. The note under it says, for the sidebar's frame,
which limit binds and by how much, the cell (species, block and region, R and Z, the v∥ and μ cells) and the shares
there, how well the rebuild reproduces the steps taken, and where each setting came from.

How (`gkview.dt_limits`; the formulas and their library sources are in the comment over that section): per block
and species, the library takes min(cfl_frac / max cflrate, cfl_frac_omegaH / ω<sub>H</sub>), the step the smallest of
all. The rebuild follows its kernels term by term (gk_collisionless_flux surface kernels: 3 / dx max |ẋ| over a face's
Gauss nodes, 5 / Δv∥ max |v̇∥| on the v∥ faces; BGK's cell-average ν; the anomalous diffusion's 2 c₀(D gˣˣJ)(2/dx)²)
from what each frame wrote (φ, n, ν) and the static files (the geometry, the velocity map, D). ω<sub>H</sub> needs the
polarization weight W = Σ n_pol m / B_ref², which no file holds: it is each block's field energy over
½ ∫ gˣˣJ (∂ₓφ)² (the field energy integrates ½ W gˣˣJ (∂ₓφ)²); every block gives the same W to 1e-15. cfl_frac and
cfl_frac_omegaH come from the run's options, else the library's defaults (1.0, 1.7); the drifts from its `drifts`
option, else whichever choice rebuilds its second frame's step. Every frame is rebuilt once and cached in runs/cache
(`<run>__dt__v1.npz`, and the note's settings in `<run>__dt__v1.json`, both copied by `sync.py` with the summary):
the first open of a run reads its frames again (Sim 1, 357 frames: 114 s; Sim 2, 31 frames: 26 s; the 2025 re-run,
401 frames: 61 s), with a progress bar; once they are all cached, opening the run reads neither the frames nor the
geometry (Sim 1: 10 s of geometry before 2026-11-19's fix).

What it finds (2026-11-19):
- **Sim 1** (prod_D1_coll): the rebuild reproduces every step (taken / rebuilt: median 0.9997, 0.9659 to 1.0013 over
  frames 1-356). The CFL limit binds throughout, ω<sub>H</sub>'s is 12× longer. At 356 of 357 frames the electrons'
  **parallel streaming** sets it (89-92 % of the rate), in block 10's outermost v∥ cell (diffusion 5 %, collisions
  2 %); at 20 µs the electrons' E∥ acceleration in block 9 (51 %).
- **Sim 2** (prod_th02q4_coll): median 0.9990, 0.9977 to 1.0183. The CFL limit binds throughout (ω<sub>H</sub>'s
  4.7× longer, with cfl_frac_omegaH 3.4). At 26 of 31 frames the electrons' **mirror force** sets it (65-98 %), in
  block 5 (private flux) at the largest μ and the fine v∥ cells near zero; at 4 frames diffusion in block 6, nearly
  tied (6.71e-10 s against 6.5-6.7e-10 s); at **90 µs, inside the heat burst, the E∥ acceleration (87 %) in block 5
  halves the CFL limit** (3.1e-10 s). The burst's deepest steps (2e-10 s) fall between frames: the rebuild sees only
  the frames, the grey line every step.
- **Not reproduced** (the chart and the note say so, and no limiter is named as fact): the 2025 re-run (median 0.57,
  0.39 to 0.59; library mm-59e6654d8-gpufix, whose time-step code is the same; cause not found: not the surface
  geometry, not ω<sub>H</sub> added to the rate); the devices' regression runs (ASDEX 0.20, DIII-D 0.48, STEP 0.69,
  TCV 1.10 and 0.58): their inputs use LBO collisions, whose ν no file holds, a diffusion with no D on record and,
  DIII-D, cfl_frac_omegaH = 1.0 set inside the input file.

Limits: the limits exist at frames only; overlays are not drawn; a neutral species' own limit and LBO collisions are
not rebuilt; ω<sub>H</sub> needs the field energy at a frame's time. Checked (`check_viewer.py` part_dtlimits): Sim 1
and Sim 2 reproduced (each frame after the first within 5 %), their leading terms, ω<sub>H</sub> binding at none, one
W per run, a frame rebuilt again equal to its cached row, the 2025 re-run flagged, the empty run's reason.

## Other devices' runs (2026-11-16)
The four regression runs of `research/tokamak/regr_devices_20261116` (ASDEX Upgrade, STEP, TCV, DIII-D) open with
every tab. What changed, each derived from the run's own files:
- **A mapc2p geometry's nodes** (DIII-D's inner-wall-limited test): `geo_corn_nodes` holds Cartesian X, Y, Z when the
  file's `geometry_type` is Gkeyll's `GKYL_GEOMETRY_MAPC2P` (3), not R, Z, phi (the tokamak geometry's): R =
  hypot(X, Y) (`gkview.physical_rz`), for the surface nodes too.
- **A species' source is not a species**: Gkeyll writes a source's moments as `<species>_source_<moment>_<frame>`
  (`gk_species_source.c`); a name `X_source` beside a species `X` is that source (DIII-D's sources wrote M0 at frame
  0 only, and counted as species broke every later frame).
- **The psi direction, block by block**: a block whose topology entry names, across a psi face, a block whose psi
  faces lie against its other psi face (nearer it, by the median distance of the face's nodes) runs against its
  topology: it is read reversed (`gkview.contrary_blocks`, `Run.reversed(b)`), so that its rows run as the topology
  says, before the run's own direction is decided as before. TCV x21's blocks 0-4 are such blocks (their topology
  joins faces 0.1 m apart: a Gkeyll-side problem, not fixed here; the header and the Run tab say so); in every other
  run here the named face lies 1e-10 cells from its partner and the other 1-300 cells away. TCV's midplane profiles
  now run outward on both sides, its separatrix at R 1.0971 / 0.6826 m.
- **A limited single block** (DIII-D): its rows split where the sheath begins, read from the range its sheath boundary
  condition's own files cover (`<species>_bc_zlower_vcutsq`: rows 5-7 of 8); its rows below close on themselves
  (core), its rows from there (`Layout.lcfs`) end on the sheath at its theta ends: two targets (lower and upper inner,
  the plate the open rows only), the outboard and inboard midplane stacks through core and SOL with the LCFS as their
  separatrix (the inboard one across the block's theta seam), one strip with core under the LCFS and SOL over it,
  regions by row (core and SOL means, the cell table, the probe), the separatrix line on its LCFS row. Gkeyll's
  parallel boundary flux of such a run covers the SOL rows only, so the targets' power is read as for any plate.
Checked, by hand (targeted, 2026-11-16): every tab builds headless on the four runs and on Sim 2; ASDEX Upgrade's and
STEP's layouts, profiles, target fluxes and volumes are identical to the snapshot's, and Sim 1's, Sim 2's and the 2025
re-run's; the DIII-D plate's pgkyl command gives the viewer's q_par (25.94, 6.14, 1.61 MW/m²); the four runs' movies
drawn in Chrome. Left: TCV's topology; a run's geqdsk when its files name only a basename (viewer.json's `geqdsk`).

## Device themes, light by default (2026-11-17)
The theme follows the shown simulation's device. A device's theme is the theme of the same name (NSTX-U, TCV, ASDEX
Upgrade, DIII-D, STEP; `viewer_plots.device_theme`, names compared as written). A device without one opens in the
default device's (viewer.json's `device`), and with none there either in NSTX-U's. Picking another simulation in the
sidebar re-themes the page. A theme asked for wins: `--theme`, the page's address, a saved view, or a click on the
sidebar's radio, which then holds for every simulation of the session. Only such a theme goes into the view (the
address, a saved view, the `./view.sh` line), so a reload of a device-themed page follows the device again.

TCV, ASDEX Upgrade and STEP have no public PowerPoint template. EPFL's and the Max Planck Society's are behind their
intranets' logins; STEP's is not linked publicly, and its communications page is password-protected. Each theme is
built from the brand guidelines and the website's stylesheet, in a PowerPoint scheme's twelve slots, and kept in
`viewer_plots.SCHEMES` like the others. The research study `tokamak/viewer_themes_20261117` holds the sources, their
terms of use, the files' SHA-256, and one JSON file per theme (each colour with its document and page, or stylesheet
and selector). `viewer.json`'s `theme_files` names the JSON files, and `check_viewer.py` compares the kept colours and
fonts with them. No logo or picture of the institutions is used. Roboto and Merriweather (ASDEX Upgrade) are free and
tracked in `fonts/`. Merriweather lacks Greek, so Viewer Symbols now carries the Greek letters too (fonts/README.md).

**Launch defaults.** A fresh launch opens light, on the default device's first simulation (Sim 1 here), in its theme,
whatever the session before ended on. Nothing persists between launches but what is given. The view comes from the
defaults, then a saved view (`--state`), the command line and the page's address, all read once when the page starts
(`viewer_state.merge`). There is no preferences file, and each launch has a new access token, so an address from an
earlier server does not open the new one. A reload of an open page keeps its address's view (dark, the simulation, a
theme chosen). What changed:
- `viewer_state.DEFAULTS`: dark is False (was True; the dark switch now reads it); theme is '' (was 'NSTX-U'), meaning
  the device's.
- `pyproject.toml`: marimo's first paint is light (was dark).
- With no `sim` asked, the page opens the default device's first simulation (was the first listed).

## Run it anywhere: a mirror (2026-11-13)
On another machine (a laptop) the viewer runs on a mirror of the runs, so it keeps working while the machine that
holds them is down. Set up there: a checkout of this branch (Setup), and its `viewer.json`: the remote's `sims`, a
`base` for the mirror here (any empty directory), and `"remote"` naming that machine: its ssh name here, the path
there of `base`, and the viewer's data directory there (`"data"`; on Perlmutter: its `gk_viewer/` checkout).
Then
```sh
./view.sh --sync      # log in to the remote once, bring the runs up to date, start the viewer
```
`sync.py` (also by hand: `python sync.py [--sim LABEL] [--run RUN]... [--dry-run]`) copies each simulation's default
run, or the runs named, to the same place under the base here: the run's files but distribution-function and source
frames and restart files (Sim 2's production run: 6,914 files, 0.14 GB of its 2.5 GB), the geqdsk it read and its
shot's `thomson_fallback` file; and from the data directory there to the one here, its summary cache (`runs/cache`)
and its shot's Flux copy (`runs/flux`). The run's files name the remote's paths; the
viewer looks for those under the base here first (`gkview.here`). Symlinks on the remote arrive as what they point
to. The first mirror of all three default runs: 193,489 files, 816 MB. After that rsync copies only what changed. The
sidebar's **sync from <host>** button, next to reload frames, copies the open run again and re-reads its frames: it
uses `--sync`'s login, or else your own ssh, never asking for a password (it fails instead). On the remote itself
(its base is this one) there is nothing to sync and no button. Needs `rsync` and `ssh` here and `python3` there.

Checked (`check_viewer.py` part_sync, with a stand-in ssh that runs the remote's side on Perlmutter): Sim 2's run
mirrored file for file without the skipped kinds, a second sync copies nothing, the viewer opens the mirror with
every tab and its own geqdsk, Thomson file and cache, and offers the button. By hand, the same stand-in:
`--sync`'s login, the first mirror of the three runs through it, every tab of each from the mirror, the login closed
on exit. Not yet run against a real second machine (and not on macOS: its rsync, `--copy-links` and `--files-from`).
Fetch from Flux on a laptop still delivers to Perlmutter's data directory (`fetch_flux.sh`), not the mirror.

## What it shows
- **Sidebar:** shot (searchable: type `203899`), run (every run directory of that simulation with a topology file
  and at least one frame; the production run by default), frame (slider + number box; it moves the movie too),
  movie frames (a frame range for the movie and the exported movie: `:`, `::2`, `-50:`, `0:60`), probe cell
  (block,i,j, or a click on the movie), quantity, value / change since the first frame / change since the previous
  frame (the last shows fronts moving), log scale, poloidal view (whole domain, whole machine, lower/upper divertor,
  outboard/inboard midplane), **drawn on the poloidal plane** (one dropdown since 2026-11-19, below: the plotted
  quantity, the grid, the separatrix, the wall, the block boundaries, the device background, the full domain of a
  half domain), **background opacity** (when `viewer.json` names a picture of the shown run's device: behind the
  poloidal plane, in the movie and every map figure), **blocks drawn** (none chosen:
  every block; see below), **draw another run over it**
  (the overlay, see below), **curves on the line charts** (a dropdown of four choices, Electrons, Ions, Simulation
  and Experiment, all chosen by default; The curves shown, below), profile x axis (distance from the separatrix or R),
  **theme** (NSTX-U, PPPL, DIII-D, TCV, ASDEX Upgrade or STEP: the shown simulation's device's until one is
  clicked; see Look and Device themes) and **dark mode** (off by default since 2026-11-17; on = the theme's dark look,
  page and figures together, no reload).
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
  - *Midplane profiles*: outboard and inboard (n, T, u_par, Mach, φ; shown frame against the first), the shot's
    measured profiles over them (Thomson n_e and T_e, CHERS T_i, TRANSP T_i and n_D; below), and the selected
    quantity at the midplane over (radius, time).
  - *Targets*: q_par against midplane distance, λq and peak q_par over time, every target.
  - *Time traces*: dt, cells with n ≤ 0, particle content, field energy, the volume-mean n_e and T_e change in
    core / SOL / PF; then **what sets the time step** (the step taken, its CFL and ω_H limits and each term's, the
    terms' shares where the CFL rate peaks; below).
  - *Probe*: every quantity at the probed cell over time (both species), with its region's volume mean dotted; q_perp
    dashed on the q_par panel.
  - *Balance & power*: first, **conservation**: each budget's imbalance over each interval between output frames
    (what the sources put in, less what the boundaries took out, less what the content gained, as a share of the
    largest of the three; log axis), each species' particles and the energy, over the whole run, with the typical
    and largest values written under it. Then the budgets' terms (source, boundary losses, rate of change,
    residual) and the power through each target's plate. See Definitions.
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
| `gk_viewer.py` | the marimo notebook (UI only) |
| `gkview.py` | data layer: run discovery (multiblock and single-block), readers (fields, dynvectors), block layout from the topology file (any layout, psi either way), derived quantities, profiles, target heat flux, per-frame cache, frame ranges, cell volumes and region means, particle / energy balance and target power, the diffusivity, the overlay's frame matching |
| `viewer_plots.py` | figures: map (matplotlib), profiles / targets / traces / probe / balance (plotly); the curves shown (each species, simulated, measured); the shared colour scale; figure and movie export; the poster theme's two looks; the page CSS |
| `movie_widget.py` | the browser-side movie (anywidget: the payload from the cache, and the canvas player in JavaScript), and LivePlot: the tab charts that follow it |
| `pgkyl_cmd.py` | the pgkyl command equivalent to each curve, cell and map the page draws (click it) |
| `viewer_state.py` | where the viewer opens: the view keys, state files, the `./view.sh` line, the blocks syntax |
| `run_options.py` | the Options tab: the driver's options read from its source, the run's values, its jobs, its start-up report |
| `thomson.py` | measured profiles of the run's shot from `runs/flux/<shot>.h5`: Thomson (MPTS) n_e and T_e, CHERS T_i, TRANSP T_i and n_D; the beam flattop, the averaging window, the placement |
| `flux_transp.py` | run on Flux by `fetch_flux.sh`: chooses the shot's TRANSP run and adds its T_i, n_D and midplane radii to the shot's file (numpy + h5py; reads TRANSP's netCDF-3 itself) |
| `sync.py` | the mirror: copies runs, caches, geqdsks and Thomson files from viewer.json's `remote` (Run it anywhere) |
| `fonts/make_symbols.py` | builds Viewer Symbols, the symbols the theme fonts lack (fonts/README.md) |
| `fetch_flux.sh` | everything about reaching PPPL Flux: copies a shot's Thomson / EFIT01 / CHERS / beam power / TRANSP run to `runs/flux/`, through the laptop's forwarded socket (or from the laptop); the login `view.sh --flux` keeps |
| `gkyl_read.py` | the `.gkyl` reader (2D fields, topology files, their MessagePack metadata; numpy only) |
| `check_gkview.py` | data checks, against the poster and movie scripts (`sims_20261022/` under `base`; with a postgkyl venv: they need postgkyl and scipy) |
| `check_viewer.py` | viewer checks: headless, exports, pgkyl commands run, and with `--browser <chrome>` the page itself (with the viewer's venv) |
| `view.sh` | launcher: writes `runs/viewer.css` next to the notebook, starts marimo with the view keys, opens the page |
| `requirements.lock.txt` | the viewer venv's packages, pinned |
| `viewer.example.json`, `flux.env.example` | a data directory's settings, to copy |
| `pyproject.toml` | `[tool.marimo.display] theme = "light"` (marimo reads the nearest pyproject.toml: the first paint, before the notebook sets the mode) |

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
theme's page CSS (`viewer_plots.page_css`) into one `<style id="gkv-theme">`, kept the body's last element. Marimo
puts its copy of `runs/viewer.css` (the first paint: the same rules in NSTX-U's theme) in the body, and of equal rules
the later wins. Until 2026-11-17 the style sat in the head and lost: every theme's page accents and dark surfaces were
NSTX-U's, while its fonts, headings and charts were its own. That sheet stays in the page under the chosen theme's, so
a rule of it that the chosen theme leaves empty still applies: until 2026-11-18 its light-mode heading colour
(`color: inherit`, more specific than a title bar's own rule) put dark text on DIII-D's and STEP's navy title bars.
Every heading rule now sets every heading property a theme sets (`viewer_plots.HEADING_RESET`), then the theme's own.

**Six themes** (`--theme`, the sidebar's radio; `viewer_plots.THEME_LOOKS`), each a PowerPoint theme, or a device's
brand where no template is public (Device themes, below), each light and dark, the same roles throughout (targets orange outer / blue-teal inner; electrons and PF the
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
- *TCV*: EPFL's brand guidelines and www.epfl.ch: white page, text #212121, the corporate colours (Rouge #FF0000 =
  the logo's Swiss red, Groseille #B51F1F, Canard #007480, Léman #00A79F, Taupe, Perle), Arial (the guidelines' font
  for office materials; Suisse Int'l is licensed). Targets red / Canard; ions and core Groseille; SOL Léman, private
  flux Canard darker (EPFL has no third hue: its two teals apart in lightness). Dark on Taupe darker 50 % (#211F1D).
  Diverging Canard ↔ red. Page controls red, links Groseille, headings black as on the site.
- *ASDEX Upgrade*: the Max Planck Society's logo guide and the stylesheet of www.mpg.de / www.ipp.mpg.de: white page,
  MPG green #006C66, the site's palette (orange #BF5300, purple #873B8F, warm blue, warm green, deep blue), the light
  grey header #EEEEEE and the slate side menu #455A64; Roboto text and Merriweather titles (the sites' fonts, both
  free and tracked). Targets orange / green; ions and core orange; SOL purple; private flux green. Dark on the site's
  dark slate #263238 darker 25 %. Diverging green ↔ orange. Headings in Merriweather and MPG green, as on the site.
- *STEP*: stepfusion.com's stylesheet: its pale page #F8FAFC, deep-space text #040740, space blues, plasma lavenders,
  fusion pinks, and the amber of its close icon (its one warm hue). Targets pink / space blue; ions and core pink;
  SOL amber; private flux blue. Dark on the site's header navy #040740, the cool hue a lavender (#BCC2FF: the brand's
  pink and its darker lavender meet under colour-vision deficiency). Diverging blue ↔ pink. Its fonts (Neue Haas
  Grotesk, STEPSans) are commercial: Liberation Sans stands in, headings in capitals as on the site, a level-2
  heading the site's navy header bar.
Charts take each theme's text font and their titles its title font (`viewer_plots.theme_fonts`: in matplotlib the
tracked file or stand-in, on the page the CSS stack), the label spacing measured in the font drawn (`text_px`).

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
mass for this driver).

Conservation (`gkview.conservation`): the three terms integrated over each interval between output frames
(cumulative trapezoids at the diagnostics' samples, read at the frame times), and `imbalance` = (in − out −
gained) / the largest of the three. Over an interval, not per sample: in some runs ∂f/∂t's moment swings sample to
sample (the 2025 re-run at 5.68 ms: +3e5, −8e5, +3e5 W/rad between neighbours, ±100 % per sample) while its
integral does not. The first sample (before any step: no flux, no rate yet) is left out. Measured 2026-11-13 (typical
= median over the intervals, signed): particles close in every healthy run (the 2025 re-run's prod 5.5e-9 %, Sim 1's
±0.002 %; Sim 2 1e-5 %, up to 57 % in its heat burst at 85 µs); energy closes in the 2025 re-run's prod (−0.04 %,
at most 6 %), not in Sim 1's production (+9.1 %, steady), Sim 2's (−41 %) or the 2025 coll_c runs (−21 %). Ruled
out as the cause: the faces between blocks (they cancel to 0.04 % of the source in Sim 1, 2 % in Sim 2: the outer
faces alone give the same residuals) and the field-energy term (≤ 1e3 W/rad against residuals of 1e4–1e5: without
it, or with the other sign, the same). Open.

Target power = the Hamiltonian boundary flux through the plate face of the leg and PF blocks,
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

## Measured profiles: Thomson, CHERS, TRANSP (2026-11-03, T_e / T_i / n_D 2026-11-16; NSTX-U shots, through PPPL Flux)
The Midplane profiles tab overlays the shot's measured profiles on the run's, each on the row and species it
compares with, above a strip of the shot's beam power and Thomson line density with the averaging window shaded:
- density row: Thomson n_e (open circles, electron colour) and TRANSP's thermal deuterium density n_D (a wide pale
  line, ion colour);
- temperature row: Thomson T_e (open circles, electron colour), CHERS T_i (open diamonds, ion colour) and TRANSP's
  T_i (wide pale line, ion colour; dotted where TRANSP predicts it, see below).
The sidebar's **Experiment** switch hides them all, and **Electrons** / **Ions** each its species' (The curves shown;
until 2026-11-17 a switch of this tab's own, **measured profiles**); a range slider picks the Thomson frames (it snaps to Thomson
frames in the beam flattop). The note under the chart says, per source, what was averaged (how many samples), how it
is placed, and why a source is missing or empty.

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
- On the laptop instead: `GK_VIEWER_DATA=<data dir> bash <(ssh perlmutter cat <viewer dir>/fetch_flux.sh) 203655
  203899` (`FLUX=`, `PM=`: the laptop's ssh names; the data directory's `flux.env` is read over ssh).

`fetch_flux.sh` runs the sim-init workflow's `tools/extract_slim_h5.py` on Flux in a temporary directory, adding
`WF:\PNB`, then this directory's `flux_transp.py`, which adds the shot's TRANSP run (`/<shot>/TRANSP`), and delivers
`runs/flux/<shot>.h5` (~10 MB, ~1 MB more with TRANSP). Flux's login prints a banner and module lines, so the files
come back base64-encoded between marker lines. The data directory's `flux.env` sets `FLUX_HOST` (your Flux
user@host), `FLUX_PY` (a Flux python with numpy and h5py; python3 if it is not there) and `FLUX_TOOL` (where that
tool is); optional (`flux.env.example`): `FLUX_TRANSP_CSV` (the sim-init workflow's candidate CSV, default: the one
of `FLUX_TOOL`'s workflow, `code/elong_tri_scan_pre_shape_candidates_latest.csv`), `FLUX_TRANSP_TOOL` (this
directory's `flux_transp.py` on Perlmutter; needed only for a fetch run on the laptop), `FLUX_TRANSP_ROOT` (TRANSP's
archive on Flux, default `/p/transparch/result/NSTU`), `FLUX_H5_ROOT` (the MDSplus dumps' directory, instead of the
extraction tool's two: a stand-in Flux). A shot file fetched before TRANSP was added (the 203655 copy here) shows
the Fetch button again. The socket only exists on the login node the laptop's connection landed on: run `./view.sh --flux` in a
terminal of that connection. sshd leaves the socket behind when the laptop disconnects, and a later connection landing
on the same node cannot make it again while the file is there (socat: "Connection refused"): `fetch_flux.sh` tests
that the socket answers, removes a dead one of yours and asks for a new connection. A VS Code window reload keeps the
old ssh connection (no new forward): close the remote connection, or ssh from a laptop terminal. Without a fetched file, 203899 falls back to the workflow's slim copy (no beam power,
so no flattop); 203655 (Sim 1, 2025 re-run) has none.

Checked here with a stand-in for ssh that prints Flux's banner on stdout (same tool, the workflow's file as the
archive): through view.sh's login (BatchMode), through the socket, a closed login falling back to the socket, and no
route (exit 4 with the fix); each delivered file equals the source dataset for dataset; `./view.sh --flux` exports the
login to marimo and logs out and removes its directory when stopped; the tab shows the button, "restart" or "start
with --flux" in the three login states. The forward itself was tested by the user (password + Duo, `hostname` =
flux-login2). Not yet run: a real fetch from Flux. 2026-11-16, TRANSP added: `check_viewer.py`'s `part_measured`
fetches 203655 and 203899 through a stand-in Flux (the same stand-in ssh, a banner on stdout; the dumps and a
TRANSP archive in a temporary directory: a synthetic run made from the real slim 203899S18 CDF with a made-up ND and
RMAJM, and runs that each step of the rule passes over: predictive, no CDF, no ND, a fatal TR.MSG).

**Definitions** (`thomson.py`, following `nstxu_sim_init_workflow` docs/02, docs/03 and docs/08):
- MPTS branch TS3, then TS2, then BEST (the first with every node); per-channel fits `FIT_NE` ± `FIT_NE_ERR` and
  `FIT_TE` ± `FIT_TE_ERR` on `FIT_RADII`, only where `VALID == 1`. Units asserted (cm⁻³, keV).
- CHERS (`\NSTX::TOP.ACTIVESPEC.CHERS.ANALYSIS.{CT1,CT2}`): `TI` ± `DTI` (keV) on `RADIUS` (cm) at `TIME` (s),
  stored (sample, channel), the axes told apart by the dims' values; a point counts where `VALID == 1`, `TI > 0`
  and `DTI ≥ 0` (the workflow's audit). CT1 and CT2 are two analyses of the same views, not independent
  measurements: a window uses the one with more valid points in it, CT2 on a tie (the audit's rule at its one time).
  An impurity temperature, taken as the deuterium one (T_D = T_CHERS, as the workflow does). Every CHERS product of
  the workflow's 16 shots reports `QUALITY=TEST` and the warning "Bay B window Transmission factor= 0.715" (docs/03).
- TRANSP (`/<shot>/TRANSP`, `flux_transp.VARIABLES`): `TI` (eV) and `ND` (cm⁻³, thermal deuterium) on the zone
  centres `X` at `TIME3` (s); `RMAJM` (cm), the midplane major radius of the zone boundaries `XB` on both sides,
  inboard edge to outboard edge through the axis (2 N + 1 points). Units and dimensions asserted; a RMAJM that is
  not increasing inboard to outboard, or an `XB` that does not end at 1, is refused with the reason in the note.
  **TRANSP covers the core only**: it models the plasma inside its boundary, so its profiles stop at the last zone
  centre inside the separatrix and it has nothing in the SOL.
- **TRANSP's run** (`flux_transp.choose`, on Flux, among `<root>/{16,15}/<shot><letter><2 digits>`):
  1. the sim-init workflow's choice where its candidate CSV lists the shot (`TRANSP_RUN_ID`: 203899 → 203899S18);
  2. else the newest run interpretive throughout (its namelist sets `lpredict_te`, `_ti`, `_ne` false in every
     block, `~update_time` blocks included);
  3. else the newest run that sets none of them, taken as TRANSP's default, interpretive. **UNVERIFIED**: that the
     default is interpretive is an assumption, not checked against TRANSP's documentation; the workflow's own rule
     requires the flags set and so found no run for 203655, whose five runs (A01-A04, F01) set none. This is the
     step that picks 203655's run; read its namelist (stored in the file, attribute `namelist`) before trusting it.

  Newest = the TR.DAT's time, A01 first on a tie (the workflow's order). The first that qualifies is taken: its CDF
  holds every variable above with those dimensions and units, and the last 200 kB of its TR.MSG holds no fatal
  marker. The group's attributes say which run (`run`), why (`why`), every run of the shot and what became of it
  (`considered`), the source CDF and its time, the namelist, and which profiles the run predicts from when
  (`predicts`, `predicts_from`).
- **203899S18 predicts T_e and T_i** from 128 ms (`lpredict_te = lpredict_ti = 1` in a `~update_time = 0.128287`
  block; the workflow's `reference_run/transp_effective_mode_audit.csv` says PREDICTIVE_AT_TARGET_TIME): in Sim 2's
  window its T_i is TRANSP's transport model, not a fit to CHERS. It is drawn dotted, its legend says "predicted by
  its model", and the note says from when. Its n_D is data-based (`lpredict_ne = 0` throughout: from the measured
  n_e). The run is kept because it is the one the workflow took its edge T_i (TIEDG) from.
- Window: the Thomson frames chosen (above); CHERS and TRANSP average their own samples within the span those frames
  stand for, half a Thomson frame interval past each end frame (`thomson.band`, the shaded band of the timeline).
  CHERS samples every 10 ms, TRANSP's profiles every ~1 ms. A window with no CHERS sample draws no CHERS and says
  where its samples are (203655: CHERS only from 126 to 216 ms).
- Placement of TRANSP: each zone centre's midplane radius is interpolated in `X` between its boundaries' (`RMAJM`),
  at each TRANSP time; its distance from TRANSP's own boundary (RMAJM's ends, same equilibrium, same time) is
  averaged over the window, and TRANSP's boundary is put on the run's separatrix. Both x modes (distance from the
  separatrix, R) use that distance. CHERS is placed as Thomson (below), each view's radius taken at the midplane.
- Beam flattop: the elongation/triangularity scan's rule on `WF:\PNB` (within 0.2 MW of the maximum, steady over 15
  samples each side). Its smallest-|DRSEP| EFIT01 time must be the run's equilibrium time; the note says if not.
- Window, opening default: the later half of the flattop after the run's time, i.e. from the midpoint of (run's
  time, flattop end) to the flattop's last frame. Point = mean over the window's frames; bars = spread over the
  frames (fit error where a channel has one valid frame).
- Placement of Thomson and CHERS: each sample's channels measured from the run's own midplane separatrix, moved by EFIT01's separatrix
  shift (RMIDOUT / RMIDIN) from the run's time to that frame. EFIT01 is used only for the shift: at 325 ms it puts
  203899's separatrix +7.1 mm (outboard) and +2.5 mm (inboard) from the run's (h1a) one, a third of the run's
  outboard SOL. 203899 moves 2-3 cm outboard and ~4 cm inboard between 0.3 and 0.9 s, so a plain R overlay would
  be wrong; EFIT01's inboard separatrix also jitters by several mm between slices (averaging frames damps it).
- Thomson does not reach the inboard SOL (203899: innermost channel 67 mm inside the separatrix); there only core
  channels inside the run's span show.

Sanity check, Sim 2 at the frame nearest 325 ms: Thomson / the run's first-frame n_e = 0.90-1.08 on 6 of 7 channels
(0.68 at +12 mm in the far SOL, where the initial profile is the workflow's spline, extrapolated).

## Checks (2026-11-18: all pass, the browser part too: `check_viewer.py --browser` 120, `check_gkview.py` 64)
Run from this directory with no environment set (its own `viewer.json`, `.venv`):
```sh
.venv/bin/python check_viewer.py [--browser <chrome>] [--parts blank,poloidal]   # --parts: some parts only
<a postgkyl venv>/bin/python check_gkview.py
```
2026-11-17, device themes and the light default (Device themes, above). `part_headless` now opens every simulation by
its label (each had been moved to the head of the list, which with the new default opens the default device's first
simulation instead) and checks that each opens in its device's theme, the view naming none.

`part_themes` checks:
- the brand themes' colours and fonts against their cited JSON files, each colour with a source;
- every theme's page CSS;
- TCV, ASDEX Upgrade and STEP light and dark, every tab built.

The new `part_devices` (`--parts devices`) checks:
- a theme asked for wins and is the only one the view keeps;
- a name of no theme counts as none;
- a device without a theme (Sim 2 offered as JET's) opens in the default device's;
- a fresh launch after a session dark on TCV in STEP's theme opens light, on Sim 1, in NSTX-U's theme, and marimo's
  first paint is light;
- with `--browser`, one page: the fresh launch; TCV by its address (red controls); the theme radio's ASDEX Upgrade
  clicked (the address keeps it); the simulation changed to STEP in the sidebar's list (the theme stays); a reload;
  then (2026-11-18) every theme by its radio, light and dark: the run's title against what it sits on (every stop of a
  title bar), WCAG contrast 3:1 or more (large text), in capitals in STEP's only. Before the `HEADING_RESET` fix it
  failed on exactly the two the visual test found: DIII-D light 1.2:1, STEP light 1.1:1 (now 9.3 and 18.1; the
  lowest of the twelve is PPPL light, 4.1, its orange titles on white).

The browser part checks light by default and the switch to dark and back. The layout audit asks for dark where it
meant the old default, and adds each other device's first view light, in its theme. The blank pages open dark, as
their screenshots expect.

The page check found a bug: the theme's `<style>` sat in the head, below marimo's copy of `runs/viewer.css` in the
body, so `--primary` stayed NSTX-U's #007382 in every theme (Look). Both suites ran in full, formatted by the hooks
first, while a 450-shot grid sweep held five cores of this login node:
- `check_viewer.py --browser`: 119 of 119 in one run (30 min, no part rerun). The seven new checks: 2 in part_themes,
  5 in part_devices. The layout audit covered 35 views.
- `check_gkview.py`: 64 of 64.
- After that, the page check's screenshots were made to wait for the movie; `--parts devices --browser` passed alone.

2026-11-17, on this data directory as it now is (the five device runs in `viewer.json`, the Flux copies of 203655
and 203899 fetched that morning, 203655's with TRANSP run 203655A04): 106 of 112 in one full `--browser` run while a
450-shot grid sweep held five cores of the same login node; the other six pass on their parts' reruns. Four of those
were the load (the two blank starts in the browser, the opacity slider, the plotly fallback's pace: the same parts
pass again unchanged). Four checks assumed the data directory of 11-16 and now read what they mean: the chart
helper skips the Thomson timeline of a run with no shot file and probes block 0 of a one-block run (`part_labels`);
the stand-in Flux archive and the "fetched before TRANSP" cases use copies without the TRANSP group, as Flux's dumps
are (`part_measured`); the mirror may open its own Flux copy (`part_sync`); `viewer.json`'s simulations without
settings of their own read as before, those with their own name their device (`part_schema`).

2026-11-17, the curves shown (The curves shown, above): the new `part_species` (`--parts species`), 6 of 6 with
`--browser`, on this directory's data and on the three simulations' alone; both suites in full, formatted by the
repo's hooks: `check_viewer.py --browser` 104 of 104 (the 98 before and part_species), `check_gkview.py` 64 of 64.
The full `check_viewer.py` ran on a data directory whose `viewer.json` offers the three NSTX-U simulations only (as
the suite asks; `base` and the cache the same) with the 2026-11-09 Flux copy of 203655 (no TRANSP). On this
directory's own data two things fail that fail the same with the code of a4005d95: `viewer.json` now offers the five
device runs too, and `part_labels` stops on the first of them (`_charts` asks `thomson.default_window` about a shot of
0, which has no Thomson file: `thomson.read` gives None); and `runs/flux/203655.h5`, fetched again on 2026-11-17 with
TRANSP, is no longer the copy without TRANSP that `part_measured` takes as its stand-in archive and as "the Flux copy
fetched before TRANSP" (3 of its checks).
2026-11-16, all four changes together (measured profiles, a blank start, the lines and the grid, other devices' runs)
and the 11-14 layout fixes, formatted by the repo's hooks: both suites in full, all pass (`check_viewer.py
--browser` 98, `check_gkview.py` 64). Fixed on the way: two checks (the opacity slider's value is now read where it
shows, marimo's `aria-valuenow` being an index into `steps=`; the strips are scrolled into view before a click probes
them, the longer sidebar having left them above the window) and two faults of the merge itself (the empty run's
movie built curved cells from no block; opening a run read each block's field file before looking for its sheath
file, which a run without that block's field file lacks: the sheath file is now looked up first).
2026-11-16 (measured T_e / T_i / n_D; only these parts run, not the whole suite): `part_thomson` 3/3 (the reader ==
the workflow's 203899 channel export for n_e and T_e, max rel 0; Thomson points on the n and T rows, both sides) and
the new `part_measured` 13/13, which needs a postgkyl venv's python (viewer.json's `pgkyl`) for scipy: flux_transp's
netCDF-3 reader == scipy on 61 variables (2 fixture files); the stand-in fetch delivers both shots, Thomson / EFIT01
/ CHERS / beam power equal to the dumps dataset for dataset; the run choice (203899 → 203899S18 from the CSV, its
prediction from 128.287 ms recorded; 203655 → A01 past S05 no CDF, A03 fatal, A02 no ND; without the CSV 203899 gets
none); the delivered TRANSP == the fixture (6 variables, units, dims) and its midplane geometry; CHERS == the
workflow's 42 direct points at 325 ms (max rel 0) and its VALID masking (3410 of 4437 points per analysis); wrong
units / axes refused with the reason, Thomson still read (5 cases); window means == plain loops (Thomson T_e, CHERS
T_i, TRANSP T_i and n_D); the profiles figure's rows, colours and dashes, 9 measured traces the same points in both x
modes; the notes for 4 missing / empty sources; the notebook on the fetched files (Sim 2) and on the copy without
TRANSP (Sim 1), every tab (11 + 11). Run: `.venv/bin/python -c 'import check_viewer as c; c.part_thomson();
c.part_measured()'`.
2026-11-16: `part_lines` (The lines over the cells, above) and `part_grid` (The grid and the quantity), run alone
with `--browser`: all pass.
2026-11-16: `check_viewer.py` part blank (A blank start): on three data directories (no `viewer.json`, `"sims": []`,
a simulation whose directory is missing) the notebook opens on the empty run with every tab built; the empty run's
charts have their panel titles, axis titles and legend and no data point, its movie no frame, cell or strip and no
view to frame, its map no mesh; with `--browser` each served by view.sh: the note at the top, the movie's four
canvases with its controls idle, every tab opened and its charts drawn (axis titles, legend, no curve or map), no
error text; screenshots dark and light. And the per-simulation and per-device settings (part_schema): this
directory's own `viewer.json` reads as before (the simulations, base, device, picture, Sim 2's title); a
simulation's own device names its runs and gets its own picture, the top-level picture staying the default
device's; a run naming its geqdsk keeps it over the simulation's (the disagreement said), one naming none takes
the simulation's (shot, time, wall); a 1-point limiter gives way to the device's wall, which gives no shot;
malformed entries left out with a note. Every figure of the three simulations built with the code before and after
this change: 102 of 102 identical (figure JSON, map pixels, movie payloads, live specs); recomputed summary rows equal
the cached ones (458 of 458 entries).
2026-11-14, later: the code reformatted to the repo's style (YAPF: 2-space indent, 80 columns; ruff's pyflakes
rules: 4 unused imports / an f-string without placeholders removed) with the repo's pre-commit hooks, all passing;
both suites then run from here as above, all pass.
2026-11-14, last: the movie's poloidal plane is drawn inside its plot area only. In a zoomed view the cells and the
wall ran on past it, over the title and the axes, and the device picture stopped at the view's box instead of
filling the area; the ticks now span the whole area, the area never shows R < 0 (the box sits at R = 0 instead of
centred when the area is wider) and a click in a margin probes nothing. `part_layout` opens every simulation in every
view (half domains mirrored too, with the device picture, light and dark) and finds nothing but text in the canvas's
margins (the drawing without the clip fails it: 13,036 and 14,466 stray pixels in two lower-divertor views). The
same audit, screenshots of every tab: the opacity slider showed 0.6000000000000001, too wide for the sidebar (which
grew a scrollbar): its steps are now two-decimal numbers; the Options tab's tables ran 1335 px wide in a 1055 px panel,
their last columns cut off: fixed column widths, the text wrapping in them (`part_layout` measures every table's
content against its panel); a chart's end labels had the zero line running through them: they have the chart's
surface behind them now.

2026-11-14: the browser part runs again (Chrome for Testing's headless shell 157; any Chrome / Chromium binary will
do). New: the open tab stays when an option re-runs the tabs (Balance & power, dark mode off and on: its charts
drawn anew each time, the tab still open, the conservation chart's log axis there). The notebook as committed in
1e19a567 fails it: the page falls back to Midplane profiles. The probe's third click now reads the strips' place just
before it clicks (it had failed 4 runs in 6, the strips never having moved: a click too soon after Esc's re-run). The
conservation chart is in the label and pgkyl audits (27 charts, 460 curves).
2026-11-13: `check_viewer.py` adds fonts (Viewer Symbols complete and reproducible, the page gets it, matplotlib
takes the tracked fonts before a system copy, no missing glyph in any theme's notebook) and sync (above).
`check_gkview.py` all pass after the geqdsk / cache path helpers moved (`geqdsk_named`, `here`, `cache_path`).

### 2026-11-12
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
  holds the view; light by default (a white page; 2026-11-17, dark before) and the switch goes dark (#1B262B) and
  back; the movie at 30 frames/s draws 60
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
