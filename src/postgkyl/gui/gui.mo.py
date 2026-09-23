import marimo

__generated_with = "0.23.11"
app = marimo.App(width="full")


@app.cell
def _():
    # --- imports & matplotlib setup -----------------------------------------
    import marimo as mo

    import matplotlib
    matplotlib.use("Agg")  # headless: figures are captured as PNG, never shown
    import matplotlib.pyplot as plt

    import base64
    import glob
    import numpy as np
    import html
    import os
    import re
    import shutil
    import tempfile
    import traceback

    from postgkyl.clap import PgkylSession

    return (
        PgkylSession, base64, glob, html, mo, np, os, plt, re, shutil,
        tempfile, traceback,
    )


@app.cell
def _(glob, os, re):
    # --- discovery: scan a directory for plottable outputs ------------------
    _FRAME_RE = re.compile(r"^(?P<stem>.+)_(?P<frame>\d+)\.gkyl$")

    def scan_outputs(directory):
        """Group a directory's *.gkyl files into named outputs.

        Files following the Gkeyll convention ``<stem>_<frame>.gkyl`` are grouped
        into frame *series* (e.g. ``...-ion_M0`` with frames ``0..5``). Files with
        no trailing frame number (geometry, integrated diagnostics, ...) are listed
        as *static* single-frame outputs. Returns ``{label: info}`` where ``info``
        has keys ``type`` ('series'|'static'), ``stem``/``path``, and ``frames``.
        """
        directory = os.path.expanduser(directory.strip())
        if not directory or not os.path.isdir(directory):
            return {}

        series, static = {}, {}
        for fn in sorted(glob.glob(os.path.join(directory, "*.gkyl"))):
            base = os.path.basename(fn)
            m = _FRAME_RE.match(base)
            if m:
                stem = os.path.join(directory, m.group("stem"))
                series.setdefault(stem, set()).add(int(m.group("frame")))
            else:
                static[fn] = base[: -len(".gkyl")]

        outputs = {}

        # Strip the common sim-prefix (text before the first '-') for readability,
        # keeping the full basename when that would be ambiguous.
        def _label(base):
            return base.split("-", 1)[1] if "-" in base else base

        for stem, frames in series.items():
            base = os.path.basename(stem)
            label = _label(base)
            if label in outputs:  # prefix collision: disambiguate with full name
                label = base
            outputs[label] = {
                "type": "series",
                "stem": stem,
                "frames": sorted(frames),
            }
        for path, base in static.items():
            label = _label(base) + "  (static)"
            outputs[label] = {"type": "static", "path": path, "frames": []}
        return outputs

    return (scan_outputs,)


@app.cell
def _(mo, os):
    # --- top-level controls (created here, displayed in the layout cell) -----
    # The starting directory can be passed on launch, e.g.
    #     marimo run dev_gui.mo.py --path /path/to/my/data/simulation
    # (also works with `marimo edit`). Falls back to the default below.
    _default_dir = "tests/test_data"
    _cli_path = mo.cli_args().get("path")
    _start_dir = os.path.expanduser(str(_cli_path)) if _cli_path else _default_dir
    dir_input = mo.ui.text(
        value=_start_dir,
        label="Data directory",
        placeholder="path to a Gkeyll data directory",
        full_width=True,
    )
    return (dir_input,)


@app.cell
def _(base64, mo, os):
    # --- header logo (embedded as base64 so it survives the raw-HTML pane) ---
    _candidates = [
        os.path.join(os.path.dirname(__file__), "logogui.png"),
    ]
    _logo = next((p for p in _candidates if os.path.exists(p)), None)
    if _logo:
        _b64 = base64.b64encode(open(_logo, "rb").read()).decode()
        header = mo.Html(
            f"<img src='data:image/png;base64,{_b64}' alt='Gkeyll Marimo GUI' "
            "style='max-width:100%;height:auto;display:block;margin:0 0 0.5rem;' />"
        )
    else:
        header = mo.md("## Gkeyll Marimo GUI")
    return (header,)


@app.cell
def _(dir_input, get_field, mo, scan_outputs, set_field):
    # --- field selection ----------------------------------------------------
    outputs = scan_outputs(dir_input.value)
    _opts = sorted(outputs.keys())

    if _opts:
        _prev = get_field()
        # Keep old value if it exists in the new directory, else fallback to M0/first
        _default = _prev if _prev in _opts else next((o for o in _opts if o.endswith("M0")), _opts[0])
        field_dropdown = mo.ui.dropdown(
            options=_opts, value=_default, label="Field", searchable=True, on_change=set_field
        )
    else:
        field_dropdown = mo.ui.dropdown(options=[], label="Field")
    return field_dropdown, outputs

@app.cell
def _():
    # --- gyrokinetic quantity registry + prefix discovery -------------------
    from postgkyl.utils.gk_quantities.registry import gk_quant_registry
    from postgkyl.commands.listoutputs import list_prefixes
    from postgkyl.tools.gk_transport import TRANSPORT_OUTPUTS

    gk_quant_list = sorted(gk_quant_registry.list())
    transport_outputs = list(TRANSPORT_OUTPUTS)

    def gk_extra(direction, extra):
        """Combine the direction + free-form extra fields into one `extra` arg
        (e.g. `dir=1,mass=0.1`). Returns None when both are blank."""
        parts = []
        if direction.value.strip():
            parts.append(f"dir={direction.value.strip()}")
        if extra.value.strip():
            parts.append(extra.value.strip())
        return ",".join(parts) or None

    return gk_extra, gk_quant_list, gk_quant_registry, list_prefixes, transport_outputs


@app.cell
def _(gk_quant_list, mo, transport_outputs):
    # --- load mode: static controls -----------------------------------------
    load_mode = mo.ui.dropdown(
        options=["load", "gk-load-quantity", "gk-transport"], value="load", label="load mode")
    quantity = mo.ui.dropdown(
        options=gk_quant_list,
        value=("M0" if "M0" in gk_quant_list else (gk_quant_list[0] if gk_quant_list else None)),
        label="quantity", searchable=True)
    species = mo.ui.text(value="ion", label="species", placeholder="ion/elc/...")
    direction = mo.ui.text(value="", label="direction", placeholder="dir, e.g. 0/1/2")
    extra = mo.ui.text(
        value="", label="extra", placeholder="mass=...,charge=...", full_width=True)
    # gk-transport: which flux-surface/time averaged radial profile to plot.
    tr_output = mo.ui.dropdown(
        options=transport_outputs, value="D", label="transport output")
    # Fluctuation about the y (binormal) or (y,z) flux-surface average, 3x data.
    # In gk-transport mode it selects the turbulent part of the fluxes instead.
    fluct = mo.ui.dropdown(options=["none", "y", "yz"], value="none", label="fluctuation about")
    return direction, extra, fluct, load_mode, quantity, species, tr_output


@app.cell
def _(dir_input, list_prefixes, mo):
    # --- load mode: dynamic prefix ------------------------------------------
    _prefixes = list_prefixes(dir_input.value.strip())
    simprefix = mo.ui.dropdown(
        options=_prefixes, value=(_prefixes[0] if _prefixes else None),
        label="sim prefix", searchable=True)
    return (simprefix,)


@app.cell
def _(mo):
    # --- persisted selection state ------------------------------------------
    get_frame, set_frame = mo.state(None)
    get_sel_en, set_sel_en = mo.state([])
    get_sel_val, set_sel_val = mo.state([])
    get_sel_mode, set_sel_mode = mo.state([])
    get_comp_en, set_comp_en = mo.state(False)
    get_comp_val, set_comp_val = mo.state(0)
    
    # New state variables for interface persistence
    get_field, set_field = mo.state(None)
    get_xidx, set_xidx = mo.state(0)
    return (
        get_comp_en, get_comp_val, get_field, get_frame, get_sel_en,
        get_sel_mode, get_sel_val, get_xidx, set_comp_en, set_comp_val, set_field,
        set_frame, set_sel_en, set_sel_mode, set_sel_val, set_xidx,
    )


@app.cell
def _(mo):
    # --- frame range / collect (field-independent, so they never reset) -----
    # Python slice over the detected frames: ':' all, '::2' every other, '-10:'
    # the last ten, '-1' the last one. Blank = the single frame of the slider.
    frame_range = mo.ui.text(value="", label="frame range", placeholder=": | ::2 | -10: | -1")
    collect_chk = mo.ui.checkbox(label="collect (time series)")
    return collect_chk, frame_range


@app.cell
def _(
    dir_input, field_dropdown, frame_range, get_frame, gk_quant_registry, load_mode, mo,
    outputs, quantity, set_frame, simprefix, species,
):
    # --- frame selection: slider with limits from the detected frames -------
    # In gk-load-quantity mode the available frames come from the registry
    # (which combination of source files exists); otherwise from the field.
    # gk-transport needs the particle flux, so its frames are those of part_flux.
    if load_mode.value in ("gk-load-quantity", "gk-transport"):
        _qname = quantity.value if load_mode.value == "gk-load-quantity" else "part_flux"
        info = {"type": "gk", "frames": []}
        frames = []
        if _qname and simprefix.value:
            try:
                frames = gk_quant_registry.get(_qname).get_avail_frames(
                    dir_input.value.strip().rstrip("/") + "/", simprefix.value,
                    species.value.strip() or None)
            except Exception:
                frames = []
    else:
        info = outputs.get(field_dropdown.value, {"type": "static", "frames": []})
        frames = info["frames"]

    if frames:
        _fval = get_frame() if get_frame() in frames else frames[0]
        # The slider is unused while a frame range is given.
        frame_slider = mo.ui.slider(
            steps=frames, value=_fval, label="frame", show_value=True,
            include_input=True, full_width=True, on_change=set_frame,
            disabled=bool(frame_range.value.strip()),
        )
    else:
        frame_slider = mo.ui.slider(steps=[0], value=0, label="frame", disabled=True)
    return frame_slider, frames, info


@app.cell
def _(
    PgkylSession, dir_input, direction, extra, field_dropdown, frames,
    gk_extra, info, load_mode, plt, quantity, simprefix, species,
):
    # --- probe the RAW field's grid (for transform bounds like x_idx) -------
    def _probe_base_grid():
        if load_mode.value == "gk-transport":
            return {"ok": False, "msg": "gk-transport gives 1D radial profiles "
                    "(flux-surface and time averaged): nothing to select."}
        try:
            plt.close("all")
            pg = PgkylSession()
            if load_mode.value == "gk-load-quantity":
                if not (quantity.value and simprefix.value and frames):
                    return {"ok": False, "msg": "Choose a quantity and a sim prefix."}
                pg.gk_load_quantity(
                    quantity=quantity.value, name=simprefix.value,
                    path=dir_input.value.strip(), frame=str(frames[0]),
                    species=species.value.strip() or None,
                    extra=gk_extra(direction, extra))
            else:
                if not field_dropdown.value:
                    return {"ok": False, "msg": "No field selected."}
                src = (f"{info['stem']}_{info['frames'][0]}.gkyl"
                       if info["type"] == "series" else info["path"])
                pg.load(src)
            dat = next(pg.data.iterator(None))
            lo, up = dat.get_bounds()
            ncells = dat.get_num_cells()
            ndim = dat.get_num_dims()
            dims = [
                {"lo": float(lo[i]), "up": float(up[i]), "n": int(ncells[i])}
                for i in range(ndim)
            ]
            return {"ok": True, "ndim": ndim, "dims": dims,
                    "ncomps": int(dat.get_num_comps())}
        except Exception as exc:
            return {"ok": False, "msg": f"{type(exc).__name__}: {exc}"}

    base_grid_info = _probe_base_grid()
    return (base_grid_info,)

@app.cell
def _(
    PgkylSession, dir_input, direction, extra, field_dropdown, frames,
    gk_extra, info, load_mode, plt, quantity, simprefix, species,
    transform, interp_pts, mapc2p_file, phi_tor_val, x_idx
):
    # --- probe the TRANSFORMED grid (for dynamic select sliders) ------------
    def _probe_transformed_grid():
        if load_mode.value == "gk-transport":
            return {"ok": False, "msg": "gk-transport gives 1D radial profiles "
                    "(flux-surface and time averaged): nothing to select."}
        try:
            plt.close("all")
            pg = PgkylSession()
            
            # 1) Load
            if load_mode.value == "gk-load-quantity":
                if not (quantity.value and simprefix.value and frames):
                    return {"ok": False, "msg": "Choose a quantity and a sim prefix."}
                pg.gk_load_quantity(
                    quantity=quantity.value, name=simprefix.value,
                    path=dir_input.value.strip(), frame=str(frames[0]),
                    species=species.value.strip() or None,
                    extra=gk_extra(direction, extra))
            else:
                if not field_dropdown.value:
                    return {"ok": False, "msg": "No field selected."}
                src = (f"{info['stem']}_{info['frames'][0]}.gkyl"
                       if info["type"] == "series" else info["path"])
                pg.load(src)

            # 2) Transform
            def _opt(widget):
                v = (widget.value or "").strip()
                return v or None

            if transform.value == "interpolate":
                pg.interpolate(interp=int(interp_pts.value) if interp_pts.value else None)
            elif transform.value == "dg-local-poly":
                pg.dg_local_poly(npoints=int(interp_pts.value) if interp_pts.value else 2)
            elif transform.value == "gk-rz":
                phi_rad = float(phi_tor_val.value) * 3.141592653589793 / 180.0
                pg.gk_rz(
                    mapc2p=_opt(mapc2p_file), 
                    phi_tor=phi_rad, 
                    nz_interp=int(interp_pts.value) if interp_pts.value else 8
                )
            elif transform.value == "gk-fluxsurf":
                pg.gk_fluxsurf(
                    mapc2p=_opt(mapc2p_file), 
                    x_idx=int(x_idx.value) if x_idx.value else 0,
                    nz_interp=int(interp_pts.value) if interp_pts.value else 8
                )
            
            # 3) Probe the resulting data bounds
            dat = next(pg.data.iterator(None))
            lo, up = dat.get_bounds()
            ncells = dat.get_num_cells()
            ndim = dat.get_num_dims()
            # This handle better axis for gk-rz plots select.
            value_coords = dat.ctx.get("value_coords")
            dims = []
            for i in range(ndim):
                if value_coords is not None and value_coords[i] is not None:
                    d_lo, d_up = float(value_coords[i].min()), float(value_coords[i].max())
                else:
                    d_lo, d_up = float(lo[i]), float(up[i])
                dims.append({"lo": d_lo, "up": d_up, "n": int(ncells[i])})
            return {"ok": True, "ndim": ndim, "dims": dims,
                    "ncomps": int(dat.get_num_comps())}
        except Exception as exc:
            print("Error probing transformed grid:", exc)
            return {"ok": False, "msg": f"{type(exc).__name__}: {exc}"}

    grid_info = _probe_transformed_grid()
    return (grid_info,)

@app.cell
def _(get_sel_mode, grid_info, mo, set_sel_mode):
    # --- per-dimension mode: select (slice at a coordinate) or average ------
    # Kept in its own cell: changing a mode updates the persisted state, which
    # re-runs the slider cell so an averaged dimension's slider is disabled.
    if grid_info.get("ok"):
        _prev_mode = get_sel_mode()
        sel_modes = mo.ui.array(
            [mo.ui.dropdown(options=["select", "average"],
                            value=_prev_mode[i] if i < len(_prev_mode) else "select")
             for i in range(grid_info["ndim"])],
            on_change=lambda vals: set_sel_mode(list(vals)),
        )
    else:
        sel_modes = mo.ui.array([])
    return (sel_modes,)


@app.cell
def _(
    get_comp_en, get_comp_val, get_sel_en, get_sel_mode, get_sel_val, grid_info, mo,
    set_comp_en, set_comp_val, set_sel_en, set_sel_val,
):
    # --- dynamic `select` sliders, limits taken from the field's grid -------
    # Limits follow the current field's grid, but the enable flags and slider
    # positions are seeded from persisted state, so changing field keeps the
    # slices (clamped into the new range) instead of resetting them. By default
    # every dimension beyond the first two is sliced, so a 3D+ field lands on a
    # viewable 2D plot immediately.
    def _clamp(v, lo, up):
        return min(max(v, lo), up)

    if grid_info.get("ok"):
        _ndim = grid_info["ndim"]
        _dims = grid_info["dims"]
        _prev_en = get_sel_en()
        _prev_val = get_sel_val()
        _prev_mode = get_sel_mode()

        def _slider(i, d):
            span = d["up"] - d["lo"]
            step = span / max(d["n"], 1)
            if i < len(_prev_val):
                val = _clamp(float(_prev_val[i]), d["lo"], d["up"])
            else:
                val = d["lo"] + span / 2.0
            # An averaged dimension has no slice coordinate.
            averaged = i < len(_prev_mode) and _prev_mode[i] == "average"
            return mo.ui.slider(
                start=d["lo"], stop=d["up"], step=step, value=val,
                show_value=True, include_input=True, full_width=True,
                disabled=averaged,
            )

        def _enabled(i):
            return bool(_prev_en[i]) if i < len(_prev_en) else (i >= 2)

        sel_sliders = mo.ui.array(
            [_slider(i, d) for i, d in enumerate(_dims)],
            on_change=lambda vals: set_sel_val(list(vals)),
        )
        sel_enables = mo.ui.array(
            [mo.ui.checkbox(value=_enabled(i)) for i in range(_ndim)],
            on_change=lambda vals: set_sel_en(list(vals)),
        )
        _cmax = max(grid_info["ncomps"] - 1, 0)
        comp_enable = mo.ui.checkbox(
            value=get_comp_en(), label="component", on_change=set_comp_en)
        comp_slider = mo.ui.slider(
            start=0, stop=_cmax, step=1, value=_clamp(int(get_comp_val()), 0, _cmax),
            show_value=True, include_input=True, on_change=set_comp_val,
        )
    else:
        sel_sliders = mo.ui.array([])
        sel_enables = mo.ui.array([])
        comp_enable = mo.ui.checkbox(value=False, label="component")
        comp_slider = mo.ui.slider(start=0, stop=0, value=0, disabled=True)
    return comp_enable, comp_slider, sel_enables, sel_sliders


@app.cell
def _(mo):
    # --- processing chain: static controls ----------------------------------
    transform = mo.ui.dropdown(
        options=["none", "interpolate", "dg-local-poly", "gk-rz", "gk-fluxsurf"],
        value="none",
        label="transform",
    )
    interp_pts = mo.ui.number(start=1, stop=32, value=2, label="points factor / nz-interp")
    mapc2p_file = mo.ui.text(value="", label="mapc2p file (optional)", full_width=True)
    
    # Toroidal angle slice slider for gk-rz
    phi_tor_val = mo.ui.slider(
        start=0, stop=360, step=1, value=0,
        label="poloidal plane angle phi-tor (degrees)",
        show_value=True, include_input=True, full_width=True
    )
    return interp_pts, mapc2p_file, phi_tor_val, transform


@app.cell
def _(base_grid_info, get_xidx, mo, set_xidx):
    # --- processing chain: dynamic flux surface slider ----------------------
    if base_grid_info.get("ok") and len(base_grid_info["dims"]) > 0:
        max_x = base_grid_info["dims"][0]["n"] - 1
        _prev = get_xidx()
        _val = _prev if _prev <= max_x else 0
        x_idx = mo.ui.slider(
            start=0, stop=max_x, step=1, value=_val, 
            label=f"flux surface x-index (0 to {max_x})", 
            show_value=True, include_input=True, full_width=True, on_change=set_xidx
        )
    else:
        x_idx = mo.ui.slider(start=0, stop=0, value=0, label="flux surface x-index", disabled=True)
        
    return (x_idx,)

@app.cell
def _(mo):
    # --- plot options -------------------------------------------------------
    surface = mo.ui.checkbox(label="surface")
    contour = mo.ui.checkbox(label="contour")
    fixaspect = mo.ui.checkbox(label="fix aspect")
    showgrid = mo.ui.checkbox(label="grid", value=True)
    logx = mo.ui.checkbox(label="logx")
    logy = mo.ui.checkbox(label="logy")
    logz = mo.ui.checkbox(label="logz")
    legend = mo.ui.checkbox(label="legend")  # off by default

    cmap = mo.ui.dropdown(
        options=[
            "(default)", "viridis", "plasma", "inferno", "twilight", "cividis",
            "twilight", "RdBu_r", "jet", "gray",
        ],
        value="(default)",
        label="cmap",
    )

    xlabel = mo.ui.text(value="", label="xlabel")
    ylabel = mo.ui.text(value="", label="ylabel")
    clabel = mo.ui.text(value="", label="clabel")
    title = mo.ui.text(value="", label="title")

    # axis / colorbar limits (blank = auto)
    xmin_t = mo.ui.text(value="", label="x min")
    xmax_t = mo.ui.text(value="", label="x max")
    ymin_t = mo.ui.text(value="", label="y min")
    ymax_t = mo.ui.text(value="", label="y max")
    cmin_t = mo.ui.text(value="", label="cbar min")
    cmax_t = mo.ui.text(value="", label="cbar max")
    # Color range symmetric about 0 (e.g. white = 0 with RdBu_r): +-|cbar max|
    # (or |cbar min|) when given, else +- the max |value| of the plotted field.
    sym_chk = mo.ui.checkbox(label="symmetric (0 centred)")

    # shift (blank = 0) and scale (blank = 1) per axis; z = value/colorbar axis
    xshift_t = mo.ui.text(value="0", label="x shift", placeholder="0")
    yshift_t = mo.ui.text(value="0", label="y shift", placeholder="0")
    zshift_t = mo.ui.text(value="0", label="z shift", placeholder="0")
    xscale_t = mo.ui.text(value="1", label="x scale", placeholder="1")
    yscale_t = mo.ui.text(value="1", label="y scale", placeholder="1")
    zscale_t = mo.ui.text(value="1", label="z scale", placeholder="1")

    plot_options = mo.vstack([
        mo.hstack(
            [surface, contour, fixaspect, showgrid, logx, logy, logz, legend],
            justify="start", gap=0.75, wrap=True,
        ),
        cmap,
        mo.hstack([xlabel, ylabel, clabel, title], justify="start", gap=0.5, wrap=True),
        mo.md("**limits** — _blank = auto_"),
        mo.hstack([xmin_t, xmax_t, ymin_t, ymax_t], justify="start", gap=0.5, wrap=True),
        mo.hstack([cmin_t, cmax_t, sym_chk], justify="start", align="center", gap=0.5,
                  wrap=True),
        mo.md("**shift / scale** _(z = value / colorbar axis)_"),
        mo.hstack([xshift_t, yshift_t, zshift_t], justify="start", gap=0.5, wrap=True),
        mo.hstack([xscale_t, yscale_t, zscale_t], justify="start", gap=0.5, wrap=True),
    ], gap=0.4)
    return (
        clabel, cmap, cmax_t, cmin_t, contour, fixaspect, legend, logx, logy,
        logz, plot_options, showgrid, surface, sym_chk, title, xlabel, xmax_t,
        xmin_t, xscale_t, xshift_t, ylabel, ymax_t, ymin_t, yscale_t, yshift_t,
        zscale_t, zshift_t,
    )


@app.cell
def _(
    PgkylSession,
    base64,
    clabel,
    cmap,
    cmax_t,
    cmin_t,
    collect_chk,
    dir_input,
    direction,
    extra,
    fluct,
    gk_extra,
    html,
    comp_enable,
    comp_slider,
    contour,
    field_dropdown,
    fixaspect,
    frame_range,
    frame_slider,
    frames,
    grid_info,
    info,
    interp_pts,
    legend,
    load_mode,
    logx,
    logy,
    logz,
    mapc2p_file,
    mo,
    np,
    os,
    plt,
    quantity,
    re,
    sel_enables,
    sel_modes,
    sel_sliders,
    showgrid,
    simprefix,
    species,
    surface,
    sym_chk,
    tempfile,
    title,
    tr_output,
    traceback,
    transform,
    xlabel,
    xmax_t,
    xmin_t,
    xscale_t,
    xshift_t,
    ylabel,
    ymax_t,
    ymin_t,
    yscale_t,
    yshift_t,
    zscale_t,
    zshift_t,
):
    # --- execute the chain & build the figure view --------------------------
    # process_frame/plot_session are exported so the movie cell replays the
    # exact same chain, one frame at a time.
    def _opt(widget):
        v = (widget.value or "").strip()
        return v or None

    def _num(widget, default=None):
        v = (widget.value or "").strip()
        return float(v) if v else default

    def pick_frames(text, what="frame range"):
        """Frames picked by a range text (a slice over the detected frames), or None if blank."""
        text = (text or "").strip()
        if not text:
            return None
        parts = text.split(":")
        try:
            if len(parts) == 1:
                return [frames[int(parts[0])]]
            if len(parts) > 3:
                raise ValueError
            return frames[slice(*[int(v) if v.strip() else None for v in parts])]
        except (ValueError, IndexError):
            raise ValueError(f"{what} '{text}' is not a valid index or slice of the "
                             f"{len(frames)} available frame(s), e.g. ':', '::2', '-10:', '-1'.")

    def process_frame(frame=None):
        """
        Run the load -> fluct -> average -> transform -> select -> collect chain.

        frame=None follows the frame slider / frame range; an explicit frame
        number loads that single frame (ignoring the frame range and collect).
        Returns (session, status, error): session is None when the chain could
        not produce plottable data, and error then explains why.
        """
        # 1) load -----------------------------------------------------------
        pg = PgkylSession()

        _frames = pick_frames(frame_range.value) if frame is None else None
        if _frames is not None and not _frames:
            return pg, "", f"Frame range `{frame_range.value.strip()}` selects no frame."
        _all = _frames is not None and len(_frames) == len(frames)
        _single = frame_slider.value if frame is None else frame

        if load_mode.value == "gk-transport":
            if not simprefix.value:
                return None, "", "Choose a simulation prefix."
            if _frames is None:
                _frame = str(_single)
            else:
                _frame = ":" if _all else ",".join(str(f) for f in _frames)
            # With 'collect', keep one profile per frame for a space-time diagram.
            pg.gk_transport(
                name=simprefix.value, path=dir_input.value.strip(), frame=_frame,
                species=species.value.strip() or "ion", outputs=tr_output.value,
                fluct=fluct.value, extra=extra.value.strip() or None,
                per_frame=bool(_frames is not None and collect_chk.value))
        elif load_mode.value == "gk-load-quantity":
            if not (quantity.value and simprefix.value):
                return None, "", "Choose a quantity and a simulation prefix."
            if _frames is None:
                _frame = str(_single)
            else:
                _frame = ":" if _all else ",".join(str(f) for f in _frames)
            pg.gk_load_quantity(
                quantity=quantity.value, name=simprefix.value,
                path=dir_input.value.strip(), frame=_frame,
                species=species.value.strip() or None,
                extra=gk_extra(direction, extra))
        else:
            if not field_dropdown.value:
                return None, "", "Select a data directory and field on the left."
            if info["type"] != "series":
                pg.load(info["path"])
            elif _frames is None:
                pg.load(f"{info['stem']}_{_single}.gkyl")
            elif _all:
                pg.load(f"{info['stem']}_[0-9]*.gkyl")
            else:
                pg.load(*[f"{info['stem']}_{f}.gkyl" for f in _frames])

        # 1b) fluctuation about the y or (y,z) average (dg-fluct, on DG data) --
        if fluct.value != "none" and load_mode.value != "gk-transport":
            _nd = [dat.get_num_dims() for dat in pg.data.iterator(None)]
            if any(nd != 3 for nd in _nd):
                return None, "", ("Fluctuations about the y or (y,z) average need 3x "
                                      "(x,y,z) configuration-space data.")
            pg.dg_fluct(z1=True, z2=fluct.value == "yz")

        # 2) average (dg-avg works on DG data, so it runs before any transform)
        avg_dirs = []
        if grid_info.get("ok"):
            avg_dirs = [i for i in range(grid_info["ndim"])
                        if sel_enables.value[i] and sel_modes.value[i] == "average"]
        if avg_dirs:
            if transform.value in ("gk-rz", "gk-fluxsurf"):
                return None, "", (f"Averaging is not available with the **{transform.value}** "
                                      "transform, which needs the full configuration space.")
            pg.dg_avg(**{f"z{i}": True for i in avg_dirs})

        # 3) transform ------------------------------------------------------
        # gk-transport profiles are already evaluated on radial nodes.
        if load_mode.value == "gk-transport" and transform.value != "none":
            return None, "", (f"The **{transform.value}** transform does not apply to "
                                  "gk-transport profiles; set transform to 'none'.")
        if transform.value == "interpolate":
            pg.interpolate(interp=int(interp_pts.value) if interp_pts.value else None)
        elif transform.value == "dg-local-poly":
            pg.dg_local_poly(npoints=int(interp_pts.value) if interp_pts.value else 2)
        elif transform.value == "gk-rz":
            # Convert degrees from slider to radians for the backend projection
            phi_rad = float(phi_tor_val.value) * 3.141592653589793 / 180.0
            pg.gk_rz(
                mapc2p=_opt(mapc2p_file), 
                phi_tor=phi_rad, 
                nz_interp=int(interp_pts.value) if interp_pts.value else 8
            )
        elif transform.value == "gk-fluxsurf":
            pg.gk_fluxsurf(
                mapc2p=_opt(mapc2p_file), 
                x_idx=int(x_idx.value) if x_idx.value else 0,
                nz_interp=int(interp_pts.value) if interp_pts.value else 8
            )

        # 4) select (slice by coordinate value from the dynamic sliders) ----
        # dg-avg removes the averaged dimensions, shifting the later ones down.
        sel_kwargs = {}
        if grid_info.get("ok"):
            for i in range(grid_info["ndim"]):
                if sel_enables.value[i] and sel_modes.value[i] == "select":
                    j = i - sum(1 for a in avg_dirs if a < i)
                    sel_kwargs[f"z{j}"] = repr(float(sel_sliders.value[i]))
            if comp_enable.value:
                sel_kwargs["comp"] = str(int(comp_slider.value))
        if sel_kwargs:
            pg.select(**sel_kwargs)

        # 5) collect --------------------------------------------------------
        if _frames is not None and collect_chk.value:
            pg.collect()

        # active dimensionality (pgkyl only plots 1D/2D) --------------------
        dims = [dat.get_num_dims(squeeze=True) for dat in pg.data.iterator(None)]
        max_dim = max(dims) if dims else 0
        status = f"{len(dims)} dataset(s) &middot; {max_dim}D after processing"

        if max_dim > 2:
            need = max_dim - 2
            hint = (
                f"This data is **{max_dim}D**; pgkyl plots only 1D/2D. Enable "
                f"**{need}** more dimension(s) on the left to select or average it down."
            )
            return pg, status, hint
        return pg, status, None

    def clean_cmd(pg):
        c = pg.get_cmd()
        c = re.sub(r"\s--saveas \S+", "", c)
        return re.sub(r"\s--no-show", "", c)

    def plotted_range(pg):
        """(min, max, max dimensionality) of the values as plotted, i.e. after the
        y (1D) or z (2D) shift and scale, over every active dataset."""
        lo, hi, max_dim = np.inf, -np.inf, 0
        for dat in pg.data.iterator(None):
            nd = dat.get_num_dims(squeeze=True)
            max_dim = max(max_dim, nd)
            shift, scale = ((_num(zshift_t, 0.0), _num(zscale_t, 1.0)) if nd >= 2
                            else (_num(yshift_t, 0.0), _num(yscale_t, 1.0)))
            vals = (np.asarray(dat.get_values(), dtype=float) + shift)*scale
            if np.isfinite(vals).any():
                lo, hi = min(lo, np.nanmin(vals)), max(hi, np.nanmax(vals))
        return lo, hi, max_dim

    def plot_session(pg, png, **overrides):
        """Plot the processed session into png with the current plot options.
        overrides replace individual pgkyl plot options (e.g. title, ymin)."""
        plt.close("all")
        if os.path.exists(png):
            os.remove(png)
        opts = dict(
            figure="0",
            surface=surface.value,
            contour=contour.value,
            fixaspect=fixaspect.value,
            showgrid=showgrid.value,
            logx=logx.value,
            logy=logy.value,
            logz=logz.value,
            cmap=None if cmap.value == "(default)" else cmap.value,
            xlabel='',
            ylabel='',
            subplot_xlabels=_opt(xlabel),
            subplot_ylabels=_opt(ylabel),
            clabel=_opt(clabel),
            title=_opt(title),
            forcelegend=legend.value,
            no_legend=not legend.value,
            xmin=_num(xmin_t), xmax=_num(xmax_t),
            ymin=_num(ymin_t), ymax=_num(ymax_t),
            zmin=_num(cmin_t), zmax=_num(cmax_t),
            diverging=sym_chk.value,
            xshift=_num(xshift_t, 0.0), yshift=_num(yshift_t, 0.0),
            zshift=_num(zshift_t, 0.0),
            xscale=_num(xscale_t, 1.0), yscale=_num(yscale_t, 1.0),
            zscale=_num(zscale_t, 1.0),
            show=False,
            saveas=png,
        )
        opts.update({k: v for k, v in overrides.items() if v is not None})
        pg.plot(**opts)
        return open(png, "rb").read() if os.path.exists(png) else None

    def _run():
        _pg, _st, _e = process_frame()
        _c = clean_cmd(_pg) if _pg is not None else ""
        if _e or _pg is None:
            return None, _c, _st, _e
        _png = os.path.join(tempfile.gettempdir(), "pgkyl_marimo.png")
        _data = plot_session(_pg, _png)
        return _data, clean_cmd(_pg), _st, None

    try:
        _png_bytes, _cmd, _status, _err = _run()
        _tb = ""
    except Exception as exc:
        _png_bytes, _cmd, _status = None, "", ""
        _err = f"**{type(exc).__name__}:** {exc}"
        _tb = traceback.format_exc()

    if _cmd:
        # Wrapping <pre> so a long command never widens the figure pane.
        _cmd_md = mo.Html(
            "<div style='margin-top:0.5rem'>"
            "<b>Equivalent command line</b>"
            f"<pre style='white-space:pre-wrap;word-break:break-word;"
            "overflow-wrap:anywhere;background:var(--gray-2,#f3f3f3);"
            "padding:0.6rem 0.8rem;border-radius:6px;margin:0.3rem 0 0;"
            "font-family:var(--monospace-font,monospace);font-size:0.85em;'>"
            f"{html.escape(_cmd)}</pre></div>"
        )
    else:
        _cmd_md = mo.md("")

    if _err:
        _parts = [mo.callout(mo.md(_err), kind="warn")]
        if _tb:
            _parts.append(mo.accordion({"Full traceback": mo.md(f"```\n{_tb}\n```")}))
        _parts.append(_cmd_md)
        plot_view = mo.vstack(_parts)
    elif _png_bytes:
        # Embed as a sized <img> so the figure scales to the pane and never
        # overflows the window (mo.image renders at native pixel size).
        _b64 = base64.b64encode(_png_bytes).decode()
        _img = mo.Html(
            f'<img src="data:image/png;base64,{_b64}" '
            'style="max-width:100%;max-height:82vh;height:auto;'
            'object-fit:contain;display:block;margin:0 auto;" />'
        )
        plot_view = mo.vstack([mo.md(f"_{_status}_"), _img, _cmd_md])
    else:
        plot_view = mo.md("_No figure produced._")
    return pick_frames, plot_session, plot_view, plotted_range, process_frame


@app.cell
def _(
    collect_chk,
    comp_enable,
    comp_slider,
    dir_input,
    field_dropdown,
    frame_range,
    frame_slider,
    grid_info,
    header,
    interp_pts,
    mapc2p_file,
    mo,
    plot_view,
    sel_enables,
    sel_modes,
    sel_sliders,
    transform,
    plot_options,
    save_button,
    save_msg,
    save_name,
    movie_button,
    movie_file,
    movie_fixed,
    movie_fps,
    movie_frames,
    movie_msg,
    direction,
    extra,
    fluct,
    load_mode,
    quantity,
    simprefix,
    species,
    tr_output,
):
    # --- assemble select rows (one per grid dimension) ----------------------
    if grid_info.get("ok"):
        _rows = [
            mo.hstack(
                [sel_enables[i], mo.md(f"**z{i}**"), sel_modes[i], sel_sliders[i]],
                justify="start", align="center", gap=0.5, widths=[0.5, 0.6, 1.8, 6],
            )
            for i in range(grid_info["ndim"])
        ]
        _rows.append(
            mo.hstack([comp_enable, comp_slider], justify="start",
                    align="center", gap=0.5, widths=[1, 6])
        )
        _select_block = mo.vstack(_rows, gap=0.25)
    else:
        _select_block = mo.callout(
            mo.md(grid_info.get("msg", "Grid unavailable.")), kind="neutral"
        )

    # --- source block: plain field, or the gk-load-quantity controls --------
    if load_mode.value == "gk-load-quantity":
        _source_block = mo.vstack([
            quantity,
            simprefix,
            mo.hstack([species, direction], justify="start", gap=0.5, wrap=True),
            extra,
        ], gap=0.4)
    elif load_mode.value == "gk-transport":
        _source_block = mo.vstack([
            simprefix,
            mo.hstack([species, tr_output], justify="start", gap=0.5, wrap=True),
            extra,
            mo.md("_Radial profiles averaged over the flux surface and the frames "
                  "(frame range); tick **collect** for one profile per frame._"),
        ], gap=0.4)
    else:
        _source_block = field_dropdown

    # --- left: stack of all controls ----------------------------------------
    _controls = mo.vstack([
        header,
        # mo.md("#### 1 · Data"),
        dir_input,
        # mo.md("#### 2 · Load"),
        load_mode,
        _source_block,
        mo.hstack([frame_range, collect_chk], justify="start", align="center", gap=1),
        frame_slider,
        # mo.md("#### 3 · Processing"),
        fluct,
        mo.hstack([transform, interp_pts], justify="start", gap=1),
        mapc2p_file,
        x_idx if transform.value == "gk-fluxsurf" else mo.md(""),
        phi_tor_val if transform.value == "gk-rz" else mo.md(""),
        mo.md("**select / average** — enable a dimension to slice it at the slider's "
              "coordinate, or to average over it (`dg-avg`)"),
        _select_block,
        mo.md("#### Plot options"),
        plot_options,
        # mo.md("#### 5 · Save"),
        save_name,
        save_button,
        save_msg,
        mo.md("#### Movie"),
        mo.md("_Replays the current figure over the frames selected below "
              "(a slice of the available frames, like the frame range)._"),
        mo.hstack([movie_frames, movie_fps], justify="start", gap=0.5, wrap=True),
        movie_file,
        movie_fixed,
        movie_button,
        movie_msg,
    ], gap=0.6)

    # Two-pane layout built as a raw flex row so the panes are resizable:
    # drag the right edge of the controls panel (CSS `resize: horizontal`) to
    # widen/narrow it; the figure pane (`flex: 1; min-width: 0`) absorbs the
    # rest and never overflows. Embedding the elements' .text keeps them live,
    # exactly like interpolating a widget into `mo.md`.
    mo.Html(
        "<div style='display:flex;gap:1.25rem;align-items:flex-start;width:100%'>"
        "<div style='flex:0 0 auto;width:30rem;min-width:16rem;max-width:80vw;"
        "resize:horizontal;overflow:auto;max-height:92vh;padding:0 1rem 1rem 0;"
        "border-right:1px solid var(--gray-4,#ddd)'>"
        f"{_controls.text}</div>"
        "<div style='flex:1 1 0;min-width:0;position:sticky;top:0.5rem'>"
        f"{plot_view.text}</div>"
        "</div>"
    )
    return


@app.cell
def _(mo):
    # --- save controls ------------------------------------------------------
    save_name = mo.ui.text(
        value="", label="save as", placeholder="figure.png", full_width=True)
    save_button = mo.ui.run_button(label="Save figure")
    return save_button, save_name


@app.cell
def _(dir_input, mo, os, save_button, save_name, shutil, tempfile):
    # --- save action: copy the current figure PNG to the chosen path --------
    # Triggered only on click (run_button.value is True just for that run).
    if save_button.value:
        _png = os.path.join(tempfile.gettempdir(), "pgkyl_marimo.png")
        if not os.path.exists(_png):
            save_msg = mo.callout(mo.md("No figure to save yet."), kind="warn")
        else:
            _name = save_name.value.strip() or "figure.png"
            if not os.path.splitext(_name)[1]:
                _name += ".png"
            _dest = os.path.expanduser(_name)
            if not os.path.isabs(_dest):
                _base = os.path.expanduser(dir_input.value.strip()) or "."
                _dest = os.path.join(_base, _name)
            try:
                shutil.copyfile(_png, _dest)
                save_msg = mo.callout(mo.md(f"Saved to `{_dest}`"), kind="success")
            except Exception as exc:
                save_msg = mo.callout(mo.md(f"Save failed: {exc}"), kind="danger")
    else:
        save_msg = mo.md("")
    return (save_msg,)


@app.cell
def _(mo):
    # --- movie controls -----------------------------------------------------
    movie_frames = mo.ui.text(value=":", label="movie frames", placeholder=": | -100: | ::2")
    movie_fps = mo.ui.number(start=1, stop=60, value=10, label="fps")
    movie_file = mo.ui.text(
        value="", label="movie file", placeholder="movie.mp4 (or .gif)", full_width=True)
    movie_fixed = mo.ui.checkbox(
        value=True, label="same y / color range on every frame (blank limits only)")
    movie_button = mo.ui.run_button(label="Make movie")
    return movie_button, movie_file, movie_fixed, movie_fps, movie_frames


@app.cell
def _(
    cmax_t, cmin_t, dir_input, mo, movie_button, movie_file, movie_fixed, movie_fps,
    movie_frames, np, os, pick_frames, plot_session, plotted_range, process_frame, shutil,
    sym_chk, tempfile, title, ymax_t, ymin_t,
):
    # --- movie action: replay the figure chain frame by frame ---------------
    # Triggered only on click. Every frame goes through process_frame, the same
    # chain as the figure; the plots are then drawn with shared axis limits
    # (where the user left them blank) and assembled like 'pgkyl animate' does.
    def _movie():
        from types import SimpleNamespace
        from postgkyl.commands.animate import _compile_movie, VIDEO_EXTS

        sel = pick_frames(movie_frames.value, what="movie frames")
        if not sel:
            return mo.callout(mo.md("The movie frames select no frame."), kind="warn"), None

        name = movie_file.value.strip() or (
            "movie.mp4" if shutil.which("ffmpeg") else "movie.gif")
        if not os.path.splitext(name)[1]:
            name += ".mp4"
        ext = os.path.splitext(name)[1].lower()
        if ext in VIDEO_EXTS and shutil.which("ffmpeg") is None:
            return mo.callout(mo.md(f"ffmpeg is not available to write a `{ext}` movie; "
                                    "use a `.gif` file name instead."), kind="warn"), None
        dest = os.path.expanduser(name)
        if not os.path.isabs(dest):
            dest = os.path.join(os.path.expanduser(dir_input.value.strip()) or ".", name)

        # Pass 1: process every frame, keeping only the small plotted datasets.
        sessions, lo, hi, max_dim = [], np.inf, -np.inf, 0
        with mo.status.progress_bar(total=len(sel), title="Processing frames",
                                    remove_on_exit=True) as bar:
            for frame in sel:
                pg, _, err = process_frame(frame)
                if err or pg is None:
                    return mo.callout(mo.md(f"Frame {frame}: {err}"), kind="warn"), None
                active = list(pg.data.iterator(None))
                pg.data.clean()  # Keep only the plotted datasets in memory.
                for dat in active:
                    pg.data.add(dat)
                f_lo, f_hi, f_dim = plotted_range(pg)
                lo, hi, max_dim = min(lo, f_lo), max(hi, f_hi), max(max_dim, f_dim)
                time = active[0].ctx.get("time") if active else None
                sessions.append((frame, time, pg))
                bar.update()

        # Shared range: y limits for 1D plots, the color range for 2D ones.
        # Limits typed in the plot options always win, so only blank ones are set.
        limits = {}
        if movie_fixed.value and np.isfinite(lo) and hi > lo:
            def _blank(widget):
                return not (widget.value or "").strip()
            if max_dim >= 2 and sym_chk.value:
                # Symmetric: a typed cbar bound already fixes +-|bound| on every frame.
                if _blank(cmin_t) and _blank(cmax_t):
                    top = max(abs(lo), abs(hi))
                    limits = {"zmin": -top, "zmax": top}
            elif max_dim >= 2:
                limits = {k: v for k, v, w in (("zmin", lo, cmin_t), ("zmax", hi, cmax_t))
                          if _blank(w)}
            else:
                pad = 0.05*(hi - lo)
                limits = {k: v for k, v, w in (("ymin", lo - pad, ymin_t),
                                               ("ymax", hi + pad, ymax_t)) if _blank(w)}

        # Pass 2: plot every frame with the same options, then assemble.
        tmpdir = tempfile.mkdtemp(prefix="pgkyl_movie_")
        try:
            pngs, skipped = [], []
            with mo.status.progress_bar(total=len(sessions), title="Plotting frames",
                                        remove_on_exit=True) as bar:
                for i, (frame, time, pg) in enumerate(sessions):
                    stamp = f"frame {frame}" + (f",  t = {time:.4g}" if time is not None else "")
                    user_title = (title.value or "").strip()
                    png = os.path.join(tmpdir, f"frame_{i:05d}.png")
                    # pgkyl draws nothing for some data (e.g. an identically zero
                    # fluctuation at t = 0); leave such frames out of the movie.
                    if plot_session(pg, png, title=f"{user_title}   {stamp}" if user_title
                                    else stamp, **limits) is None:
                        skipped.append(frame)
                    else:
                        pngs.append(png)
                    bar.update()
            if not pngs:
                return mo.callout(mo.md("No frame produced a figure."), kind="warn"), None
            _compile_movie(pngs, dest, fps=movie_fps.value, duration=1000.0/movie_fps.value,
                           ctx=SimpleNamespace(obj={"verbose": False}))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        note = (f" Skipped {len(skipped)} frame(s) with nothing to plot: {skipped}."
                if skipped else "")
        msg = mo.callout(mo.md(f"Saved {len(pngs)} frames to `{dest}`.{note}"), kind="success")
        return msg, dest

    movie_msg = mo.md("")
    if movie_button.value:
        try:
            _msg, _dest = _movie()
            _parts = [_msg]
            # Preview the movie in place when it is small enough to embed.
            if _dest and os.path.getsize(_dest) < 50e6:
                if _dest.lower().endswith(".gif"):
                    _parts.append(mo.image(src=_dest))
                else:
                    _parts.append(mo.video(src=_dest, controls=True, loop=True))
            movie_msg = mo.vstack(_parts)
        except Exception as exc:
            movie_msg = mo.callout(mo.md(f"**{type(exc).__name__}:** {exc}"), kind="danger")
    return (movie_msg,)


if __name__ == "__main__":
    app.run()
