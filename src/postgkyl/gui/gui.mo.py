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
    import html
    import os
    import re
    import shutil
    import tempfile
    import traceback

    from postgkyl.clap import PgkylSession

    return (
        PgkylSession, base64, glob, html, mo, os, plt, re, shutil,
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
    refresh = mo.ui.run_button(label="Reset figure")
    _default_dir = "tests/test_data"
    _cli_path = mo.cli_args().get("path")
    _start_dir = os.path.expanduser(str(_cli_path)) if _cli_path else _default_dir
    dir_input = mo.ui.text(
        value=_start_dir,
        label="Data directory",
        placeholder="path to a Gkeyll data directory",
        full_width=True,
    )
    return dir_input, refresh


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

    gk_quant_list = sorted(gk_quant_registry.list())

    def gk_extra(direction, extra):
        """Combine the direction + free-form extra fields into one `extra` arg
        (e.g. `dir=1,mass=0.1`). Returns None when both are blank."""
        parts = []
        if direction.value.strip():
            parts.append(f"dir={direction.value.strip()}")
        if extra.value.strip():
            parts.append(extra.value.strip())
        return ",".join(parts) or None

    return gk_extra, gk_quant_list, gk_quant_registry, list_prefixes


@app.cell
def _(gk_quant_list, mo):
    # --- load mode: static controls -----------------------------------------
    load_mode = mo.ui.dropdown(
        options=["load", "gk-load-quantity"], value="load", label="load mode")
    quantity = mo.ui.dropdown(
        options=gk_quant_list,
        value=("M0" if "M0" in gk_quant_list else (gk_quant_list[0] if gk_quant_list else None)),
        label="quantity", searchable=True)
    species = mo.ui.text(value="ion", label="species", placeholder="ion/elc/...")
    direction = mo.ui.text(value="", label="direction", placeholder="dir, e.g. 0/1/2")
    extra = mo.ui.text(
        value="", label="extra", placeholder="mass=...,charge=...", full_width=True)
    return direction, extra, load_mode, quantity, species


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
    get_comp_en, set_comp_en = mo.state(False)
    get_comp_val, set_comp_val = mo.state(0)
    
    # New state variables for interface persistence
    get_field, set_field = mo.state(None)
    get_xidx, set_xidx = mo.state(0)
    return (
        get_comp_en, get_comp_val, get_field, get_frame, get_sel_en,
        get_sel_val, get_xidx, set_comp_en, set_comp_val, set_field,
        set_frame, set_sel_en, set_sel_val, set_xidx,
    )


@app.cell
def _(mo):
    # --- all-frames / collect (field-independent, so they never reset) ------
    all_frames = mo.ui.checkbox(label="all frames")
    collect_chk = mo.ui.checkbox(label="collect (time series)")
    return all_frames, collect_chk


@app.cell
def _(
    dir_input, field_dropdown, get_frame, gk_quant_registry, load_mode, mo,
    outputs, quantity, set_frame, simprefix, species,
):
    # --- frame selection: slider with limits from the detected frames -------
    # In gk-load-quantity mode the available frames come from the registry
    # (which combination of source files exists); otherwise from the field.
    if load_mode.value == "gk-load-quantity":
        info = {"type": "gk", "frames": []}
        frames = []
        if quantity.value and simprefix.value:
            try:
                frames = gk_quant_registry.get(quantity.value).get_avail_frames(
                    dir_input.value.strip().rstrip("/") + "/", simprefix.value,
                    species.value.strip() or None)
            except Exception:
                frames = []
    else:
        info = outputs.get(field_dropdown.value, {"type": "static", "frames": []})
        frames = info["frames"]

    if frames:
        _fval = get_frame() if get_frame() in frames else frames[0]
        frame_slider = mo.ui.slider(
            steps=frames, value=_fval, label="frame", show_value=True,
            include_input=True, full_width=True, on_change=set_frame,
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
def _(
    get_comp_en, get_comp_val, get_sel_en, get_sel_val, grid_info, mo,
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

        def _slider(i, d):
            span = d["up"] - d["lo"]
            step = span / max(d["n"], 1)
            if i < len(_prev_val):
                val = _clamp(float(_prev_val[i]), d["lo"], d["up"])
            else:
                val = d["lo"] + span / 2.0
            return mo.ui.slider(
                start=d["lo"], stop=d["up"], step=step, value=val,
                show_value=True, include_input=True, full_width=True,
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
    showgrid = mo.ui.checkbox(label="grid")
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
        mo.hstack([cmin_t, cmax_t], justify="start", gap=0.5, wrap=True),
        mo.md("**shift / scale** _(z = value / colorbar axis)_"),
        mo.hstack([xshift_t, yshift_t, zshift_t], justify="start", gap=0.5, wrap=True),
        mo.hstack([xscale_t, yscale_t, zscale_t], justify="start", gap=0.5, wrap=True),
    ], gap=0.4)
    return (
        clabel, cmap, cmax_t, cmin_t, contour, fixaspect, legend, logx, logy,
        logz, plot_options, showgrid, surface, title, xlabel, xmax_t,
        xmin_t, xscale_t, xshift_t, ylabel, ymax_t, ymin_t, yscale_t, yshift_t,
        zscale_t, zshift_t,
    )


@app.cell
def _(
    PgkylSession,
    all_frames,
    base64,
    clabel,
    cmap,
    cmax_t,
    cmin_t,
    collect_chk,
    dir_input,
    direction,
    extra,
    gk_extra,
    html,
    comp_enable,
    comp_slider,
    contour,
    field_dropdown,
    fixaspect,
    frame_slider,
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
    os,
    plt,
    quantity,
    re,
    refresh,
    sel_enables,
    sel_sliders,
    showgrid,
    simprefix,
    species,
    surface,
    tempfile,
    title,
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
    refresh.value  # clicking the refresh button forces this cell to re-run

    def _opt(widget):
        v = (widget.value or "").strip()
        return v or None

    def _num(widget, default=None):
        v = (widget.value or "").strip()
        return float(v) if v else default

    def _run():
        # 1) load -----------------------------------------------------------
        plt.close("all")
        pg = PgkylSession()

        if load_mode.value == "gk-load-quantity":
            if not (quantity.value and simprefix.value):
                return None, "", "", "Choose a quantity and a simulation prefix."
            _frame = ":" if all_frames.value else str(frame_slider.value)
            pg.gk_load_quantity(
                quantity=quantity.value, name=simprefix.value,
                path=dir_input.value.strip(), frame=_frame,
                species=species.value.strip() or None,
                extra=gk_extra(direction, extra))
        else:
            if not field_dropdown.value:
                return None, "", "", "Select a data directory and field on the left."
            if info["type"] == "series":
                src = (f"{info['stem']}_[0-9]*.gkyl" if all_frames.value
                       else f"{info['stem']}_{frame_slider.value}.gkyl")
            else:
                src = info["path"]
            pg.load(src)

        # 2) transform ------------------------------------------------------
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

        # 3) select (slice by coordinate value from the dynamic sliders) ----
        sel_kwargs = {}
        if grid_info.get("ok"):
            for i in range(grid_info["ndim"]):
                if sel_enables.value[i]:
                    sel_kwargs[f"z{i}"] = repr(float(sel_sliders.value[i]))
            if comp_enable.value:
                sel_kwargs["comp"] = str(int(comp_slider.value))
        if sel_kwargs:
            pg.select(**sel_kwargs)

        # 4) collect --------------------------------------------------------
        if all_frames.value and collect_chk.value:
            pg.collect()

        # active dimensionality (pgkyl only plots 1D/2D) --------------------
        dims = [dat.get_num_dims(squeeze=True) for dat in pg.data.iterator(None)]
        max_dim = max(dims) if dims else 0
        status = f"{len(dims)} dataset(s) &middot; {max_dim}D after processing"

        def _clean_cmd():
            c = pg.get_cmd()
            c = re.sub(r"\s--saveas \S+", "", c)
            return re.sub(r"\s--no-show", "", c)

        if max_dim > 2:
            need = max_dim - 2
            hint = (
                f"This data is **{max_dim}D**; pgkyl plots only 1D/2D. Enable "
                f"**{need}** more `select` slider(s) on the left to slice it down."
            )
            return None, _clean_cmd(), status, hint

        # 5) plot -> PNG ----------------------------------------------------
        png = os.path.join(tempfile.gettempdir(), "pgkyl_marimo.png")
        if os.path.exists(png):
            os.remove(png)
        pg.plot(
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
            xshift=_num(xshift_t, 0.0), yshift=_num(yshift_t, 0.0),
            zshift=_num(zshift_t, 0.0),
            xscale=_num(xscale_t, 1.0), yscale=_num(yscale_t, 1.0),
            zscale=_num(zscale_t, 1.0),
            show=False,
            saveas=png,
        )
        data = open(png, "rb").read() if os.path.exists(png) else None
        return data, _clean_cmd(), status, None

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
    return (plot_view,)


@app.cell
def _(
    all_frames,
    collect_chk,
    comp_enable,
    comp_slider,
    dir_input,
    field_dropdown,
    frame_slider,
    grid_info,
    header,
    interp_pts,
    mapc2p_file,
    mo,
    plot_view,
    refresh,
    sel_enables,
    sel_sliders,
    transform,
    plot_options,
    save_button,
    save_msg,
    save_name,
    direction,
    extra,
    load_mode,
    quantity,
    simprefix,
    species,
):
    # --- assemble select rows (one per grid dimension) ----------------------
    if grid_info.get("ok"):
        _rows = [
            mo.hstack(
                [sel_enables[i], mo.md(f"**z{i}**"), sel_sliders[i]],
                justify="start", align="center", gap=0.5, widths=[0.5, 0.6, 6],
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
    else:
        _source_block = field_dropdown

    # --- left: stack of all controls ----------------------------------------
    _controls = mo.vstack([
        header,
        # mo.md("#### 1 · Data"),
        refresh,
        dir_input,
        # mo.md("#### 2 · Load"),
        load_mode,
        _source_block,
        mo.hstack([all_frames, collect_chk], justify="start", gap=1),
        frame_slider,
        # mo.md("#### 3 · Processing"),
        mo.hstack([transform, interp_pts], justify="start", gap=1),
        mapc2p_file,
        x_idx if transform.value == "gk-fluxsurf" else mo.md(""),
        phi_tor_val if transform.value == "gk-rz" else mo.md(""),
        mo.md("**select** — enable a dimension to slice it at the slider's coordinate"),
        _select_block,
        mo.md("#### Plot options"),
        plot_options,
        # mo.md("#### 5 · Save"),
        save_name,
        save_button,
        save_msg,
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


if __name__ == "__main__":
    app.run()
