import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full", app_title="Postgkyl GUI")


@app.cell
def _():
    # --- imports ------------------------------------------------------------
    # The processing chain lives in postgkyl.gui.pipeline; these cells only
    # turn widget values into its Settings and its results into pixels.
    import marimo as mo

    import matplotlib

    matplotlib.use("Agg")  # Headless: figures are rendered to PNG bytes.

    import base64
    import dataclasses
    import html
    import math
    import os
    import shutil
    import tempfile
    import traceback

    import postgkyl as pg
    from postgkyl.cli import gui_launch as launch
    from postgkyl.gui import pipeline as gp

    return (
        base64, dataclasses, gp, html, launch, math, mo, os, pg, shutil,
        tempfile, traceback,
    )


@app.cell
def _(launch, mo, os):
    # --- data directory, from 'pgkyl-gui --path <dir>' or the default -------
    _cli_path = mo.cli_args().get("path")
    dir_input = mo.ui.text(
        value=os.path.expanduser(str(_cli_path)) if _cli_path else launch.default_path(),
        label="Data directory",
        full_width=True,
    )
    return (dir_input,)


@app.cell
def _(base64, gp, mo, os):
    # --- header logo ----------------------------------------------------------
    _logo = os.path.join(os.path.dirname(gp.__file__), "logo.png")
    if os.path.exists(_logo):
        with open(_logo, "rb") as _f:
            _b64 = base64.b64encode(_f.read()).decode()
        header = mo.Html(
            f"<img src='data:image/png;base64,{_b64}' alt='Postgkyl GUI' "
            "style='max-width:100%;height:auto;display:block;margin:0 0 0.5rem;' />"
        )
    else:
        header = mo.md("## Postgkyl GUI")
    return (header,)


@app.cell
def _(mo):
    # --- selections that survive a change of field or directory -------------
    get_field, set_field = mo.state(None)
    get_frame, set_frame = mo.state(None)
    get_sel_en, set_sel_en = mo.state([])
    get_sel_val, set_sel_val = mo.state([])
    get_sel_mode, set_sel_mode = mo.state([])
    get_comp_en, set_comp_en = mo.state(False)
    get_comp_val, set_comp_val = mo.state(0)
    get_xidx, set_xidx = mo.state(0)
    return (
        get_comp_en, get_comp_val, get_field, get_frame, get_sel_en,
        get_sel_mode, get_sel_val, get_xidx, set_comp_en, set_comp_val,
        set_field, set_frame, set_sel_en, set_sel_mode, set_sel_val, set_xidx,
    )


@app.cell
def _(dir_input, get_field, gp, mo, set_field):
    # --- field of 'file' mode -----------------------------------------------
    outputs = gp.scan_outputs(dir_input.value)
    _opts = sorted(outputs)
    if _opts:
        _prev = get_field()
        _default = _prev if _prev in _opts else next(
            (o for o in _opts if o.endswith("M0")), _opts[0])
        field_dropdown = mo.ui.dropdown(options=_opts, value=_default,
                                        label="field", searchable=True,
                                        on_change=set_field)
    else:
        field_dropdown = mo.ui.dropdown(options=[], label="field")
    return field_dropdown, outputs


@app.cell
def _(dir_input, gp, mo, pg):
    # --- load mode and the gyrokinetic loaders' controls --------------------
    load_mode = mo.ui.dropdown(
        options={"file": "file", "GK quantity": "quantity"},
        value="file", label="load")
    _quantities = pg.gk.available_quantities()
    quantity = mo.ui.dropdown(options=_quantities,
                              value="M0" if "M0" in _quantities else None,
                              label="quantity", searchable=True)
    _sims = gp.list_simulations(dir_input.value)
    simprefix = mo.ui.dropdown(options=_sims,
                               value=_sims[0] if _sims else None,
                               label="simulation", searchable=True)
    species = mo.ui.text(value="ion", label="species",
                         placeholder="ion, or elc,ion")
    direction = mo.ui.text(value="", label="direction", placeholder="0, 1 or 2")
    gk_options = mo.ui.text(value="", label="options", full_width=True,
                            placeholder="mass=... ti_over_te=2 den_ref=1e19,1e19")
    fluct = mo.ui.dropdown(options=["none", "y", "yz"], value="none",
                           label="fluctuation about")
    frame_range = mo.ui.text(value="", label="frame range",
                             placeholder=": | ::2 | -10: | -1")
    collect_chk = mo.ui.checkbox(label="collect (time series)")
    return (
        collect_chk, direction, fluct, frame_range, gk_options, load_mode,
        quantity, simprefix, species,
    )


@app.cell
def _(
    dir_input, field_dropdown, frame_range, get_frame, gp, load_mode, mo,
    outputs, quantity, set_frame, simprefix, species,
):
    # --- frames: from the file family, or from the GK registry --------------
    output = outputs.get(field_dropdown.value)
    if load_mode.value == "file":
        frames = output.frames if output else []
    elif simprefix.value and quantity.value:
        frames = gp.quantity_frames(dir_input.value, simprefix.value,
                                    quantity.value, species.value.strip() or None)
    else:
        frames = []

    if frames:
        frame_slider = mo.ui.slider(
            steps=frames, value=get_frame() if get_frame() in frames else frames[0],
            label="frame", show_value=True, include_input=True, full_width=True,
            on_change=set_frame, disabled=bool(frame_range.value.strip()))
    else:
        frame_slider = mo.ui.slider(steps=[0], value=0, label="frame",
                                    disabled=True)
    return frame_slider, frames, output


@app.cell
def _(mo):
    # --- transform controls -------------------------------------------------
    transform = mo.ui.dropdown(
        options=["interpolate", "local_poly", "map_to_rz",
                 "extract_flux_surface", "none"],
        value="interpolate", label="transform")
    interp_pts = mo.ui.number(start=1, stop=32, value=2,
                              label="points per cell")
    mapc2p_file = mo.ui.text(value="", label="mapc2p file (optional)",
                             full_width=True)
    phi_tor_deg = mo.ui.slider(start=0, stop=360, step=1, value=0,
                               label="toroidal angle (degrees)",
                               show_value=True, include_input=True,
                               full_width=True)
    return interp_pts, mapc2p_file, phi_tor_deg, transform


@app.cell
def _(
    collect_chk, dataclasses, dir_input, direction, fluct, frame_range,
    frame_slider, frames, gk_options, gp, load_mode, output, quantity,
    simprefix, species,
):
    # --- the settings every later cell refines ------------------------------
    def _base_settings():
        _picked = gp.pick_frames(frames, frame_range.value)
        if load_mode.value == "file" and output is not None and not output.frames:
            _frames = (None,)
        elif _picked is not None:
            _frames = tuple(_picked)
        else:
            _frames = (frame_slider.value,) if frames else ()
        if not _frames:
            raise ValueError("No frame to load: choose a field, or a simulation "
                             "and species with output for this quantity.")
        _dir = direction.value.strip()
        return gp.Settings(
            directory=dir_input.value.strip(),
            mode=load_mode.value,
            frames=_frames,
            available=tuple(frames),
            output=output,
            sim=output.sim if load_mode.value == "file" and output else simprefix.value,
            quantity=quantity.value,
            species=species.value.strip() or None,
            direction=int(_dir) if _dir else None,
            options=tuple(gp.parse_options(gk_options.value).items()),
            fluct=fluct.value,
            collect=collect_chk.value,
        )

    try:
        base_settings = _base_settings()
        settings_error = None
    except Exception as _exc:
        base_settings = None
        settings_error = f"{type(_exc).__name__}: {_exc}"
    replace = dataclasses.replace
    return base_settings, replace, settings_error


@app.cell
def _(base_settings, gp, pg, replace):
    # --- the raw layout: weight compatibility and the flux-surface index ----
    base_grid = None
    weight_path = None
    if base_settings is not None:
        try:
            base_grid = gp.probe(gp.build_chain(
                replace(base_settings, fluct="none", transform="none"),
                probe_only=True))
            _w = gp.weight_file(base_settings.directory, base_settings.sim or "")
            if _w and pg.load(_w).num_dims == base_grid.ndim:
                weight_path = _w
        except Exception:
            base_grid = None
    return base_grid, weight_path


@app.cell
def _(base_grid, get_xidx, mo, set_xidx):
    # --- radial cell of extract_flux_surface --------------------------------
    if base_grid is not None:
        _max = base_grid.cells[0] - 1
        x_idx = mo.ui.slider(start=0, stop=_max, step=1,
                             value=min(get_xidx(), _max),
                             label=f"flux surface x-index (0 to {_max})",
                             show_value=True, include_input=True,
                             full_width=True, on_change=set_xidx)
    else:
        x_idx = mo.ui.slider(start=0, stop=0, value=0,
                             label="flux surface x-index", disabled=True)
    return (x_idx,)


@app.cell
def _(
    base_grid, base_settings, gp, interp_pts, mapc2p_file, math, phi_tor_deg,
    replace, settings_error, transform, weight_path, x_idx,
):
    # --- settings through the transform, and the layout they produce --------
    def _transform_settings():
        _mapc2p = (mapc2p_file.value or "").strip() or None
        _points = int(interp_pts.value) if interp_pts.value else None
        _geometry = transform.value in ("map_to_rz", "extract_flux_surface")
        return replace(
            base_settings,
            weight=weight_path,
            source_ndim=base_grid.ndim if base_grid is not None else 0,
            transform=transform.value,
            num_interp=None if _geometry else _points,
            nz_interp=_points if _geometry else None,
            mapc2p=_mapc2p,
            phi_tor=math.radians(float(phi_tor_deg.value)),
            x_idx=int(x_idx.value),
        )

    if base_settings is None:
        transform_settings, grid_info = None, None
        grid_msg = settings_error or "Choose data to plot."
    else:
        transform_settings = _transform_settings()
        try:
            grid_info = gp.probe(gp.build_chain(transform_settings,
                                                probe_only=True))
            grid_msg = None
        except Exception as _exc:
            grid_info = None
            grid_msg = f"{type(_exc).__name__}: {_exc}"
    return grid_info, grid_msg, transform_settings


@app.cell
def _(get_sel_mode, grid_info, mo, set_sel_mode):
    # --- per-dimension mode: select (slice) or average ----------------------
    if grid_info is not None:
        _prev = get_sel_mode()
        sel_modes = mo.ui.array(
            [mo.ui.dropdown(options=["select", "average"],
                            value=_prev[i] if i < len(_prev) else "select")
             for i in range(grid_info.ndim)],
            on_change=lambda vals: set_sel_mode(list(vals)))
    else:
        sel_modes = mo.ui.array([])
    return (sel_modes,)


@app.cell
def _(
    get_comp_en, get_comp_val, get_sel_en, get_sel_mode, get_sel_val,
    grid_info, mo, set_comp_en, set_comp_val, set_sel_en, set_sel_val,
):
    # --- selection sliders sized from the processed layout ------------------
    # Limits follow the current layout; the positions and enable flags persist
    # across fields (clamped). Every dimension beyond the first two is sliced
    # by default, so 3D+ data land on a 2D plot.
    def _clamp(v, lo, up):
        return min(max(v, lo), up)

    if grid_info is not None:
        _prev_en, _prev_val, _prev_mode = get_sel_en(), get_sel_val(), get_sel_mode()

        def _slider(i):
            lo, up, n = grid_info.lower[i], grid_info.upper[i], grid_info.cells[i]
            step = 1.0 if grid_info.curvilinear[i] else (up - lo) / max(n, 1)
            val = _clamp(float(_prev_val[i]), lo, up) if i < len(_prev_val) else lo + (up - lo) / 2.0
            if grid_info.curvilinear[i]:
                val = float(round(val))
            return mo.ui.slider(
                start=lo, stop=up, step=step, value=val, show_value=True,
                include_input=True, full_width=True,
                disabled=i < len(_prev_mode) and _prev_mode[i] == "average")

        sel_sliders = mo.ui.array([_slider(i) for i in range(grid_info.ndim)],
                                  on_change=lambda vals: set_sel_val(list(vals)))
        sel_enables = mo.ui.array(
            [mo.ui.checkbox(value=bool(_prev_en[i]) if i < len(_prev_en) else i >= 2)
             for i in range(grid_info.ndim)],
            on_change=lambda vals: set_sel_en(list(vals)))
        _cmax = max(grid_info.num_fields - 1, 0)
        comp_enable = mo.ui.checkbox(value=get_comp_en(), label="component",
                                     on_change=set_comp_en)
        comp_slider = mo.ui.slider(start=0, stop=max(_cmax, 1), step=1,
                                   value=_clamp(int(get_comp_val()), 0, _cmax),
                                   show_value=True, include_input=True,
                                   disabled=_cmax == 0, on_change=set_comp_val)
    else:
        sel_sliders = sel_enables = mo.ui.array([])
        comp_enable = mo.ui.checkbox(value=False, label="component")
        comp_slider = mo.ui.slider(start=0, stop=1, value=0, disabled=True)
    return comp_enable, comp_slider, sel_enables, sel_sliders


@app.cell
def _(mo):
    # --- plot options (named as pg.plot's) ----------------------------------
    surface = mo.ui.checkbox(label="surface")
    contour = mo.ui.checkbox(label="contour")
    contourf = mo.ui.checkbox(label="contourf")
    fixaspect = mo.ui.checkbox(label="fix aspect")
    showgrid = mo.ui.checkbox(label="grid")
    logx = mo.ui.checkbox(label="logx")
    logy = mo.ui.checkbox(label="logy")
    logz = mo.ui.checkbox(label="logz")
    legend = mo.ui.checkbox(label="legend")
    diverging = mo.ui.checkbox(label="diverging cmap")
    # Contour levels: evenly spaced over the colour range with cbar bounds or a
    # diverging cmap (centred on 0), else about this many automatic levels.
    nlevels = mo.ui.number(start=2, stop=200, step=1, value=11,
                           label="contour levels")
    cmap = mo.ui.dropdown(
        options=["(default)", "viridis", "plasma", "inferno", "cividis",
                 "twilight", "RdBu_r", "jet", "gray"],
        value="(default)", label="cmap")
    # A ui.dictionary, so marimo tracks every field (a plain dict is inert).
    texts = mo.ui.dictionary({
        key: mo.ui.text(value="", label=label)
        for key, label in (("xlabel", "xlabel"), ("ylabel", "ylabel"),
                           ("clabel", "clabel"), ("title", "title"),
                           ("xmin", "x min"), ("xmax", "x max"),
                           ("ymin", "y min"), ("ymax", "y max"),
                           ("zmin", "cbar min"), ("zmax", "cbar max"),
                           ("xshift", "x shift"), ("yshift", "y shift"),
                           ("zshift", "z shift"), ("xscale", "x scale"),
                           ("yscale", "y scale"), ("zscale", "z scale"))
    })

    def _row(*keys):
        return mo.hstack([texts[k] for k in keys], justify="start", gap=0.5,
                         wrap=True)

    plot_options_view = mo.vstack([
        mo.hstack([surface, contour, contourf, fixaspect, showgrid, logx,
                   logy, logz, legend, diverging], justify="start", gap=0.75,
                  wrap=True),
        cmap,
        _row("xlabel", "ylabel", "clabel", "title"),
        mo.md("**limits** _(blank = auto)_"),
        _row("xmin", "xmax", "ymin", "ymax"),
        _row("zmin", "zmax"),
        nlevels,
        mo.md("**shift / scale** _(z = value or colour axis; blank = none)_"),
        _row("xshift", "yshift", "zshift"),
        _row("xscale", "yscale", "zscale"),
    ], gap=0.4)
    return (
        cmap, contour, contourf, diverging, fixaspect, legend, logx, logy,
        logz, nlevels, plot_options_view, showgrid, surface, texts,
    )


@app.cell
def _(
    cmap, contour, contourf, diverging, fixaspect, legend, logx, logy, logz,
    nlevels, showgrid, surface, texts,
):
    # --- pg.plot keyword options from the widgets ---------------------------
    def plot_options(**overrides):
        def _num(key):
            v = (texts.value[key] or "").strip()
            return float(v) if v else None

        def _text(key):
            return (texts.value[key] or "").strip() or None

        opts = dict(
            surface=surface.value, contour=contour.value,
            contourf=contourf.value,
            cnlevels=(int(nlevels.value) if contour.value or contourf.value
                      else None),
            fixaspect=fixaspect.value, no_showgrid=not showgrid.value,
            logx=logx.value, logy=logy.value, logz=logz.value,
            no_legend=not legend.value, forcelegend=legend.value,
            diverging=diverging.value,
            cmap=None if cmap.value == "(default)" else cmap.value,
            # Blank axes unless labels are typed, then on every subplot.
            xlabel="", ylabel="",
            subplot_xlabels=_text("xlabel"), subplot_ylabels=_text("ylabel"),
            # Given (blank unless typed), so the z scale never changes it.
            clabel=_text("clabel") or "", title=_text("title"),
            **{k: _num(k) for k in ("xmin", "xmax", "ymin", "ymax", "zmin",
                                    "zmax", "xshift", "yshift", "zshift",
                                    "xscale", "yscale", "zscale")},
        )
        opts.update(overrides)
        # Leave defaults implicit, so the shown script states only choices.
        return {k: v for k, v in opts.items() if v is not None and v is not False}

    return (plot_options,)


@app.cell
def _(
    comp_enable, comp_slider, grid_info, replace, sel_enables, sel_modes,
    sel_sliders, transform_settings,
):
    # --- final settings: selections and averages ----------------------------
    if transform_settings is not None and grid_info is not None:
        _on = [i for i in range(grid_info.ndim) if sel_enables.value[i]]
        settings = replace(
            transform_settings,
            ndim=grid_info.ndim,
            average=tuple(i for i in _on if sel_modes.value[i] == "average"),
            select=tuple(
                (i, int(sel_sliders.value[i]) if grid_info.curvilinear[i]
                 else float(sel_sliders.value[i]))
                for i in _on if sel_modes.value[i] == "select"),
            comp=int(comp_slider.value) if comp_enable.value else None,
        )
    else:
        settings = None
    return (settings,)


@app.cell
def _(base64, gp, grid_msg, mo, plot_options, settings, traceback):
    # --- run the chain; show the figure, its Python and its command line ----
    figure_bytes, _err, _tb, _status = None, None, "", ""
    _script, _command = "", ""
    if settings is None:
        _err = grid_msg or "Choose data to plot."
    else:
        try:
            _steps = gp.build_chain(settings)
            _script = gp.python_script(_steps)
            _session = gp.run(_steps)
            _command = _session.command()
            _data = _session.datasets
            _dims = max(len([n for n in d.values.shape[:-1] if n > 1]) for d in _data)
            _status = f"{len(_data)} dataset(s) · {_dims}D after processing"
            if _dims > 2:
                _err = (f"This data is **{_dims}D** and plots are 1D or 2D: enable "
                        f"**{_dims - 2}** more dimension(s) to select or average.")
            else:
                _plot = gp.plot_step(plot_options(), _data)
                gp.apply(_session, _plot)
                _script = gp.python_script(_steps + (_plot,))
                _command = _session.command()
                figure_bytes = gp.figure_png(_session.result)
        except Exception as _exc:
            _err = f"**{type(_exc).__name__}:** {_exc}"
            _tb = traceback.format_exc()

    def _code_view(title, code, language):
        # A fenced block, which marimo gives a copy-to-clipboard button; long
        # lines wrap instead of scrolling out of view.
        if not code:
            return mo.md("")
        _block = mo.md(f"**{title}**\n\n```{language}\n{code}\n```").text
        return mo.Html(
            "<style>.pgkyl-code pre{white-space:pre-wrap;"
            "word-break:break-all}</style>"
            f"<div class='pgkyl-code'>{_block}</div>")

    _code = [_code_view("Equivalent Python", _script, "python"),
             _code_view("Equivalent command line", _command, "bash")]
    if _err:
        _parts = [mo.callout(mo.md(_err), kind="warn")]
        if _tb:
            _parts.append(mo.accordion({"Full traceback": mo.md(f"```\n{_tb}\n```")}))
        plot_view = mo.vstack(_parts + _code)
    else:
        _b64 = base64.b64encode(figure_bytes).decode()
        plot_view = mo.vstack([
            mo.md(f"_{_status}_"),
            mo.Html(f'<img src="data:image/png;base64,{_b64}" style="max-width:100%;'
                    'max-height:82vh;height:auto;object-fit:contain;display:block;'
                    'margin:0 auto;" />'),
            *_code,
        ])
    return figure_bytes, plot_view


@app.cell
def _(mo):
    # --- save and movie controls --------------------------------------------
    save_name = mo.ui.text(value="", label="save as", placeholder="figure.png",
                           full_width=True)
    save_button = mo.ui.run_button(label="Save figure")
    movie_frames = mo.ui.text(value=":", label="movie frames",
                              placeholder=": | -100: | ::2")
    movie_fps = mo.ui.number(start=1, stop=60, value=10, label="fps")
    movie_file = mo.ui.text(value="", label="movie file", full_width=True,
                            placeholder="movie.mp4 (or .gif)")
    movie_fixed = mo.ui.checkbox(value=True, label="same y / colour range on "
                                 "every frame (blank limits only)")
    movie_button = mo.ui.run_button(label="Make movie")
    return (movie_button, movie_file, movie_fixed, movie_fps, movie_frames,
            save_button, save_name)


@app.cell
def _(dir_input, figure_bytes, mo, os, save_button, save_name):
    # --- save the shown figure ----------------------------------------------
    def destination(name, default):
        name = name.strip() or default
        if not os.path.splitext(name)[1]:
            name += os.path.splitext(default)[1]
        path = os.path.expanduser(name)
        return path if os.path.isabs(path) else os.path.join(
            os.path.expanduser(dir_input.value.strip()) or ".", path)

    save_msg = mo.md("")
    if save_button.value:
        if figure_bytes is None:
            save_msg = mo.callout(mo.md("No figure to save yet."), kind="warn")
        else:
            _dest = destination(save_name.value, "figure.png")
            try:
                with open(_dest, "wb") as _f:
                    _f.write(figure_bytes)
                save_msg = mo.callout(mo.md(f"Saved to `{_dest}`"), kind="success")
            except OSError as _exc:
                save_msg = mo.callout(mo.md(f"Save failed: {_exc}"), kind="danger")
    return save_msg, destination


@app.cell
def _(
    destination, frames, gp, mo, movie_button, movie_file, movie_fixed,
    movie_fps, movie_frames, os, plot_options, replace, settings, shutil,
):
    # --- movie: the shown chain, frame by frame, through pg.animate ---------
    def _movie():
        _picked = gp.pick_frames(frames, movie_frames.value)
        if not _picked:
            return mo.callout(mo.md("The movie frames select no frame."), kind="warn"), None
        _default = "movie.mp4" if shutil.which("ffmpeg") else "movie.gif"
        _dest = destination(movie_file.value, _default)
        _groups = []
        with mo.status.progress_bar(total=len(_picked), title="Processing frames",
                                    remove_on_exit=True) as _bar:
            for _frame in _picked:
                _groups.append(gp.processed(gp.build_chain(
                    replace(settings, frames=(_frame,), collect=False))))
                _bar.update()
        gp.make_movie(_groups, _dest, fps=int(movie_fps.value),
                      plot_options=plot_options(), fixed_range=movie_fixed.value)
        return mo.callout(mo.md(f"Saved {len(_groups)} frames to `{_dest}`."),
                          kind="success"), _dest

    movie_msg = mo.md("")
    if movie_button.value:
        if settings is None:
            movie_msg = mo.callout(mo.md("Choose data to plot first."), kind="warn")
        else:
            try:
                _msg, _dest = _movie()
                _parts = [_msg]
                if _dest and os.path.getsize(_dest) < 50e6:
                    _parts.append(mo.image(src=_dest) if _dest.lower().endswith(".gif")
                                  else mo.video(src=_dest, controls=True, loop=True))
                movie_msg = mo.vstack(_parts)
            except Exception as _exc:
                movie_msg = mo.callout(mo.md(f"**{type(_exc).__name__}:** {_exc}"),
                                       kind="danger")
    return (movie_msg,)


@app.cell
def _(
    collect_chk, comp_enable, comp_slider, dir_input, direction, field_dropdown,
    fluct, frame_range, frame_slider, gk_options, grid_info, grid_msg, header,
    interp_pts, load_mode, mapc2p_file, mo, movie_button, movie_file,
    movie_fixed, movie_fps, movie_frames, movie_msg, phi_tor_deg, plot_options_view,
    plot_view, quantity, save_button, save_msg, save_name, sel_enables,
    sel_modes, sel_sliders, simprefix, species, transform, x_idx,
):
    # --- layout: controls on a resizable left pane, the figure on the right -
    if grid_info is not None:
        _rows = [
            mo.hstack([sel_enables[i], mo.md(f"**z{i}**"), sel_modes[i], sel_sliders[i]],
                      justify="start", align="center", gap=0.5,
                      widths=[0.5, 0.6, 1.8, 6])
            for i in range(grid_info.ndim)
        ]
        _rows.append(mo.hstack([comp_enable, comp_slider], justify="start",
                               align="center", gap=0.5, widths=[1, 6]))
        _select_block = mo.vstack(_rows, gap=0.25)
    else:
        _select_block = mo.callout(mo.md(grid_msg or "Layout unavailable."),
                                   kind="neutral")

    if load_mode.value == "quantity":
        _source = mo.vstack([quantity, simprefix,
                             mo.hstack([species, direction], justify="start",
                                       gap=0.5, wrap=True), gk_options], gap=0.4)
    else:
        _source = field_dropdown

    _controls = mo.vstack([
        header,
        dir_input,
        load_mode,
        _source,
        mo.hstack([frame_range, collect_chk], justify="start", align="center", gap=1),
        frame_slider,
        fluct,
        mo.hstack([transform, interp_pts], justify="start", gap=1),
        mapc2p_file,
        x_idx if transform.value == "extract_flux_surface" else mo.md(""),
        phi_tor_deg if transform.value == "map_to_rz" else mo.md(""),
        mo.md("**select / average**: enable a dimension to slice it at the "
              "slider's coordinate, or to average over it"),
        _select_block,
        mo.md("#### Plot options"),
        plot_options_view,
        save_name,
        save_button,
        save_msg,
        mo.md("#### Movie"),
        mo.md("_Replays the figure over the selected frames (a slice of the "
              "available frames, like the frame range)._"),
        mo.hstack([movie_frames, movie_fps], justify="start", gap=0.5, wrap=True),
        movie_file,
        movie_fixed,
        movie_button,
        movie_msg,
    ], gap=0.6)

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


if __name__ == "__main__":
    app.run()
