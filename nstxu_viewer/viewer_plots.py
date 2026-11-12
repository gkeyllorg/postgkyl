"""Figures of the NSTX-U run viewer (nstxu_viewer.py), light or dark. The poloidal plane and the unrolled strips are
drawn with matplotlib (the cells are curvilinear quadrilaterals); profiles, target heat flux and time traces with
plotly (hover readouts, zoom).

Colours and font: the theme of the APS DPP 2025 poster (its .pptx slide-master theme, read here as
gyro_grid_20261018/poster/plot_suite_poster.py reads it; that module imports scipy, which this venv does not have),
in PowerPoint's tone steps (L10 = lighter 10 %, D25 = darker 25 %, ...). As on the posters: inferno for magnitudes
(the DPP poster's n_e and T_e maps), the progress posters' slate-blue <-> crimson map for signed values and changes,
the theme's orange and teal for the targets. The dark look is the same theme on its slate (dk2 D50).

Validated with the data-viz validator on each look's surface (white; #1B262B):
- targets: hue = side (outer accent2 D25, inner hlink), line style = upper (dashed, squares) or lower (solid,
  circles). Two hues: all-pairs CVD dE 19.0, normal 26.6 in both looks. Four theme hues fail all-pairs (accent2 and
  accent3 coincide under CVD, dE 0.6; the theme's blues read gray), so the near-symmetric upper/lower pair shares a hue;
- species: electrons hlink, ions accent1 (dark: accent1 L10): all-pairs CVD 11.0 / 9.2, normal 29.7 / 28.3;
- regions: core accent1, SOL accent3 D50 (dark: D40), private flux hlink: dark passes all-pairs; light has
  crimson/gold at CVD 7.9 (the 6-8 band), so the region lines also carry direct end labels.
"""
import colorsys
import os
import re
import zipfile
from dataclasses import dataclass

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
from matplotlib import font_manager, patheffects  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize  # noqa: E402
import numpy as np  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

import gkview as gv  # noqa: E402
import pgkyl_cmd as pc  # noqa: E402
import thomson  # noqa: E402

# metric-compatible stand-ins, used only where the theme font is not installed (Arial is not, on Perlmutter)
STANDIN = {'Arial': 'Liberation Sans', 'Helvetica': 'Liberation Sans'}


def read_theme(pptx):
    """Colour scheme {slot: hex} and body (minor) font of the poster's slide-master theme."""
    with zipfile.ZipFile(pptx) as z:
        rels = z.read('ppt/slideMasters/_rels/slideMaster1.xml.rels').decode()
        xml = z.read('ppt/' + re.search(r'Target="\.\./(theme/theme\d+\.xml)"', rels).group(1)).decode()
    scheme = re.search(r'<a:clrScheme.*?</a:clrScheme>', xml, re.S).group(0)
    colours = dict(re.findall(r'<a:(\w+)><a:(?:srgbClr val|sysClr [^>]*lastClr)="([0-9A-Fa-f]{6})"', scheme))
    return colours, re.search(r'<a:minorFont><a:latin typeface="([^"]+)"', xml).group(1)


# the page's themes: each a PowerPoint file's slide-master theme (colour scheme and body font, as read_theme reads it),
# kept here so the viewer needs none of the files; check_viewer.py re-reads those viewer.json's "theme_files" names.
# NSTX-U: the DPP 2025 poster; PPPL: the lab's 2025 slide template; DIII-D: the DIII-D talks (negD_CEDA_aug28.pptx;
# bernard_poster_aps25.pdf, the same Office scheme), plus 'navy': their title bar's colour, which the slides set
# directly, not a scheme slot.
THEME_SOURCES = {'NSTX-U': 'aps_dpp_2025_poster_jjosephyzack.pptx', 'PPPL': 'PPPL Powerpoint Slide Template.pptx',
                 'DIII-D': 'negD_CEDA_aug28.pptx'}
SCHEMES = {
    'NSTX-U': (dict(dk1='000000', lt1='FFFFFF', dk2='364C56', lt2='E8E8E8', accent1='C63264', accent2='F57F24',
                    accent3='FFDC00', accent4='364C56', accent5='698E9F', accent6='8DBBCF', hlink='009AAD',
                    folHlink='C95773'), 'Arial'),
    'PPPL': (dict(dk1='FFFFFF', lt1='212121', dk2='29404B', lt2='98B8C7', accent1='F58025', accent2='008C99',
                  accent3='7AE68C', accent4='E7FF37', accent5='AE2659', accent6='FFDD00', hlink='4DD0E1',
                  folHlink='4DD0E1'), 'Arial'),
    'DIII-D': (dict(dk1='000000', lt1='FFFFFF', dk2='1F497D', lt2='EEECE1', accent1='4F81BD', accent2='C0504D',
                    accent3='9BBB59', accent4='8064A2', accent5='4BACC6', accent6='F79646', hlink='0000FF',
                    folHlink='800080', navy='00217B'), 'Century Gothic'),
}
THEME, FONT = SCHEMES['NSTX-U']
RENDER_FONT = FONT if FONT in {f.name for f in font_manager.fontManager.ttflist} else STANDIN.get(FONT, 'DejaVu Sans')
CSS_FONT = ', '.join(f"'{f}'" for f in dict.fromkeys([FONT, STANDIN.get(FONT, FONT)])) + ', Helvetica, sans-serif'
STEPS = {'': (1, 0), 'L60': (0.4, 0.6), 'L40': (0.6, 0.4), 'L20': (0.8, 0.2), 'L10': (0.9, 0.1), 'D10': (0.9, 0),
         'D25': (0.75, 0), 'D40': (0.6, 0), 'D50': (0.5, 0), 'D75': (0.25, 0)}


def tone(slot, step='', theme='NSTX-U'):
    """A theme colour, or a PowerPoint colour-picker step of it (HLS lightness scaled, as plot_suite_poster.tone)."""
    mod, off = STEPS[step]
    h, l, s = colorsys.rgb_to_hls(*(int(SCHEMES[theme][0][slot][i:i + 2], 16) / 255 for i in (0, 2, 4)))
    return '#' + ''.join(f'{round(255 * c):02X}' for c in colorsys.hls_to_rgb(h, min(1.0, l * mod + off), s))


@dataclass(frozen=True)
class Look:
    name: str
    surf: str
    ink: str
    ink2: str
    muted: str
    grid: str
    axis: str
    outer: str        # outer targets
    inner: str        # inner targets
    elc: str
    ion: str
    core: str
    sol: str
    pf: str
    div: tuple        # diverging steps, negative pole -> neutral -> positive pole
    theme: str = 'NSTX-U'
    mode: str = ''    # 'light' | 'dark' ('' : the name)
    accent: str = ''  # the page's controls (the movie's play button, its ring outline, the probe); '': hlink

    @property
    def dark(self):
        return (self.mode or self.name) == 'dark'

    @property
    def plotly_template(self):
        return 'plotly_dark' if self.dark else 'plotly_white'


LOOKS = {
    # the posters: white page, black ink, lt2 grays; the progress posters' diverging map
    'light': Look('light', '#FFFFFF', tone('dk1'), tone('lt2', 'D50'), tone('lt2', 'D25'), tone('lt2'),
                  tone('lt2', 'D25'), tone('accent2', 'D25'), tone('hlink'), tone('hlink'), tone('accent1'),
                  tone('accent1'), tone('accent3', 'D50'), tone('hlink'),
                  (tone('accent4', 'D25'), tone('accent6', 'D25'), tone('accent6', 'L40'), tone('lt2', 'L40'),
                   tone('accent1', 'L60'), tone('accent1'), tone('accent1', 'D50'))),
    # the same theme on its slate: dk2 D50 surface, lt2 inks, poles light and the neutral dark
    'dark': Look('dark', tone('dk2', 'D50'), tone('lt2'), tone('lt2', 'D25'), tone('lt2', 'D50'), tone('dk2', 'D25'),
                 tone('dk2'), tone('accent2', 'D25'), tone('hlink'), tone('hlink'), tone('accent1', 'L10'),
                 tone('accent1', 'L10'), tone('accent3', 'D40'), tone('hlink'),
                 (tone('accent6', 'L40'), tone('accent6', 'D25'), tone('accent6', 'D50'), tone('lt2', 'D75'),
                  tone('accent1', 'D25'), tone('accent1'), tone('accent1', 'L60'))),
}
if 'PPPL' in SCHEMES:
    def _p(slot, step=''):
        return tone(slot, step, 'PPPL')
    # The PPPL 2025 template ('Simple Dark': text dk1 white on slate dk2, titles accent1 orange; its 'White' slides
    # the light look). The poster looks' roles: targets orange (outer) / teal (inner), electrons and private flux
    # teal, ions and core crimson, SOL olive (accent6 D40 in both looks); diverging teal <-> orange about a gray.
    # Validated (data-viz validator, all pairs, each on its surface): targets CVD dE 16.5, species and regions 8.1
    # dark / 10.2 light, normal vision >= 18.7, every mark >= 3:1. The slides' own slate (dk2) is too light a
    # surface for marks inside the dark lightness band to reach 3:1: the dark charts sit on dk2 D25 (the slides'
    # gradient at its dark end), with dk2 for the cards.
    LOOKS['pppl-light'] = Look('pppl-light', '#FFFFFF', _p('lt1'), _p('dk2'), _p('lt2', 'D25'), _p('lt2', 'L60'),
                               _p('lt2'), _p('accent1', 'D25'), _p('accent2'), _p('accent2'), _p('accent5'),
                               _p('accent5'), _p('accent6', 'D40'), _p('accent2'),
                               (_p('accent2', 'D50'), _p('accent2'), _p('hlink', 'L40'), '#F3F3F3',
                                _p('accent1', 'L40'), _p('accent1'), _p('accent1', 'D50')),
                               theme='PPPL', mode='light', accent=_p('accent1', 'D25'))
    LOOKS['pppl-dark'] = Look('pppl-dark', _p('dk2', 'D25'), _p('dk1'), _p('lt2'), _p('lt2', 'D40'), _p('dk2'),
                              _p('dk2', 'L20'), _p('accent1', 'D25'), _p('accent2'), _p('accent2'), _p('accent5', 'L20'),
                              _p('accent5', 'L20'), _p('accent6', 'D40'), _p('accent2'),
                              (_p('hlink', 'L40'), _p('accent2'), _p('accent2', 'D50'), _p('lt1', 'L10'),
                               _p('accent1', 'D50'), _p('accent1'), _p('accent1', 'L60')),
                              theme='PPPL', mode='dark', accent=_p('accent1'))
if 'DIII-D' in SCHEMES:
    def _d(slot, step=''):
        return tone(slot, step, 'DIII-D')
    # The DIII-D talks: Office 2007 scheme, Century Gothic, white slides under a navy title bar (white titles; body
    # text tx2 = dk2), red / blue for the two cases compared, orange for what a slide stresses. The poster looks'
    # roles: targets orange (outer) / blue (inner), electrons and private flux blue, ions and core red, SOL green;
    # diverging blue <-> red about a gray. Validated (data-viz validator, all pairs, each on its surface): light
    # targets CVD dE 21.8, species 20.0, regions 15.1; dark 21.8 / 18.5 / 8.3 (its red one step lighter, accent2
    # D10); normal vision >= 20.6; every mark >= 3:1. The dark look sits on the title bar's navy at D50 (no passing
    # set on its lighter steps), the cards on D25.
    LOOKS['diii-d-light'] = Look('diii-d-light', '#FFFFFF', _d('dk1'), _d('dk2'), _d('dk2', 'L60'), _d('lt2'),
                                 _d('lt2', 'D10'), _d('accent6', 'D25'), _d('accent1'), _d('accent1'),
                                 _d('accent2', 'D25'), _d('accent2', 'D25'), _d('accent3', 'D25'), _d('accent1'),
                                 (_d('dk2'), _d('accent1'), _d('dk2', 'L60'), _d('lt1', 'D10'), _d('accent2', 'L60'),
                                  _d('accent2'), _d('accent2', 'D50')),
                                 theme='DIII-D', mode='light', accent=_d('navy'))
    LOOKS['diii-d-dark'] = Look('diii-d-dark', _d('navy', 'D50'), _d('lt1'), _d('dk2', 'L60'), _d('dk2', 'L20'),
                                _d('dk2', 'D25'), _d('dk2'), _d('accent6', 'D25'), _d('accent1'), _d('accent1'),
                                _d('accent2', 'D10'), _d('accent2', 'D10'), _d('accent3', 'D25'), _d('accent1'),
                                (_d('dk2', 'L60'), _d('accent1'), _d('dk2', 'D25'), _d('lt1', 'D75'), _d('accent2', 'D50'),
                                 _d('accent2'), _d('accent2', 'L60')),
                                theme='DIII-D', mode='dark', accent=_d('accent6'))
THEME_LOOKS = {name: {'light': LOOKS[f'{key}-light' if key else 'light'], 'dark': LOOKS[f'{key}-dark' if key else 'dark']}
               for name, key in (('NSTX-U', ''), ('PPPL', 'pppl'), ('DIII-D', 'diii-d')) if name in SCHEMES}


def look(theme='NSTX-U', dark=True):
    """The look of a theme (THEME_LOOKS; an unknown one: NSTX-U), light or dark."""
    return THEME_LOOKS.get(theme, THEME_LOOKS['NSTX-U'])['dark' if dark else 'light']


SEQ = 'inferno'
REGION_NAME = {'core': 'core', 'sol': 'SOL', 'pf': 'private flux'}


def _page(theme):
    """The page's look in a theme: fonts (text, headings, an @import), heading styles, the light mode's accents
    and the dark mode's surfaces (marimo's CSS variables)."""
    if theme == 'PPPL':
        d, li = LOOKS['pppl-dark'], LOOKS['pppl-light']
        # the template's 'Fonts in Use': Roboto Condensed for titles (a Google font; Arial Narrow, Arial otherwise)
        head = "'Roboto Condensed', 'Arial Narrow', " + CSS_FONT
        return dict(src=THEME_SOURCES['PPPL'], text=CSS_FONT, head=head,
                    imp="https://fonts.googleapis.com/css2?family=Roboto+Condensed:wght@400;700;900&display=swap",
                    light=dict(primary=li.accent, link=li.inner), h_light=f'color: {li.accent};', h_dark=f'color: {d.accent};',
                    h2='', dark=dict(background=d.surf, card=d.grid, popover=d.grid, muted=d.surf, border=d.axis,
                                     input=tone('dk2', 'L10', 'PPPL'), foreground=d.ink, muted_foreground=d.ink2,
                                     primary=d.accent, accent=d.grid, link=tone('hlink', '', 'PPPL')))
    if theme == 'DIII-D':
        d, li = LOOKS['diii-d-dark'], LOOKS['diii-d-light']
        # Century Gothic, as the talks (a Microsoft font: geometric look-alikes where it is not installed)
        text = "'Century Gothic', 'URW Gothic', 'Avant Garde', Futura, " + CSS_FONT
        # a level-2 heading as the slides' title bar: the navy gradient of the rendered deck (darker at its ends)
        bar = (f'background: linear-gradient(90deg, #172250, #21409A 50%, #172250); color: {tone("lt1", "", "DIII-D")}; '
               'padding: 0.2em 0.6em; border-radius: 2px;')
        return dict(src=THEME_SOURCES['DIII-D'], text=text, head=text, imp='',
                    light=dict(primary=li.accent, link='#0070C0'), h_light='color: inherit;', h_dark='color: inherit;',
                    h2=bar, dark=dict(background=d.surf, card=tone('navy', 'D25', 'DIII-D'),
                                      popover=tone('navy', 'D25', 'DIII-D'), muted=d.surf, border=d.axis,
                                      input=tone('navy', '', 'DIII-D'), foreground=d.ink, muted_foreground=d.ink2,
                                      primary=d.accent, accent=tone('navy', 'D25', 'DIII-D'), link=d.ink2))
    d = LOOKS['dark']
    return dict(src=THEME_SOURCES['NSTX-U'], text=CSS_FONT, head=CSS_FONT, imp='',
                light=dict(primary=tone('hlink', 'D25'), link=tone('hlink', 'D25')), h_light='color: inherit;',
                h_dark='color: inherit;', h2='',
                dark=dict(background=d.surf, card=tone('dk2', 'D25'), popover=tone('dk2', 'D25'), muted=tone('dk2', 'D50'),
                          border=tone('dk2'), input=tone('accent5', 'D25'), foreground=tone('lt2'),
                          muted_foreground=tone('lt2', 'D25'), primary=tone('hlink'), accent=tone('dk2'),
                          link=tone('hlink', 'L20')))


def page_css(theme='NSTX-U'):
    """CSS for the marimo page in a theme (_page): its fonts and headings, its light-mode accents, and in dark mode
    its surfaces (body.dark is marimo's). The page puts the chosen theme's into the document at run time
    (nstxu_viewer.py); view.sh writes the NSTX-U one to runs/viewer.css, the first paint. Every theme sets every
    rule, so switching from one to another leaves nothing behind."""
    t = _page(theme if theme in SCHEMES else 'NSTX-U')
    var = lambda d: '\n'.join(f'  --{k.replace("_", "-")}: {v};' for k, v in d.items())  # noqa: E731
    dark = dict(t['dark'])
    dark['card_foreground'] = dark['popover_foreground'] = dark['foreground']
    return f"""/* NSTX-U run viewer, {theme} theme ({os.path.basename(t['src'])}), written by viewer_plots.page_css */
{f"@import url('{t['imp']}');" if t['imp'] else ''}
:root {{
  --marimo-text-font: {t['text']};
  --marimo-heading-font: {t['head']};
}}
.markdown :is(h1, h2, h3, h4) {{ font-family: {t['head']}; }}
.markdown h2 {{ {t['h2'] or 'background: none; padding: 0; color: inherit;'} }}
:root:has(body:not(.dark)) {{
{var(t['light'])}
}}
:root:has(body:not(.dark)) .markdown :is(h1, h3) {{ {t['h_light']} }}
:root:has(body:not(.dark)) .markdown h2 {{ {'' if t['h2'] else t['h_light']} }}
:root:has(body.dark) {{
{var(dark)}
}}
:root:has(body.dark) .markdown :is(h1, h3) {{ {t['h_dark']} }}
:root:has(body.dark) .markdown h2 {{ {'' if t['h2'] else t['h_dark']} }}
"""


def species_colour(pal, s):
    return {'elc': pal.elc, 'ion': pal.ion}.get(s, pal.ink2)


def target_style(pal, tg):
    """Hue by side (outer / inner), dash and marker by upper / lower."""
    up = tg.updown == 'upper'
    return dict(color=pal.outer if tg.side == 'outer' else pal.inner, dash='dash' if up else 'solid',
                symbol='square' if up else 'circle', lam_dash='dashdot' if up else 'dot')


def mpl_style(pal):
    return {'figure.facecolor': pal.surf, 'axes.facecolor': pal.surf, 'savefig.facecolor': pal.surf,
            'text.color': pal.ink, 'axes.labelcolor': pal.ink2, 'xtick.color': pal.ink2, 'ytick.color': pal.ink2,
            'axes.edgecolor': pal.axis, 'font.family': [RENDER_FONT], 'font.size': 10.5, 'axes.titlesize': 12,
            'axes.titlecolor': pal.ink}


def _scale(steps):
    return [[i / (len(steps) - 1), c] for i, c in enumerate(steps)]


def time_unit(t_max):
    """(scale, label) of the time axes: us for runs shorter than 1 ms."""
    return (1e6, 'µs') if t_max < 1e-3 else (1e3, 'ms')


def nice_limit(x):
    """Smallest 1, 2, 2.5 or 5 x 10^n >= x (a colour limit; for a relative change 1 = 100 %)."""
    if not np.isfinite(x) or x <= 0:
        return 1.0
    e = 10.0 ** np.floor(np.log10(x))
    return float(next(m * e for m in (1, 2, 2.5, 5, 10) if x <= m * e * (1 + 1e-12)))


SINCE = {'change': 'the first frame', 'step': 'the previous frame'}   # modes other than 'value'


def values(run, fr, fr0, qty, mode, fr_prev=None):
    """block -> values to colour: the quantity in its unit, or its change since the first frame ('change') or the
    previous frame ('step'): relative for positive quantities, a difference in the unit for signed ones."""
    _, _, unit, scale, kind = gv.quantities(run)[qty]
    ref = fr_prev if mode == 'step' and fr_prev is not None else fr0
    out = {}
    for b in range(run.nb):
        v, v0 = fr[b][qty], ref[b][qty]
        with np.errstate(divide='ignore', invalid='ignore'):
            if mode == 'value':
                out[b] = v * scale
            elif kind == 'pos':
                out[b] = v / v0 - 1.0
            else:
                out[b] = (v - v0) * scale
    return out


def mode_values(F, rows, mode, kind, scale):
    """The shown values of cached frames F (frames x cells) at rows: the quantity in its unit, or its change since
    the run's first frame ('change') or the previous written frame ('step'), relative for positive quantities."""
    rows = np.asarray(rows)
    with np.errstate(divide='ignore', invalid='ignore'):
        if mode == 'value':
            return F[rows] * scale
        ref = F[np.maximum(rows - 1, 0)] if mode == 'step' else F[:1]
        return F[rows] / ref - 1.0 if kind == 'pos' else (F[rows] - ref) * scale


def fixed_scale(run, V, qty, mode, log, step_first=False):
    """One colour scale over a set of frames' values V (a per-frame scale makes colours jump between frames):
    dict(cmap 'seq' | 'div', norm, ticks, fmt, label, extend). 99.5th percentile; step_first = the first row is a
    'step' change against itself (zero), left out."""
    label, sym, unit, scale, kind = gv.quantities(run)[qty]
    rel = mode != 'value' and kind == 'pos'
    u = f' ({unit})' if unit else ''
    fin = V[1:] if step_first and len(V) > 1 else V
    fin = fin[np.isfinite(fin)]
    if mode == 'value' and kind == 'pos':
        hi = float(np.percentile(fin, 99.5)) if len(fin) else 1.0
        if log:
            pos = fin[fin > 0]
            lo = max(float(np.percentile(pos, 0.5)) if len(pos) else hi * 1e-4, hi * 1e-4)
            norm, ticks = LogNorm(lo, hi), matplotlib.ticker.LogLocator(subs=(1.0, 2.0, 5.0)).tick_values(lo, hi)
        else:
            norm, ticks = Normalize(0.0, hi), matplotlib.ticker.MaxNLocator(5).tick_values(0.0, hi)
        cmap, cblabel = 'seq', f'{label}{u}'
    else:
        lim = float(np.percentile(np.abs(fin), 99.5)) if len(fin) else 1.0
        lim = nice_limit(lim) if mode != 'value' else (lim or 1.0)
        norm = Normalize(-lim, lim)
        ticks = matplotlib.ticker.MaxNLocator(5, symmetric=True).tick_values(-lim, lim)
        cmap = 'div'
        cblabel = f'{label}{u}' if mode == 'value' else f'{sym} change since {SINCE[mode]}' + ('' if rel else u)
    lo_, hi_ = norm.vmin, norm.vmax
    ticks = [t for t in ticks if lo_ - 1e-12 * abs(hi_) <= t <= hi_ * (1 + 1e-12) + 1e-300]
    fmt = (lambda t: f'{t * 100:+.3g} %' if t else '0') if rel else (lambda t: f'{t:.4g}')
    return dict(cmap=cmap, norm=norm, ticks=ticks, fmt=fmt, label=cblabel, rel=rel,
                extend=[bool((fin < lo_).any()), bool((fin > hi_).any())])


def colour_norm(run, vals, qty, mode, whole_run, log, summ, pal, cells=None):
    """(cmap, norm, colour-bar label, tick formatter, extend) for the map. cells: the shown cells (cell_offsets
    order), for the whole-run scale of a block selection (None: the run's percentiles, every cell)."""
    label, sym, unit, scale, kind = gv.quantities(run)[qty]
    u = f' ({unit})' if unit else ''
    div = LinearSegmentedColormap.from_list(f'div_{pal.name}', pal.div)
    allv = np.concatenate([v.ravel() for v in vals.values()])
    allv = allv[np.isfinite(allv)]
    whole = whole_run and summ is not None and f'pct|{qty}' in summ
    pct = None
    if whole:
        if cells is None:
            pct = summ[f'pct|{qty}']
        else:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)   # a frame with no finite value in the selection
                pct = np.nanpercentile(summ[f'fld|{qty}'][:, cells].astype(float), [0, 1, 50, 99, 100], axis=1).T
    if mode == 'value' and kind == 'pos':
        if whole:
            p = pct * scale
            lo, hi = np.nanmin(p[:, 1]), np.nanmax(p[:, 3])
        else:
            lo, hi = np.percentile(allv, [1, 99]) if len(allv) else (0.0, 1.0)
        if log:
            lo = max(lo, hi * 1e-4, np.min(allv[allv > 0]) if (allv > 0).any() else hi * 1e-4)
            return SEQ, LogNorm(lo, hi), f'{label}{u}', None, 'both'
        return SEQ, Normalize(0.0, hi), f'{label}{u}', None, 'max'
    if mode == 'value' or kind != 'pos':
        if whole:
            p = pct * scale
            lim = max(abs(np.nanmin(p[:, 1])), abs(np.nanmax(p[:, 3])))
        else:
            lim = np.percentile(np.abs(allv), 99) if len(allv) else 1.0
        lab = f'{label}{u}' if mode == 'value' else f'{sym} change since {SINCE[mode]}{u}'
        return div, Normalize(-lim, lim), lab, None, 'both'
    lim = 1.0 if whole_run and mode == 'change' else nice_limit(np.percentile(np.abs(allv), 99) if len(allv) else 1.0)
    fmt = lambda x, _: f'{x * 100:+.3g} %' if x else '0'  # noqa: E731
    return div, Normalize(-lim, lim), f'{sym} change since {SINCE[mode]}', fmt, 'both'


def shown_blocks(run, blocks=None):
    """The blocks to draw: the chosen ones (None or none chosen: every block), in block order."""
    out = [b for b in range(run.nb) if blocks is None or b in blocks]
    return out or list(range(run.nb))


def zoom_boxes(run, blocks=None):
    """name -> (R0, R1, Z0, Z1) views of the poloidal plane, from the layout ('whole domain': the chosen blocks)."""
    nd = run.nodes
    allp = np.concatenate([nd[b].reshape(-1, 2) for b in shown_blocks(run, blocks)])

    def box(pts, pad=0.08, square=False):
        r0, z0 = pts.min(0); r1, z1 = pts.max(0)
        dr, dz = r1 - r0, z1 - z0
        if square:   # midplane views: a narrow radial slab, widened to a readable aspect
            dz = dr = max(dr, dz)
            zc = 0.5 * (z0 + z1); z0, z1 = zc - dz / 2, zc + dz / 2
        return (r0 - pad * dr, r1 + pad * dr, z0 - pad * dz, z1 + pad * dz)

    out = {'whole domain': box(allp, 0.03)}
    for ud in ('lower', 'upper'):
        div = [b for t in run.layout.targets if t.updown == ud for b in (t.leg, t.pf) if b is not None]
        if div:
            out[f'{ud} divertor'] = box(np.concatenate([nd[b].reshape(-1, 2) for b in div]))
    for side, st in run.layout.stacks.items():
        R = np.concatenate([gv.mid_edges(nd[b], st.j) for b in gv.stack_blocks(st)])
        h = 0.6 * np.ptp(R)
        out[f'{side} midplane'] = box(np.array([[R.min(), -h], [R.max(), h]]), 0.1, square=True)
    return out


def map_figure(run, fr, fr0, qty, mode='value', whole_run=False, log=False, summ=None, zoom='whole domain', title='',
               pal=LOOKS['dark'], fr_prev=None, scale=None, blocks=None, vals=None):
    """Poloidal plane (true R-Z, wall, separatrix) + every cell unrolled; x = cells with n <= 0 (any species).
    scale: a fixed_scale() dict (one colour scale over a movie's frames), else colour_norm's. blocks: draw only these
    (None: all; the colour scale and the 'whole domain' view are theirs). vals: block -> values to draw instead of the
    quantity (with scale; no n <= 0 marks), e.g. the diffusivity."""
    shown = shown_blocks(run, blocks)
    sel = None if len(shown) == run.nb else set(shown)
    if vals is None:
        vals = values(run, fr, fr0, qty, mode, fr_prev)
    vals = {b: vals[b] for b in shown}
    if scale is None:
        cells = None if sel is None else gv.region_cells(run, sel)
        cmap, norm, cblabel, fmt, extend = colour_norm(run, vals, qty, mode, whole_run, log, summ, pal, cells)
    else:
        cmap = SEQ if scale['cmap'] == 'seq' else LinearSegmentedColormap.from_list(f'div_{pal.name}', pal.div)
        norm, cblabel, extend = scale['norm'], scale['label'], {(0, 0): 'neither', (1, 0): 'min', (0, 1): 'max',
                                                                (1, 1): 'both'}[tuple(map(int, scale['extend']))]
        fmt = (lambda x, _: scale['fmt'](x)) if scale['rel'] else None
    halo = [patheffects.withStroke(linewidth=2.4, foreground=pal.surf)]   # x marks readable on light and dark cells
    lay = run.layout
    strips = gv.select_strips(lay, sel)
    nstrip = max(len(strips), 1)
    bad = {b: np.zeros(vals[b].shape, bool) for b in shown}
    if fr is not None and qty is not None:
        for b in shown:
            for s in run.species:
                bad[b] |= fr[b][f'n_{s}'] <= 0
    height = {1: 5.2, 2: 6.4}.get(nstrip, 6.4 + 1.7 * (nstrip - 2))
    # a title (the movie's frames) takes a band of its own above the strips' titles: 13 pt, 6 pt from the top edge
    title_band = (6 + 13 * 1.25 + 6) / 72 / height if title else 0.0
    with plt.rc_context(mpl_style(pal)):
        fig = plt.figure(figsize=(15.5, height), dpi=110)
        gs = fig.add_gridspec(nstrip + 1, 2, width_ratios=[1.0, 3.3], height_ratios=[1] * nstrip + [0.09],
                              wspace=0.14, hspace=0.62 if nstrip >= 2 else 0.45, left=0.05, right=0.985,
                              top=(0.92 if nstrip >= 2 else 0.86) - title_band, bottom=0.08 if nstrip >= 2 else 0.12)
        ax = fig.add_subplot(gs[:, 0])
        kw = dict(cmap=cmap, norm=norm, shading='flat')
        for b in shown:
            nd = run.nodes[b]
            m = ax.pcolormesh(nd[..., 0], nd[..., 1], np.ma.masked_invalid(vals[b]), edgecolors='none', **kw)
            outline = np.concatenate([nd[0, :], nd[:, -1], nd[-1, ::-1], nd[::-1, 0]])
            ax.plot(outline[:, 0], outline[:, 1], color=pal.muted, lw=0.35, alpha=0.7)
            if lay.region[b] == 'sol' and lay.under.get(b) is not None:   # its inner edge: the separatrix
                ax.plot(nd[0, :, 0], nd[0, :, 1], color=pal.ink, lw=0.8, ls=(0, (4, 2)))
        w = gv.wall(run)
        if w is not None:
            ax.plot(w[:, 0], w[:, 1], color=pal.ink2, lw=1.0)
        pts = np.array([gv.cell_centres(run.nodes[b])[i, j] for b in shown for i, j in np.argwhere(bad[b])])
        if len(pts):
            ax.scatter(pts[:, 0], pts[:, 1], marker='x', s=16, lw=0.9, color=pal.ink, zorder=5, path_effects=halo)
        boxes = zoom_boxes(run, sel)
        r0, r1, z0, z1 = boxes.get(zoom, boxes['whole domain'])
        ax.set_xlim(r0, r1); ax.set_ylim(z0, z1); ax.set_aspect('equal')
        # few, short tick labels: a zoomed view's 0.0025 m steps ('−0.0075') ran into each other and pushed the Z
        # label off the figure
        for axis, n in ((ax.xaxis, 4), (ax.yaxis, 5)):
            axis.set_major_locator(matplotlib.ticker.MaxNLocator(n, steps=[1, 2, 2.5, 5, 10]))
            axis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:.6g}'.replace('-', '−')))
        ax.set_xlabel('R (m)'); ax.set_ylabel('Z (m)', labelpad=1)
        ax.set_title('poloidal plane' + ('' if zoom == 'whole domain' else f': {zoom}') +
                     ('' if sel is None else f' (blocks {block_list(shown)})'), loc='left')
        for s in ax.spines.values():
            s.set_visible(False)

        axs = [fig.add_subplot(gs[i, 1]) for i in range(nstrip)]
        for axk, st in zip(axs, strips):
            for b, x0, band in st.items:
                nr, nt = vals[b].shape
                y0, h = gv.BANDS[band]
                X, Y = np.meshgrid(x0 + np.arange(nt + 1), y0 + h * np.arange(nr + 1) / nr)
                axk.pcolormesh(X, Y, np.ma.masked_invalid(vals[b]), edgecolors=pal.surf,
                               linewidth=0.3 if nr <= 8 else 0.0, **kw)
                yc = y0 + h * (np.arange(nr) + 0.5) / nr
                p = [(x0 + j + 0.5, yc[i]) for i, j in np.argwhere(bad[b])]
                if p:
                    axk.scatter(*np.array(p).T, marker='x', s=18, lw=0.9, color=pal.ink, zorder=5, path_effects=halo)
            if st.sep is not None:
                axk.axhline(st.sep, color=pal.ink, lw=0.9, ls=(0, (4, 2)))
            for xa, xb, lab in st.segments:
                axk.text(0.5 * (xa + xb), 8.25, lab, ha='center', va='bottom', fontsize=9.5, color=pal.ink2)
                if xa > 0:
                    axk.axvline(xa, color=pal.muted, lw=0.6)
            if st.mirror_x is not None:
                axk.axvline(st.mirror_x, color=pal.ink, lw=1.4)
                # x runs right to left (below): the strip's end is at the left
                axk.text(st.mirror_x, -0.35, 'midplane', va='top', fontsize=9, color=pal.ink,
                         ha='left' if st.mirror_x >= st.width - 0.5 else 'right' if st.mirror_x <= 0.5 else 'center')
            # reflected left-right: turned a quarter turn (outboard strip clockwise, inboard anticlockwise) a strip
            # lies as its cells do in the poloidal plane (upper leg up, core inside, SOL outside)
            axk.set_xlim(st.width, 0); axk.set_ylim(0, 8)
            axk.set_yticks([y for _, y in st.ylabels]); axk.set_yticklabels([lab for lab, _ in st.ylabels])
            axk.tick_params(axis='y', length=0); axk.set_xticks([])
            axk.set_title(st.title, loc='left', pad=24)
            for s in axk.spines.values():
                s.set_visible(False)
        cax = fig.add_subplot(gs[nstrip, 1])
        cb = fig.colorbar(m, cax=cax, orientation='horizontal', extend=extend)
        cb.outline.set_visible(False)
        if fmt:
            cb.formatter = matplotlib.ticker.FuncFormatter(fmt); cb.update_ticks()
        cb.set_label(cblabel + ('   (x: n ≤ 0)' if len(pts) else ''), color=pal.ink2)
        if title:
            fig.text(0.05, 1 - 6 / 72 / height, title, fontsize=13, color=pal.ink, va='top')
    return fig


def block_list(blocks):
    """'0-3, 7, 9-10' for a list of block numbers."""
    b = sorted(set(int(x) for x in blocks))
    out, i = [], 0
    while i < len(b):
        j = i
        while j + 1 < len(b) and b[j + 1] == b[j] + 1:
            j += 1
        out.append(f'{b[i]}' if i == j else f'{b[i]}–{b[j]}')
        i = j + 1
    return ', '.join(out)


def value_scale(V, label, log=False):
    """A fixed_scale()-like dict for a field of positive values that is not a run quantity (the diffusivity): 0 to the
    largest value (log: over its span when that is wide)."""
    fin = np.asarray(V, float)
    fin = fin[np.isfinite(fin)]
    hi = float(fin.max()) if len(fin) and fin.max() > 0 else 1.0
    lo = float(fin[fin > 0].min()) if (fin > 0).any() else hi
    if log and lo < hi:
        norm, ticks = LogNorm(lo, hi), matplotlib.ticker.LogLocator(subs=(1.0, 2.0, 5.0)).tick_values(lo, hi)
    else:
        norm, ticks = Normalize(0.0, hi * 1.0000001 if hi == lo else hi), matplotlib.ticker.MaxNLocator(5).tick_values(0, hi)
    return dict(cmap='seq', norm=norm, ticks=[t for t in ticks if norm.vmin <= t <= norm.vmax], fmt=lambda t: f'{t:.4g}',
                label=label, rel=False, extend=[False, False])


# ------------------------------------------------------------------------------------------------------------ plotly
# The band above a chart holds the legend and the subplot titles only. Plotly widens the top margin to hold its
# legend (its rows wrap at the plot width) but never for a free annotation, which is clipped at the figure's edge or
# drawn over the legend: so what a line style means is a legend entry of its own (legend_key()), and a longer note
# goes on the page next to the chart.
SUBTITLE = 13      # subplot titles (make_subplots' 16 overran narrow columns)
LEGEND_GAP = 28    # px from the legend's lowest row to the plots' top: clear of the subplot titles (13 px, 1.3 lines)
END_LABEL_COLUMN = 310   # px of plot per column of the time traces on a ~760 px wide chart (sizes the end labels' room)
_FT = {}


def text_px(s, size=12):
    """Width (px) of s in the theme font (its metric twin where the theme's is not installed); HTML tags dropped."""
    from matplotlib.ft2font import FT2Font, LoadFlags
    path = font_manager.findfont(font_manager.FontProperties(family=RENDER_FONT))
    f = _FT.get(path) or _FT.setdefault(path, FT2Font(path))
    f.set_size(size, 72)
    f.set_text(re.sub(r'<[^>]+>', '', s), 0.0, flags=LoadFlags.NO_HINTING)
    return f.get_width_height()[0] / 64.0


def subplots(**kw):
    """make_subplots, its subplot titles at SUBTITLE px."""
    fig = make_subplots(**kw)
    for a in fig.layout.annotations:
        a.font.size = SUBTITLE
    return fig


def legend_key(fig, name, line=None, marker=None, row=1, col=1):
    """A legend entry that only says what a line or marker style means (no data)."""
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines' if marker is None else 'markers', line=line,
                             marker=marker, name=name, legendgroup='key', showlegend=True, hoverinfo='skip'),
                  row=row, col=col)


def _layout(fig, height, pal, t=60, b=50, l=60, r=20, gap=LEGEND_GAP, **kw):
    """The look, and the legend in rows above the plots, gap px above them (8 where there are no subplot titles)."""
    fig.update_layout(template=pal.plotly_template, paper_bgcolor=pal.surf, plot_bgcolor=pal.surf, height=height,
                      font=dict(family=CSS_FONT, size=12, color=pal.ink2), margin=dict(l=l, r=r, t=t, b=b),
                      hoverlabel=dict(bgcolor=pal.grid, font_color=pal.ink, font_family=CSS_FONT),
                      legend=dict(orientation='h', yanchor='bottom', y=1 + gap / (height - t - b), xanchor='left', x=0,
                                  font_color=pal.ink2),
                      **kw)
    # no x zero line: at distance 0 it would hide the dashed separatrix
    fig.update_xaxes(gridcolor=pal.grid, zeroline=False, linecolor=pal.axis, showspikes=True,
                     spikecolor=pal.muted, spikethickness=1, spikedash='solid', spikemode='across')
    fig.update_yaxes(gridcolor=pal.grid, zerolinecolor=pal.axis, linecolor=pal.axis)
    for a in fig.layout.annotations:
        if a.font.color is None:
            a.font.color = pal.ink
    return fig


def envelope(t, v, nmax=8000):
    """At most ~nmax points: the min and max of each bin, so dips and spikes survive."""
    if len(t) <= nmax:
        return t, v
    w = int(np.ceil(len(t) / (nmax // 2)))
    n = len(t) // w * w
    T, V = t[:n].reshape(-1, w), v[:n].reshape(-1, w)
    i0, i1 = V.argmin(1), V.argmax(1)
    r = np.arange(len(V))
    first, last = np.minimum(i0, i1), np.maximum(i0, i1)
    tt = np.column_stack([T[r, first], T[r, last]]).ravel()
    vv = np.column_stack([V[r, first], V[r, last]]).ravel()
    return np.append(tt, t[n:]), np.append(vv, v[n:])


def side_coordinate(R, rsep, side, x_mode):
    """Distance from the separatrix (mm, positive into the SOL on either side) or R (m)."""
    if x_mode == 'R':
        return R, rsep, 'R at the midplane (m)'
    sign = 1.0 if side == 'outboard' else -1.0
    return sign * (R - rsep) * 1e3, 0.0, 'distance from the separatrix (mm)'


# plotly labels: HTML sub/superscripts and ASCII bars (the theme font, Arial, has no parallel sign U+2225)
PROFILE_ROWS = [('density', 'n', 'n (10<sup>19</sup> m<sup>-3</sup>)'), ('temperature', 'T', 'T (eV)'),
                ('parallel flow', 'upar', 'u<sub>||</sub> (km/s)'),
                ('ion Mach number', 'mach', 'u<sub>||,i</sub> / c<sub>s</sub>'), ('potential', 'phi', 'φ (V)')]


def thomson_trace(th, t0, t1, eq_s, side, rsep, x_mode, span, scale, pal, showlegend):
    """Thomson n_e over the frames t0..t1 (s) on one side, in the profile figure's x coordinate (thomson.window_profile:
    from the run's separatrix rsep at its time eq_s, moved by EFIT01's shift to each frame); None when no channel lies
    in the run's span (Thomson does not reach the inboard SOL). Bars: the spread over the frames, or the fit error where
    a channel has one valid frame."""
    p = thomson.window_profile(th, side, t0, t1, eq_s, rsep)
    sign = 1.0 if side == 'outboard' else -1.0
    x, _, _ = side_coordinate(rsep + sign * p['d'], rsep, side, x_mode)
    inside = (x >= np.min(span)) & (x <= np.max(span))
    if not inside.any():
        return None
    one = p['n'] == 1
    bar = np.where(one, p['err'], p['ne_sd'])
    xbar = np.where(one, 0.0, p['d_sd']) * (1.0 if x_mode == 'R' else 1e3)
    kind = np.where(one, 'fit error, 1 frame', [f'spread over {n} frames' for n in p['n']])
    cd = [[float(a), float(b), str(k)] for a, b, k in zip(p['R'][inside], (bar * scale)[inside], kind[inside])]
    return go.Scatter(x=x[inside], y=p['ne'][inside] * scale, mode='markers', customdata=cd,
                      marker=dict(symbol='circle-open', size=7, color=pal.elc, line=dict(width=1.5)),
                      error_y=dict(type='data', array=(bar * scale)[inside], color=pal.elc, thickness=1, width=3),
                      error_x=dict(type='data', array=xbar[inside], color=pal.elc, thickness=1, width=0),
                      name=f'Thomson n<sub>e</sub> {t0 * 1e3:.0f}–{t1 * 1e3:.0f} ms', legendgroup='thomson',
                      showlegend=showlegend, meta=dict(pgkyl=pc.NONE['thomson']),
                      hovertemplate='Thomson n_e: %{y:.4g} ± %{customdata[1]:.2g} (%{customdata[2]}), channel at R '
                                    '%{customdata[0]:.3f} m<extra></extra>')


def thomson_note(run, fr, th, t0, t1):
    """The line under the Thomson overlay: what is averaged, the beam flattop and the scan's equilibrium time against
    the run's, how far the separatrix moves in the window, and EFIT01's separatrix against the run's own."""
    k = (th.t >= t0) & (th.t <= t1)
    out = [f'Thomson n_e (MPTS {th.branch}, per-channel fits where VALID): the mean of {k.sum()} frame(s), '
           f'{t0 * 1e3:.1f} to {t1 * 1e3:.1f} ms. Each frame\'s channels are placed from the run\'s separatrix, moved by '
           f'how far EFIT01\'s separatrix moves between {run.eq_ms} ms and that frame.']
    key = f'n_{run.species[0]}'
    for side in [s for s in ('outboard', 'inboard') if s in run.layout.stacks]:
        shift = (th.separatrix(side, th.t[k]) - th.separatrix(side, run.eq_ms / 1e3)) * 1e3
        off = (th.separatrix(side, run.eq_ms / 1e3) - gv.midplane_profile(run, fr, side, key)[2]) * 1e3
        out.append(f'{side.capitalize()}: EFIT01\'s separatrix moves {shift.min():+.1f} to {shift.max():+.1f} mm (in R) '
                   f'from {run.eq_ms} ms over the window; its own position there, {off:+.1f} mm from the run\'s, is not '
                   'used.')
    if th.pnb is None:
        out.append('This file has no beam power (the sim-init workflow\'s copy), so no flattop: the window offers every '
                   'Thomson frame. Fetching the shot (fetch_flux.sh) adds it.')
    elif th.flat is None:
        out.append('No beam flattop by the scan\'s rule: the window offers every Thomson frame.')
    else:
        same = th.t_eq is not None and round(th.t_eq * 1e3) == run.eq_ms
        found = f'{th.t_eq * 1e3:.1f} ms' if th.t_eq is not None else 'none'
        out.append(f'Beam flattop (the elongation/triangularity scan\'s rule): {th.flat[0] * 1e3:.0f} to '
                   f'{th.flat[1] * 1e3:.0f} ms; its smallest-|DRSEP| EFIT01 time, {found}, '
                   + ('is the run\'s equilibrium time.' if same else f'is NOT the run\'s {run.eq_ms} ms.'))
    return ' '.join(out)


def thomson_timeline(th, t0, t1, eq_s, pal=LOOKS['dark']):
    """The shot around the averaging window: beam power with its flattop (when the file has it) and the Thomson line
    density per frame; the window shaded, the run's equilibrium time dashed."""
    rows = 2 if th.pnb is not None else 1
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.1)
    if th.pnb is not None:
        fig.add_trace(go.Scatter(x=th.pnb_t * 1e3, y=th.pnb, mode='lines', line=dict(color=pal.ink2, width=1.2),
                                 name='beam power', showlegend=False, hovertemplate='%{y:.3g} MW<extra></extra>',
                                 meta=dict(pgkyl=pc.NONE['thomson'])),
                      row=1, col=1)
        if th.flat is not None:
            fig.add_vrect(x0=th.flat[0] * 1e3, x1=th.flat[1] * 1e3, fillcolor=pal.grid, opacity=0.6, line_width=0,
                          row=1, col=1)
            legend_key(fig, 'beam flattop', marker=dict(symbol='square', size=12, color=pal.grid, opacity=0.6))
        fig.update_yaxes(title_text='P<sub>NBI</sub> (MW)', row=1, col=1)
    fig.add_trace(go.Scatter(x=th.t * 1e3, y=th.ld * 1e-19, mode='lines+markers', line=dict(color=pal.elc, width=1),
                             marker=dict(color=pal.elc, size=5), name='Thomson line density', showlegend=False,
                             meta=dict(pgkyl=pc.NONE['thomson']),
                             hovertemplate='%{x:.1f} ms: %{y:.3g} × 10^19 m^-2<extra></extra>'), row=rows, col=1)
    fig.update_yaxes(title_text='∫n<sub>e</sub> dl (10<sup>19</sup> m<sup>-2</sup>)', row=rows, col=1)
    half = 0.5 * np.median(np.diff(th.t)) * 1e3   # the band reaches half a frame past the window's end frames
    for r in range(1, rows + 1):
        fig.add_vrect(x0=t0 * 1e3 - half, x1=t1 * 1e3 + half, fillcolor=pal.elc, opacity=0.18, line_width=0, row=r, col=1)
        fig.add_vline(x=eq_s * 1e3, line=dict(color=pal.muted, width=1, dash='dash'), row=r, col=1)
    fig.update_xaxes(title_text='time in the shot (ms)', row=rows, col=1)
    legend_key(fig, 'averaging window', marker=dict(symbol='square', size=12, color=pal.elc, opacity=0.35))
    legend_key(fig, 'the run\'s equilibrium time', line=dict(color=pal.muted, width=1, dash='dash'))
    return _layout(fig, 150 * rows + 110, pal, gap=8, showlegend=True, hovermode='x unified')


@dataclass
class Cmp:
    """A second run drawn over the first (the overlay): run, its summary, its label on the page; fr: its frame matched
    in time to the shown one (gkview.match_frames), None where it has none; fr0: its first frame; k: that frame's
    number (-1: none); dt: its time minus the shown time (s); hist: (dt history, integrated moments, field energy);
    idx: its matched frame (row) for each of the first run's frames, -1 where none."""
    run: object
    summ: object
    label: str
    fr: object = None
    fr0: object = None
    k: int = -1
    dt: float = float('nan')
    hist: tuple = None
    idx: object = None


CMP_LEGEND = 'long dashes'


def cmp_line(line):
    """The overlay's line: the first run's colour, long dashes (dash-dot where the first run's is dashed), thinner."""
    out = dict(line)
    dash = out.get('dash') or 'solid'
    out['dash'] = {'dash': 'longdashdot', 'dashdot': 'longdashdot', 'dot': 'dot'}.get(dash, 'longdash')
    out['width'] = max(1.1, float(out.get('width', 2)) - 0.5)
    return out


def in_span(t, t_ref, *ys):
    """(t, ys...) of an overlay time series cut to the first run's time span t_ref (2 % margin): the time axes stay
    the shown run's, and an overlay running on far past it widens no y axis."""
    t = np.asarray(t, float)
    lo, hi = float(np.min(t_ref)), float(np.max(t_ref))
    pad = 0.02 * (hi - lo)
    k = (t >= lo - pad) & (t <= hi + pad)
    return (t[k],) + tuple(np.asarray(y)[k] for y in ys)


def cmp_note(cmp, tu_scale=None):
    """One line saying what the overlay is and which of its frames is shown."""
    if cmp is None:
        return ''
    if cmp.k < 0:
        return f'{CMP_LEGEND}: {cmp.label} (no frame of it near the shown time)'
    ts, tu = tu_scale or time_unit(float(cmp.summ['t'][-1]))
    return f'{CMP_LEGEND}: {cmp.label}, its frame {cmp.k} ({cmp.dt * ts:+.3g} {tu} from the shown time)'


def profiles_figure(run, fr, fr0, frame_label, x_mode='distance', pal=LOOKS['dark'], measured=None, cmp=None,
                    blocks=None):
    """Outboard and inboard midplane profiles: rows = quantity, both species; solid = shown frame, dotted = first.
    measured = (Thomson data, t0, t1 in s): its n_e over that window on the density row, placed from the run's time.
    cmp: a Cmp, its profiles at its matched frame (long dashes, following the movie). blocks: only the rows of these
    blocks (None: all)."""
    qs = gv.quantities(run)
    rows = [r for r in PROFILE_ROWS if (r[1] in ('mach', 'phi') and r[1] in qs) or any(f'{r[1]}_{s}' in qs
                                                                                       for s in run.species)]
    sides = [s for s in ('outboard', 'inboard') if s in run.layout.stacks and gv.stack_rows(run, s, blocks).any()]
    if not sides:
        raise ValueError('none of the chosen blocks is on a midplane stack (the core and SOL blocks at Z = 0)')
    seen = set()
    fig = subplots(rows=len(rows), cols=len(sides), shared_xaxes='columns', vertical_spacing=0.045,
                   horizontal_spacing=0.07, column_titles=[f'{s} midplane' for s in sides])
    qb = gv.quantities(cmp.run) if cmp is not None else {}
    k0 = run.frames()[0]
    for c, side in enumerate(sides, 1):
        keep = np.nonzero(gv.stack_rows(run, side, blocks))[0]
        for r, (title, base, ylab) in enumerate(rows, 1):
            keys = [(base, None)] if base in ('mach', 'phi') else [(f'{base}_{s}', s) for s in run.species]
            for key, s in keys:
                _, _, unit, scale, _ = qs[key]
                R, v, rsep = gv.midplane_profile(run, fr, side, key)
                _, v0, _ = gv.midplane_profile(run, fr0, side, key)
                x, xsep, xlab = side_coordinate(R, rsep, side, x_mode)
                if base == 'mach':
                    s = 'ion'   # an ion quantity: the ion colour, no legend entry of its own
                col = species_colour(pal, s) if s else pal.ink2
                name = gv.species_tag(s)[0] + 's' if s else 'potential'
                first = c == 1 and name not in seen
                seen.add(name)
                fig.add_trace(go.Scatter(x=x[keep], y=v0[keep] * scale, mode='lines',
                                         line=dict(color=col, width=1.5, dash='dot'),
                                         opacity=0.6, name=name, legendgroup=name, showlegend=False,
                                         meta=dict(pgkyl=pc.midplane(run, side, key, k0, blocks)),
                                         hovertemplate=f'{name}, first frame: %{{y:.4g}} {unit}<extra></extra>'),
                              row=r, col=c)
                fig.add_trace(go.Scatter(x=x[keep], y=v[keep] * scale, mode='lines', line=dict(color=col, width=2),
                                         name=name, legendgroup=name, showlegend=first,
                                         meta=dict(live=f'mid|{side}|{key}', scale=scale,
                                                   keep=None if len(keep) == len(x) else keep.tolist(),
                                                   pgkyl=pc.midplane(run, side, key, '{frame}', blocks)),
                                         hovertemplate=f'{name}: %{{y:.4g}} {unit}<extra></extra>'), row=r, col=c)
                if cmp is not None and side in cmp.run.layout.stacks and key in qb:
                    Rb, vb, rsb = gv.midplane_profile(cmp.run, cmp.fr if cmp.fr is not None else cmp.fr0, side, key)
                    xb, _, _ = side_coordinate(Rb, rsb, side, x_mode)
                    yb = vb * qb[key][3] if cmp.fr is not None else np.full(len(vb), np.nan)
                    fig.add_trace(go.Scatter(x=xb, y=yb, mode='lines', line=cmp_line(dict(color=col, width=2)),
                                             name=cmp.label, legendgroup='cmp', showlegend='cmp' not in seen,
                                             meta=dict(live=f'cmp|mid|{side}|{key}', scale=qb[key][3],
                                                       pgkyl=pc.midplane(cmp.run, side, key, '{cmp_frame}')),
                                             hovertemplate=f'{cmp.label}, {name}: %{{y:.4g}} {unit}<extra></extra>'),
                                  row=r, col=c)
                    seen.add('cmp')
            if measured is not None and base == 'n':
                tr = thomson_trace(*measured, run.eq_ms / 1e3, side, rsep, x_mode, x, scale, pal,
                                   'thomson' not in seen)
                if tr is not None:
                    seen.add('thomson')
                    fig.add_trace(tr, row=r, col=c)
            fig.add_vline(x=xsep, line=dict(color=pal.muted, width=1, dash='dash'), row=r, col=c)
            fig.update_yaxes(title_text=ylab if c == 1 else None, row=r, col=c)
        fig.update_xaxes(title_text=xlab, row=len(rows), col=c)
    # solid: the shown frame (the species' own entries); the overlay has its entry (long dashes)
    legend_key(fig, 'first frame', line=dict(color=pal.ink2, width=1.5, dash='dot'))
    legend_key(fig, 'separatrix', line=dict(color=pal.muted, width=1, dash='dash'))
    return _layout(fig, 190 * len(rows) + 120, pal, hovermode='x unified')


def history_figure(run, summ, qty, mode, t_now, x_mode='distance', pal=LOOKS['dark']):
    """The quantity at the outboard and inboard midplane over (radius, time); the shown frame's time marked."""
    label, sym, unit, scale, kind = gv.quantities(run)[qty]
    ts, tu = time_unit(summ['t'][-1])
    sides = [s for s in ('outboard', 'inboard') if s in run.layout.stacks and f'mid|{s}|{qty}' in summ]
    fig = subplots(rows=1, cols=len(sides), horizontal_spacing=0.08, subplot_titles=[f'{s} midplane' for s in sides])
    for c, side in enumerate(sides, 1):
        Z = summ[f'mid|{side}|{qty}']
        R, _, rsep = gv.midplane_profile(run, gv.frame(run, int(summ['frames'][0])), side, qty)
        x, xsep, xlab = side_coordinate(R, rsep, side, x_mode)
        with np.errstate(divide='ignore', invalid='ignore'):
            if mode == 'value':
                z, zl = Z * scale, f'{sym} ({unit})' if unit else sym
                cs = 'Inferno' if kind == 'pos' else _scale(pal.div)
            else:
                ref = np.vstack([Z[:1], Z[:-1]]) if mode == 'step' else Z[:1]   # the previous frame, or the first
                lab = 'change since the previous frame' if mode == 'step' else 'change'
                if kind == 'pos':
                    z, cs, zl = (Z / ref - 1) * 100, _scale(pal.div), f'{sym} {lab} (%)'
                else:
                    z, cs, zl = (Z - ref) * scale, _scale(pal.div), f'{sym} {lab}' + (f' ({unit})' if unit else '')
        o = np.argsort(x)
        fin = z[np.isfinite(z)]
        lim = np.percentile(np.abs(fin), 99) if len(fin) else 1.0
        rng = dict(zmin=0, zmax=np.percentile(fin, 99)) if (mode == 'value' and kind == 'pos') else dict(zmin=-lim, zmax=lim)
        cmd = pc.midplane(run, side, qty, history=True)
        if mode != 'value':
            cmd = '# the chart shows the change; this, the values\n' + cmd
        fig.add_trace(go.Heatmap(x=x[o], y=summ['t'] * ts, z=z[:, o], colorscale=cs, **rng, meta=dict(pgkyl=cmd),
                                 colorbar=dict(title=dict(text=zl, side='right'), len=0.9), showscale=(c == len(sides)),
                                 hovertemplate=f'%{{x:.4g}}, t = %{{y:.4g}} {tu}: %{{z:.4g}}<extra></extra>'), row=1, col=c)
        fig.add_hline(y=t_now * ts, line=dict(color=pal.ink, width=1), row=1, col=c, name='cursor-y')
        fig.add_vline(x=xsep, line=dict(color=pal.ink2, width=1, dash='dash'), row=1, col=c)
        fig.update_xaxes(title_text=xlab, row=1, col=c)
    fig.update_yaxes(title_text=f't ({tu})', row=1, col=1)
    fig = _layout(fig, 380, pal, t=80)
    fig.update_layout(meta=dict(tscale=ts))
    fig.update_layout(title=dict(text=f'{label} at the midplane over time (line: shown frame)',
                                 font=dict(color=pal.ink, size=14), x=0, xref='paper', xanchor='left'))
    return fig


def _cursor(fig, t_now, ts, rows_cols, pal):
    """The shown frame's time on these subplots: shapes named 'cursor-x', moved by a LivePlot as the movie plays
    (layout.meta.tscale: the figure's time unit)."""
    fig.update_layout(meta=dict(fig.layout.meta or {}, tscale=ts))
    for r, c in rows_cols:
        fig.add_vline(x=t_now * ts, line=dict(color=pal.ink, width=1), row=r, col=c, name='cursor-x')


def shown_targets(run, blocks=None):
    """The targets whose leg (the SOL block holding the plate) is among the chosen blocks (None: all)."""
    return [tg for tg in run.layout.targets if blocks is None or tg.leg in blocks]


def targets_figure(run, fr, summ, t_now, pal=LOOKS['dark'], cmp=None, blocks=None):
    """q_par at each target against the midplane distance of its flux surface (shown frame, lambda_q as vertical
    lines); lambda_q and the peak q_par over time. cmp: a Cmp, its targets of the same name in long dashes. blocks:
    the targets whose leg is chosen."""
    ts, tu = time_unit(summ['t'][-1])
    tgs = shown_targets(run, blocks)
    if not tgs:
        raise ValueError('no target among the chosen blocks (a target belongs to the SOL block holding its plate)'
                         if run.layout.targets else 'this run has no target plate (no physical theta face off Z = 0)')
    fig = subplots(rows=1, cols=3, horizontal_spacing=0.07, column_widths=[0.38, 0.31, 0.31],
                   subplot_titles=['heat flux at the targets (shown frame)', 'heat-flux width λ<sub>q</sub>',
                                   'peak heat flux'])
    flow = summ['t'] > 0   # t = 0 has no flow yet (q_par = 0): left out of the time traces, as on the posters
    t = summ['t'][flow] * ts
    fl_frames = [int(f) for f in np.asarray(summ['frames'])[flow]]   # a click on a time trace: that frame's plate
    cmp_tg = {tg.name: tg for tg in cmp.run.layout.targets} if cmp is not None else {}
    shown_cmp = False
    for tg in tgs:
        st = target_style(pal, tg)
        line = dict(color=st['color'], width=2, dash=st['dash'])
        q, x, lam = gv.target_heat_flux(run, fr, tg)
        mk = dict(size=8, symbol=st['symbol'], line=dict(color=pal.surf, width=1.5), maxdisplayed=12)
        fig.add_trace(go.Scatter(x=x, y=q / 1e6, mode='lines+markers', marker=mk, line=line, name=tg.name,
                                 legendgroup=tg.name, meta=dict(live=f'q|{tg.name}', scale=1e-6,
                                                                pgkyl=pc.plate(run, tg, '{frame}')),
                                 hovertemplate=f'{tg.name}: %{{y:.4g}} MW/m² at %{{x:.3g}} mm<extra></extra>'), row=1, col=1)
        # always drawn, transparent where lambda_q is undefined (a LivePlot moves it and shows it per frame); parked
        # on the curve's middle then, so it widens no axis
        fig.add_vline(x=lam if np.isfinite(lam) else float(x[len(x) // 2]), row=1, col=1, name=f'lam|{tg.name}',
                      line=dict(color=st['color'], width=1.4, dash=st['lam_dash']), opacity=1.0 if np.isfinite(lam) else 0.0)
        for c, y, fmt in ((2, summ[f'lam|{tg.name}'][flow], '%{y:.3f} mm'),
                          (3, np.nanmax(summ[f'q|{tg.name}'][flow], axis=1) / 1e6, '%{y:.4g} MW/m²')):
            why = pc.NONE['lambda'] if c == 2 else 'the largest |q_par| of this, at the clicked frame'
            fig.add_trace(go.Scatter(x=t, y=y, mode='lines+markers', marker=mk, line=line, name=tg.name,
                                     legendgroup=tg.name, showlegend=False,
                                     meta=dict(pgkyl=f'# {why}\n' + pc.plate(run, tg, '{point_frame}'),
                                               pgkyl_frames=fl_frames),
                                     hovertemplate=f'{tg.name}: {fmt}<extra></extra>'), row=1, col=c)
        tb = cmp_tg.get(tg.name)
        if tb is None:
            continue
        lb = cmp_line(line)
        qb, xb, lamb = gv.target_heat_flux(cmp.run, cmp.fr if cmp.fr is not None else cmp.fr0, tb)
        yb = qb / 1e6 if cmp.fr is not None else np.full(len(qb), np.nan)
        fig.add_trace(go.Scatter(x=xb, y=yb, mode='lines', line=lb, name=cmp.label, legendgroup='cmp',
                                 showlegend=not shown_cmp, meta=dict(live=f'cmp|q|{tg.name}', scale=1e-6,
                                                                     pgkyl=pc.plate(cmp.run, tb, '{cmp_frame}')),
                                 hovertemplate=f'{cmp.label}, {tg.name}: %{{y:.4g}} MW/m² at %{{x:.3g}} mm<extra></extra>'),
                      row=1, col=1)
        shown_cmp = True
        ok = cmp.fr is not None and np.isfinite(lamb)
        fig.add_vline(x=lamb if ok else float(xb[len(xb) // 2]), row=1, col=1, name=f'cmp|lam|{tg.name}',
                      line=dict(color=st['color'], width=1.0, dash='longdashdot'), opacity=0.8 if ok else 0.0)
        fb = cmp.summ['t'] > 0
        for c, y, fmt in ((2, cmp.summ[f'lam|{tg.name}'][fb], '%{y:.3f} mm'),
                          (3, np.nanmax(cmp.summ[f'q|{tg.name}'][fb], axis=1) / 1e6, '%{y:.4g} MW/m²')):
            tt, y, kf = in_span(cmp.summ['t'][fb], summ['t'], y, np.asarray(cmp.summ['frames'])[fb])
            fig.add_trace(go.Scatter(x=tt * ts, y=y, mode='lines', line=lb, name=cmp.label,
                                     legendgroup='cmp', showlegend=False,
                                     meta=dict(pgkyl=pc.plate(cmp.run, tb, '{point_frame}'),
                                               pgkyl_frames=[int(f) for f in kf]),
                                     hovertemplate=f'{cmp.label}, {tg.name}: {fmt}<extra></extra>'), row=1, col=c)
    outb = run.layout.stacks.get('outboard')
    if outb is not None:
        sol = gv.mid_edges(run.nodes[outb.sol], outb.j)
        cell = np.abs(np.diff(sol)).max() * 1e3
        if len(sol) - 1 <= 8:   # a coarse outboard SOL: is lambda_q resolved at all?
            fig.add_hline(y=cell, line=dict(color=pal.muted, width=1, dash='dot'), row=1, col=2,
                          annotation_text='one outboard SOL cell', annotation_font_color=pal.ink2,
                          annotation_position='bottom right')
    _cursor(fig, t_now, ts, [(1, 2), (1, 3)], pal)
    # where a leg's SOL never reaches Z = 0 its surfaces sit along the plate (targets_along: said under the chart)
    fig.update_xaxes(title_text='midplane distance from the separatrix (mm)' if not targets_along(run, blocks) else
                     'distance from the separatrix (mm)', row=1, col=1)
    fig.update_xaxes(title_text=f't ({tu})', row=1, col=2); fig.update_xaxes(title_text=f't ({tu})', row=1, col=3)
    fig.update_yaxes(title_text='|q<sub>||</sub>| at the plate (MW/m²)', rangemode='tozero', row=1, col=1)
    fig.update_yaxes(title_text='λ<sub>q</sub> (mm)', rangemode='tozero', row=1, col=2)
    fig.update_yaxes(title_text='max |q<sub>||</sub>| (MW/m²)', type='log', exponentformat='power', dtick='D2',
                     row=1, col=3)
    # each target's entry shows its hue (side) and dash (upper / lower); the overlay has its own entry
    legend_key(fig, 'λ<sub>q</sub> (vertical lines)', line=dict(color=pal.ink2, width=1.4, dash='dot'))
    return _layout(fig, 450, pal, hovermode='x unified')


def targets_along(run, blocks=None):
    """Whether a shown target's leg never reaches Z = 0 (its surfaces then sit at their distance along the plate)."""
    return any(tg.mid is None for tg in shown_targets(run, blocks))


def timetraces_figure(run, summ, dt, integ, fenergy, t_now, pal=LOOKS['dark'], cmp=None, blocks=None):
    """Time step; cells with n <= 0; particle content; field energy; volume-mean change by region. cmp: a Cmp (with
    its hist), the same traces in long dashes. blocks: a choice of blocks, its volume mean added to the region rows."""
    ts, tu = time_unit(summ['t'][-1])
    fig = subplots(rows=3, cols=2, vertical_spacing=0.11, horizontal_spacing=0.08,
                   subplot_titles=['time step', 'cells with n ≤ 0', 'particles (relative to the first sample)',
                                   'field energy (sum over blocks)', 'density change by region (volume mean)',
                                   'temperature change by region (volume mean)'])
    td, vd = envelope(*dt)
    fig.add_trace(go.Scatter(x=td * ts, y=vd, mode='lines', line=dict(color=pal.ink2, width=1.5), name='time step',
                             showlegend=False, hovertemplate='dt = %{y:.4g} s<extra></extra>',
                             meta=dict(pgkyl=pc.dt(run))), row=1, col=1)
    fig.update_yaxes(type='log', exponentformat='power', title_text='dt (s)', row=1, col=1)
    t = summ['t'] * ts
    for s in run.species:
        name = gv.species_tag(s)[0] + 's'
        col = species_colour(pal, s)
        fig.add_trace(go.Scatter(x=t, y=summ[f'neg|{s}'], mode='lines', line=dict(color=col, width=2), name=name,
                                 legendgroup=s, hovertemplate=f'{name}: %{{y:.0f}} cells<extra></extra>',
                                 meta=dict(pgkyl=pc.NONE['neg'])), row=1, col=2)
        if s in integ:
            ti, d = integ[s]
            x, y = envelope(ti, d[:, 0] / d[0, 0])
            fig.add_trace(go.Scatter(x=x * ts, y=y, mode='lines', line=dict(color=col, width=2), name=name,
                                     legendgroup=s, showlegend=False, meta=dict(pgkyl=pc.particles(run, s)),
                                     hovertemplate=f'{name}: %{{y:.4f}}<extra></extra>'), row=2, col=1)
    if fenergy is not None:
        x, y = envelope(*fenergy)
        fig.add_trace(go.Scatter(x=x * ts, y=y, mode='lines', line=dict(color=pal.ink2, width=1.5), name='field energy',
                                 showlegend=False, hovertemplate='%{y:.4g}<extra></extra>',
                                 meta=dict(pgkyl=pc.field_energy(run))), row=2, col=2)
        top = np.nanmax(y)
        if top > 0:   # it grows from ~0: show eight decades below its largest value
            fig.update_yaxes(range=[np.log10(top) - 8, np.log10(top) + 0.5], row=2, col=2)
    fig.update_yaxes(type='log', exponentformat='power', title_text='field energy', row=2, col=2)
    if cmp is not None and cmp.hist is not None:   # the comparison's traces, its own times
        dt_b, integ_b, fen_b = cmp.hist
        T = summ['t']
        kw = dict(name=cmp.label, legendgroup='cmp', showlegend=False)
        x, y = envelope(*in_span(dt_b[0], T, dt_b[1]))
        fig.add_trace(go.Scatter(x=x * ts, y=y, mode='lines', line=cmp_line(dict(color=pal.ink2, width=1.5)),
                                 hovertemplate=f'{cmp.label}: dt = %{{y:.4g}} s<extra></extra>',
                                 meta=dict(pgkyl=pc.dt(cmp.run)), **kw), row=1, col=1)
        for s in cmp.run.species:
            col = species_colour(pal, s)
            tb, yb = in_span(cmp.summ['t'], T, cmp.summ[f'neg|{s}'])
            fig.add_trace(go.Scatter(x=tb * ts, y=yb, mode='lines', line=cmp_line(dict(color=col, width=2)),
                                     hovertemplate=f'{cmp.label}, {s}: %{{y:.0f}} cells<extra></extra>',
                                     meta=dict(pgkyl=pc.NONE['neg']), **kw),
                          row=1, col=2)
            if s in integ_b:
                ti, d = integ_b[s]
                x, y = envelope(*in_span(ti, T, d[:, 0] / d[0, 0]))
                fig.add_trace(go.Scatter(x=x * ts, y=y, mode='lines', line=cmp_line(dict(color=col, width=2)),
                                         hovertemplate=f'{cmp.label}, {s}: %{{y:.4f}}<extra></extra>',
                                         meta=dict(pgkyl=pc.particles(cmp.run, s)), **kw), row=2, col=1)
        if fen_b is not None:
            x, y = envelope(*in_span(fen_b[0], T, fen_b[1]))
            fig.add_trace(go.Scatter(x=x * ts, y=y, mode='lines', line=cmp_line(dict(color=pal.ink2, width=1.5)),
                                     hovertemplate=f'{cmp.label}: %{{y:.4g}}<extra></extra>',
                                     meta=dict(pgkyl=pc.field_energy(cmp.run)), **kw), row=2, col=2)
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=cmp_line(dict(color=pal.ink2, width=2)),
                                 name=cmp.label, legendgroup='cmp', showlegend=True, hoverinfo='skip'), row=1, col=2)
    tag = gv.species_tag('elc')[1] if 'elc' in run.species else gv.species_tag(run.species[0])[1]
    s0 = 'elc' if 'elc' in run.species else run.species[0]
    chosen = None if blocks is None or set(blocks) >= set(range(run.nb)) else sorted(blocks)
    for c, base in ((1, 'n'), (2, 'T')):
        ends = []
        for reg in ('core', 'sol', 'pf') + (('chosen',) if chosen else ()):
            sel = chosen if reg == 'chosen' else reg
            if not len(gv.region_cells(run, sel)):
                continue
            m = gv.region_mean(run, summ, f'{base}_{s0}', sel)   # Jacobian-weighted over the region's cells
            if np.isfinite(m).any():
                y = (m / m[0] - 1) * 100
                name = f'blocks {block_list(chosen)}' if reg == 'chosen' else REGION_NAME[reg]
                colr = pal.ink if reg == 'chosen' else getattr(pal, reg)
                fig.add_trace(go.Scatter(x=t, y=y, mode='lines',
                                         line=dict(color=colr, width=2, dash='dashdot' if reg == 'chosen' else None),
                                         name=name, legendgroup='r' + reg, showlegend=(c == 1),
                                         meta=dict(pgkyl=pc.NONE['region']),
                                         hovertemplate=f'{name}: %{{y:+.1f}} %<extra></extra>'), row=3, col=c)
                k = np.where(np.isfinite(y))[0][-1]
                ends.append([y[k], t[k], 'chosen blocks' if reg == 'chosen' else name, np.nanmin(y), np.nanmax(y)])
            if cmp is not None and reg != 'chosen' and len(gv.region_cells(cmp.run, reg)) and \
                    f'{base}_{s0}' in gv.quantities(cmp.run):
                mb = gv.region_mean(cmp.run, cmp.summ, f'{base}_{s0}', reg)
                if np.isfinite(mb).any():
                    tb, yb = in_span(cmp.summ['t'], summ['t'], (mb / mb[0] - 1) * 100)
                    fig.add_trace(go.Scatter(x=tb * ts, y=yb, mode='lines',
                                             line=cmp_line(dict(color=getattr(pal, reg), width=2)), name=cmp.label,
                                             legendgroup='cmp', showlegend=False, meta=dict(pgkyl=pc.NONE['region']),
                                             hovertemplate=f'{cmp.label}, {REGION_NAME[reg]}: %{{y:+.1f}} %<extra></extra>'),
                                  row=3, col=c)
        # direct labels at the line ends (crimson/gold sit close in CVD), pushed apart to stay readable and kept
        # within the curves' span (above it they would reach the subplot title)
        room = 0.14
        if ends:
            lo_all, hi_all = min(e[3] for e in ends), max(e[4] for e in ends)
            span = hi_all - lo_all or 1.0
            ends.sort()
            for a, b in zip(ends, ends[1:]):
                b[0] = max(b[0], a[0] + 0.08 * span)
            over = ends[-1][0] - hi_all
            if over > 0:   # the stack moves down as a whole, its spacing kept
                for e in ends:
                    e[0] -= over
            for yl, tl, name, _, _ in ends:
                fig.add_annotation(x=tl, y=yl, text=f' {name}', showarrow=False, xanchor='left',
                                   font=dict(color=pal.ink2, size=11), row=3, col=c)
            # the time axis runs on past the last frame far enough that the longest label ends inside its column,
            # at the narrowest page the viewer is laid out for (END_LABEL_COLUMN px of plot per column)
            w = max(text_px(f' {e[2]}', 11) for e in ends) + 6
            room = w / max(END_LABEL_COLUMN - w, 1.0)
        fig.update_yaxes(title_text=f'{base}_{tag} change (%)', row=3, col=c)
        fig.update_xaxes(range=[t[0], t[-1] + room * (t[-1] - t[0])], row=3, col=c)
    fig.update_yaxes(title_text='N / N(first)', row=2, col=1)
    fig.update_yaxes(title_text='cells', rangemode='tozero', row=1, col=2)
    for r in (1, 2, 3):
        for c in (1, 2):
            fig.update_xaxes(title_text=f't ({tu})', row=r, col=c)
    _cursor(fig, t_now, ts, [(r, c) for r in (1, 2, 3) for c in (1, 2)], pal)
    return _layout(fig, 900, pal, hovermode='x unified')


def cell_rows(run, fr, fr0, qty, blocks=None):
    """One row per cell for the table view: block, region, psi and theta index (the file's), R, Z, value, first frame,
    change; the chosen blocks only (None: all)."""
    label, sym, unit, scale, kind = gv.quantities(run)[qty]
    u = f' ({unit})' if unit else ''
    cols = {'block': [], 'region': [], 'i (psi)': [], 'j (theta)': [], 'R (m)': [], 'Z (m)': [],
            f'{sym}{u}': [], f'{sym}, first frame{u}': [], 'change': []}
    for b in shown_blocks(run, blocks):
        c = gv.cell_centres(run.nodes[b])
        v, v0 = fr[b][qty] * scale, fr0[b][qty] * scale
        for (i, j), x in np.ndenumerate(v):
            cols['block'].append(b); cols['region'].append(REGION_NAME[run.layout.region[b]])
            cols['i (psi)'].append(run.file_row(b, i)); cols['j (theta)'].append(j)
            cols['R (m)'].append(round(float(c[i, j, 0]), 5)); cols['Z (m)'].append(round(float(c[i, j, 1]), 5))
            cols[f'{sym}{u}'].append(float(f'{x:.5g}')); cols[f'{sym}, first frame{u}'].append(float(f'{v0[i, j]:.5g}'))
            if kind == 'pos':
                cols['change'].append(f'{(x / v0[i, j] - 1) * 100:+.2f} %' if v0[i, j] else '')
            else:
                cols['change'].append(f'{x - v0[i, j]:+.4g}')
    return cols


# ------------------------------------------------------------------------------- options, side by side
def _rgba(hexc, a):
    return 'rgba({}, {}, {}, {})'.format(*(int(hexc[i:i + 2], 16) for i in (1, 3, 5)), a)


def options_side_by_side(ro, va, vb):
    """Rows (compare, option, this run, overlay, set by) of every option either run used (run_options.effective: set
    at launch, else the driver source's default), those that differ first. A path is shown in full where its
    shortened forms would look the same."""
    ea, eb = ro.effective(va), ro.effective(vb)
    rows = []
    for k in dict.fromkeys(list(ea) + list(eb)):
        a, b = str(ea.get(k, '')), str(eb.get(k, ''))
        sa, sb = ro.shown_value(a) or '–', ro.shown_value(b) or '–'
        if a != b and sa == sb:
            sa, sb = a or '–', b or '–'
        rows.append(dict(compare='same' if a == b else 'differs', option=k, this_run=sa, overlay=sb,
                         set_by=', '.join(n for n, vv in (('this run', va['launch']), ('overlay', vb['launch']))
                                          if k in vv) or 'neither (defaults)'))
    return sorted(rows, key=lambda r: r['compare'] != 'differs')   # stable: each group in the driver's order


def _compare_tints(pal):
    # the targets' two hues (validated apart under CVD in every look): what differs orange, what agrees teal
    return {'differs': _rgba(pal.outer, 0.30), 'same': _rgba(pal.inner, 0.13)}


def compare_style(pal):
    """marimo table style_cell for options_side_by_side rows: each row tinted by its first column ('differs' |
    'same'), that cell bold where they differ. marimo styles a row's cells in column order, so the first column's
    value names the row (rowId is the row's place in the shown order, which a sort changes)."""
    tint, state = _compare_tints(pal), {}

    def style(row, col, value):
        if col == 'compare':
            state[row] = value
        st = state.get(row)
        out = {'backgroundColor': tint[st]} if st in tint else {}
        if col == 'compare' and st == 'differs':
            out['fontWeight'] = 'bold'
        return out
    return style


def compare_legend(pal):
    """The side-by-side table's key (markdown): the two tints, named."""
    t = _compare_tints(pal)
    chip = lambda name, what: (f'<span style="background: {t[name]}; padding: 1px 8px; border-radius: 3px">{name}'  # noqa: E731
                               f'</span> {what}')
    return '<small>' + chip('differs', 'the two runs used different values') + ' &nbsp; ' + \
        chip('same', 'the same value') + ' &nbsp; (set at launch, else the driver source\'s default)</small>'


# ------------------------------------------------------------------------------------------- balance and probe
def balance_figure(run, B, t_now, pal=LOOKS['dark'], cmp=None, B_cmp=None):
    """Particle balance (each species) and energy balance, whole device; the power through each target's plate.
    Terms: source (dashed gray), boundary losses (crimson), rate of change (gold, dash-dot), residual (ink, dotted):
    hues clear of the targets' orange and teal. The y ranges fit the second half of the run (the start-up transient
    is tens of times larger; double-click a panel to see it)."""
    ts, tu = time_unit(B['t'][-1])
    fd = gv.full_device(run)
    sp = list(B['species'])
    fig = subplots(rows=2, cols=2, vertical_spacing=0.16, horizontal_spacing=0.09,
                   subplot_titles=[f'{gv.species_tag(s)[0]} particles (whole device)' for s in sp[:2]] +
                   ['energy (whole device)', 'power to each target (its full toroidal ring)'])
    term = [('source', pal.muted, 'dash'), ('boundary losses', pal.core, 'solid'), ('rate of change', pal.sol, 'dashdot'),
            ('residual', pal.ink, 'dot')]

    def fit(r, c, ys):
        late = np.concatenate([np.abs(np.asarray(y)[len(y) // 2:]) for y in ys])
        top = 1.25 * np.nanmax(late) if late.size and np.nanmax(late) > 0 else None
        lo = min(0.0, min(np.nanmin(np.asarray(y)[len(y) // 2:]) for y in ys))
        if top:
            fig.update_yaxes(range=[1.25 * lo if lo < 0 else -0.05 * top, top], row=r, col=c)
    notes = []

    cmp_seen = []

    def four(r, c, t, src, out, dot, scale, unit, show, faint=None, cmd=''):
        res = src - out - dot
        if faint is None:
            fit(r, c, [y * scale for y in (src, out, dot, res)])
        for (name, col, dash), y in zip(term, (src, out, dot, res)):
            x, v = envelope(*(in_span(t, B['t'], y * scale) if faint is not None else (t, y * scale)))
            x = x * ts
            if faint is None:
                fig.add_trace(go.Scatter(x=x, y=v, mode='lines', line=dict(color=col, width=2, dash=dash), name=name,
                                         legendgroup=name, showlegend=show, meta=dict(pgkyl=cmd),
                                         hovertemplate=f'{name}: %{{y:.4g}} {unit}<extra></extra>'), row=r, col=c)
            else:   # the comparison: the same terms, thin and faint
                fig.add_trace(go.Scatter(x=x, y=v, mode='lines', line=dict(color=col, width=1.2, dash=dash),
                                         opacity=0.5, name=faint, legendgroup='cmp', showlegend=not cmp_seen,
                                         meta=dict(pgkyl=cmd),
                                         hovertemplate=f'{faint}, {name}: %{{y:.4g}} {unit}<extra></extra>'),
                              row=r, col=c)
                cmp_seen.append(1)
        k = len(t) - 1
        return res[k] / src[k] * 100 if src[k] else np.nan

    for c, s in enumerate(sp[:2], 1):
        d = B['species'][s]
        notes.append(f"{gv.species_tag(s)[0]}s {four(1, c, d['t'], d['src'], d['out'], d['dot'], fd, '/s', c == 1, cmd=pc.balance(run)):+.2f} %")
        fig.update_yaxes(title_text='particles / s', exponentformat='power', row=1, col=c)
    E = B['E']
    notes.append(f"energy {four(2, 1, B['t'], E['src'], E['out'], E['dot'], fd / 1e6, 'MW', False, cmd=pc.balance(run, True)):+.1f} %")
    if B_cmp is not None:
        fdb = gv.full_device(cmp.run)
        for c, s in enumerate(sp[:2], 1):
            d = B_cmp['species'].get(s)
            if d is not None:
                four(1, c, d['t'], d['src'], d['out'], d['dot'], fdb, '/s', False, faint=cmp.label, cmd=pc.balance(cmp.run))
        Eb = B_cmp['E']
        four(2, 1, B_cmp['t'], Eb['src'], Eb['out'], Eb['dot'], fdb / 1e6, 'MW', False, faint=cmp.label,
             cmd=pc.balance(cmp.run, True))
        for name, v in B_cmp['targets'].items():
            tg = next((t for t in run.layout.targets if t.name == name), None)
            if tg is None:
                continue
            st = target_style(pal, tg)
            x, y = envelope(*in_span(B_cmp['t'], B['t'], v['power'] * 2 * np.pi / 1e6))
            x = x * ts
            tgb = next((t for t in cmp.run.layout.targets if t.name == name), None)
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line=cmp_line(dict(color=st['color'], width=2, dash=st['dash'])),
                                     name=cmp.label, legendgroup='cmp', showlegend=False,
                                     meta=dict(pgkyl=(pc.target_power(cmp.run, tgb) if tgb else None) or ''),
                                     hovertemplate=f'{cmp.label}, {name}: %{{y:.4g}} MW<extra></extra>'), row=2, col=2)
    fig.update_yaxes(title_text='MW', row=2, col=1)
    fit(2, 2, [v['power'] * 2 * np.pi / 1e6 for v in B['targets'].values()])
    for name, v in B['targets'].items():
        tg = next(t for t in run.layout.targets if t.name == name)
        st = target_style(pal, tg)
        x, y = envelope(B['t'] * ts, v['power'] * 2 * np.pi / 1e6)
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line=dict(color=st['color'], width=2, dash=st['dash']),
                                 name=name, hovertemplate=f'{name}: %{{y:.4g}} MW<extra></extra>',
                                 meta=dict(pgkyl=pc.target_power(run, tg) or '')), row=2, col=2)
    fig.update_yaxes(title_text='MW', row=2, col=2)
    for r in (1, 2):
        for c in (1, 2):
            fig.update_xaxes(title_text=f't ({tu})', row=r, col=c)
    _cursor(fig, t_now, ts, [(r, c) for r in (1, 2) for c in (1, 2)], pal)
    return _layout(fig, 720, pal, hovermode='x unified'), notes


def diffusivity_figure(run, Dinfo, x_mode='distance', pal=LOOKS['dark'], cmp=None, D_cmp=None, blocks=None):
    """The anomalous diffusivity the run used (gkview.diffusivity) at the outboard and inboard midplane, node by node
    across the core and SOL blocks: the D the driver gave at each node row's psi. One curve when every species has the
    same D, one per species otherwise. cmp / D_cmp: a Cmp and its diffusivity, in long dashes. blocks: the rows of the
    chosen blocks. A log y axis when D spans more than a factor 20."""
    sides = [s for s in ('outboard', 'inboard') if s in run.layout.stacks and gv.stack_rows(run, s, blocks).any()]
    if not sides:
        raise ValueError('no midplane stack among the chosen blocks: see the map below')
    fig = subplots(rows=1, cols=len(sides), horizontal_spacing=0.08, subplot_titles=[f'{s} midplane' for s in sides])
    allD, seen = [], set()

    def curves(r, D, label, style, c, side, first_label):
        sp = list(D['species'])
        for s in (sp[:1] if D['same'] else sp):
            st = r.layout.stacks[side]
            Rn, Dm, Bn = gv.midplane_D(r, D['species'][s], side)
            if blocks is not None and r is run:   # node rows of the chosen blocks
                keep = np.isin(Bn, list(blocks))
                Rn, Dm = Rn[keep], Dm[keep]
            x, xsep, xlab = side_coordinate(Rn, gv.mid_edges(r.nodes[st.sol], st.j)[0], side, x_mode)
            col = pal.ink2 if D['same'] else species_colour(pal, s)
            name = label + ('' if D['same'] else f', {gv.species_tag(s)[0]}s')
            o = np.argsort(x)
            cmd = (pc.diffusivity(r, None, s, side) if D['kind'] == 'file' else
                   f'no pgkyl equivalent: no anom_diff file, a uniform D ({D["source"]})')
            fig.add_trace(go.Scatter(x=x[o], y=Dm[o], mode='lines+markers' if len(x) <= 40 else 'lines',
                                     line=style(dict(color=col, width=2)), marker=dict(size=6, color=col),
                                     name=name, legendgroup=name, showlegend=name not in seen, meta=dict(pgkyl=cmd),
                                     hovertemplate=f'{name}: D = %{{y:.4g}} m²/s at %{{x:.4g}}<extra></extra>'),
                          row=1, col=c)
            seen.add(name)
            allD.append(Dm[np.isfinite(Dm)])
            if first_label:
                fig.add_vline(x=xsep, line=dict(color=pal.muted, width=1, dash='dash'), row=1, col=c)
                fig.update_xaxes(title_text=xlab, row=1, col=c)

    for c, side in enumerate(sides, 1):
        if Dinfo['kind']:
            curves(run, Dinfo, run.name, lambda d: d, c, side, True)
        if cmp is not None and D_cmp is not None and D_cmp['kind'] and side in cmp.run.layout.stacks:
            curves(cmp.run, D_cmp, cmp.label, cmp_line, c, side, not Dinfo['kind'])
    allD = np.concatenate(allD) if allD else np.zeros(0)
    pos = allD[allD > 0]
    log = len(pos) and pos.max() / pos.min() > 20
    fig.update_yaxes(title_text='D (m²/s)', row=1, col=1)
    fig.update_yaxes(type='log' if log else 'linear', rangemode=None if log else 'tozero', exponentformat='power')
    # what the curves are (the D at each node row, core then SOL) is said under the chart; the overlay has its entry
    legend_key(fig, 'separatrix', line=dict(color=pal.muted, width=1, dash='dash'))
    return _layout(fig, 400, pal, hovermode='x unified')


def diffusivity_map(run, Dinfo, zoom='whole domain', pal=LOOKS['dark'], blocks=None, species=None):
    """The diffusivity of every cell (its cell average) on the poloidal plane and the unrolled strips (map_figure)."""
    s = species or next(iter(Dinfo['species']))
    cells = Dinfo['species'][s]['cells']
    V = np.concatenate([cells[b].ravel() for b in shown_blocks(run, blocks)])
    pos = V[np.isfinite(V) & (V > 0)]
    log = len(pos) and pos.max() / pos.min() > 20
    who = '' if Dinfo['same'] else f', {gv.species_tag(s)[0]}s'
    sc = value_scale(V, f'anomalous diffusivity D{who} (m²/s), cell averages', log=bool(log))
    return map_figure(run, None, None, None, zoom=zoom, pal=pal, scale=sc, blocks=blocks, vals=cells)


PROBE_ROWS = [('n', 'n (10<sup>19</sup> m<sup>-3</sup>)'), ('T', 'T (eV)'), ('upar', 'u<sub>||</sub> (km/s)'),
              ('mach', 'u<sub>||,i</sub> / c<sub>s</sub>'), ('phi', 'φ (V)'), ('q_par', 'q<sub>||</sub> (MW m<sup>-2</sup>)')]


def probe_cell(run, c):
    """(block, i, j, region, R, Z) of a cell index (cell_offsets order)."""
    off = gv.cell_offsets(run)
    b = max(bb for bb in range(run.nb) if off[bb] <= c)
    nt = run.nodes[b].shape[1] - 1
    i, j = divmod(c - off[b], nt)
    R, Z = gv.cell_centres(run.nodes[b])[i, j]
    return b, run.file_row(b, i), j, run.layout.region[b], float(R), float(Z)


def probe_figure(run, summ, c, t_now, pal=LOOKS['dark'], cmp=None):
    """Every quantity at one cell over time (both species where there are two), with the volume mean of its region
    (dotted) for density and temperature. cmp: a Cmp, the same at its cell nearest this one (long dashes)."""
    qs = gv.quantities(run)
    b, i, j, reg, R, Z = probe_cell(run, c)
    cb, cd = gv.nearest_cell(cmp.run, R, Z, reg) if cmp is not None else (-1, np.nan)
    qb = gv.quantities(cmp.run) if cmp is not None else {}
    ts, tu = time_unit(summ['t'][-1])
    t = summ['t'] * ts
    rows = [r for r in PROBE_ROWS if r[0] in qs or any(f'{r[0]}_{s}' in qs for s in run.species)]
    nrow = (len(rows) + 1) // 2
    fig = subplots(rows=nrow, cols=2, vertical_spacing=0.11, horizontal_spacing=0.09, shared_xaxes=True,
                   subplot_titles=[lab for _, lab in rows])
    seen = set()
    for n, (base, _) in enumerate(rows):
        r, cc = n // 2 + 1, n % 2 + 1
        keys = [(base, 'ion' if base == 'mach' else None)] if base in qs else [(f'{base}_{s}', s) for s in run.species]
        for key, s in keys:
            _, _, unit, scale, _ = qs[key]
            col = species_colour(pal, s) if s else pal.ink2
            name = gv.species_tag(s)[0] + 's' if s else qs[key][0]
            show = name not in seen and s is not None
            seen.add(name)
            fig.add_trace(go.Scatter(x=t, y=summ[f'fld|{key}'][:, c] * scale, mode='lines', line=dict(color=col, width=2),
                                     name=name, legendgroup=name, showlegend=show,
                                     meta=dict(pgkyl=pc.cell_history(run, b, i, j, key)),
                                     hovertemplate=f'{name}: %{{y:.4g}} {unit}<extra></extra>'), row=r, col=cc)
            if base in ('n', 'T'):
                m = gv.region_mean(run, summ, key, reg) * scale
                fig.add_trace(go.Scatter(x=t, y=m, mode='lines', line=dict(color=col, width=1.4, dash='dot'),
                                         opacity=0.7, name=f'{name}, {REGION_NAME[reg]} mean', showlegend=False,
                                         meta=dict(pgkyl=pc.NONE['region']),
                                         hovertemplate=f'{REGION_NAME[reg]} volume mean: %{{y:.4g}} {unit}<extra></extra>'),
                              row=r, col=cc)
            if cb >= 0 and key in qb:
                tb, yb = in_span(cmp.summ['t'], summ['t'], cmp.summ[f'fld|{key}'][:, cb] * qb[key][3])
                fig.add_trace(go.Scatter(x=tb * ts, y=yb, mode='lines', line=cmp_line(dict(color=col, width=2)), name=cmp.label,
                                         legendgroup='cmp', showlegend='cmp' not in seen,
                                         meta=dict(pgkyl=pc.cell_history(cmp.run, *probe_cell(cmp.run, cb)[:3], key)),
                                         hovertemplate=f'{cmp.label}, {name}: %{{y:.4g}} {unit}<extra></extra>'),
                              row=r, col=cc)
                seen.add('cmp')
        if base == 'q_par' and 'q_perp' in qs:   # the part of q_par carried by perpendicular energy, same axes
            _, _, unit, scale, _ = qs['q_perp']
            fig.add_trace(go.Scatter(x=t, y=summ['fld|q_perp'][:, c] * scale, mode='lines',
                                     line=dict(color=pal.ink2, width=1.6, dash='dash'), name=qs['q_perp'][0],
                                     showlegend=False, meta=dict(pgkyl=pc.cell_history(run, b, i, j, 'q_perp')),
                                     hovertemplate=f'q_perp: %{{y:.4g}} {unit}<extra></extra>'), row=r, col=cc)
        if r == nrow:
            fig.update_xaxes(title_text=f't ({tu})', row=r, col=cc)
    _cursor(fig, t_now, ts, [(n // 2 + 1, n % 2 + 1) for n in range(len(rows))], pal)
    legend_key(fig, f'{REGION_NAME[reg]} volume mean', line=dict(color=pal.ink2, width=1.4, dash='dot'))
    if 'q_perp' in qs and any(base == 'q_par' for base, _ in rows):
        legend_key(fig, 'q<sub>⊥</sub> part of q<sub>||</sub>', line=dict(color=pal.ink2, width=1.6, dash='dash'))
    return _layout(fig, 250 * ((len(rows) + 1) // 2) + 90, pal, hovermode='x unified')


def probe_note(run, c, cmp=None):
    """The probed cell, and the overlay's cell nearest it (markdown, for the page above the probe chart)."""
    b, i, j, reg, R, Z = probe_cell(run, c)
    txt = f'**Cell:** block {b}, i = {i} (psi), j = {j} (theta): {REGION_NAME[reg]}, R = {R:.4f} m, Z = {Z:.4f} m.'
    if cmp is not None:
        cb, cd = gv.nearest_cell(cmp.run, R, Z, reg)
        if cb >= 0:
            bb, ib, jb, rb, _, _ = probe_cell(cmp.run, cb)
            txt += (f' **Overlay** ({CMP_LEGEND}): {cmp.label}, its nearest cell: block {bb}, i = {ib}, j = {jb} '
                    f'({REGION_NAME[rb]}, {cd * 1e3:.3g} mm away).')
    return txt


# --------------------------------------------------------------------------------------------------- exports
def save_figure(fig, path, dpi=200):
    """Write a matplotlib figure by the path's extension (png / pdf / svg); the SVG names the theme font first, the
    metric-twin stand-in it was set in as the fallback (the progress posters' rule). Returns the path."""
    ext = os.path.splitext(path)[1].lower().lstrip('.')
    with plt.rc_context({'svg.fonttype': 'none', 'pdf.fonttype': 42}):
        fig.savefig(path, dpi=dpi if ext == 'png' else None, format=ext)
    if ext == 'svg' and RENDER_FONT != FONT:
        txt = open(path).read()
        open(path, 'w').write(txt.replace(f"'{RENDER_FONT}'", f"'{FONT}', '{RENDER_FONT}'")
                              .replace(f'"{RENDER_FONT}"', f'"{FONT}", "{RENDER_FONT}"'))
    return path


def render_movie(run, summ, frames, qty, mode, log, zoom, pal, path, fps=10, progress=None, title='', blocks=None):
    """An MP4 of the map over the given frame numbers, one colour scale for all of them, the time and frame on each.
    H.264 from imageio-ffmpeg's ffmpeg (the system ffmpeg has no H.264 encoder). progress(iterable, total) wraps the
    loop. Returns the path."""
    import imageio_ffmpeg
    allf = [int(k) for k in summ['frames']]
    rows = [allf.index(k) for k in frames]
    label, sym, unit, scale_, kind = gv.quantities(run)[qty]
    V = mode_values(summ[f'fld|{qty}'].astype(np.float64), rows, mode, kind, scale_)
    sel = shown_blocks(run, blocks)
    if len(sel) < run.nb:   # one colour scale over the chosen blocks' cells
        V = V[:, gv.region_cells(run, sel)]
    scale = fixed_scale(run, V, qty, mode, log, step_first=(mode == 'step' and rows[0] == 0))
    ts, tu = time_unit(float(summ['t'][-1]))
    fr0 = gv.frame(run, allf[0])
    writer = None
    try:
        it = progress(list(zip(frames, rows)), len(frames)) if progress else zip(frames, rows)
        for k, row in it:
            fig = map_figure(run, gv.frame(run, k), fr0, qty, mode, False, log, None, zoom, pal=pal,
                             fr_prev=gv.frame(run, allf[max(row - 1, 0)]), scale=scale, blocks=blocks,
                             title=f'{title}   t = {summ["t"][row] * ts:.4g} {tu}   (frame {k})')
            fig.canvas.draw()
            img = np.asarray(fig.canvas.buffer_rgba())[..., :3]
            plt.close(fig)
            h, w = img.shape[:2]
            img = np.pad(img, ((0, h % 2), (0, w % 2), (0, 0)), mode='edge')   # yuv420p needs even sizes
            if writer is None:
                writer = imageio_ffmpeg.write_frames(path, (img.shape[1], img.shape[0]), fps=fps, codec='libx264',
                                                     pix_fmt_out='yuv420p', quality=8, macro_block_size=1)
                writer.send(None)
            writer.send(np.ascontiguousarray(img))
    finally:
        if writer is not None:
            writer.close()
    return path


# ---------------------------------------------------------------------------------------------- live updates
def _axis_key(ref):
    """'y3' -> 'yaxis3' (a trace's axis reference -> its layout key)."""
    return ref[0] + 'axis' + ref[1:]


def _far_out_kept(ext, upper):
    """The extreme over the frames whose own extreme ext is not far out (Tukey: beyond 3 interquartile ranges past
    the quartile): a few frames standing far from the rest (a start-up transient) do not set the axis."""
    q1, q3 = np.percentile(ext, [25, 75])
    keep = ext <= q3 + 3 * (q3 - q1) if upper else ext >= q1 - 3 * (q3 - q1)
    return float(ext[keep].max() if upper else ext[keep].min())


def live_spec(fig, summ, frames=None, shown=None):
    """What a LivePlot (movie_widget.py) changes per frame while the movie plays, read from the figure's tags:
    traces with meta {'live': a per-frame summary key, 'scale'} (their y at every frame), shapes named 'cursor-x' /
    'cursor-y' (the frame's time, in the figure's unit layout.meta.tscale) and 'lam|<target>' (its lambda_q; drawn
    at op, transparent where undefined). So that the axes hold still while the curves and lines move (modifies fig):
    each axis a live trace is on gets a fixed range spanning it over frames (the movie's; default every frame),
    leaving out frames that stand far out (_far_out_kept), the frame shown (paused there, all of it is seen) and the
    axis's other curves (the first frame dotted); over[j]: frame j runs past such an axis. Each axis a live shape moves along, where some frame's position lies
    outside the curves (plotly's autorange counts shapes, so that frame's figure would have other axes), gets one
    spanning the curves and every position."""
    import base64
    import warnings
    allf = [int(k) for k in summ['frames']]
    span = [allf.index(int(f)) for f in (frames if frames is not None else allf) if int(f) in allf] or list(range(len(allf)))
    tscale = (fig.layout.meta or {}).get('tscale', 1.0)
    traces, shapes, live, other = [], [], {}, {}
    for i, tr in enumerate(fig.data):
        m = tr.meta if isinstance(tr.meta, dict) else None
        if m and m.get('live') and m['live'] in summ:   # a meta may hold only a pgkyl command
            Y = np.asarray(summ[m['live']], dtype=float)
            if m.get('keep') is not None:   # the rows of the chosen blocks
                Y = Y[:, m['keep']]
            Y = np.ascontiguousarray(Y * m.get('scale', 1.0), dtype=np.float32)
            traces.append(dict(index=i, n=int(Y.shape[1]), y=base64.b64encode(Y.tobytes()).decode('ascii')))
            live.setdefault(_axis_key(tr.yaxis or 'y'), []).append(Y.astype(float))
    for i, tr in enumerate(fig.data):
        key = _axis_key(tr.yaxis or 'y')
        if tr.type == 'scatter' and key in live and tr.y is not None and i not in [t['index'] for t in traces]:
            other.setdefault(key, []).append(np.asarray(tr.y, dtype=float).ravel())
    over = np.zeros(len(allf), dtype=bool)
    for key, Ys in live.items():
        ax = fig.layout[key]
        if ax.range is not None:
            continue
        log, tozero = ax.type == 'log', ax.rangemode == 'tozero'
        A, o = np.hstack(Ys), np.concatenate(other.get(key, [np.array([])]))
        with np.errstate(divide='ignore', invalid='ignore'), warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)   # a frame with no finite value
            if log:
                A, o = np.where(A > 0, np.log10(A), np.nan), np.log10(o[o > 0])
            hi_f, lo_f = np.nanmax(A[span], axis=1), np.nanmin(A[span], axis=1)
        hi_f, lo_f, o = hi_f[np.isfinite(hi_f)], lo_f[np.isfinite(lo_f)], o[np.isfinite(o)]
        if not len(hi_f):
            continue
        lo, hi = _far_out_kept(lo_f, upper=False), _far_out_kept(hi_f, upper=True)
        if shown is not None and int(shown) in allf:
            o = np.r_[o, A[allf.index(int(shown))]]
            o = o[np.isfinite(o)]
        if len(o):
            lo, hi = min(lo, float(o.min())), max(hi, float(o.max()))
        if tozero and not log:
            lo, hi = min(lo, 0.0), max(hi, 0.0)
        pad = 0.05 * (hi - lo) if hi > lo else max(abs(hi) * 0.05, 1e-30)
        rng = [lo if (tozero and lo == 0) else lo - pad, hi + pad]
        fig.layout[key].update(range=rng, autorange=False)
        with np.errstate(invalid='ignore'):
            over |= ((A < rng[0]) | (A > rng[1])).any(axis=1)
    moves = {}
    for i, sh in enumerate(fig.layout.shapes or ()):
        if sh.name in ('cursor-x', 'cursor-y'):
            vals, axis = summ['t'] * tscale, sh.name[-1]
        elif sh.name and sh.name.startswith(('lam|', 'cmp|lam|')) and sh.name in summ:
            vals, axis = summ[sh.name], 'x'
        else:
            continue
        op = sh.opacity if sh.opacity else 1.0   # 0: hidden at the figure's frame only
        shapes.append(dict(index=i, axis=axis, op=float(op), v=[float(v) if np.isfinite(v) else None for v in vals]))
        ref = sh.xref if axis == 'x' else sh.yref
        if ref and ' ' not in ref and ref != 'paper':
            moves.setdefault(_axis_key(ref), []).append(np.asarray(vals, dtype=float))
    for key, arrs in moves.items():
        ax = fig.layout[key]
        if ax.range is not None or ax.type == 'log':
            continue
        ref, along = key[0] + key[5:], key[0]   # 'xaxis2' -> 'x2', 'x'
        data, markers = [], False
        for tr in fig.data:
            if (getattr(tr, f'{along}axis') or along) == ref and getattr(tr, along) is not None:
                c = np.asarray(getattr(tr, along), dtype=float).ravel()
                c = np.sort(c[np.isfinite(c)])
                if tr.type == 'heatmap' and len(c) > 1:   # its cells reach half a cell past the end centres
                    c = np.r_[c[0] - (c[1] - c[0]) / 2, c, c[-1] + (c[-1] - c[-2]) / 2]
                data.append(c)
                markers |= 'markers' in (getattr(tr, 'mode', None) or '')
        v = np.concatenate(arrs)
        v, d = v[np.isfinite(v)], np.concatenate(data) if data else np.array([])
        # a line on or near the curves' end widens the axis too (by half its width): inside = 1 % of the span clear
        clear = 0.01 * (d.max() - d.min()) if len(d) else 0.0
        if not len(v) or not len(d) or (v.min() > d.min() + clear and v.max() < d.max() - clear):
            continue
        lo, hi = min(v.min(), d.min()), max(v.max(), d.max())
        pad = 0.05 * (hi - lo) if markers else 0.0   # plotly pads an axis with markers on it, not one with lines only
        fig.layout[key].update(range=[float(lo - pad), float(hi + pad)], autorange=False)
    # the overlay's frame matched to each of the first run's (pgkyl commands of its curves), -1 where none
    cmp_frames = ([int(summ.b['frames'][j]) if j >= 0 else -1 for j in summ.idx] if isinstance(summ, gv.Compared)
                  else None)
    return dict(frames=allf, traces=traces, shapes=shapes, over=[int(v) for v in over], cmp_frames=cmp_frames)
