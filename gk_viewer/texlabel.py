"""Labels of physical quantities, written once in LaTeX and set as LaTeX would set them in each place they appear.

A label is text with math between dollar signs: 'electron density ($10^{19}\\,\\mathrm{m^{-3}}$)', '$n_\\mathrm{e}$
change (%)'. Inside the dollars: letters italic, \\mathrm{...} upright (descriptive subscripts, units), _ and ^ sub- and
superscripts (one token or a {group}), Greek letters and \\parallel, \\perp, \\int, \\times, \\pm, the spaces \\, \\; and
'\\ '. Three renderings:
- mpl: matplotlib's mathtext, as written (viewer_plots.mpl_style sets its fonts to the theme's);
- html: plotly's subset (<i>, <sub>, <sup>), in the chart's font;
- plain: Unicode text (pgkyl commands, the cells table, the movie's canvas): sub- and superscripts as Unicode where
  every character has one (n_e -> nₑ, 10^{19} -> 10¹⁹), else _x and ^x.
"""
import re

GREEK = {
    'alpha': 'α',
    'beta': 'β',
    'gamma': 'γ',
    'delta': 'δ',
    'epsilon': 'ε',
    'zeta': 'ζ',
    'eta': 'η',
    'theta': 'θ',
    'kappa': 'κ',
    'lambda': 'λ',
    'mu': 'μ',
    'nu': 'ν',
    'xi': 'ξ',
    'pi': 'π',
    'rho': 'ρ',
    'sigma': 'σ',
    'tau': 'τ',
    'phi': 'φ',
    'chi': 'χ',
    'psi': 'ψ',
    'omega': 'ω',
    'Gamma': 'Γ',
    'Delta': 'Δ',
    'Theta': 'Θ',
    'Lambda': 'Λ',
    'Pi': 'Π',
    'Sigma': 'Σ',
    'Phi': 'Φ',
    'Psi': 'Ψ',
    'Omega': 'Ω'
}
SYMBOL = {
    'parallel': '∥',
    'perp': '⊥',
    'int': '∫',
    'times': '×',
    'pm': '±',
    'cdot': '·',
    ',': ' ',
    ';': ' ',
    ' ': ' ',
    'leq': '≤',
    'geq': '≥',
    'approx': '≈'
}
SUB = dict(
    zip('0123456789+-−=()aeiorsxjklmnpt', '₀₁₂₃₄₅₆₇₈₉₊₋₋₌₍₎ₐₑᵢₒᵣₛₓⱼₖₗₘₙₚₜ'))
SUP = dict(zip('0123456789+-−=()in', '⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁻⁼⁽⁾ⁱⁿ'))
_TOKEN = re.compile(r'\\([A-Za-z]+|.)|([{}_^])|(.)', re.S)


def _parse(s):
  """A math string as nested items: ('chr', c, upright) | ('sub' | 'sup', items) | ('group', items)."""
  toks = [m.groups() for m in _TOKEN.finditer(s)]
  pos = 0

  def atom(upright):
    nonlocal pos
    cmd, op, ch = toks[pos]
    pos += 1
    if op == '{':
      return ('group', seq(upright, '}'))
    if cmd in ('mathrm', 'text'):
      return atom(True) if pos < len(toks) else ('group', [])
    if cmd is not None:
      if cmd in GREEK:
        return ('chr', GREEK[cmd], upright or cmd[0].isupper())
      return ('chr', SYMBOL.get(cmd, cmd), True)
    if ch == ' ':  # math ignores spaces (\, and \; make them)
      return ('group', [])
    return ('chr', ch, upright or not ch.isalpha())

  def seq(upright, end=None):
    nonlocal pos
    out = []
    while pos < len(toks):
      op = toks[pos][1]
      if end is not None and op == end:
        pos += 1
        return out
      if op in ('_', '^'):
        pos += 1
        out.append(('sub' if op == '_' else 'sup', [atom(upright)]))
      else:
        out.append(atom(upright))
    return out

  return seq(False)


def _html(items):
  out, italic = [], []

  def flush():
    if italic:
      out.append('<i>' + ''.join(italic) + '</i>')
      italic.clear()

  for it in items:
    if it[0] == 'chr':
      c = '−' if it[1] == '-' else it[1]
      if it[2]:
        flush()
        out.append(c)
      else:
        italic.append(c)
    elif it[0] == 'group':
      flush()
      out.append(_html(it[1]))
    else:
      flush()
      tag = 'sub' if it[0] == 'sub' else 'sup'
      out.append(f'<{tag}>{_html(it[1])}</{tag}>')
  flush()
  return ''.join(out)


def _plain(items):
  out = []
  for it in items:
    if it[0] == 'chr':
      out.append('−' if it[1] == '-' else it[1])
    elif it[0] == 'group':
      out.append(_plain(it[1]))
    else:
      inner = _plain(it[1])
      table, mark = (SUB, '_') if it[0] == 'sub' else (SUP, '^')
      out.append(''.join(
          table[c] for c in inner) if inner and all(c in table
                                                    for c in inner) else mark +
                 (inner if len(inner) == 1 else f'({inner})' if mark ==
                  '^' else inner))
  return ''.join(out)


def _render(s, math):
  parts = re.split(r'(?<!\\)\$', s or '')
  return ''.join(math(_parse(p)) if k % 2 else p for k, p in enumerate(parts))


def html(s):
  """A label for plotly: math italic, sub- and superscripts as <sub> / <sup>."""
  return _render(s, _html)


def plain(s):
  """A label as Unicode text."""
  return _render(s, _plain)


def mpl(s):
  """A label for matplotlib (mathtext reads the dollars itself)."""
  return s or ''


def unit(text, u):
  """'text (unit)' with the unit (math, no dollars) set as math; text alone without a unit."""
  return f'{text} (${u}$)' if u else text
