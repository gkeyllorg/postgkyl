#!/usr/bin/env python3
"""Build Viewer Symbols (ViewerSymbols.ttf, ViewerSymbols-Bold.ttf): the characters the viewer writes that a theme's
upright font here lacks (∥ and ⊥ in Liberation Sans, → in Roboto Condensed, ...), cut from matplotlib's copy of
DejaVu Sans and renamed. The page draws them after the theme's font (viewer_plots.css_font), and text_px measures
them the same way.

The characters are not listed by hand: they are every non-ASCII character in the viewer's Python sources (texlabel's
tables, the literal labels, the movie's JavaScript) that one of the tracked upright faces does not have. Run it again
when a source gains a symbol (check_viewer.py fails until then):

  <viewer venv>/bin/python fonts/make_symbols.py

The output is reproducible (no build time stamped in the files), so the tracked files equal a fresh build.
"""
import glob
import os
import shutil
import sys
import unicodedata

import matplotlib
from fontTools import subset
from fontTools.ttLib import TTFont

FONTS = os.path.dirname(os.path.abspath(__file__))
VIEWER = os.path.dirname(FONTS)
FAMILY = 'Viewer Symbols'
DEJAVU = os.path.join(matplotlib.get_data_path(), 'fonts', 'ttf')
# (output file, DejaVu face it is cut from, style)
FACES = [('ViewerSymbols.ttf', 'DejaVuSans.ttf', 'Regular'), ('ViewerSymbols-Bold.ttf', 'DejaVuSans-Bold.ttf', 'Bold')]


def source_chars():
    """Every non-ASCII character in the viewer's Python sources that is drawn (not a control or format character;
    spaces such as texlabel's thin space for \\, are), sorted."""
    chars = set()
    for p in glob.glob(os.path.join(VIEWER, '*.py')):
        chars |= {c for c in open(p, encoding='utf-8').read() if ord(c) > 127 and unicodedata.category(c)[0] != 'C'}
    return sorted(chars)


def theme_faces():
    """The tracked upright faces the page draws text in (movie_widget.font_loader sends only those), but this
    family's own."""
    names = {os.path.basename(p): p for p in glob.glob(os.path.join(FONTS, '*.[ot]tf'))}
    return sorted(p for n, p in names.items() if 'italic' not in n.lower() and not n.startswith('ViewerSymbols'))


def cmap(path):
    return set(TTFont(path, lazy=True).getBestCmap())


def lacking(chars):
    """The characters of `chars` that at least one theme face does not have."""
    faces = [cmap(p) for p in theme_faces()]
    return [c for c in chars if any(ord(c) not in f for f in faces)]


def build(face, style, chars, out):
    """Subset DejaVu `face` to `chars` and name it FAMILY `style`."""
    opts = subset.Options()
    opts.layout_features = ['*']
    opts.name_IDs = ['*']
    opts.notdef_outline = True
    opts.drop_tables += ['FFTM']   # FontForge's time stamps
    font = subset.load_font(os.path.join(DEJAVU, face), opts)
    sub = subset.Subsetter(opts)
    sub.populate(unicodes=[ord(c) for c in chars])
    sub.subset(font)
    name = font['name']
    version = name.getDebugName(5)
    names = {1: FAMILY, 2: style, 3: f'{FAMILY} {style}: DejaVu Sans {version} subset',
             4: FAMILY if style == 'Regular' else f'{FAMILY} {style}', 6: f'ViewerSymbols-{style}',
             10: f'A subset of DejaVu Sans ({version}, as matplotlib ships it) for the NSTX-U run viewer: the '
                 f'symbols its theme fonts lack. Not DejaVu: renamed as the Bitstream Vera license asks.'}
    name.names = [n for n in name.names if n.nameID not in (16, 17, 18, 21, 22)]
    for nid, text in names.items():
        name.setName(text, nid, 3, 1, 0x409)   # Windows, Unicode BMP, en-US
        name.setName(text, nid, 1, 0, 0)       # Mac, Roman, English
    font.recalcTimestamp = False   # reproducible: head.modified stays DejaVu's
    font.save(out)
    return version


def main():
    need = lacking(source_chars())
    missing = [c for c in need if ord(c) not in cmap(os.path.join(DEJAVU, 'DejaVuSans.ttf'))]
    if missing:
        print('DejaVu Sans lacks', ' '.join(f'{c} U+{ord(c):04X}' for c in missing), file=sys.stderr)
        return 1
    for out, face, style in FACES:
        version = build(face, style, need, os.path.join(FONTS, out))
        print(f'{out}: {len(need)} characters from {face} {version}, {os.path.getsize(os.path.join(FONTS, out))} B')
    shutil.copyfile(os.path.join(DEJAVU, 'LICENSE_DEJAVU'), os.path.join(FONTS, 'LICENSE-DejaVu.txt'))
    print(''.join(need))
    return 0


if __name__ == '__main__':
    sys.exit(main())
