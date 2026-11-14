# Fonts the themes use, tracked so every machine renders them

The themes' own fonts are proprietary (Arial, Century Gothic: Monotype/Microsoft) and cannot be shipped here. These
free fonts stand in for them wherever the real one is not installed: in the matplotlib figures and movies (registered
with matplotlib at import, `viewer_plots.FONT_FILES`, ahead of any system copy of the same family: Perlmutter's
Liberation Sans 1.07.4 lacks the thin space), and in the page and the slide downloads of its charts (sent
to the browser once and registered with `FontFace`, `movie_widget.FontLoader`). An exported SVG names the real font
first, so PowerPoint uses it where it is installed.

| family | stands in for | theme, role | version | source | license |
|---|---|---|---|---|---|
| Liberation Sans | Arial (metric-compatible: same advance widths) | NSTX-U, PPPL: text | 2.1.5 | github.com/liberationfonts/liberation-fonts, release 2.1.5 (liberation-fonts-ttf-2.1.5.tar.gz) | SIL OFL 1.1, `LICENSE-LiberationSans.txt` |
| Roboto Condensed | itself (PPPL's title font; a Google font) | PPPL: titles | 3.008 | fonts.google.com (static instances served by the Google Fonts CSS API) | SIL OFL 1.1, `LICENSE-RobotoCondensed.txt` |
| TeX Gyre Adventor | Century Gothic (a look-alike: URW Gothic, the same geometric design; not metric-compatible) | DIII-D: text and titles | 2.501 | CTAN, fonts/tex-gyre/opentype | GUST Font License, `LICENSE-TeXGyreAdventor.txt` |
| Viewer Symbols | the symbols the fonts above lack | every theme, after its font | DejaVu Sans 2.35 subset | built here: `make_symbols.py` | Bitstream Vera license (DejaVu changes: public domain), `LICENSE-DejaVu.txt` |

Regular, bold, italic and bold italic of each of the first three; Viewer Symbols regular and bold.

**Viewer Symbols** holds what a theme's upright face here lacks of the characters the viewer's sources write: ∥ ⊥
(Liberation Sans, Roboto Condensed), → (Roboto Condensed), ˣ (Roboto Condensed, TeX Gyre Adventor), and Unicode sub-
and superscripts of plain-text labels (texlabel.plain; each of the three lacks some, TeX Gyre Adventor most). Without it the page drew ∥ and ⊥ as empty boxes ("tofu") in the
NSTX-U and PPPL themes, and text_px measured them in a font without them (matplotlib's "Glyph 8741 (\N{PARALLEL TO})
missing from font(s) Liberation Sans"). `make_symbols.py` finds those characters (every non-ASCII character in the
viewer's `*.py` that one upright face lacks: no hand-kept list), cuts them from matplotlib's copy of DejaVu Sans
(fontTools subset) and renames the result, as the Vera license asks of a modified font. The page lists it after each
theme font (viewer_plots.css_font) and text_px measures with it the same way. Matplotlib figures fall back to the
whole DejaVu Sans instead (viewer_plots.FALLBACK). The build is reproducible; `check_viewer.py` fails when a source
gains a symbol the font lacks, or when the files differ from a fresh build: then run

```
<viewer venv>/bin/python fonts/make_symbols.py
```

SHA-256 (fetched 2026-11-12; Viewer Symbols built 2026-11-13 from matplotlib 3.11.0's DejaVu Sans, fontTools 4.68.0):
```
1d371637a4df5bb34899c16d1485ddb11ab1712239301fd543926fe9c99fa5fb  ViewerSymbols-Bold.ttf
241aa929794c65000c4ce28917575493aee37c3b45b5a5ebc41dcfe4a4905fc5  ViewerSymbols.ttf
698da70fc191cc5f33ad4d6d3fe830fe4624b898ea2e3169955928b7c491f1ee  LiberationSans-BoldItalic.ttf
788abee4c806d660e8aee46689dd8540cd4bb98da03dcc9d171ce3efd99a9173  LiberationSans-Bold.ttf
e5bae5c4cde31f22142753855f4f8fb86da6ff39955ed3c0a11248b0d16948b0  LiberationSans-Italic.ttf
76d04c18ea243f426b7de1f3ad208e927008f961dc5945e5aad352d0dfde8ee8  LiberationSans-Regular.ttf
6b2a812369cb0b5484023afd8d9886b513c15e9e450b856f31c5660f9017c452  RobotoCondensed-BoldItalic.ttf
35140bfdde80295fbcd04dad086f51688abca96cb8c87529daf781c4462b6e33  RobotoCondensed-Bold.ttf
135aea79fc5725b1688f9ea6007666f0eb426231ead6a87e1573c5fc03874b38  RobotoCondensed-Italic.ttf
ef9d48725ff2bd72cd569ca50555fb781f99cf8f3a88a3bb3c8a4c85fd574458  RobotoCondensed-Regular.ttf
d524c5596812aeee8fed5c81739056ada7a305b84591ea30e30c025b860b311d  texgyreadventor-bolditalic.otf
e373ae29c808c6d9149e3add46564f9acdb6ebfcd0ee311f6a55560226994720  texgyreadventor-bold.otf
1f21a5a07c207f70785dd6cbe4fcb2e0a3f60823b219efb01790a72d8a61b3cb  texgyreadventor-italic.otf
ee265c1c9728be9491c7f0d7fcdcb2f97bc35040a6dc1f9815265cee333e155b  texgyreadventor-regular.otf
```
