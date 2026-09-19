Load many files, collect, and animate
=======================================

The generated travelling-wave examples represent the same positive density
wave at successive times. These analytic fixtures demonstrate file handling
and visualization; they are not numerical simulation results. Each file has
its own frame number and time metadata.

Download the :download:`example bundle <downloads/postgkyl-examples.zip>`.
The zero-padded names (``travelling_wave_000.gkyl`` through frame 015) make
lexicographic sorting match time order. The script uses ``sorted(Path.glob(...))``;
the CLI accepts the quoted wildcard. Use ``sort`` with your own files if
their names or input order do not match time order.

``collect(frames)`` makes a single dataset whose leading grid axis is time.
A 1-D sequence therefore becomes a 2-D space–time plot. Its timestamps are
sorted using the files' metadata. Interpolate before collecting.

``animate(frames)`` instead draws the original individual frames in sequence.
To play a collected space–time dataset, use
``animate([collected_data], collected=True)``.
The example keeps a fixed value range across all frames, so a changing color
does not merely reflect a changing scale. The second animation shows a 2-D
travelling wave. ``saveframes`` also saves each frame as a PNG; ``fps`` controls
playback speed, independently of the physical times stored in the files.

Both GIFs below are generated and checked frame by frame during the build.
Use the browser's image controls or open the images separately to inspect them.

.. include:: _pairs/07_collect_animate.inc

Animation controls
------------------

Animation accepts the plotting controls for contours, quiver, streamlines,
scatter, line styling, labels, legends, colorbars, axis transforms, logarithmic
scales, aspect ratio, and mesh edges. ``subplots=True`` gives each dataset in a
frame its own panels; ``squeeze=True`` overlays components in one panel.
``group=0`` or ``group=1`` draws field lineouts along that coordinate.

``xlim``, ``ylim``, and ``zlim`` accept pairs (or comma-separated strings) and
supersede individual minimum/maximum bounds. Fixed ranges include value shifts
and scaling. ``variable_range=True`` recomputes the range over all blocks of
each frame, while respecting explicit bounds. ``cutoffglobalrange`` selects a
central fraction of the dataset extrema, between zero and one.

The generated CLI uses the public API's names: the former ``--float`` control
is ``--variable_range``, ``--nsubplotrow`` / ``--nsubplotcol`` are
``--num_subplot_row`` / ``--num_subplot_col``, and ``--fix-aspect`` is
``--fixaspect``. Use ``--no_legend``, ``--no_colorbar``, ``--no_showgrid``, and
``--no_show`` to disable default display features. ``--forcelegend`` forces a
legend. All options are listed by ``pgkyl animate --help``.

``use`` selects a tag; ``grouptags=True`` produces separate animations with
tag suffixes on output filenames. ``multiblock=True`` combines blocks with the
same frame index. ``saveframes`` writes numbered PNG files; ``nproc`` selects
the number of frame workers and ``tmpdir`` places temporary frames. GIF, WebP,
and APNG output use Pillow; MP4, MOV, AVI, and MKV require ffmpeg.

Video encoders on clusters
------------------------------

Finding an ``ffmpeg`` executable is not enough: it must contain an encoder
for the requested video format. Animation checks encoders before generating
frames. It prefers software H.264 (``libx264`` or ``libopenh264``) in the
executable on ``PATH``, then tries the executable supplied by
``imageio-ffmpeg``. If neither provides software H.264, it uses ``mpeg4``.
Hardware encoders are used only when explicitly requested.

Use ``--codec mpeg4`` (Python: ``codec="mpeg4"``) to choose MPEG-4 explicitly,
or ``--codec libx264`` to require that encoder. Explicit choices produce a
clear error if unavailable. For example::

    pgkyl "frames_*.gkyl" interpolate animate --saveas movie.mp4 --nproc 10 --codec mpeg4

``pip install ffmpeg`` installs a Python package, not an ffmpeg executable.
A pip installation that includes an executable is available through
``python -m pip install -U imageio-ffmpeg``. GIF, WebP, and APNG require no
ffmpeg installation. Video export reports the chosen executable and encoder
if encoding fails, uses a noninteractive canvas when compiling saved frames,
and pads odd-sized H.264/MPEG-4 frames to even dimensions.
