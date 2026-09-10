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
Do not pass the collected space–time dataset as though it were a sequence.
The example keeps a fixed value range across all frames, so a changing color
does not merely reflect a changing scale. The second animation shows a 2-D
travelling wave. ``saveframes`` also saves each frame as a PNG; ``fps`` controls
playback speed, independently of the physical times stored in the files.

Both GIFs below are generated and checked frame by frame during the build.
Use the browser's image controls or open the images separately to inspect them.

.. include:: _pairs/07_collect_animate.inc
