Build a command line from Python
================================

``PostgkylSession`` drives the ``pgkyl`` command line from Python, one
command per call, and records the equivalent command line. Develop a figure
in a script or notebook, where an IDE shows each command's options and
docstring as you type, then print the one ``pgkyl`` command that redraws it.

Install Postgkyl using :doc:`installation`. Download and unzip the
:download:`example bundle <downloads/postgkyl-examples.zip>`, then run the
script below from the extracted directory.

The script loads the electron density of a 3x2v gyrokinetic turbulence
simulation and takes its fluctuation about the average over the binormal
direction :math:`y`,

.. math::

   \delta n_e = n_e - \langle n_e \rangle_y .

It maps the fluctuation onto the poloidal (R-Z) plane, up-sampling the
parallel direction by 12 points per cell, and plots it with a diverging color
map centered on zero. ``map_to_rz`` infers the geometry from the simulation
prefix. ``s.print_cli()`` then prints the command line that draws the same
figure: the command below, with the script's absolute paths. The build
compares the pixels of the two figures.

Every command is a method with that command's options, so
``help(s.map_to_rz)`` lists them. Each returns the session, so calls chain or
stand alone; what the last command returned, here ``plot``'s figure, is
``s.result``. Each call parses and runs the very tokens it records, on the
same working set as the command line, so ``s.command()`` is exactly what ran.
The current datasets are ``s.datasets``, which the fluent API can inspect at
any step: the script checks that the fluctuation's :math:`y` average vanishes
and that the mapped field matches the fluent API's.

A dataset passed to an option such as ``average --weight`` is named by its
``tag``. A value with no command-line spelling, such as a NumPy array or a
``slice``, is refused with a ``TypeError``; use the string selector ``"1:3"``
instead. A failed call is not recorded.

.. include:: _pairs/13_postgkyl_session.inc
