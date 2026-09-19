Troubleshooting and documentation development
===============================================

If an operation reports missing Gkeyll support, run:

.. code-block:: bash

   python -c "from postgkyl import gpython; gpython.require()"

Follow the bridge rebuild instructions in :doc:`installation`. NumPy must
be installed before building the bridge, and the same NumPy ABI must be used
at runtime. If an operation instead rejects modal coefficients, use the
explicit representation change appropriate to the calculation; see
:doc:`concepts`.

For a machine without a display, set ``MPLBACKEND=Agg`` and use
``no_show=True`` or ``--no_show``. Example scripts already save their plots
without opening windows.

Build the documentation
-------------------------

From a Postgkyl checkout, using Python 3.12:

.. code-block:: bash

   python -m pip install --upgrade numpy setuptools wheel
   python -m pip install --no-build-isolation -e '.[docs,test]'
   POSTGKYL_REQUIRE_GKEYLL=1 MPLBACKEND=Agg python -m pytest tests/test_examples.py tests/test_documentation.py tests/test_docs_build.py
   python scripts/build_docs.py
   python -m sphinx -W --keep-going -b html -c docs build/docs/source build/docs/html

Open ``build/docs/html/index.html``. The preparation step regenerates fixtures
and figures and replaces only an output directory bearing its ownership marker.
The example bundle preserves all paths used by the scripts and CLI walkthrough.

``examples/figure_commands.json`` owns the CLI commands paired with the scripts.
``examples/compare_interfaces.py`` executes both interfaces and fails on changed
pixels, animation timing, or Plotly data/layout. The generated website includes
that comparison report and both outputs. To run just the paired gallery:

.. code-block:: bash

   python examples/compare_interfaces.py

The PyVista examples require OpenGL even when no window is shown. Headless
CI installs Mesa/EGL and sets ``VTK_DEFAULT_OPENGL_WINDOW=vtkEGLRenderWindow``
and ``LIBGL_ALWAYS_SOFTWARE=1``. See :doc:`interactive` for local setup.

Edit guides in ``docs/source/``, examples in ``examples/``, and API descriptions
in implementing function docstrings. The command reference reads the actual
compiled CLI. Both websites use this same preparation step; generated files
are not maintained by hand.

The Gkeyll host fetches Postgkyl ``main`` on each build. Its scheduled GitHub
Actions check detects breakage from upstream changes even when the host
repository has no new commits. Postgkyl pull requests build their proposed
code and documentation together before merge.
