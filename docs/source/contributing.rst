Troubleshooting and documentation development
===============================================

If an operation reports missing Gkeyll support, run:

.. code-block:: bash

   python -c "from postgkyl import gpython; gpython.require()"

Follow the bridge rebuild instructions in :doc:`installation`. If an operation
instead rejects modal coefficients, use the explicit representation change
appropriate to the calculation; see
:doc:`concepts`.

For a machine without a display, set ``MPLBACKEND=Agg`` and use
``no_show=True`` or ``--no_show``. Example scripts already save their plots
without opening windows.

Build the documentation
-------------------------

Follow the documentation development setup in :doc:`installation`, then run
from the Postgkyl checkout:

.. code-block:: bash

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

The PyVista examples require OpenGL even when no window is shown. See the
rendering dependencies in :doc:`installation` for headless setup.

Edit guides in ``docs/source/``, examples in ``examples/``, and API descriptions
in implementing function docstrings. The command reference reads the actual
compiled CLI. Both websites use this same preparation step; generated files
are not maintained by hand.

The Gkeyll host fetches Postgkyl ``main`` on each build. Its scheduled GitHub
Actions check detects breakage from upstream changes even when the host
repository has no new commits. Postgkyl pull requests build their proposed
code and documentation together before merge.

Publish the documentation
-------------------------

After each push to Postgkyl ``main``, including merged pull requests, the
Documentation workflow requests a build of the ``gkeyll`` Read the Docs
project's ``latest`` version after its tests and preview build pass. This
build uses the separate ``gkeyllorg/gkyl-doc`` repository, whose preparation
script fetches Postgkyl ``main``. Pull requests only build previews.

To request another build, run Postgkyl's Documentation workflow manually on
``main``. The trigger logs a link to the hosted build; successful acceptance
does not mean that the hosted build has finished. Read the Docs publishes
the updated site when that build succeeds. Each hosted build fetches the
current ``main``, which may include newer merges than the triggering commit.
