Measure exponential growth
==========================

All scripts below run against the :download:`example bundle
<downloads/postgkyl-examples.zip>`. Unzip it, activate the environment where
Postgkyl is installed, and run the commands from the extracted directory.
The Python and CLI outputs below are both generated and compared when this website is built.

The generator supplies an analytic energy history with a known exponential
rate. Fit the logarithm of positive energy to time; its slope is the energy
growth rate. If energy is proportional to squared mode amplitude, the
amplitude growth rate is half this slope. This exact example checks the
calculation; real data require selecting a justified growth interval and
examining residuals before saturation dominates. The script checks the known rate and residuals numerically, then plots the reconstructed fitted energy.

.. include:: _pairs/06_growth.inc

Growth spectrum from field frames
---------------------------------

Plain ``growth`` prints the fitted growth rate and returns its fitted curve.
For spatial field frames, ``growth --spectrum`` returns growth rate versus
positive wavenumber. ``frequency --spectrum`` returns the corresponding signed
phase frequency. Both use DG cell averages by default; ``--interpolated``
evaluates modal fields before transforming them. Name results with ``-t`` and
select a tag later in the chain. Each command activates only its result;
previous inputs are set aside and can be restored by their load-time tag::

   pgkyl "run-field_[0-9]*.gkyl" --tag raw growth --spectrum -d 1 -t gamma \
     --time_norm R_axis/c_s --time_window 1.,5.0 \
     --space_norm rho_s/Cy plot

Append ``activate --tags raw`` later in the same chain to restore the source
frames. Without ``-t``, results use the ``growth`` or ``frequency`` tag.

For a one-dimensional oscillating time trace, plain ``frequency`` prints its
dominant angular frequency and returns a fitted sinusoid. ``frequency --fft``
instead prints the dominant temporal FFT peak and returns FFT magnitude versus
angular frequency. Both plain fits use a best leading window unless
``--time_window`` selects an explicit interval.

``--time_norm`` and ``--space_norm`` accept numbers (for example ``0.5``),
numeric metadata names, or expressions using ``+``, ``-``, ``*``, ``/``, and
parentheses. With ``--time_norm``, window bounds are in normalized time.
Without ``--time_window``, spatial spectra share a best leading fit window
chosen from the dominant positive mode.
