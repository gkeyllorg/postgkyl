postgkyl notebooks
==================

Notebooks using Marimo or Jupyter are useful collections of operations to pre- or post-process Gkeyll simulations, or better understand their algorithms. This folder contains some such notebooks.
File convention: We use the `*.mo.py` extension for Marimo notebooks, which are Python scripts that can be executed in a notebook-like environment.

- twist_shift.mo.py: Analyze the impact of twist-shift boundary conditions on mode shearing and aliasing, and test possible filtering strategies to mitigate these effects.
- dg_twist_shift.mo.py: Apply DG twistshift to a quantity produced by the initialization of a Gkeyll gyrokinetic simulation.


### How to use Marimo notebooks

1. Install Marimo: e.g. `pip install marimo`.
2. Run the notebook: e.g. `marimo run <notebook_name>`.

This will execute the notebook in a web browser, allowing you to interact with the code and visualize the results.
