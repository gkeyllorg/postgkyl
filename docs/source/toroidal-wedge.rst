Render a toroidal wedge
=======================

All scripts below run against the :download:`example bundle
<downloads/postgkyl-examples.zip>`. Unzip it, activate the environment where
Postgkyl is installed, and run the commands from the extracted directory.
The Python and CLI outputs below are both generated and compared when this website is built.

``map_to_torus`` places a three-dimensional field-aligned field on the boundary
of a toroidal wedge: two poloidal caps at the toroidal angles ``phi_tor`` joined
by the inner and outer flux surfaces selected with ``x_idx``. Negative radial
indices count back from the outer edge. The result is one closed surface in
Cartesian X, Y, Z, which ``pyvista`` colors with ``surface=True``; pass
``no_normalize=True`` so the torus keeps its physical proportions.

Every face samples the field as :doc:`map_to_rz <physical-rz>` does, so the
caps are its poloidal slices at the wedge angles and the faces agree where
they meet. Geometry is inferred from
the simulation prefix as for ``map_to_rz``. Build
``pg.resolve_torus_projection(data, geometry, ...)`` once and pass it as
``data.map_to_torus(projection=projection)`` to reuse the wedge for other
fields or frames on the same computational grid.

.. include:: _pairs/14_map_to_torus.inc
