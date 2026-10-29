Plot in physical R–Z coordinates
================================

All scripts below run against the :download:`example bundle
<downloads/postgkyl-examples.zip>`. Unzip it, activate the environment where
Postgkyl is installed, and run the commands from the extracted directory.
The Python and CLI outputs below are both generated and compared when this website is built.

Computational coordinates are useful for analysis, but a poloidal view places
the field in its physical geometry. Keep the companion
``rt_gk_tcv_nt_iwl_3x2v_p1-geo_int_mapc2p.gkyl`` beside the electron density
file so ``pg.map_to_rz`` can resolve it by the simulation prefix. The mapping
operation uses the same geometry filenames for every Gkeyll equation system.
The matching CLI verb is ``map_to_rz``.

Load reusable geometry with ``pg.resolve_geometry(data.file_name)`` or construct
``pg.Geometry`` from coordinate arrays, build
``pg.resolve_rz_projection(data, geometry)``, and apply it with
``data.map_to_rz(projection=projection)``. The projection can be reused for
fields on the same computational grid. A three-dimensional field is sliced
at ``phi_tor`` by evaluating its DG expansion where the geometry places that
toroidal angle along the binormal coordinate. Angles that the simulated
binormal domain covers are exact. When that domain is not a whole fraction of
the torus, the rest of the plane repeats it, and the copies meet with a jump
along field lines.

``data.map_to_phitheta(x_idx=...)`` (CLI verb ``map_to_phitheta``) samples one
flux surface the same way over a full toroidal turn, on its (phi, theta) plane.
To render poloidal cuts and flux surfaces together in 3-D, see
:doc:`toroidal-wedge`.

.. include:: _pairs/05_map_to_rz.inc
