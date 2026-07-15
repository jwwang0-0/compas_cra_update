********************************************************************************
Full-Arch Discretization Comparison
********************************************************************************

This example studies how the complete example-19 arch safe-load region changes
with block discretization. It is not a construction-process simulation: each
case is a full arch with block ``0`` fixed and the visible ``Fx``-``Fz`` load
applied to the last block.

The block count is swept from ``15`` through ``30`` while keeping span, height,
thickness, depth, friction, density, and hidden load-application radial-face model fixed.
For each discretization, the script solves the dual robust support problem and
prints a compact metrics table with boundedness, polygon area, and visible
``Fx``/``Fz`` bounds.

The script saves two SVG files:

* ``19_5_rbe_discretization_arch.svg`` overlays all solved safe-load regions.
* ``19_5_rbe_discretization_arch_overlay.svg`` is a clean visual-only overlay
  of all solved safe-load regions in one graph.
* ``19_5_rbe_discretization_arch_metrics.svg`` plots polygon area and load
  bounds versus block count.

.. literalinclude:: 19_5_rbe_discretization_arch.py
    :language: python
