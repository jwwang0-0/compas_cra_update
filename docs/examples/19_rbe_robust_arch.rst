********************************************************************************
Arch Construction Robust RBE Force Range
********************************************************************************

Simulate left-to-right construction of the example-06 arch. Stage ``n`` contains
blocks ``0`` through ``n - 1`` only, block 0 is fixed, and the free two-dimensional
load is applied to the current last block.

At each stage, the visible ``Fx``-``Fz`` load is realized by hidden point forces
at the four vertices of the current last block's exposed rightmost face. The
resulting hidden moment is generated geometrically by ``r x q``. Each hidden
force component uses a large ``1e6`` placeholder bound; this should later be
replaced by real actuator, contact, or material limits.

The script saves one construction-stage SVG for the default thickness and one
SVG for each comparison thickness ``0.3``, ``0.5``, and ``0.8``. Stages with an
empty safe-load set are marked directly in the corresponding subplot, and small
safe regions are shown with local zoomed axes.

.. literalinclude:: 19_rbe_robust_arch.py
    :language: python
