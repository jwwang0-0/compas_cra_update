Three-Block RBE With Geometry Uncertainty
=========================================

This example uses the same three-block geometry as example 18. Block 0 is
fixed, and the visible safe load remains the ``(Fx, Fz)`` load on block 2. The
load is applied through four candidate points on the rightmost side of block 2,
with the same large ``1e6`` placeholder hand-force bound used by the robust
examples.

The geometry-uncertainty solver enumerates finite scenarios and intersects the
corresponding safe-load sets. The plotted cases compare nominal geometry,
region shrinkage, local point offsets, interface-frame tilts, single
contact-point loss, and a small combined shrink-plus-tilt scenario set. These
are finite scenario checks, not a continuous nonlinear geometry uncertainty
model.

.. literalinclude:: 20_rbe_uncertainty_geometry_3block.py
    :language: python
