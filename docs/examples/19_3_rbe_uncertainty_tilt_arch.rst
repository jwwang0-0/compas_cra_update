********************************************************************************
Four-Block Arch RBE Foundation-Tilt Force Range
********************************************************************************

This example uses the same fixed four-block arch geometry as example 19_2:
``height=2``, ``span=4``, blocks ordered left to right, and block 0 fixed. The
visible safe load is the ``Fx``-``Fz`` load on block 3.

Foundation tilt is modeled as a change in gravity direction in the structure
coordinate frame. Contact geometry, contact normals, friction cones, support
conditions, and the block 3 load-application points remain fixed.

The plot compares the certainty case with symmetric tilt intervals
``±2.5°``, ``±5°``, ``±7.5°``, and ``±10°``. Each nonzero interval checks the two
endpoint tilt scenarios. The block 3 visible load is realized through hidden
point forces at the four vertices of block 3's rightmost face, with the same
large ``1e6`` placeholder force bound used by the robust examples.

.. literalinclude:: 19_3_rbe_uncertainty_tilt_arch.py
    :language: python
