********************************************************************************
Four-Block Arch RBE Disturbance-Uncertainty Force Range
********************************************************************************

This example uses the example-06 arch geometry with ``height=2`` and ``span=4``,
but keeps only four left-to-right blocks. Block 0 is fixed, blocks 1 and 2 carry
uncertain external disturbances, and the visible safe load is the ``Fx``-``Fz``
load on block 3.

The disturbance on blocks 1 and 2 is modeled as independent two-dimensional
boxes:

``Fx_i, Fz_i in [-r Wi, r Wi]``

where ``Wi`` is the corresponding block weight and ``r`` is ``5%``, ``10%``,
``15%``, or ``20%``. A ``0%`` case is included as the baseline. This is a fixed
four-block analysis, not a construction-process simulation.

The block 3 visible load is realized through hidden point forces at the four
vertices of block 3's right exposed radial face. The hidden force bound remains the large
``1e6`` placeholder used by the robust examples.

.. literalinclude:: 19_2_rbe_uncertainty_disturb_arch.py
    :language: python
