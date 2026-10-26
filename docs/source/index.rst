topologysegmentation
=====================

PyTorch losses and metrics for topology-aware segmentation. Each loss is a
class. :doc:`losses` lists every class and the meaning of its parameters.

`Repository <https://github.com/ProninEgorDm/TopologyLosses>`_ ·
`PyPI <https://pypi.org/project/topologysegmentation>`_

Install::

   pip install topologysegmentation
   pip install "topologysegmentation[ph]"

The first command provides Dice, clDice, T-Loss, projected pooling, and
critical points. The second adds Hu, Betti matching, and SAT.

.. code-block:: python

   from topologysegmentation import CLDiceLoss, DiceLoss, SumLoss

   criterion = SumLoss((DiceLoss(), 1.0), (CLDiceLoss(), 0.1))
   value = criterion(logits, target)

By default every loss applies a sigmoid. Pass logits, or set
``sigmoid=False`` when the tensor is already a probability.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   losses
