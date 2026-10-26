Losses
======

Every loss is a class. Construct it with the parameters below, then call it
as ``loss(logits, target)``. The default return value is one scalar, so
``loss.backward()`` works directly.

.. code-block:: python

   from topologysegmentation import CLDiceLoss, DiceLoss, SumLoss

   criterion = SumLoss((DiceLoss(), 1.0), (CLDiceLoss(alpha=0.5), 0.1))
   value = criterion(logits, target)
   value.backward()

A one-off mix is ordinary addition:
``DiceLoss()(logits, target) + 0.1 * CLDiceLoss()(logits, target)``.

Calling convention
------------------

A frame is ``(N, 1, H, W)``. A volume is ``(N, 1, D, H, W)``. One frame may
be ``(H, W)`` or a channel-free batch ``(N, H, W)``. A tensor ``(D, H, W)``
without a channel is a batch of frames. A side of length 1 is still that
axis.

``sigmoid=True`` (the default on every loss) reads logits and applies a
sigmoid. ``sigmoid=False`` keeps values that are already probabilities in
``[0, 1]``. Set only one of ``sigmoid``, ``softmax``, and ``other_act`` on
``DiceLoss``.

``reduction`` is accepted by ``DiceLoss``, ``BCELoss``, ``FocalLoss``,
``TLoss``, ``HuTopoLoss``, and ``BettiMatchingLoss``.

``"mean"``
    Average the remaining values. This is the default.
``"sum"``
    Add them.
``"none"``
    Return the unreduced tensor: per image, per class, or per voxel,
    depending on the loss. Useful while debugging. ``SumLoss`` adds scalars
    and raises if a term returns a tensor.

``loss_shapes()`` maps each short name to the spatial ranks it accepts.
``2`` is a frame, ``3`` is a volume. ``sat`` and ``bm`` raise ``ValueError``
on a volume. For a volume, Betti matching is ``BettiMatching3DLoss``.

SumLoss
-------

Weighted sum of loss modules. ``forward`` returns one scalar. After the
call, ``parts`` maps each name to that term already multiplied by its weight.

A term is a module, a ``(module, weight)`` pair, or a
``(module, weight, name)`` triple. The name defaults to the class attribute
``name`` (``dice``, ``cldice``, and so on). A repeated name becomes
``dice_2``. The name has to be a Python identifier.

``weights``
    Dict of the current weights. Assign ``criterion.weights["hu"] = 0.05``
    between epochs. A weight of ``0`` skips that term, so an expensive loss
    can stay in the module and start on a later epoch.
``parts``
    Dict filled by the last ``forward``. Values are the weighted terms.
``terms``
    ``ModuleDict`` of the loss modules. ``parameters()`` includes learnable
    tensors such as ``TLoss.nu``.

DiceLoss
--------

Soft Dice. The default is binary logits: sigmoid, one Dice per image, then
the mean,

``1 - (2 * intersection + smooth) / (prediction_sum + target_sum + smooth)``.

With ``smooth=1e-6`` this is the overlap term used next to the topology
losses. Several channels are kept. ``(N, C, H, W)`` or ``(N, C, D, H, W)``
is a multi-class map. A rank-3 tensor is still a batch of single-channel
frames.

``include_background`` : bool, default ``True``
    When ``False`` and there is more than one channel, channel 0 is left
    out of the sum. On a single channel the flag is ignored.
``to_onehot_y`` : bool, default ``False``
    ``target`` is a one-channel map of class indices and is expanded to the
    channel count of the prediction. On a single channel the flag is ignored.
``sigmoid`` : bool, default ``True``
    Apply a sigmoid. With several channels this is multi-label: each channel
    is an independent probability.
``softmax`` : bool, default ``False``
    Apply a softmax over channels. This is the setting for exclusive classes.
    On a single channel the flag is ignored.
``other_act`` : callable or ``None``, default ``None``
    A function applied to the prediction instead of sigmoid or softmax, for
    example ``torch.tanh``.
``squared_pred`` : bool, default ``False``
    Square the prediction inside the overlap sums (the squared Dice used by
    V-Net).
``jaccard`` : bool, default ``False``
    Compute soft IoU. The denominator is the union. With ``False`` the
    denominator is the sum of the prediction and the target, which is Dice.
``reduction`` : ``"mean"``, ``"sum"``, or ``"none"``, default ``"mean"``
    See the calling convention. ``"none"`` returns a tensor of shape
    ``(N, C)``, or ``(C,)`` when ``batch=True``.
``smooth`` : float, default ``1e-6``
    Added to the numerator and the denominator. ``smooth_nr`` and
    ``smooth_dr`` override the two sides separately.
``smooth_nr`` : float or ``None``, default ``None``
    Numerator constant. ``None`` uses ``smooth``.
``smooth_dr`` : float or ``None``, default ``None``
    Denominator constant. ``None`` uses ``smooth``.
``batch`` : bool, default ``False``
    Sum the overlap over the batch before dividing, so the batch contributes
    one Dice per class. With ``False`` each image is scored on its own and
    the scores are then reduced.
``weight`` : float, sequence, tensor, or ``None``, default ``None``
    Per-class multiplier, applied when there is more than one channel.
    Values are ``>= 0``. A single number is repeated for every channel.
    The length matches the channel count after ``include_background`` drops
    channel 0.
``soft_label`` : bool, default ``False``
    ``target`` contains values between 0 and 1. False positives and false
    negatives are weighted by the gap between prediction and target
    (Wang et al., MICCAI 2023). With ``False`` the ordinary overlap sums
    are used, including for soft targets.
``ignore_index`` : int or ``None``, default ``None``
    Drop voxels whose label equals this value. The target is a one-channel
    label map. Use a sentinel that is not a class id, for example ``-1``.

BCELoss
-------

Binary cross-entropy, one value per voxel. Several channels are several
labels. Exclusive classes belong in ``DiceLoss(softmax=True, to_onehot_y=True)``.

``sigmoid`` : bool, default ``True``
    Use ``binary_cross_entropy_with_logits``. ``False`` uses
    ``binary_cross_entropy`` on probabilities.
``reduction`` : ``"mean"``, ``"sum"``, or ``"none"``, default ``"mean"``
    ``"none"`` returns the per-voxel map. With ``ignore_index``, ``"mean"``
    divides by the number of kept voxels.
``pos_weight`` : float, sequence, tensor, or ``None``, default ``None``
    Multiplier of the positive class, forwarded to
    ``binary_cross_entropy_with_logits``. The length is 1 or the channel
    count. Requires ``sigmoid=True``.
``ignore_index`` : int or ``None``, default ``None``
    Drop voxels whose label equals this value.

FocalLoss
---------

Focal loss (Lin et al., RetinaNet). Easy voxels, where the predicted
probability of the true label is already high, are down-weighted by
``(1 - p) ** gamma``.

``alpha`` : float, default ``0.25``
    Constant multiplier of the whole loss.
``gamma`` : float, default ``2.0``
    Focusing strength. ``0`` leaves a scaled cross-entropy.
``sigmoid`` : bool, default ``True``
    Same switch as ``BCELoss``.
``reduction`` : ``"mean"``, ``"sum"``, or ``"none"``, default ``"mean"``
    Same meaning as ``BCELoss``.
``ignore_index`` : int or ``None``, default ``None``
    Drop voxels whose label equals this value.

CombinedLoss
------------

``bce_weight * BCELoss + dice_weight * DiceLoss``. Both terms share
``sigmoid``. For any other mix, use ``SumLoss``.

``bce_weight`` : float, default ``0.5``
    Weight of the cross-entropy term.
``dice_weight`` : float, default ``0.5``
    Weight of the Dice term.
``smooth`` : float, default ``1e-6``
    Passed to ``DiceLoss``.
``sigmoid`` : bool, default ``True``
    Passed to both terms.

CLDiceLoss
----------

Soft centerline Dice (Shit et al., `arXiv:2003.07311 <https://arxiv.org/abs/2003.07311>`_).
The prediction and the target are skeletonized together with ten pooling
iterations, on a frame or a volume. The result is one scalar for the whole
batch.

``alpha`` : float, default ``0.5``
    Share of the centerline term. The soft Dice share is ``1 - alpha``.
``smooth`` : float, default ``1.0``
    Constant added inside the skeleton precision and sensitivity. This is
    the constant from the original implementation.
``exclude_background`` : bool, default ``False``
    Drop channel 0 before the skeleton, for a map that stores background
    and foreground in separate channels.
``sigmoid`` : bool, default ``True``
    Read logits, or probabilities when ``False``.

A weight next to Dice in the range 0.1–0.3 keeps clDice a topology term.

TLoss
-----

Student-t likelihood of a noisy mask (González-Jiménez et al.), divided by
the voxel count so the scale stays comparable across resolutions. The value
is a negative log-likelihood: on a perfect mask it settles near a constant
that depends on the number of voxels.

``nu`` : float, default ``1.0``
    Stored as ``nn.Parameter``. ``exp(nu)`` is the degrees of freedom. An
    optimizer that receives ``criterion.parameters()`` learns ``nu``.
``epsilon`` : float, default ``1e-8``
    Stabilizer inside the exponentials and the precision term.
``sigmoid`` : bool, default ``True``
    Read logits, or probabilities when ``False``.
``reduction`` : ``"mean"``, ``"sum"``, or ``"none"``, default ``"mean"``
    Combines the per-image values. ``"none"`` returns a tensor of shape
    ``(N,)``.

A weight near 0.1 is the usual scale next to Dice.

ProjectedPoolingLoss
--------------------

Multi-scale occupancy. The mask is max-pooled at kernel sizes that start at
a quarter of the shortest side and halve down to 1, and those pooled maps
are compared with the prediction. A frame also adds the two axis
max-projections. A volume adds the three orthogonal ones. A thin break
changes a projection more than it changes Dice.

``sigmoid`` : bool, default ``True``
    Read logits, or probabilities when ``False``.

The usual weight next to Dice is 1.

CriticalPointLoss
------------------

Compares three soft maps of local topology: endpoints, junctions, and
isolated voxels. The neighbor count is one convolution, 8 neighbors on a
frame and 26 on a volume.

``sigma`` : float, default ``0.5``
    Width of the soft peaks around the target neighbor counts. A smaller
    ``sigma`` concentrates the loss on voxels whose degree is already close
    to an endpoint, a junction, or an isolated pixel.
``sigmoid`` : bool, default ``True``
    Read logits, or probabilities when ``False``.

The usual weight next to Dice is 1.

HuTopoLoss
----------

Diagram matching on critical pixels (Hu, Li, Samaras, and Chen, NeurIPS
2019, `arXiv:1906.05404 <https://arxiv.org/abs/1906.05404>`_). In each
homology dimension, the predicted bars closest to birth 1 and death 0 are
pulled toward that corner, one per ground-truth feature that is alive at
``threshold``. Every other bar is pulled to the diagonal. The gradient
reaches the birth and death voxels of those bars. Frames and volumes.
Requires the ``ph`` extra (``gudhi``).

The value grows with the number of bars, so it grows with resolution. A
fixed weight of 0.1 is a starting point next to Dice, not a scale that
stays constant when the image gets larger.

``threshold`` : float, default ``0.5``
    A ground-truth bar counts as a feature when its birth is at least this
    value and its death is below it.
``epsilon`` : float, default ``1e-6``
    Padding of the filtration values passed to the cubical complex.
``sigmoid`` : bool, default ``True``
    Read logits, or probabilities when ``False``.
``reduction`` : ``"mean"``, ``"sum"``, or ``"none"``, default ``"mean"``
    Combines the per-image values.

BettiMatchingLoss
------------------

Betti matching (Stucki et al., ICML 2023). Frames only. Requires the ``ph``
extra. On a volume this class raises ``ValueError``; use
``BettiMatching3DLoss``.

``filtration`` : ``"superlevel"``, ``"sublevel"``, or ``"bothlevel"``, default ``"superlevel"``
    Which filtration is matched. ``"bothlevel"`` adds the superlevel and
    sublevel costs.
``relative`` : bool, default ``False``
    Ask the matcher for relative homology.
``comparison`` : str, default ``"union"``
    How the two diagrams are paired. ``"union"`` is the training objective
    from the paper.
``construction`` : str, default ``"V"``
    Cell complex. ``"V"`` is the vertex grid. The fast path implements this
    construction.
``sigmoid`` : bool, default ``True``
    Read logits, or probabilities when ``False``.
``reduction`` : ``"mean"``, ``"sum"``, or ``"none"``, default ``"mean"``
    Combines the per-image values.

A weight near 0.1 is the usual scale next to Dice. The cost follows the
number of bars: a vessel mask is far cheaper than a noisy field of the same
size.

BettiMatching3DLoss
--------------------

The same Betti-matching objective through the C++ extension, on a frame or
a volume. Build `Betti-Matching-3D <https://github.com/nstucki/Betti-Matching-3D>`_
and put the ``betti_matching`` module on ``PYTHONPATH``. It is not installed
by pip.

``filtration`` : ``"superlevel"`` or ``"sublevel"``, default ``"superlevel"``
    The C++ matcher builds sublevel sets. ``"superlevel"`` inverts the
    probabilities before that call, which is the superlevel filtration of
    the original map.
``sigmoid`` : bool, default ``True``
    Read logits, or probabilities when ``False``.

SATLoss
-------

Spatially aware matching of persistence diagrams (Wen et al.,
`arXiv:2412.02076 <https://arxiv.org/abs/2412.02076>`_). Frames only.
Requires the ``ph`` extra (``gudhi``, ``pot``, and ``torch-topological``).
The transport cost is divided by the side length of the image, so a weight
near 0.1 stays a regularizer when the resolution changes.

``p`` : int, default ``2``
    Order of the ground distance between diagram points.
``sigmoid`` : bool, default ``True``
    Read logits, or probabilities when ``False``.
