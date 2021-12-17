import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss

def condor_loss(logits, levels, weight, reduction='mean'):
    r"""Compute the CORN loss used in `Universally rank consistent ordinal regression in neural networks
    (ICLR 2022 under review) <https://arxiv.org/abs/2111.08851>`_.

    Args:
        logits (torch.Tensor): torch.tensor, shape(num_examples, num_classes-1)
        levels (torch.Tensor): torch.tensor, shape(num_examples, num_classes-1)
            True labels represented as extended binary vectors
        weight (torch.Tensor, optional): a manual rescaling weight given to the loss
            of each batch element. If given, has to be a Tensor of size `nbatch`.
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``
    
    Returns:
        loss (torch.Tensor): A torch.tensor containing a single loss value (if `reduction='mean'` or '`sum'`)
        or a loss value for each data record (if `reduction=None`).

    """

    if not logits.shape == levels.shape:
        raise ValueError("Please ensure that logits (%s) has the same shape as levels (%s). "
                         % (logits.shape, levels.shape))

    logprobs = torch.cumsum(F.logsigmoid(logits), dim = 1)
    term1 = (logprobs*levels
             + torch.log(1 - torch.exp(logprobs)+torch.finfo(torch.float32).eps)*(1-levels))

    if weight is not None:
        term1 *= weight

    val = (-torch.sum(term1, dim=1))

    if reduction == 'mean':
        loss = torch.mean(val)
    elif reduction == 'sum':
        loss = torch.sum(val)
    elif reduction is None:
        loss = val
    else:
        s = ('Invalid value for `reduction`. Should be "mean", '
             '"sum", or None. Got %s' % reduction)
        raise ValueError(s)

    return loss


class CondorLoss(_WeightedLoss):
    r"""Creates a criterion that measures The CONDOR loss between the target and the output.
    
    Args:
        weight (Tensor, optional): a manual rescaling weight given to the loss
            of each batch element. If given, has to be a Tensor of size `nbatch`.
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there are multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when reduce is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

    Examples::

        >>> loss = CondorLoss()
        >>> logits = torch.tensor(
        ...    [[1., 1., 0., 0.],
        ...     [1., 0., 0., 0.],
        ...    [1., 1., 1., 1.]])
        >>> target = torch.tensor(
        ...    [[2.1, 1.8, -2.1, -1.8],
        ...     [1.9, -1., -1.5, -1.3],
        ...     [1.9, 1.8, 1.7, 1.6]])
        >>> output = loss(logits, target)
        >>> output.backward()
    """

    def __init__(self, weight: Optional[torch.Tensor] = None, size_average=None, 
                 reduce=None, reduction: str = 'mean') -> None:
        super(CondorLoss, self).__init__(weight, size_average, reduce, reduction)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return condor_loss(input, target, weight=self.weight, reduction=self.reduction)