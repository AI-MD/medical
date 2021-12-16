import torch
from utils.util import prob_to_label


def accuracy(output, target):
    r"""Computes the accuracy.

    Args:
        output (tensor): Classification outputs
        target (tensor): :math:`(N)` where each value is :math:`0 \leq \text{targets}[i] \leq C-1`

    Returns:
        Accuracies (float)
    """
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def ordinal_accuracy(output, target):
    r"""Computes the accuracy in ordinal classification scenario.

    Args:
        output (tensor): Ordinal classification outputs
        target (tensor): :math:`(N)` where each value is :math:`0 \leq \text{targets}[i] \leq C-1`

    Returns:
        Accuracies (float)
    """
    with torch.no_grad():
        pred = prob_to_label(output)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)