import torch


def prepare_device(n_gpu_use: int):
    r"""Get gpu device indices which are used for DataParallel.
    
    Args:
        n_gpu_use (int): number of gpu to use for DataParallel

    Returns:
        device, list_ids (Tuple[Any, List]): gpu device and indices
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids


def label_to_levels(label: torch.Tensor, num_classes: int, dtype=torch.float32) -> torch.Tensor:
    r"""Converts class indices to categorical binary label vectors.
    
    Args:
        label (torch.Tensor): torch. target label 
        num_classes (int): The number of class in dataset
        dtype (torch.TensorType)

    Returns:
        levels (torch.Tensor): shape=(num_labels, num_classes-1)
    """
    assert label <= num_classes - 1

    int_label = label.item()
    levels = [1]*(int_label) + [0]*(num_classes - 1 - int_label)
    levels = torch.tensor(levels, dtype=dtype)
    
    return levels


def levels_from_labelbatch(labels: torch.Tensor, num_classes: int, dtype=torch.float32) -> torch.Tensor:
    r"""Converts target label to ordinal categorical target in batch.
    
    Args:
        label (torch.Tensor): 1D torch.Tensor target label 
        num_classes (int): The number of class in dataset
        dtype (torch.TensorType)

    Returns:
        levels (torch.Tensor): shape=(num_labels, num_classes-1)
    """
    levels = []
    for label in labels:
        level = label_to_levels(label, num_classes)
        levels.append(level)
    
    levels = torch.stack(levels)

    return levels


def prob_to_label(probs):
    r"""Converts predicted probabilities from extended binary format to integer class labels.
    
    Args:
        probas (torch.tensor): probabilities returned by ordinal model.

    Returns:
        class (torch.tensor): predicted class label
    """
    predict_levels = probs > 0.5
    return torch.sum(predict_levels, dim=1)


def logits_to_label(logits):
    r"""Converts predicted logits returned by ordinal model to integer class labels.
    
    Args:
        logits (torch.tensor): logits returned by ordinal model.

    Returns:
        class (torch.tensor): predicted class label
    """
    probs = torch.cumpord(torch.sigmoid(logits), dim=1)
    predicted_labels = prob_to_label(probs)
    return predicted_labels