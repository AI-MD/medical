import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
def VideoCrossEntropyLoss(output, target):
    return F.cross_entropy(output, target)

def CrossEntropyLoss(output, target):
    return F.cross_entropy(output, target)

def nll_loss(output, target):
    return F.nll_loss(output, target)

def bce_loss(output, target):
    return F.binary_cross_entropy_with_logits(output, target)

def smooth_crossentropy(pred, gold, smoothing=0.1):
    n_class = pred.size(1)

    one_hot = torch.full_like(pred, fill_value=smoothing / (n_class - 1))
    one_hot.scatter_(dim=1, index=gold.unsqueeze(1), value=1.0 - smoothing)
    log_prob = F.log_softmax(pred, dim=1)

    return F.kl_div(input=log_prob, target=one_hot, reduction='none').sum(-1)
    
def smooth_multilabel_loss(output, target, smoothing=0.05):
    soft_target = (1 - target) * smoothing + target * (1 - smoothing)
    return F.multilabel_soft_margin_loss(output, soft_target)

class CosineLoss(nn.Module):
    def __init__(self, xent=.5, reduction="mean"):
        super(CosineLoss, self).__init__()
        self.xent = xent
        self.reduction = reduction
        
        self.y = torch.Tensor([1.]).cuda()
        
    def forward(self, input, target):
        cosine_loss = F.cosine_embedding_loss(input, target, self.y, reduction=self.reduction)
        cent_loss = F.binary_cross_entropy_with_logits(F.normalize(input), target, reduction=self.reduction)
        
        return cosine_loss + self.xent * cent_loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha 

    def forward(self, output, target):
        bce_loss = F.binary_cross_entropy_with_logits(output, target)
        pt = torch.exp(-bce_loss)
        alpha_tensor = (1 - self.alpha) + target * (2 * self.alpha - 1)
        f_loss = alpha_tensor * (1 - pt) ** self.gamma * bce_loss
        return f_loss.mean()

def CB_loss(logits, labels, samples_per_cls =  [900, 9000, 9000, 9000, 4500], no_of_classes = 5, loss_type = "sigmoid" , beta = 0.9999, gamma =  1.0):
    """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
    Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
    where Loss is one of the standard losses used for Neural Networks.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      samples_per_cls: A python list of size [no_of_classes].
      no_of_classes: total number of classes. int
      loss_type: string. One of "sigmoid", "focal", "softmax".
      beta: float. Hyperparameter for Class balanced loss.
      gamma: float. Hyperparameter for Focal loss.
    Returns:
      cb_loss: A float tensor representing class balanced loss
    """
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * no_of_classes
    
    labels_one_hot = labels #F.one_hot(labels, no_of_classes).float()
   

    weights = torch.tensor(weights).float().cuda()
    weights = weights.unsqueeze(0)
    weights = weights.repeat(labels_one_hot.shape[0], 1) * labels_one_hot
    weights = weights.sum(1)
    weights = weights.unsqueeze(1)
    weights = weights.repeat(1, no_of_classes)

    if loss_type == "focal":
        cb_loss = focal_loss(labels_one_hot, logits, weights, gamma)
    elif loss_type == "sigmoid":
        cb_loss = F.binary_cross_entropy_with_logits(input = logits,target = labels_one_hot, weight = weights)
    elif loss_type == "softmax":
        pred = logits.softmax(dim = 1)
        cb_loss = F.binary_cross_entropy(input = pred, target = labels_one_hot, weight = weights)
    return cb_loss