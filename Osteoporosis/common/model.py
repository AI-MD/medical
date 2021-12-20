import torch
import torch.nn as nn
from common.resnet import resnet18

from typing import Optional, Tuple


class StandardModel(nn.Module):
    r"""Multi-class single label classification.

    Args:
        num_classes (int): number of class labels.
        pretrained (bool): whether use the pretrained model. Default: True
        use_feat (bool): whether return the representation feature. Default: False
    """

    def __init__(self, num_classes: int, pretrained: Optional[bool]=True, use_feat: Optional[bool]=False) -> None:
        super().__init__()
        self.use_feat = use_feat
        self.cnn = resnet18(pretrained=pretrained)
        self.fc = nn.Linear(self.cnn.num_features, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        x = self.cnn(x)
        out = self.fc(x)

        if self.use_feat:
            out = (x, out)

        return out


class OrdinalModel(nn.Module):
    r"""Ordinal classification.

    Args:
        num_classes (int): number of class labels.
        pretrained (bool): whether use the pretrained model. Default: True
        use_feat (bool): whether return the representation feature. Default: False
    """

    def __init__(self, num_classes: int, pretrained: Optional[bool]=True, use_feat: Optional[bool]=False) -> None:
        super().__init__()
        self.num_classes = num_classes - 1
        self.use_feat = use_feat
        self.cnn = resnet18(pretrained=pretrained)
        self.fc = nn.Linear(self.cnn.num_features, self.num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        x = self.cnn(x)
        out = self.fc(x)

        if self.use_feat:
            out = (x, out)

        return out