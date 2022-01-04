import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.common_types import _size_2_t
from typing import Optional, List, Tuple

from base import BaseModel

class SeparableConv2d(nn.Module):
    def __init__(
        self, 
        in_channels:int, 
        out_channels:int, 
        kernel_size:_size_2_t, 
        stride:_size_2_t=1, 
        padding:_size_2_t=0, 
        dilation:_size_2_t=1
    ) -> None:
        super(SeparableConv2d, self).__init__()

        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups=in_channels,
            bias=False
        )
        self.poitwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False
        )

    def forward(self, x:Tensor) -> Tensor:
        x = self.depthwise(x)
        x = self.poitwise(x)
        return x

class Block(nn.Module):
    def __init__(
        self,
        in_channels:int,
        out_channels:int,
        reps:int,
        strides:_size_2_t,
        start_with_relu:bool=True,
        grow_first:bool=True
    ):
        super(Block, self).__init__()

        if out_channels != in_channels or strides != 1:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_channels)
        else:
            self.skip = None

        rep = []
        for i in range(reps):
            if grow_first:
                inc = in_channels if i == 0 else out_channels
                outc = out_channels
            else:
                inc = in_channels
                outc = in_channels if i < (reps - 1) else out_channels
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(inc, outc, 3, stride=1, padding=1))
            rep.append(nn.BatchNorm2d(outc))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, x):
        out = self.rep(x)

        if self.skip is not None:
            skip = self.skip(x)
            skip = self.skipbn(skip)
        else:
            skip = x

        out += skip
        return out

