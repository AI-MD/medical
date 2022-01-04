import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from model.module import *
import torch.utils.model_zoo as model_zoo
import timm




class Model(BaseModel):
    def __init__(self, model, pretrained:bool = False, **kwargs):
        super(Model, self).__init__()
        if pretrained:
            self.model = timm.create_model(model, pretrained=True, **kwargs)

        else:
            self.model = timm.create_model(model, pretrained=False, **kwargs)

    def forward(self, x):

        

        return self.model(x) # out b, c
                             # torch.sigmoid(output)
                             # bce_loss = nn.BCELoss( )
                             # target = 2 
                             # layer4 : [0, 0]
                             # layer2 : [1, 1]
                             # layer1 : [1, 0]
                             # layer3 : [0, 1]


class CustomModel(BaseModel):
    def __init__(self, model, pretrained: bool = False, **kwargs):
        super(CustomModel, self).__init__()
        if pretrained:
            self.model = timm.create_model(model, pretrained=True, **kwargs)
        else:
            self.model = timm.create_model(model, pretrained=False, **kwargs)

        self.fc = nn.Linear(1281, 4)


    def logits(self, features:Tensor, time_index:Tensor) -> Tensor:
        x = nn.ReLU(inplace=True)(features)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = torch.cat([x,time_index],dim= 1)

        x = self.fc(x)

        return x

    def forward(self, x, time_index):
        feature = self.model.forward_features(x)  # feature out

        out = self.logits(feature, time_index)

        return out # out b, c
        # torch.sigmoid(output)
        # bce_loss = nn.BCELoss( )
        # target = 2
        # layer4 : [0, 0]
        # layer2 : [1, 1]
        # layer1 : [1, 0]
        # layer3 : [0, 1]