import timm

from base import BaseModel


class Cls_Noisy_Model(BaseModel):
    def __init__(self, model, pretrained:bool = False , **kwargs):
        super(Cls_Noisy_Model, self).__init__()
        if pretrained:
            self.model = timm.create_model(model, pretrained = True, **kwargs)
        else:
            self.model = timm.create_model(model, pretrained = False, **kwargs)
    
    def forward(self, x):
        return self.model(x)


class Cls_Model(BaseModel):
    def __init__(self, model, pretrained:bool = False, **kwargs):
        super(Cls_Model, self).__init__()
        if pretrained:
            self.model = timm.create_model(model, pretrained=True, **kwargs)
        else:
            self.model = timm.create_model(model, pretrained=False, **kwargs)

    def forward(self, x):
        return self.model(x)
