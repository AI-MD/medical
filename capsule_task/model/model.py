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
        return self.model(x)

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

        return out


class CRNN(BaseModel):
    def __init__(self, model, embed_size, LSTM_UNITS, num_layer ,num_classes, pretrained, device):
        super(CRNN, self).__init__()

        if pretrained:
            self.model = timm.create_model(model, pretrained, num_classes = num_classes)
        else:
            self.model = timm.create_model(model, pretrained, num_classes = num_classes)

        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.hidden_size = LSTM_UNITS
        self.num_layer = num_layer
        self.lstm1 = nn.LSTM(embed_size, LSTM_UNITS, num_layer, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(LSTM_UNITS * 2, LSTM_UNITS, num_layer, bidirectional=True, batch_first=True)

        self.linear= nn.Linear(LSTM_UNITS * 2, LSTM_UNITS * 2)

        self.linear_out = nn.Linear(LSTM_UNITS * 2, num_classes)
        self.device = device

    def forward(self, x_3d):

        cnn_embed_seq = []
        for t in range(x_3d.size(1)):
            with torch.no_grad():
                x = self.model.forward_features(x_3d[:, t, :, :, :])
                x = self.avgpool(x)
                feature = x .view(x .size(0), -1)

            cnn_embed_seq.append(feature)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)

        h0 = torch.zeros(2 * self.num_layer, cnn_embed_seq.size(0), self.hidden_size).to(self.device)  # (BATCH SIZE, SEQ_LENGTH, HIDDEN_SIZE)
        c0 = torch.zeros(2 * self.num_layer, cnn_embed_seq.size(0), self.hidden_size).to(self.device)   # hidden state와 동일

        self.lstm1.flatten_parameters()
        h_lstm1, (h1, c1) = self.lstm1(cnn_embed_seq, (h0, c0))
        self.lstm2.flatten_parameters()
        h_lstm2, _ = self.lstm2(h_lstm1, (h1, c1))

        hidden_result = F.relu(self.linear(h_lstm2))

        output = self.linear_out(hidden_result)

        return output