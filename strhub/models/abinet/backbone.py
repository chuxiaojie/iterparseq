import torch.nn as nn
from torch.nn import TransformerEncoderLayer, TransformerEncoder

from .resnet import resnet45
from .transformer import PositionalEncoding


class ResTranformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, d_inner=2048, dropout=0.1, activation='relu', backbone_ln=2, n_pixels=8*32, resnet_kwargs=dict()):
        super().__init__()
        self.resnet = resnet45(**resnet_kwargs)
        self.channels = self.resnet.channels
        self.pos_encoder = PositionalEncoding(d_model, max_len=n_pixels)
        encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                dim_feedforward=d_inner, dropout=dropout, activation=activation)
        self.transformer = TransformerEncoder(encoder_layer, backbone_ln)

    def forward_transformer(self, feature):
        n, c, h, w = feature.shape
        feature = feature.view(n, c, -1).permute(2, 0, 1)
        feature = self.pos_encoder(feature)
        feature = self.transformer(feature)
        feature = feature.permute(1, 2, 0).view(n, c, h, w)
        return feature

    def forward(self, images, **kwargs):
        feature = self.resnet(images, **kwargs)
        feature = self.forward_transformer(feature)
        return feature