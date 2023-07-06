from torch import nn

from .attention import PositionAttention, Attention
from .backbone import ResTranformer
from .model import Model
from .resnet import resnet45

import copy
import torch

class BaseVision(Model):
    def __init__(self, dataset_max_length, null_label, num_classes,
                 attention='position', attention_mode='nearest', loss_weight=1.0,
                 d_model=512, nhead=8, d_inner=2048, dropout=0.1, activation='relu',
                 backbone='transformer', backbone_ln=2, h=8, w=32, resnet_kwargs=dict()):
        super().__init__(dataset_max_length, null_label)
        self.loss_weight = loss_weight
        self.out_channels = d_model

        resnet_kwargs = copy.deepcopy(resnet_kwargs)
        resnet_kwargs['output_channels'] = d_model

        if backbone == 'transformer':
            self.backbone = ResTranformer(d_model, nhead, d_inner, dropout, activation, backbone_ln, n_pixels=h*w, resnet_kwargs=resnet_kwargs)
        else:
            self.backbone = resnet45(**resnet_kwargs)

        if attention == 'position':
            self.attention = PositionAttention(
                in_channels=d_model,
                max_length=self.max_length,
                mode=attention_mode,
                h=h, w=w
            )
        elif attention == 'attention':
            self.attention = Attention(
                in_channels=d_model,
                max_length=self.max_length,
                n_feature=h*w,
            )
        else:
            raise ValueError(f'invalid attention: {attention}')

        self.cls = nn.Linear(self.out_channels, num_classes)

    def _forward(self, b_features):
        attn_vecs, attn_scores = self.attention(b_features)  # (N, T, E), (N, T, H, W)
        logits = self.cls(attn_vecs) # (N, T, C)
        pt_lengths = self._get_length(logits)

        return {'feature': attn_vecs, 'logits': logits, 'pt_lengths': pt_lengths,
                'attn_scores': attn_scores, 'loss_weight':self.loss_weight, 'name': 'vision', 'b_features':b_features}

    def forward(self, images):
        features = self.backbone(images)  # (N, E, H, W)
        return self._forward(features)


class BaseIterVision(BaseVision):
    def __init__(self, *args, num_iters=2, share_weights=False, deep_supervision=True, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(self.backbone, ResTranformer) #only support ResTranformer
        self.backbones = nn.ModuleList()
        self.trans = nn.ModuleList()
        self.share_weights = share_weights
        self.deep_supervision = deep_supervision
        for i in range(num_iters-1):
            B = None if self.share_weights else copy.deepcopy(self.backbone)
            self.backbones.append(B)
            self.split_sizes=[self.backbone.channels[0]] + self.backbone.channels
            self.trans.append(nn.Conv2d(self.split_sizes[-1], sum(self.split_sizes), 1))
            torch.nn.init.zeros_(self.trans[-1].weight)
        
    def forward_test(self, images):
        l_feats = self.backbone.resnet(images)
        b_feats = self.backbone.forward_transformer(l_feats)
        cnt = len(self.backbones)
        if cnt == 0:
            v_res = super()._forward(b_feats)
        for B,T in zip(self.backbones, self.trans):
            cnt -= 1
            extra_feats = T(b_feats).split(self.split_sizes, dim=1)
            if self.share_weights:
                v_res = super().forward(images, extra_feats=extra_feats)
            else:
                b_feats = B(images, extra_feats=extra_feats)
                v_res = super()._forward(b_feats) if cnt==0 else None
        return v_res

    def forward_train(self, images):
        l_feats = self.backbone.resnet(images)
        b_feats = self.backbone.forward_transformer(l_feats)
        v_res = super()._forward(b_feats)
        # v_res = super().forward(images)
        all_v_res = [v_res]
        for B,T in zip(self.backbones, self.trans):
            extra_feats = T(v_res['b_features']).split(self.split_sizes, dim=1)
            if self.share_weights:
                v_res = super().forward(images, extra_feats=extra_feats)
            else:
                b_feats = B(images, extra_feats=extra_feats)
                v_res = super()._forward(b_feats)
            all_v_res.append(v_res)
        return all_v_res

    def forward(self, images):
        if self.training and self.deep_supervision:
            return self.forward_train(images)
        else:
            return self.forward_test(images)