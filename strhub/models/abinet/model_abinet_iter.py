import torch
from torch import nn

from .model_alignment import BaseAlignment
from .model_language import BCNLanguage
from .model_vision import BaseVision, BaseIterVision


class ABINetIterModel(nn.Module):
    def __init__(self, dataset_max_length, null_label, num_classes, iter_size=1,
                 d_model=512, nhead=8, d_inner=2048, dropout=0.1, activation='relu',
                 v_loss_weight=1., v_attention='position', v_attention_mode='nearest',
                 v_backbone='transformer', v_num_layers=2,
                 l_loss_weight=1., l_num_layers=4, l_detach=True, l_use_self_attn=False,
                 a_loss_weight=1., img_size=[32, 128]):
        super().__init__()
        self.iter_size = iter_size
        self.vision = BaseVision(dataset_max_length, null_label, num_classes, v_attention, v_attention_mode,
                                 v_loss_weight, d_model, nhead, d_inner, dropout, activation, v_backbone, v_num_layers, h=img_size[0]//4, w=img_size[1]//4)
        self.language = BCNLanguage(dataset_max_length, null_label, num_classes, d_model, nhead, d_inner, dropout,
                                    activation, l_num_layers, l_detach, l_use_self_attn, l_loss_weight)
        self.alignment = BaseAlignment(dataset_max_length, null_label, num_classes, d_model, a_loss_weight)

    def forward(self, images):
        v_res = self.vision(images)
        a_res = v_res
        all_l_res, all_a_res = [], []
        for _ in range(self.iter_size):
            tokens = torch.softmax(a_res['logits'], dim=-1)
            lengths = a_res['pt_lengths']
            lengths.clamp_(2, self.language.max_length)  # TODO:move to langauge model
            l_res = self.language(tokens, lengths)
            all_l_res.append(l_res)
            a_res = self.alignment(l_res['feature'], v_res['feature'])
            all_a_res.append(a_res)
        if self.training:
            return all_a_res, all_l_res, v_res
        else:
            return a_res, all_l_res[-1], v_res



class IterNetModel(nn.Module):
    def __init__(self, dataset_max_length, null_label, num_classes, iter_size=1,
                 d_model=512, nhead=8, d_inner=2048, dropout=0.1, activation='relu',
                 v_loss_weight=1., v_attention='position', v_attention_mode='nearest',
                 v_backbone='transformer', v_num_layers=2,
                 l_loss_weight=1., l_num_layers=4, l_detach=True, l_use_self_attn=False,
                 a_loss_weight=1., num_iters=2, share_weights=False, deep_supervision=True, resnet_kwargs: dict=dict(), img_size=[32, 128]):
        super().__init__()
        self.iter_size = iter_size
        self.deep_supervision = deep_supervision
        self.vision = BaseIterVision(dataset_max_length, null_label, num_classes, v_attention, v_attention_mode,
                                 v_loss_weight, d_model, nhead, d_inner, dropout, activation, v_backbone, v_num_layers, h=img_size[0]//4, w=img_size[1]//4,
                                 num_iters=num_iters, share_weights=share_weights, deep_supervision=deep_supervision, resnet_kwargs=resnet_kwargs)
        self.language = BCNLanguage(dataset_max_length, null_label, num_classes, d_model, nhead, d_inner, dropout,
                                    activation, l_num_layers, l_detach, l_use_self_attn, l_loss_weight)
        self.alignment = BaseAlignment(dataset_max_length, null_label, num_classes, d_model, a_loss_weight)

    def forward(self, images):
        list_v_res = self.vision(images)
        if not isinstance(list_v_res, (list, tuple)):
            list_v_res = [list_v_res]
        all_l_res, all_a_res = [], []

        for v_res in list_v_res:
            a_res = v_res
            for _ in range(self.iter_size):
                tokens = torch.softmax(a_res['logits'], dim=-1)
                lengths = a_res['pt_lengths']
                lengths.clamp_(2, self.language.max_length)  # TODO:move to langauge model
                l_res = self.language(tokens, lengths)
                all_l_res.append(l_res)
                a_res = self.alignment(l_res['feature'], v_res['feature'])
                all_a_res.append(a_res)
        if self.training and self.deep_supervision:
            return all_a_res, all_l_res, list_v_res
        else:
            return a_res, all_l_res[-1], list_v_res[-1]