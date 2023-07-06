# Scene Text Recognition Model Hub
# Copyright 2022 Darwin Bautista
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import Optional

import torch
from torch import nn as nn, Tensor
from torch.nn import functional as F
from torch.nn.modules import transformer

from timm.models.vision_transformer import VisionTransformer, checkpoint_seq
from .patch_embeds import PatchEmbed, MultiPatchEmbed


class DecoderLayer(nn.Module):
    """A Transformer decoder layer supporting two-stream attention (XLNet)
       This implements a pre-LN decoder, as opposed to the post-LN default in PyTorch."""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='gelu',
                 layer_norm_eps=1e-5):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm_q = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm_c = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = transformer._get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.gelu
        super().__setstate__(state)

    def forward_stream(self, tgt: Tensor, tgt_norm: Tensor, tgt_kv: Tensor, memory: Tensor, tgt_mask: Optional[Tensor],
                       tgt_key_padding_mask: Optional[Tensor]):
        """Forward pass for a single stream (i.e. content or query)
        tgt_norm is just a LayerNorm'd tgt. Added as a separate parameter for efficiency.
        Both tgt_kv and memory are expected to be LayerNorm'd too.
        memory is LayerNorm'd by ViT.
        """
        tgt2, sa_weights = self.self_attn(tgt_norm, tgt_kv, tgt_kv, attn_mask=tgt_mask,
                                          key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)

        tgt2, ca_weights = self.cross_attn(self.norm1(tgt), memory, memory)
        tgt = tgt + self.dropout2(tgt2)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(self.norm2(tgt)))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt, sa_weights, ca_weights

    def forward(self, query, content, memory, query_mask: Optional[Tensor] = None, content_mask: Optional[Tensor] = None,
                content_key_padding_mask: Optional[Tensor] = None, update_content: bool = True):
        query_norm = self.norm_q(query)
        content_norm = self.norm_c(content)
        query = self.forward_stream(query, query_norm, content_norm, memory, query_mask, content_key_padding_mask)[0]
        if update_content:
            content = self.forward_stream(content, content_norm, content_norm, memory, content_mask,
                                          content_key_padding_mask)[0]
        return query, content


class Decoder(nn.Module):
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm):
        super().__init__()
        self.layers = transformer._get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, query, content, memory, query_mask: Optional[Tensor] = None, content_mask: Optional[Tensor] = None,
                content_key_padding_mask: Optional[Tensor] = None):
        for i, mod in enumerate(self.layers):
            last = i == len(self.layers) - 1
            query, content = mod(query, content, memory, query_mask, content_mask, content_key_padding_mask,
                                 update_content=not last)
        query = self.norm(query)
        return query


from functools import partial
import copy
class Encoder(VisionTransformer):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_kwargs=dict(type='PatchEmbed'), **kwargs):
        embed_kwargs = copy.deepcopy(embed_kwargs)
        embed_layer = eval(embed_kwargs.pop('type'))
        super().__init__(img_size, patch_size, in_chans, embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                         mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                         drop_path_rate=drop_path_rate, embed_layer=partial(embed_layer, **embed_kwargs),
                         num_classes=0, global_pool='', class_token=False)  # these disable the classifier head

    def forward(self, x):
        # Return all tokens
        return self.forward_features(x)

class SubEncoder(Encoder):
    def forward(self, x, extra_feats=None, patch_size=None):
        if extra_feats is not None:
            assert len(self.blocks)%len(extra_feats) == 0
            num_interval = len(self.blocks)//len(extra_feats)

        x = self.patch_embed(x, patch_size=patch_size)
        x = self._pos_embed(x)
        x = self.norm_pre(x)
        for i, block in enumerate(self.blocks):
            if extra_feats is not None and i%num_interval==0:
                x = x + extra_feats[i//num_interval]
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint_seq(block, x)
            else:
                x = block(x)
        x = self.norm(x)
        return x

class IterEncoder(nn.Module):
    def __init__(self, num_iters=2, shared_connection=False, shared_encoder=False, num_extra_feats=None, num_test_iters=None, **kwargs):
        super().__init__()

        self.shared_encoder = shared_encoder
        self.num_iters = num_iters
        # assert num_iters > 1
        if num_test_iters is None:
            num_test_iters = num_iters
        self.num_test_iters = num_test_iters

        if self.shared_encoder:
            encoder = SubEncoder(**kwargs)
            self.encoders = nn.ModuleList([encoder for _ in range(self.num_iters)])
        else:
            encoders = [SubEncoder(**kwargs) for _ in range(self.num_iters)]
            self.encoders = nn.ModuleList(encoders)

        self.depth = len(self.encoders[0].blocks)
        self.embed_dim = self.encoders[0].embed_dim
        if num_extra_feats is None:
            num_extra_feats = self.depth
        self.num_extra_feats = num_extra_feats

        if shared_connection:
            connection = nn.Linear(self.embed_dim, self.num_extra_feats*self.embed_dim)
            self.connections = nn.ModuleList([connection for _ in range(self.num_iters-1)])
        else:
            connections = [nn.Linear(self.embed_dim, self.num_extra_feats*self.embed_dim) for _ in range(self.num_iters-1)]
            self.connections = nn.ModuleList(connections)

    @torch.jit.ignore
    def no_weight_decay(self):
        param_names = set()
        for i, encoder in enumerate(self.encoders):
            enc_param_names = {f'encoders.{i}.{n}' for n in encoder.no_weight_decay()}
            param_names = param_names.union(enc_param_names)
        return param_names

    def sample_patch_size(self, rng):
        patch_sizes = [enc.patch_embed.sample_patch_size(rng) for enc in self.encoders]
        return patch_sizes

    def forward(self, x, patch_sizes=None):
        if not isinstance(patch_sizes, list):
            patch_sizes = [patch_sizes] * len(self.encoders)
        out = self.encoders[0](x, patch_size=patch_sizes[0])
        outs = [out]
        for encoder, connect, patch_size in zip(self.encoders[1:self.num_test_iters], self.connections, patch_sizes[1:]):
            extra_feats = connect(outs[-1]).chunk(self.num_extra_feats, dim=-1)
            out = encoder(x, extra_feats=extra_feats, patch_size=patch_size)
            outs.append(out)
        return outs


class TokenEmbedding(nn.Module):

    def __init__(self, charset_size: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(charset_size, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, tokens: torch.Tensor):
        return math.sqrt(self.embed_dim) * self.embedding(tokens)
