""" Image to Patch Embedding using Conv2d

A convolution based approach to patchifying a 2D image w/ embedding projection.

Based on the impl in https://github.com/google-research/vision_transformer

Hacked together by / Copyright 2020 Ross Wightman
"""
from torch import nn as nn

from timm.models.layers.helpers import to_2tuple
from timm.models.layers.trace_utils import _assert
from timm.models.layers.patch_embed import PatchEmbed as _PatchEmbed

class PatchEmbed(_PatchEmbed):
    """ 2D Image to Patch Embedding
    """
    def __init__(
            self,
            kernel_size=None,
            padding=None,
            **kwargs
    ):
        super().__init__(**kwargs)
        if kernel_size is not None:
            kernel_size = to_2tuple(kernel_size)
            stride = self.proj.stride
            self.proj = nn.Conv2d(self.proj.in_channels, self.proj.out_channels, kernel_size=kernel_size, stride=stride, bias=self.proj.bias is not None, padding=padding)

    def sample_patch_size(self, rng):
        return None
        
    def forward(self, x, patch_size=None):
        return super().forward(x)

from torch.nn import LayerNorm
class MultiPatchEmbed(nn.Module):
    def __init__(self,
            img_size=224,
            patch_size=[16],
            default_patch_size=None,
            in_chans=3,
            embed_dim=768,
            norm_layer=None,
            flatten=True,
            bias=True,
    ):
        super().__init__()
        if norm_layer is not None:
            norm_layer = eval(norm_layer)
        self.patch_embed = nn.ModuleDict()
        patch_size_list = patch_size
        for patch_size in patch_size_list:
            patch_size = to_2tuple(patch_size)
            self.patch_embed[str(patch_size)] = PatchEmbed(
                img_size=img_size, patch_size=patch_size,
                in_chans=in_chans, embed_dim=embed_dim,
                norm_layer=norm_layer, flatten=flatten, 
                bias=bias
            )
        if default_patch_size is None:
            default_patch_size = patch_size_list[-1]
        self.default_patch_size = to_2tuple(default_patch_size)
        assert str(self.default_patch_size) in self.patch_embed

        self.patch_sizes = list(self.patch_embed.keys())
        self.num_embed = len(self.patch_sizes)

    def sample_patch_size(self, rng):
        return rng.choice(self.patch_sizes)

    def forward(self, x, patch_size=None):
        if patch_size is None:
            patch_size = self.default_patch_size
        else:
            patch_size = to_2tuple(patch_size)
        return self.patch_embed[str(patch_size)](x)