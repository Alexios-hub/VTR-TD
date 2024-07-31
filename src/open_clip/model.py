""" CLIP Model

Adapted from https://github.com/openai/CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.
"""
import copy
import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, autocast, nn
from torch.utils.checkpoint import checkpoint
from functools import partial

from open_clip.module_ta import TaModuleV1,TaModuleV2

from .hf_model import HFTextEncoder
from .modified_resnet import ModifiedResNet
from .timm_model import TimmModel
from .transformer import LayerNormFp32, LayerNorm, QuickGELU, Attention, VisionTransformer, TextTransformer,\
    text_global_pool
from .utils import to_2tuple

from transformers.models.videomae.modeling_videomae import get_sinusoid_encoding_table

from timm.layers.norm_act import _create_act
from timm.layers.trace_utils import _assert
from timm.layers import DropPath, use_fused_attn
from open_clip.modeling_perceiver_xattn import Perceiver
from typing import OrderedDict

@dataclass
class CLIPVisionCfg:
    layers: Union[Tuple[int, int, int, int], int] = 12
    width: int = 768
    head_width: int = 64
    mlp_ratio: float = 4.0
    patch_size: int = 16
    image_size: Union[Tuple[int, int], int] = 224

    ls_init_value: Optional[float] = None  # layer scale initial value
    patch_dropout: float = 0.  # what fraction of patches to dropout during training (0 would mean disabled and no patches dropped) - 0.5 to 0.75 recommended in the paper for optimal results
    attentional_pool: bool = False  # whether to use attentional pooler in the last embedding layer (overrides pool_type)
    attn_pooler_queries: int = 256  # n_queries for attentional pooler
    attn_pooler_heads: int = 8  # n heads for attentional_pooling
    no_ln_pre: bool = False  # disable pre transformer LayerNorm
    pos_embed_type: str = 'learnable'
    final_ln_after_pool: bool = False  # apply final LayerNorm after pooling
    pool_type: str = 'tok'
    output_tokens: bool = False
    act_kwargs: Optional[dict] = None
    norm_kwargs: Optional[dict] = None

    timm_model_name: Optional[str] = None  # a valid model name overrides layers, width, patch_size
    timm_model_pretrained: bool = False  # use (imagenet) pretrained weights for named model
    timm_pool: str = 'avg'  # feature pooling for timm model ('abs_attn', 'rot_attn', 'avg', '')
    timm_proj: str = 'linear'  # linear projection for timm model output ('linear', 'mlp', '')
    timm_proj_bias: bool = False  # enable bias final projection
    timm_drop: float = 0.  # head dropout
    timm_drop_path: Optional[float] = None  # backbone stochastic depth


@dataclass
class CLIPTextCfg:
    context_length: int = 77
    vocab_size: int = 49408
    hf_tokenizer_name: Optional[str] = None
    tokenizer_kwargs: Optional[dict] = None

    width: int = 512
    heads: int = 8
    layers: int = 12
    mlp_ratio: float = 4.0
    ls_init_value: Optional[float] = None  # layer scale initial value
    embed_cls: bool = False
    pad_id: int = 0
    no_causal_mask: bool = False  # disable causal masking
    final_ln_after_pool: bool = False  # apply final LayerNorm after pooling
    pool_type: str = 'argmax'
    proj_bias: bool = False
    output_tokens: bool = False
    act_kwargs: dict = None
    norm_kwargs: dict = None

    # HuggingFace specific text tower config
    hf_model_name: Optional[str] = None
    hf_model_pretrained: bool = True
    hf_proj_type: str = 'mlp'
    hf_pooler_type: str = 'mean_pooler'  # attentional pooling for HF models


def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == 'bf16':
        cast_dtype = torch.bfloat16
    elif precision == 'fp16':
        cast_dtype = torch.float16
    return cast_dtype


def get_input_dtype(precision: str):
    input_dtype = None
    if precision in ('bf16', 'pure_bf16'):
        input_dtype = torch.bfloat16
    elif precision in ('fp16', 'pure_fp16'):
        input_dtype = torch.float16
    return input_dtype


def _build_vision_tower(
        embed_dim: int,
        vision_cfg: CLIPVisionCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None
):
    if isinstance(vision_cfg, dict):
        vision_cfg = CLIPVisionCfg(**vision_cfg)

    # OpenAI models are pretrained w/ QuickGELU but native nn.GELU is both faster and more
    # memory efficient in recent PyTorch releases (>= 1.10).
    # NOTE: timm models always use native GELU regardless of quick_gelu flag.
    act_layer = QuickGELU if quick_gelu else nn.GELU

    if vision_cfg.timm_model_name:
        visual = TimmModel(
            vision_cfg.timm_model_name,
            pretrained=vision_cfg.timm_model_pretrained,
            pool=vision_cfg.timm_pool,
            proj=vision_cfg.timm_proj,
            proj_bias=vision_cfg.timm_proj_bias,
            drop=vision_cfg.timm_drop,
            drop_path=vision_cfg.timm_drop_path,
            patch_drop=vision_cfg.patch_dropout if vision_cfg.patch_dropout > 0 else None,
            embed_dim=embed_dim,
            image_size=vision_cfg.image_size,
        )
    elif isinstance(vision_cfg.layers, (tuple, list)):
        vision_heads = vision_cfg.width * 32 // vision_cfg.head_width
        visual = ModifiedResNet(
            layers=vision_cfg.layers,
            output_dim=embed_dim,
            heads=vision_heads,
            image_size=vision_cfg.image_size,
            width=vision_cfg.width,
        )
    else:
        vision_heads = vision_cfg.width // vision_cfg.head_width
        norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
        if vision_cfg.norm_kwargs:
            norm_layer = partial(norm_layer, **vision_cfg.norm_kwargs)
        if vision_cfg.act_kwargs is not None:
            act_layer = partial(act_layer, **vision_cfg.act_kwargs)

        visual = VisionTransformer(
            image_size=vision_cfg.image_size,
            patch_size=vision_cfg.patch_size,
            width=vision_cfg.width,
            layers=vision_cfg.layers,
            heads=vision_heads,
            mlp_ratio=vision_cfg.mlp_ratio,
            ls_init_value=vision_cfg.ls_init_value,
            patch_dropout=vision_cfg.patch_dropout,
            attentional_pool=vision_cfg.attentional_pool,
            attn_pooler_queries=vision_cfg.attn_pooler_queries,
            attn_pooler_heads=vision_cfg.attn_pooler_heads,
            pos_embed_type=vision_cfg.pos_embed_type,
            no_ln_pre=vision_cfg.no_ln_pre,
            final_ln_after_pool=vision_cfg.final_ln_after_pool,
            pool_type=vision_cfg.pool_type,
            output_tokens=vision_cfg.output_tokens,
            output_dim=embed_dim,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )

    return visual


def _build_text_tower(
        embed_dim: int,
        text_cfg: CLIPTextCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None,
):
    if isinstance(text_cfg, dict):
        text_cfg = CLIPTextCfg(**text_cfg)

    if text_cfg.hf_model_name:
        text = HFTextEncoder(
            text_cfg.hf_model_name,
            output_dim=embed_dim,
            proj_type=text_cfg.hf_proj_type,
            pooler_type=text_cfg.hf_pooler_type,
            pretrained=text_cfg.hf_model_pretrained,
            output_tokens=text_cfg.output_tokens,
        )
    else:
        act_layer = QuickGELU if quick_gelu else nn.GELU
        norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
        if text_cfg.norm_kwargs:
            norm_layer = partial(norm_layer, **text_cfg.norm_kwargs)
        if text_cfg.act_kwargs is not None:
            act_layer = partial(act_layer, **text_cfg.act_kwargs)

        text = TextTransformer(
            context_length=text_cfg.context_length,
            vocab_size=text_cfg.vocab_size,
            width=text_cfg.width,
            heads=text_cfg.heads,
            layers=text_cfg.layers,
            mlp_ratio=text_cfg.mlp_ratio,
            ls_init_value=text_cfg.ls_init_value,
            output_dim=embed_dim,
            embed_cls=text_cfg.embed_cls,
            no_causal_mask=text_cfg.no_causal_mask,
            pad_id=text_cfg.pad_id,
            pool_type=text_cfg.pool_type,
            proj_bias=text_cfg.proj_bias,
            output_tokens=text_cfg.output_tokens,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
    return text


class CLIP(nn.Module):
    output_dict: torch.jit.Final[bool]

    def __init__(
            self,
            embed_dim: int,
            vision_cfg: CLIPVisionCfg,
            text_cfg: CLIPTextCfg,
            quick_gelu: bool = False,
            init_logit_scale: float = np.log(1 / 0.07),
            init_logit_bias: Optional[float] = None,
            cast_dtype: Optional[torch.dtype] = None,
            output_dict: bool = False,
            text_output_tokens = False
    ):
        super().__init__()
        self.text_output_tokens = text_output_tokens
        self.output_dict = output_dict

        self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype)

        text = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)
        self.transformer = text.transformer
        self.context_length = text.context_length
        self.vocab_size = text.vocab_size
        self.token_embedding = text.token_embedding
        self.positional_embedding = text.positional_embedding
        self.ln_final = text.ln_final
        self.text_projection = text.text_projection
        self.text_pool_type = text.pool_type
        self.register_buffer('attn_mask', text.attn_mask, persistent=False)

        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)
        if init_logit_bias is not None:
            self.logit_bias = nn.Parameter(torch.ones([]) * init_logit_bias)
        else:
            self.logit_bias = None

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        self.visual.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.transformer.grad_checkpointing = enable

    def encode_image(self, image, normalize: bool = False):
        if self.visual.output_tokens == True:
            features, token_features = self.visual(image)
            return (F.normalize(features, dim=-1), token_features) if normalize else (features, token_features)
        features = self.visual(image)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, normalize: bool = False):
        cast_dtype = self.transformer.get_cast_dtype()

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.to(cast_dtype)
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        # x, _ = text_global_pool(x, text, self.text_pool_type)
        x, tokens = text_global_pool(x, text, self.text_pool_type)
        if self.text_projection is not None:
            if isinstance(self.text_projection, nn.Linear):
                x = self.text_projection(x)
            else:
                x = x @ self.text_projection
        if self.text_output_tokens:
            return (F.normalize(x, dim=-1),tokens) if normalize else (x,tokens)

        return F.normalize(x, dim=-1) if normalize else x

    def get_logits(self, image, text):
        image_features = self.encode_image(image, normalize=True)
        text_features = self.encode_text(text, normalize=True)
        image_logits = self.logit_scale.exp() * image_features @ text_features.T
        if self.logit_bias is not None:
            image_logits += self.logit_bias
        text_logits = image_logits.T
        return image_logits, text_logits

    def forward(
            self,
            image: Optional[torch.Tensor] = None,
            text: Optional[torch.Tensor] = None,
    ):
        image_features = self.encode_image(image, normalize=True) if image is not None else None
        text_features = self.encode_text(text, normalize=True) if text is not None else None

        if self.output_dict:
            out_dict = {
                "image_features": image_features,
                "text_features": text_features,
                "logit_scale": self.logit_scale.exp()
            }
            if self.logit_bias is not None:
                out_dict['logit_bias'] = self.logit_bias
            return out_dict

        if self.logit_bias is not None:
            return image_features, text_features, self.logit_scale.exp(), self.logit_bias
        return image_features, text_features, self.logit_scale.exp()


class CustomTextCLIP(nn.Module):
    output_dict: torch.jit.Final[bool]

    def __init__(
            self,
            embed_dim: int,
            vision_cfg: CLIPVisionCfg,
            text_cfg: CLIPTextCfg,
            quick_gelu: bool = False,
            init_logit_scale: float = np.log(1 / 0.07),
            init_logit_bias: Optional[float] = None,
            cast_dtype: Optional[torch.dtype] = None,
            output_dict: bool = False,
    ):
        super().__init__()
        self.output_dict = output_dict
        self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype)
        self.text = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)
        self.context_length = self.text.context_length
        self.vocab_size = self.text.vocab_size
        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)
        if init_logit_bias is not None:
            self.logit_bias = nn.Parameter(torch.ones([]) * init_logit_bias)
        else:
            self.logit_bias = None

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        self.visual.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)

    def lock_text_tower(self, unlocked_layers: int = 0, freeze_layer_norm: bool = True):
        self.text.lock(unlocked_layers, freeze_layer_norm)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.text.set_grad_checkpointing(enable)

    def encode_image(self, image, normalize: bool = False):
        # features, stages_features = self.visual(image)
        # return (F.normalize(features, dim=-1), stages_features) if normalize else (features, stages_features)
        features = self.visual(image)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, normalize: bool = False):
        features = self.text(text)
        # features, tokens = self.text(text)
        # return (F.normalize(features, dim=-1),tokens) if normalize else (features,tokens)
        return F.normalize(features, dim=-1) if normalize else features

    def get_logits(self, image, text):
        image_features = self.encode_image(image, normalize=True)
        text_features = self.encode_text(text, normalize=True)
        image_logits = self.logit_scale.exp() * image_features @ text_features.T
        if self.logit_bias is not None:
            image_logits += self.logit_bias
        text_logits = image_logits.T
        return image_logits, text_logits

    def forward(
            self,
            image: Optional[torch.Tensor] = None,
            text: Optional[torch.Tensor] = None,
    ):
        # image_features, stages_features = self.encode_image(image, normalize=True) if image is not None else None
        # text_features, text_tokens = self.encode_text(text, normalize=True) if text is not None else None

        image_features = self.encode_image(image, normalize=True) if image is not None else None
        text_features = self.encode_text(text, normalize=True) if text is not None else None
        if self.output_dict:
            out_dict = {
                "image_features": image_features,
                "text_features": text_features,
                "logit_scale": self.logit_scale.exp(),
                # "stages_features":stages_features,
                # "text_tokens":text_tokens
            }
            if self.logit_bias is not None:
                out_dict['logit_bias'] = self.logit_bias
            return out_dict

        if self.logit_bias is not None:
            return image_features, text_features, self.logit_scale.exp(), self.logit_bias
        return image_features, text_features, self.logit_scale.exp()


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x_input = x
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x + x_input

class AdaptMLP(nn.Module):
    def __init__(self, original_mlp, in_dim, mid_dim, dropout=0.0, s=0.1):
        super().__init__()
        self.original_mlp = original_mlp

        self.c_fc = nn.Linear(in_dim, mid_dim)
        # self.act = nn.ReLU()
        self.up_proj = nn.Linear(mid_dim, in_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = s

        #initialization
        nn.init.kaiming_uniform_(self.c_fc.weight)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.c_fc.bias)
        nn.init.zeros_(self.up_proj.bias)


        #freeze original MLP
        for _, p in self.original_mlp.named_parameters():
            p.requires_grad = False

    # def forward(self, x):
    #     original_mlp_x = self.original_mlp(x)
    #     if len(x.shape) == 4:#B,C,H,W
    #         x = x.transpose(1,3)
    #     down = self.c_fc(x)
    #     # down = self.act(down)
    #     down = self.dropout(down)
    #     up = self.up_proj(down)
    #     if len(up.shape) == 4:
    #         up = up.transpose(1,3)

    #     output = original_mlp_x + up * self.scale
    #     return output

    #after FFN
    def forward(self, x):
        original_mlp_x = self.original_mlp(x)
        if len(original_mlp_x.shape) == 4:#B,C,H,W
            original_mlp_x = original_mlp_x.transpose(1,3)
        down = self.c_fc(original_mlp_x)
        # down = self.act(down)
        down = self.dropout(down)
        up = self.up_proj(down)
        if len(up.shape) == 4:
            up = up.transpose(1,3)

        output = original_mlp_x + up * self.scale
        return output


class Attention3D(nn.Module):
    """Multi-headed Self Attention module.

    Source modified from:
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    """
    fused_attn: torch.jit.Final[bool]

    def __init__(
            self,
            dim: int,
            head_dim: int = 32,
            qkv_bias: bool = False,
            attn_drop: float = 0.0,
            proj_drop: float = 0.0,
            num_frames: int = 4
    ) -> None:
        """Build MHSA module that can handle 3D or 4D input tensors.

        Args:
            dim: Number of embedding dimensions.
            head_dim: Number of hidden dimensions per head. Default: ``32``
            qkv_bias: Use bias or not. Default: ``False``
            attn_drop: Dropout rate for attention tensor.
            proj_drop: Dropout rate for projection tensor.
        """
        super().__init__()
        assert dim % head_dim == 0, "dim should be divisible by head_dim"
        self.head_dim = head_dim
        self.num_heads = dim // head_dim
        self.scale = head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.num_frames = num_frames
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B_T, C, H, W = x.shape
        B = B_T // self.num_frames

        N = self.num_frames * H * W
        x = x.reshape(B, self.num_frames, C,H,W).transpose(1,2)#[B,C,T,H,W]

        x = x.flatten(2).transpose(-2, -1)  # (B, N, C)
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        if self.fused_attn:
            x = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.transpose(-2, -1).reshape(B, C, self.num_frames, H, W).transpose(1,2).reshape(B*self.num_frames,C,H,W)

        return x

class ShapedAttention3D(nn.Module):
    """Shaped Multi-headed Self Attention module.
    """
    fused_attn: torch.jit.Final[bool]

    def __init__(
            self,
            dim: int,
            head_dim: int = 32,
            qkv_bias: bool = False,
            attn_drop: float = 0.0,
            num_frames: int = 4,
            width: int = 8
    ) -> None:
        """Build MHSA module that can handle 3D or 4D input tensors.

        Args:
            dim: Number of embedding dimensions.
            head_dim: Number of hidden dimensions per head. Default: ``32``
            qkv_bias: Use bias or not. Default: ``False``
            attn_drop: Dropout rate for attention tensor.
            proj_drop: Dropout rate for projection tensor.
        """
        super().__init__()
        assert dim % head_dim == 0, "dim should be divisible by head_dim"
        self.head_dim = head_dim
        self.num_heads = dim // head_dim
        self.scale = head_dim ** -0.5
        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.num_frames = num_frames

        n_c = num_frames * width * width
        self.alpha = nn.Parameter(torch.zeros((1, self.num_heads, 1, 1)))
        self.beta = nn.Parameter(torch.zeros((1, self.num_heads, 1, 1)))
        self.gamma = nn.Parameter(torch.zeros((1, self.num_heads, 1, 1)))

        self.register_buffer('C',torch.ones((n_c, n_c)) / n_c)
        self.register_buffer('it',torch.eye(n_c))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B_T, C, H, W = x.shape
        B = B_T // self.num_frames

        N = self.num_frames * H * W
        x = x.reshape(B, self.num_frames, C,H,W).transpose(1,2)#[B,C,T,H,W]
        x = x.flatten(2).transpose(-2, -1)  # (B, N, C)
        qk = (
            self.qk(x)
            .reshape(B, N, 2, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k = qk.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
        v = x.reshape(B, N, 1, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).squeeze()

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn = self.alpha * self.it + self.beta * attn - self.gamma * self.C
        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)

        x = x.transpose(-2, -1).reshape(B, C, self.num_frames, H, W).transpose(1,2).reshape(B*self.num_frames,C,H,W)

        return x

from timm.models.layers import trunc_normal_

class ConvNormAct(nn.Module):
    """ Custom ConvNormAct for 3D convolutions. """
    def __init__(self, in_chs, out_chs, kernel_size, groups, apply_act=True):
        super().__init__()
        self.conv = nn.Conv3d(in_chs, out_chs, kernel_size, padding=kernel_size // 2, groups=groups)
        self.bn = nn.BatchNorm3d(out_chs)
        self.act = nn.GELU() if apply_act else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x
    
class ConvMlp3D(nn.Module):
    """3D Convolutional FFN Module."""

    def __init__(
            self,
            in_chs: int,
            hidden_channels: Optional[int] = None,
            out_chs: Optional[int] = None,
            act_layer: nn.Module = nn.GELU,
            drop: float = 0.0,
            num_frames: int=4
    ) -> None:
        """Build 3D convolutional FFN module.

        Args:
            in_chs: Number of input channels.
            hidden_channels: Number of channels after expansion. Default: None
            out_chs: Number of output channels. Default: None
            act_layer: Activation layer. Default: `GELU`
            drop: Dropout rate. Default: `0.0`.
        """
        super().__init__()
        out_chs = out_chs or in_chs
        hidden_channels = hidden_channels or in_chs
        self.conv = ConvNormAct(
            in_chs,
            out_chs,
            kernel_size=7,
            groups=in_chs,
            apply_act=False,
        )
        self.fc1 = nn.Conv3d(in_chs, hidden_channels, kernel_size=1)
        self.act = act_layer()
        self.fc2 = nn.Conv3d(hidden_channels, out_chs, kernel_size=1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        self.num_frames = num_frames

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Conv3d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B_T,C,H,W = x.shape
        B = B_T // self.num_frames
        x = x.reshape(B, self.num_frames, C, H, W).transpose(1,2)
        x = self.conv(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = x.transpose(1,2).reshape(B*self.num_frames,C,H,W)
        return x
    
class AttentionBlock3D(nn.Module):
    """Implementation of metaformer block with MHSA as token mixer.

    For more details on Metaformer structure, please refer to:
    `MetaFormer Is Actually What You Need for Vision <https://arxiv.org/pdf/2111.11418.pdf>`_
    """

    def __init__(
            self,
            dim: int,
            mlp_ratio: float = 4.0,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.BatchNorm3d,
            proj_drop: float = 0.0,
            drop_path: float = 0.0,
            num_frames: int = 4,
            use_pos_emb = False,
            width=8
    ):
        """Build Attention Block.

        Args:
            dim: Number of embedding dimensions.
            mlp_ratio: MLP expansion ratio. Default: 4.0
            act_layer: Activation layer. Default: ``nn.GELU``
            norm_layer: Normalization layer. Default: ``nn.BatchNorm2d``
            proj_drop: Dropout rate. Default: 0.0
            drop_path: Drop path rate. Default: 0.0
            layer_scale_init_value: Layer scale value at initialization. Default: 1e-5
        """

        super().__init__()

        self.norm = norm_layer(dim)
        self.token_mixer = Attention3D(dim=dim,num_frames=num_frames)

        self.layer_scale_1 = nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.mlp = ConvMlp3D(
            in_chs=dim,
            hidden_channels=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
            num_frames=num_frames,
        )

        self.layer_scale_2 = nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        if use_pos_emb:
            n_positions = width * width * num_frames
            self.pos_emb = get_sinusoid_encoding_table(n_position=n_positions, d_hid=dim)
        self.use_pos_emb = use_pos_emb
        self.num_frames = num_frames

    def forward(self, x):
        B_T, C, H, W = x.shape
        B = B_T // self.num_frames
        N = self.num_frames * H * W
        if self.use_pos_emb:
             x = x.reshape(B,self.num_frames,C,H,W).transpose(1,2).flatten(2).transpose(1,2)#[B,N,C]
             x = x + self.pos_emb[:,:N].type_as(x).to(x.device).clone().detach()
             x = x.reshape(B*self.num_frames, H, W, C).permute(0,3,1,2)#[B_T,C,H,W]
        x = x + self.drop_path1(self.layer_scale_1(self.token_mixer(self.norm(x.reshape(B,self.num_frames,C,H,W).transpose(1,2)).transpose(1,2).reshape(B*self.num_frames,C,H,W))))
        x = x + self.drop_path2(self.layer_scale_2(self.mlp(x)))
        return x
    
class ParallelAttentionBlock3D(nn.Module):
    """Implementation of metaformer block with MHSA as token mixer.

    For more details on Metaformer structure, please refer to:
    `MetaFormer Is Actually What You Need for Vision <https://arxiv.org/pdf/2111.11418.pdf>`_
    """

    def __init__(
            self,
            dim: int,
            mlp_ratio: float = 4.0,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.BatchNorm3d,
            proj_drop: float = 0.0,
            drop_path: float = 0.0,
            num_frames: int = 4,
            use_pos_emb = False,
            width = 8
    ):
        """Build Attention Block.

        Args:
            dim: Number of embedding dimensions.
            mlp_ratio: MLP expansion ratio. Default: 4.0
            act_layer: Activation layer. Default: ``nn.GELU``
            norm_layer: Normalization layer. Default: ``nn.BatchNorm2d``
            proj_drop: Dropout rate. Default: 0.0
            drop_path: Drop path rate. Default: 0.0
            layer_scale_init_value: Layer scale value at initialization. Default: 1e-5
        """

        super().__init__()

        self.norm = norm_layer(dim)
        self.token_mixer = ShapedAttention3D(dim=dim,num_frames=num_frames,width=width)

        self.layer_scale_1 = nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.mlp = ConvMlp3D(
            in_chs=dim,
            hidden_channels=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
            num_frames=num_frames,
        )

        self.layer_scale_2 = nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        if use_pos_emb:
            n_positions = width*width*num_frames
            self.pos_emb = get_sinusoid_encoding_table(n_position=n_positions, d_hid=dim)
        self.use_pos_emb = use_pos_emb
        self.num_frames = num_frames

    def forward(self, x):
        B_T, C, H, W = x.shape
        B = B_T // self.num_frames
        N = self.num_frames * H * W
        if self.use_pos_emb:
             x = x.reshape(B,self.num_frames,C,H,W).transpose(1,2).flatten(2).transpose(1,2)#[B,N,C]
             x = x + self.pos_emb[:,:N].type_as(x).to(x.device).clone().detach()
             x = x.reshape(B*self.num_frames, H, W, C).permute(0,3,1,2)#[B_T,C,H,W]
        x_norm = self.norm(x.reshape(B,self.num_frames,C,H,W).transpose(1,2)).transpose(1,2).reshape(B*self.num_frames,C,H,W)
        x_att = self.drop_path1(self.layer_scale_1(self.token_mixer(x_norm)))
        x_mlp = self.drop_path2(self.layer_scale_2(self.mlp(x_norm)))
        out = x_att + x_mlp
        return out
    
class AdaptAttention(nn.Module):
    def __init__(self, original_mlp, in_dim, mid_dim, dropout=0.0, s=0.1, use_pos_emb=False, width=8, num_frames=4):
        super().__init__()
        self.original_mlp = original_mlp
        self.in_dim = in_dim
        self.mid_dim = mid_dim
        self.down_proj = nn.Linear(in_dim, mid_dim)
        self.up_proj = nn.Linear(mid_dim, in_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = s
        self.use_pos_emb = use_pos_emb
        self.encoder = AttentionBlock3D(dim=mid_dim,num_frames=num_frames,use_pos_emb=use_pos_emb,width=width)
        self.num_frames = num_frames

        #initialization
        nn.init.kaiming_uniform_(self.down_proj.weight)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.bias)
    def forward(self, x):
        original_mlp_x = self.original_mlp(x)#[B*T,C,H,W]
        x = self.down_proj(original_mlp_x.transpose(1,3)).transpose(1,3)
        x = self.dropout(self.encoder(x))
        x = self.up_proj(x.transpose(1,3)).transpose(1,3)
        output = original_mlp_x + self.scale * x
        return output

class AdaptParallelAttention(nn.Module):
    def __init__(self, original_mlp, in_dim, mid_dim, dropout=0.0, s=0.1, use_pos_emb=False, width=8, num_frames=4):
        super().__init__()
        self.original_mlp = original_mlp
        self.in_dim = in_dim
        self.mid_dim = mid_dim
        self.down_proj = nn.Linear(in_dim, mid_dim)
        self.up_proj = nn.Linear(mid_dim, in_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = s
        self.use_pos_emb = use_pos_emb
        self.encoder = ParallelAttentionBlock3D(dim=mid_dim,num_frames=num_frames,use_pos_emb=use_pos_emb,width=width)
        self.num_frames = num_frames

        #initialization
        nn.init.kaiming_uniform_(self.down_proj.weight)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.bias)
    def forward(self, x):
        original_mlp_x = self.original_mlp(x)#[B*T,C,H,W]
        x = self.down_proj(original_mlp_x.transpose(1,3)).transpose(1,3)
        x = self.dropout(self.encoder(x))
        x = self.up_proj(x.transpose(1,3)).transpose(1,3)
        output = original_mlp_x + self.scale * x
        return output

class AdaptTextParallelAttention(nn.Module):
    def __init__(self, original_mlp, in_dim, mid_dim, dropout=0.0, s=0.1):
        super().__init__()
        self.original_mlp = original_mlp
        self.in_dim = in_dim
        self.mid_dim = mid_dim
        self.c_fc = nn.Linear(in_dim, mid_dim)
        self.up_proj = nn.Linear(mid_dim, in_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = s
        encoder_layer = nn.TransformerEncoderLayer(d_model=mid_dim,nhead=8,dim_feedforward=mid_dim*4,activation=F.gelu,batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer,num_layers=1)

        #initialization
        nn.init.kaiming_uniform_(self.c_fc.weight)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.c_fc.bias)
        nn.init.zeros_(self.up_proj.bias)
    def forward(self, x):
        original_mlp_x = self.original_mlp(x)
        x = self.c_fc(original_mlp_x)
        x = self.dropout(self.encoder(x))
        x = self.up_proj(x)
        output = original_mlp_x + self.scale * x
        return output
# class myTemporalTransformer(nn.Module):
#     def __init__(self, d_model=512, nhead=8, n_layers=3, n_position=4):
#         super().__init__()
#         self.num_frames = n_position
#         encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,nhead=nhead,batch_first=True)
#         self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=n_layers)
#         self.position_embeddings = get_sinusoid_encoding_table(n_position=n_position,d_hid=d_model)

#     def forward(self, x):
#         #x:[B*T,C,H,W]
#         n, C, H, W = x.shape
#         B = n // self.num_frames
#         x = x.reshape(B, self.num_frames, C, H*W).permute(0, 3, 1, 2) + self.position_embeddings.type_as(x).to(x.device).clone().detach()#[B,H*W,T,C]
#         x = x.transpose(1, 2).reshape(B, self.num_frames*H*W, C)#[B,T*H*W,C]
#         x = self.encoder(x).reshape(B*self.num_frames, H, W, C).permute(0, 3, 1, 2)#[B*T,C,H,W]
#         return x

def patchify_feature_map(x):
    """
    对输入的特征图进行patch化，堆叠相邻的2x2 tokens。
    
    参数:
    x: 输入的特征图，形状为[B, C, H, W]，其中C=64, H=64, W=64
    
    返回:
    patchified_x: patch化后的特征图，形状为[B, 4C, 32, 32]
    """
    # unfold操作，将每个2x2的区域展开为单独的维度
    # 结果的形状将是[B, C, 32, 32, 2, 2]
    patches = x.unfold(2, 2, 2).unfold(3, 2, 2)
    
    # 调整形状，合并通道维度和最后两个维度
    # 新的形状将是[B, 4C, 32, 32]
    patchified_x = patches.reshape(x.size(0), x.size(1) * 4, 32, 32)
    
    return patchified_x

class AdaptConv(nn.Module):
    def __init__(self, original_mlp, in_channels, adapter_channels, kernel_size, padding=True, num_frames=4, scale=0.1):
        super().__init__()
        self.original_mlp = original_mlp
        self.num_frames = num_frames
        self.scale = scale
        self.fc1 = nn.Linear(in_channels, adapter_channels)
        self.conv = nn.Conv3d(
            adapter_channels, adapter_channels,
            kernel_size=kernel_size,
            stride=(1, 1, 1),
            padding=tuple(x // 2 for x in kernel_size) if padding else(kernel_size[0]//2,0,0),
            groups=adapter_channels,
        )


        self.fc2 = nn.Linear(adapter_channels, in_channels)
        nn.init.constant_(self.conv.weight, 0.)
        nn.init.constant_(self.conv.bias, 0.)
        nn.init.constant_(self.fc1.bias, 0.)
        nn.init.constant_(self.fc2.bias, 0.)
    
    def forward(self, x):
        BT, C, H, W = x.shape
        B = BT // self.num_frames
        x_mlp = self.original_mlp(x)
        x = x.view(B, self.num_frames, C, H, W).permute(0,1,3,4,2).contiguous()#[B,T,H,W,C]
        x = self.fc1(x)
        x = x.permute(0,4,1,2,3).contiguous()#[B,C_adapter,T,H,W]
        x = self.conv(x)

        x = x.permute(0,2,3,4,1)#[B,T,H,W,C_adapter]
        x = self.fc2(x)#[B,T,H,W,C]
        x = x.permute(0,1,4,2,3).contiguous().view(BT,C,H,W)
        x = self.scale * x + x_mlp
        return x

class AdapterConv(nn.Module):

    def __init__(self, in_channels, adapter_channels, kernel_size, num_frames=4, scale=1.0):
        super().__init__()
        self.num_frames = num_frames
        self.scale = scale
        self.fc1 = nn.Linear(in_channels, adapter_channels)
        self.conv = nn.Conv3d(
            adapter_channels, adapter_channels,
            kernel_size=kernel_size,
            stride=(1, 1, 1),
            padding=tuple(x // 2 for x in kernel_size),
            groups=adapter_channels,
        )


        self.fc2 = nn.Linear(adapter_channels, in_channels)
        nn.init.constant_(self.conv.weight, 0.)
        nn.init.constant_(self.conv.bias, 0.)
        nn.init.constant_(self.fc1.bias, 0.)
        nn.init.constant_(self.fc2.bias, 0.)

    def forward(self, x):
        BT, C, H, W = x.shape

        B = BT // self.num_frames
        x_id = x
        x = x.view(B, self.num_frames, C, H, W).permute(0,1,3,4,2).contiguous()#[B,T,H,W,C]
        x = self.fc1(x)
        x = x.permute(0,4,1,2,3).contiguous()#[B,C_adapter,T,H,W]
        x = self.conv(x)

        # x = self.bn(x)
        # x = self.act(x)

        x = x.permute(0,2,3,4,1)#[B,T,H,W,C_adapter]
        x = self.fc2(x)#[B,T,H,W,C]
        x = x.permute(0,1,4,2,3).contiguous().view(BT,C,H,W)
        x_id = x_id + self.scale * x
        return x_id

class RepMixerBlock3D(nn.Module):
    def __init__(self, in_channels, adapter_channels, hidden_rate, kernel_size=(3,1,1), num_frames=4, scale=0.1):
        super().__init__()
        self.num_frames=num_frames
        self.scale = scale

        #temporal mixer
        self.norm1 = nn.BatchNorm3d(num_features=in_channels)
        self.conv1 = nn.Conv3d(in_channels=in_channels,out_channels=adapter_channels,kernel_size=kernel_size,padding=tuple(x // 2 for x in kernel_size), bias=False,groups=adapter_channels)
        self.norm2 = nn.BatchNorm3d(num_features=adapter_channels)

        #mlp
        self.conv2 = nn.Conv3d(in_channels=adapter_channels,out_channels=adapter_channels,kernel_size=(7,7,7),padding=(3,3,3),groups=adapter_channels)
        self.norm3 = nn.BatchNorm3d(num_features=adapter_channels)
        self.fc1 = nn.Conv3d(in_channels=adapter_channels,out_channels=adapter_channels*hidden_rate,kernel_size=(1,1,1),bias=False)
        self.act = nn.GELU()
        self.fc2 = nn.Conv3d(in_channels=adapter_channels*hidden_rate,out_channels=adapter_channels,kernel_size=(1,1,1),bias=False)
        self.proj = nn.Conv3d(in_channels=adapter_channels,out_channels=in_channels,kernel_size=(1,1,1),bias=False)
        nn.init.constant_(self.norm1.weight,0)
        nn.init.constant_(self.norm1.bias,0)
        nn.init.constant_(self.conv1.weight,0)
        nn.init.constant_(self.norm2.weight,0)
        nn.init.constant_(self.norm2.bias,0)
        nn.init.constant_(self.conv2.weight,0)
        nn.init.constant_(self.conv2.bias,0)
        nn.init.constant_(self.norm3.weight,0)
        nn.init.constant_(self.norm3.bias,0)
        nn.init.constant_(self.fc1.weight,0)
        nn.init.constant_(self.fc2.weight,0)
        nn.init.constant_(self.proj.weight,0)
    
    def forward(self,x):
        x_org = x #[BT, C, H, W]
        BT, C, H, W = x.shape
        B = BT // self.num_frames
        x = x.reshape(B, self.num_frames, C, H, W).transpose(1,2)#[B,C,T,H,W]
        x = x + self.norm2(self.conv1(self.norm1(x)))
        x = x + self.fc2(self.act(self.fc1(self.norm3(self.conv2(x)))))#[B,C_adapter,T,H,W]
        x = self.proj(x).transpose(1,2)#[B,T,C,H,W]
        x = x.reshape(BT,C,H,W)
        x = self.scale*x + x_org
        return x


class ResAdapterBlock(nn.Module):
    def __init__(self, original_block, d_model, adapter_channels, kernel_size, num_frames=4, scale=1.0):
        super().__init__()
        self.original_block = original_block
        for _, p in self.original_block.named_parameters():
            p.requires_grad = False
        self.adapter = AdapterConv(in_channels=d_model,adapter_channels=adapter_channels,kernel_size=kernel_size,num_frames=num_frames,scale=scale)
        # self.adapter = RepMixerBlock3D(in_channels=d_model,adapter_channels=adapter_channels,hidden_rate=3,kernel_size=kernel_size,num_frames=num_frames,scale=scale)

    def forward(self,x):
        x = self.adapter(x)
        x = self.original_block(x)
        return x

class AdapterTransBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_type=None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_type = attn_type

    def make_attn_mask(self, x):
        attn_mask = None
        if self.attn_type == 'uni':
            attn_mask = torch.zeros(x.size(0), x.size(0)).to(x.device)
            attn_mask.fill_(float("-inf"))
            attn_mask.triu_(1)  # zero out the lower diagonal
        return attn_mask

    def forward(self, x, attn_mask=None):
        if attn_mask is None:
            attn_mask = self.make_attn_mask(x)
        
        ln_x = self.ln_1(x)
        x = x + self.attn(ln_x, ln_x, ln_x, need_weights=False, attn_mask=attn_mask)[0]
        x = x + self.mlp(self.ln_2(x))
        return x
    
class TextTransfAdapter(nn.Module):
    def __init__(
        self, original_block, embed_dim=512, mid_dim=64, n_head=2, idx=-1,
        attn_type=None, scale=0.1, pos=None, seq_len=12,
        pca=False, **kwargs
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.mid_dim = mid_dim
        self.n_head = n_head
        self.idx = idx
        self.up_share, self.down_share = False, False

        self.pca = pca
        self.original_block = original_block
        if pca:
            def pca_down(x):
                seq_len, bs, _ =  x.shape
                group_dim = int(embed_dim // mid_dim)
                assert group_dim * mid_dim == embed_dim
                x = x.permute(1, 0, 2)
                x = x.reshape(bs, seq_len, mid_dim, group_dim)   # bs * seq * dim
                pca_dim = min(*x.shape[-2:], mid_dim)
                u, s, v = torch.pca_lowrank(x.float(), pca_dim)
                x_down = torch.matmul(x, v[:, :, :, :1].half()).reshape(bs, seq_len, mid_dim)
                x_down = x_down.permute(1, 0, 2)
                return x_down
            self.down = pca_down
        else:
            self.down = nn.Linear(embed_dim, mid_dim)
        self.c_fc = nn.Linear(mid_dim, embed_dim)
        self.act = QuickGELU()
        self.scale = scale
        self.ln_pre = LayerNorm(mid_dim)
        self.block = AdapterTransBlock(mid_dim, n_head, attn_type=attn_type)
        self.positional_embedding = nn.Parameter(torch.randn(seq_len, mid_dim))
        if pos is not None:
            self.positional_embedding = nn.Parameter(pos.clone()[:,:mid_dim])
        else:
            nn.init.normal_(self.positional_embedding, std=0.01)

        self.ln_in_flag = kwargs.get('lnin', False)
        if self.ln_in_flag:
            self.ln_in = LayerNorm(embed_dim)
    
        self.modal_dynamic = False
        if kwargs.get('modal_dynamic', {}):
            md_cfg = kwargs['modal_dynamic']
            if idx in md_cfg['adapter_idx']:
                self.modal_dynamic = True
                md_idx = md_cfg['adapter_idx'].index(idx)
                self.shared_factor = md_cfg['factor'](md_idx)
                self.md_pos = md_cfg['position']
                if md_cfg['position'] in ['up in', 'down out']:
                    self.md_proj = nn.Linear(self.shared_factor.shape[0], mid_dim)
                else:
                    self.md_proj = nn.Linear(self.shared_factor.shape[0], embed_dim)
        self.init()    
        
    def forward(self, x: Tensor, *args, **kwargs):
        x_mlp = self.original_block(x)

        if self.ln_in_flag:
            x = self.ln_in(x)

        down_weight = self.down.weight
        if self.modal_dynamic and self.md_pos.startswith('down'):
            factor = self.md_proj(self.shared_factor)
            if self.md_pos == 'down in':
                down_weight = torch.einsum('i,oi->oi', factor, down_weight)
            elif self.md_pos == 'down out':
                down_weight = torch.einsum('o,oi->oi', factor, down_weight)
        down = x @ down_weight.T + self.down.bias

        pos_emd = self.positional_embedding[:, None].to(x.dtype)
        down = down + pos_emd
        
        down = self.ln_pre(down)

        down = self.block(down)
        down = self.act(down)
        
        up_weight = self.c_fc.weight
        if self.modal_dynamic and self.md_pos.startswith('up'):
            factor = self.md_proj(self.shared_factor)
            if self.md_pos == 'up in':
                up_weight = torch.einsum('i,oi->oi', factor, up_weight)
            elif self.md_pos == 'up out':
                up_weight = torch.einsum('o,oi->oi', factor, up_weight)
        up = down @ up_weight.T + self.c_fc.bias
        delta_x = self.scale * up
        return delta_x + x_mlp

    def init(self):
        proj_std = ((2 *self.embed_dim) ** -0.5)
        attn_std = self.embed_dim ** -0.5
        fc_std = (2 * self.embed_dim) ** -0.5
        nn.init.normal_(self.block.attn.in_proj_weight, std=attn_std)
        nn.init.normal_(self.block.attn.out_proj.weight, std=proj_std)
        nn.init.normal_(self.block.mlp.c_fc.weight, std=fc_std)
        nn.init.normal_(self.block.mlp.c_proj.weight, std=proj_std)

        if not self.pca:
            nn.init.normal_(self.down.weight, std=fc_std)
            nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.c_fc.weight)
        nn.init.zeros_(self.c_fc.bias)


class VideoAdapter(nn.Module):
    def __init__(
        self, embed_dim=512, cls_mid=64, n_head=2,
        attn_type=None, scale=0.1, pos=None, idx=0, seq_len=12, 
        pca='pca', calibrate_func='v1', ratio=[0.5, 0.5], **kwargs
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.cls_mid = cls_mid
        self.n_head = n_head
        self.temporal = kwargs.get('temporal', [])
        assert all([t in ['p', 'c'] for t in self.temporal])
        self.idx = idx
        self.no_cc = kwargs.get('no_cc', False)

        self.ln_in_flag = kwargs.get('lnin', False)
        if self.ln_in_flag:
            self.ln_in = LayerNorm(embed_dim)

        cal_func = {
            'v1': TaModuleV1,
            'v2': TaModuleV2,
        }
        if 'c' in self.temporal:
            self.conv_rf_c = cal_func[calibrate_func](
                c_in=cls_mid,            # number of input filters
                c_out=cls_mid,
                cc_dim=0 if self.no_cc else cls_mid,
                mid_dim=16,            # reduction ratio for MLP
                kernels=3,      # list of temporal kernel sizes
                concat=not self.no_cc
            )
        if 'p' in self.temporal:
            self.conv_rf_p = cal_func[calibrate_func](
                c_in=cls_mid,            # number of input filters
                c_out=cls_mid,
                cc_dim=0 if self.no_cc else cls_mid,
                mid_dim=16,            # reduction ratio for MLP
                kernels=3,      # list of temporal kernel sizes
                concat=not self.no_cc
            )

        self.down = nn.Linear(embed_dim, cls_mid)
        self.up = nn.Linear(cls_mid, embed_dim)

        self.act = QuickGELU()
        self.scale = scale
        self.ln_pre = LayerNorm(cls_mid)
        self.block = AdapterTransBlock(cls_mid, n_head, attn_type)

        self.positional_embedding = nn.Parameter(torch.randn(seq_len * 2 + 1, cls_mid))
        nn.init.normal_(self.positional_embedding, std=0.01)
        self.cc = nn.Parameter(cls_mid ** -.5 * torch.randn(cls_mid))
        self.patch_pooling = pca

        if ratio[1] is None:
            self.ratio_p = nn.Parameter(torch.tensor(0.))
        else:
            self.ratio_p = ratio[1]
                    
        self.modal_dynamic = False
        if kwargs.get('modal_dynamic', {}):
            md_cfg = kwargs['modal_dynamic']
            if idx in md_cfg['adapter_idx']:
                self.modal_dynamic = True
                md_idx = md_cfg['adapter_idx'].index(idx)
                self.shared_factor = md_cfg['factor'](md_idx)
                self.md_pos = md_cfg['position']
                if 'up in' in md_cfg['position'] or 'down out' in md_cfg['position']:
                    self.md_proj = nn.Linear(self.shared_factor.shape[0], cls_mid)
                else:
                    self.md_proj = nn.Linear(self.shared_factor.shape[0], embed_dim)
        self.init()

    def forward(self, x: Tensor, casual_mask, **kwargs):
        """
        input shape v16 197 * (bs * frames) * 512
        cls 1 * (bs * frames) * 512
        video input: cls bs * frames
        """
        # ----- utils ----- #
        B, F = casual_mask.shape
        patch_num = int((x.shape[0] - 1) ** .5)
        # ----- utils ----- #

        # ----- downsample the cls and patch ----- #
        if self.ln_in_flag:
            x = self.ln_in(x)

        down_weight = self.down.weight
        if self.modal_dynamic and self.md_pos.startswith('down'):
            factor = self.md_proj(self.shared_factor)                
            if 'down in' in self.md_pos:
                down_weight = torch.einsum('i,oi->oi', factor, down_weight)
            elif 'down out' in self.md_pos:
                down_weight = torch.einsum('o,oi->oi', factor, down_weight)

        if self.modal_dynamic and self.md_pos in ['down in cls', 'down out cls']:
            cls_emb = x[0].reshape(B, -1, self.embed_dim)
            patch_emb = x[1:].reshape(patch_num ** 2, B, -1, self.embed_dim)
            cls_down = self.act(cls_emb @ down_weight.T + self.down.bias)
            patch_down = self.act(self.down(patch_emb))
        else:
            down = x @ down_weight.T + self.down.bias
            down = self.act(down)
            cls_down = down[0].reshape(B, -1, self.cls_mid)
            patch_down = down[1:].reshape(patch_num ** 2, B, -1, self.cls_mid) 
        # ----- downsample the cls and patch ----- #
        
        # ----- use pca or mean pooling to down sample patch ----- #
        if self.patch_pooling == 'pca':
            patch_emb_bnel = patch_down.permute(1, 2 ,3, 0)  # L * B * N * E -> B * N * E * L
            with autocast(dtype=torch.float32):
                u, s, v = torch.pca_lowrank(patch_emb_bnel, 1)
            patch_pca = torch.matmul(patch_emb_bnel, v).reshape(B, F, -1)
        elif self.patch_pooling == 'tconv':
            patch_2d = patch_down.reshape(patch_num, patch_num, B, F, -1)
            patch_bcfpp = patch_2d.permute(2, 4, 3, 0, 1)   # B, E, frame, pn, pn
            offset = (F + 1) % 2
            patch_pca = self.tconv(patch_bcfpp)[:, :, offset:]     # B, E, fn, pn, pn
            patch_pca = patch_pca.reshape(B, -1, F, patch_num ** 2)
            patch_pca = patch_pca.mean(dim=-1).permute(0, 2, 1)     # B, F, E
        else:
            patch_pca = patch_down.mean(dim=0)
        # ----- use pca or mean pooling to down sample patch ----- #

        # ----- temporal modeling ----- #
        dt, device = cls_down.dtype, cls_down.device
        cls_mid = cls_down.shape[-1]
        if self.no_cc:
            temporal_seq = torch.cat([cls_down, patch_pca], dim=1)
        else:
            cc = self.cc.to(dt) + torch.zeros(B, 1, cls_mid, dtype=dt, device=device)
            temporal_seq = torch.cat([cc, cls_down, patch_pca], dim=1)
        pos_emd = self.positional_embedding[:temporal_seq.size(1), :].to(x.dtype)
        temporal_seq = temporal_seq + pos_emd

        temporal_seq = self.ln_pre(temporal_seq)
        temporal_seq = temporal_seq.permute(1, 0, 2)
        # ----- temporal modeling ----- #

        # ----- get attention mask for video ----- #
        # TODO
        v_mask = casual_mask
        mask = torch.cat([torch.ones(B, int(not self.no_cc)).to(device), casual_mask.repeat(1, 2)], dim=1)
        e_mask = (1.0 - mask.unsqueeze(1)) * -1000000.0
        e_mask = e_mask.expand(-1, mask.size(1), -1)
        attn_mask_ = e_mask.repeat_interleave(self.n_head, dim=0)
        # ----- get attention mask for video ----- $

        # ----- temporal modeling for cls ----- #
        temporal_seq = self.block(temporal_seq, attn_mask_)
        temporal_seq = temporal_seq.permute(1, 0, 2)    # bs, frames + 1, e_a
        temporal_seq = self.act(temporal_seq)
        
        if self.no_cc:
            cc_temp = 0
            cls_temp = temporal_seq[:, :F]
            patch_temp = temporal_seq[:, F:]
        else:
            cc_temp = temporal_seq[:, 0]
            cls_temp = temporal_seq[:, 1:F+1]
            patch_temp = temporal_seq[:, F+1:]
        # ----- temporal modeling for cls ----- #

        # ----- patch for patch calibrate ----- #
        if 'p' in self.temporal:
            calibrate_input = cls_temp + patch_temp
            
            alpha_p = self.conv_rf_p(cc_temp, calibrate_input, v_mask)
            delta_patch = torch.einsum('bfi,oi,pbfi->pbfo', alpha_p, self.up.weight, patch_down)
            delta_patch = delta_patch.reshape(patch_num ** 2, -1, self.embed_dim)
        else:
            delta_patch = torch.einsum('pbfi,oi->pbfo', patch_down, self.up.weight).reshape(patch_num ** 2, -1, self.embed_dim)
        patch_up = delta_patch + self.up.bias
        # ----- patch for patch calibrate ----- #

        # ----- cls up sample ----- #
        up_weight = self.up.weight
        if self.modal_dynamic and self.md_pos.startswith('up'):
            factor = self.md_proj(self.shared_factor)
            if self.md_pos == 'up in':
                up_weight = torch.einsum('i,oi->oi', factor, up_weight)
            elif self.md_pos == 'up out':
                up_weight = torch.einsum('o,oi->oi', factor, up_weight)
        if 'c' in self.temporal:
            calibrate_input = cls_temp + patch_temp
            alpha_c = self.conv_rf_c(cc_temp, calibrate_input, v_mask)
            cls_up = torch.einsum('bfi,oi,bfi->bfo', alpha_c, up_weight, cls_temp)
        else:
            cls_up = cls_temp @ up_weight.T
        cls_up = (cls_up + self.up.bias).reshape(1, -1, self.embed_dim)
        # ----- cls up sample ----- #

        delta_x = self.scale * torch.cat([cls_up, patch_up])
        return delta_x
    
    def init(self):
        proj_std = ((2 *self.embed_dim) ** -0.5)
        attn_std = self.embed_dim ** -0.5
        fc_std = (2 * self.embed_dim) ** -0.5
        nn.init.normal_(self.block.attn.in_proj_weight, std=attn_std)
        nn.init.normal_(self.block.attn.out_proj.weight, std=proj_std)
        nn.init.normal_(self.block.mlp.c_fc.weight, std=fc_std)
        nn.init.normal_(self.block.mlp.c_proj.weight, std=proj_std)


        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        
class VideoCLIP(nn.Module):
    def __init__(
            self,
            clip_2d,
            num_frames = 4,
            init_logit_scale: float = np.log(1 / 0.07)
    ):
        super().__init__()
        self.num_frames = num_frames
        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)

        for param in clip_2d.parameters():
            param.requires_grad = False
        # clip_2d.logit_scale.requires_grad = False

        for i in range(1,len(clip_2d.visual.trunk.stem)):
            clip_2d.visual.trunk.stem[i] = ResAdapterBlock(original_block=clip_2d.visual.trunk.stem[i],d_model=64,adapter_channels=64//4,kernel_size=(3,3,3),num_frames=num_frames,scale=0.1).to(dtype=torch.bfloat16)

        # clip_2d.visual.trunk.stem[1] = AdaptConv(original_mlp=clip_2d.visual.trunk.stem[1],in_channels=64,adapter_channels=64//4,kernel_size=(3,1,1),padding=False,num_frames=num_frames,scale=0.1).to(dtype=torch.bfloat16)
        # clip_2d.visual.trunk.stem[2] = AdaptConv(original_mlp=clip_2d.visual.trunk.stem[2],in_channels=64,adapter_channels=64//4,kernel_size=(5,1,1),num_frames=num_frames,scale=0.1).to(dtype=torch.bfloat16)
        
        dims = [64,128,256,512]
        for i, dim in zip(range(len(clip_2d.visual.trunk.stages)),dims):
            for j in range(len(clip_2d.visual.trunk.stages[i].blocks)):
                clip_2d.visual.trunk.stages[i].blocks[j].token_mixer = AdaptConv(original_mlp=clip_2d.visual.trunk.stages[i].blocks[j].token_mixer,in_channels=dim,adapter_channels=dim//4,kernel_size=(3,3,3),num_frames=num_frames,scale=0.1).to(dtype=torch.bfloat16)
                clip_2d.visual.trunk.stages[i].blocks[j].mlp = AdaptConv(original_mlp=clip_2d.visual.trunk.stages[i].blocks[j].mlp,in_channels=dim,adapter_channels=dim//4, kernel_size=(3,3,3),num_frames=num_frames,scale=0.1).to(dtype=torch.bfloat16)

        clip_2d.visual.trunk.stages[3] = AdaptParallelAttention(original_mlp=clip_2d.visual.trunk.stages[3],in_dim=512,mid_dim=512//4,use_pos_emb=True,width=8,num_frames=num_frames).to(dtype=torch.bfloat16)
        clip_2d.visual.trunk.final_conv = AdaptParallelAttention(original_mlp=clip_2d.visual.trunk.final_conv,in_dim=1024,mid_dim=1024//4,width=8,num_frames=num_frames).to(dtype=torch.bfloat16)
        clip_2d.visual.trunk.head = AdaptMLP(original_mlp=clip_2d.visual.trunk.head,in_dim=512,mid_dim=4,s=0.1).to(dtype=torch.bfloat16)
        
        for block in clip_2d.text.transformer.resblocks:
            block.mlp = AdaptMLP(original_mlp=block.mlp,in_dim=512,mid_dim=4,s=0.1).to(dtype=torch.bfloat16)
            # block.mlp = AdaptTextParallelAttention(original_mlp=block.mlp,in_dim=512,mid_dim=256).to(dtype=torch.bfloat16)
            # block.mlp = TextTransfAdapter(original_block=block.mlp,embed_dim=512,mid_dim=512//4,n_head=2,scale=0.1,seq_len=77)


        
        self.clip_2d = clip_2d


    def forward(
            self,
            video: Optional[torch.Tensor] = None,
            text: Optional[torch.Tensor] = None,
    ):
        """
        video:shape=[B,num_frames,C,H,W]
        """
        B,T,C,H,W = video.shape#[B,4,3,224,224]

        video = video.reshape(B*T,C,H,W)
        multi_text = False
        if len(text.shape) == 3:
            b, n, d = text.shape
            text = text.reshape(b*n,d)
            multi_text = True

        out_dict = self.clip_2d(video, text)
        if multi_text:
            out_dict['text_features'] = out_dict['text_features'].reshape(b,n,-1)

        out_dict['image_features'] = out_dict['image_features'].view(B,T,-1).mean(dim=1)

        output = {}
        output['image_features'] = F.normalize(out_dict['image_features'],dim=-1)
        output['text_features'] = F.normalize(out_dict['text_features'],dim=-1)
        output['logit_scale'] = self.logit_scale.exp()
        return output

def convert_weights_to_lp(model: nn.Module, dtype=torch.float16):
    """Convert applicable model parameters to low-precision (bf16 or fp16)"""

    def _convert_weights(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.to(dtype)
            if l.bias is not None:
                l.bias.data = l.bias.data.to(dtype)

        if isinstance(l, (nn.MultiheadAttention, Attention)):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.to(dtype)

        if isinstance(l, (CLIP, TextTransformer)):
            # convert text nn.Parameter projections
            attr = getattr(l, "text_projection", None)
            if attr is not None:
                attr.data = attr.data.to(dtype)

        if isinstance(l, VisionTransformer):
            # convert vision nn.Parameter projections
            attr = getattr(l, "proj", None)
            if attr is not None:
                attr.data = attr.data.to(dtype)

    model.apply(_convert_weights)


convert_weights_to_fp16 = convert_weights_to_lp  # backwards compat


# used to maintain checkpoint compatibility
def convert_to_custom_text_state_dict(state_dict: dict):
    if 'text_projection' in state_dict:
        # old format state_dict, move text tower -> .text
        new_state_dict = {}
        for k, v in state_dict.items():
            if any(k.startswith(p) for p in (
                'text_projection',
                'positional_embedding',
                'token_embedding',
                'transformer',
                'ln_final',
            )):
                k = 'text.' + k
            new_state_dict[k] = v
        return new_state_dict
    return state_dict


def build_model_from_openai_state_dict(
        state_dict: dict,
        quick_gelu=True,
        cast_dtype=torch.float16,
):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_size = vision_patch_size * grid_size
    else:
        counts: list = [
            len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_size = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    vision_cfg = CLIPVisionCfg(
        layers=vision_layers,
        width=vision_width,
        patch_size=vision_patch_size,
        image_size=image_size,
    )
    text_cfg = CLIPTextCfg(
        context_length=context_length,
        vocab_size=vocab_size,
        width=transformer_width,
        heads=transformer_heads,
        layers=transformer_layers,
    )
    model = CLIP(
        embed_dim,
        vision_cfg=vision_cfg,
        text_cfg=text_cfg,
        quick_gelu=quick_gelu,  # OpenAI models were trained with QuickGELU
        cast_dtype=cast_dtype,
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        state_dict.pop(key, None)
    convert_weights_to_fp16(model)  # OpenAI state dicts are partially converted to float16
    model.load_state_dict(state_dict)
    return model.eval()


def trace_model(model, batch_size=256, device=torch.device('cpu')):
    model.eval()
    image_size = model.visual.image_size
    example_images = torch.ones((batch_size, 3, image_size, image_size), device=device)
    example_text = torch.zeros((batch_size, model.context_length), dtype=torch.int, device=device)
    model = torch.jit.trace_module(
        model,
        inputs=dict(
            forward=(example_images, example_text),
            encode_text=(example_text,),
            encode_image=(example_images,)
        ))
    model.visual.image_size = image_size
    return model


def resize_pos_embed(state_dict, model, interpolation: str = 'bicubic', antialias: bool = True):
    # Rescale the grid of position embeddings when loading from state_dict
    old_pos_embed = state_dict.get('visual.positional_embedding', None)
    if old_pos_embed is None or not hasattr(model.visual, 'grid_size'):
        return
    grid_size = to_2tuple(model.visual.grid_size)
    extra_tokens = 1  # FIXME detect different token configs (ie no class token, or more)
    new_seq_len = grid_size[0] * grid_size[1] + extra_tokens
    if new_seq_len == old_pos_embed.shape[0]:
        return

    if extra_tokens:
        pos_emb_tok, pos_emb_img = old_pos_embed[:extra_tokens], old_pos_embed[extra_tokens:]
    else:
        pos_emb_tok, pos_emb_img = None, old_pos_embed
    old_grid_size = to_2tuple(int(math.sqrt(len(pos_emb_img))))

    logging.info('Resizing position embedding grid-size from %s to %s', old_grid_size, grid_size)
    pos_emb_img = pos_emb_img.reshape(1, old_grid_size[0], old_grid_size[1], -1).permute(0, 3, 1, 2)
    pos_emb_img = F.interpolate(
        pos_emb_img,
        size=grid_size,
        mode=interpolation,
        antialias=antialias,
        align_corners=False,
    )
    pos_emb_img = pos_emb_img.permute(0, 2, 3, 1).reshape(1, grid_size[0] * grid_size[1], -1)[0]
    if pos_emb_tok is not None:
        new_pos_embed = torch.cat([pos_emb_tok, pos_emb_img], dim=0)
    else:
        new_pos_embed = pos_emb_img
    state_dict['visual.positional_embedding'] = new_pos_embed


def resize_text_pos_embed(state_dict, model, interpolation: str = 'linear', antialias: bool = False):
    old_pos_embed = state_dict.get('positional_embedding', None)
    if old_pos_embed is None:
        return
    # FIXME add support for text cls_token
    model_pos_embed = getattr(model, 'positional_embedding', None)
    if model_pos_embed is None:
        model_pos_embed = getattr(model.text, 'positional_embedding', None)

    old_num_pos = old_pos_embed.shape[0]
    old_width = old_pos_embed.shape[1]
    num_pos = model_pos_embed.shape[0]
    width = model_pos_embed.shape[1]
    assert old_width == width, 'text pos_embed width changed!'
    if old_num_pos == num_pos:
        return

    logging.info('Resizing text position embedding num_pos from %s to %s', old_num_pos, num_pos)
    old_pos_embed = old_pos_embed.reshape(1, old_num_pos, old_width).permute(0, 2, 1)
    old_pos_embed = F.interpolate(
        old_pos_embed,
        size=num_pos,
        mode=interpolation,
        antialias=antialias,
        align_corners=False,
    )
    old_pos_embed = old_pos_embed.permute(0, 2, 1)[0]
    new_pos_embed = old_pos_embed

    state_dict['positional_embedding'] = new_pos_embed


def get_model_preprocess_cfg(model):
    module = getattr(model, 'visual', model)
    preprocess_cfg = getattr(module, 'preprocess_cfg', {})
    if not preprocess_cfg:
        # use separate legacy attributes if preprocess_cfg dict not found
        size = getattr(module, 'image_size')
        if size is not None:
            preprocess_cfg['size'] = size
        mean = getattr(module, 'image_mean', None)
        if mean is not None:
            preprocess_cfg['mean'] = mean
        std = getattr(module, 'image_std', None)
        if std is not None:
            preprocess_cfg['std'] = std
    return preprocess_cfg


def set_model_preprocess_cfg(model, preprocess_cfg: Dict[str, Any]):
    module = getattr(model, 'visual', model)
    module.image_mean = preprocess_cfg['mean']  # legacy attribute, keeping for bwd compat
    module.image_std = preprocess_cfg['std']  # legacy attribute, keeping for bwd compat
    module.preprocess_cfg = copy.deepcopy(preprocess_cfg)  # new attr, package all pp cfg as dict


def get_model_tokenize_cfg(model):
    module = getattr(model, 'text', model)
    cfg = {}
    context_length = getattr(module, 'context_length', None)
    if context_length is not None:
        cfg['context_length'] = context_length
    vocab_size = getattr(module, 'vocab_size', None)
    if vocab_size is not None:
        cfg['vocab_size'] = vocab_size
    return cfg


class Custom3DConvModule(nn.Module):
    def __init__(self, in_channels, num_frames=4):
        super(Custom3DConvModule, self).__init__()
        self.num_frames = num_frames
        
        # 第一个3x1x1的3D卷积层
        self.conv3d_1 = nn.Conv3d(in_channels, in_channels*4, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0))
        self.bn_act3d_1 = BatchNormAct3d(in_channels*4)
        
        # 第二个3x1x1的3D卷积层
        self.conv3d_2 = nn.Conv3d(in_channels*4, in_channels, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0))
        self.bn_act3d_2 = BatchNormAct3d(in_channels)
                
    def forward(self, x):
        # 输入 x 的形状为 (B*T, C, H, W)
        x_in = x
        B = x.shape[0] // self.num_frames
        C, H, W = x.shape[1], x.shape[2], x.shape[3]
        
        # 重排列数据维度以匹配3D卷积层的输入需求
        x = x.reshape(B, self.num_frames, C, H, W).transpose(1, 2)

        # 通过第一个3D卷积层和批归一化激活层
        x = self.bn_act3d_1(self.conv3d_1(x))
        
        # 通过第二个3D卷积层和批归一化激活层
        x = self.bn_act3d_2(self.conv3d_2(x))
        
        # 恢复原始维度顺序并合并批次和帧数维度
        x = x.transpose(1, 2).reshape(B * self.num_frames, C, H, W)

        return x + x_in


class BatchNormAct3d(nn.BatchNorm3d):
    """BatchNorm + Activation for 3D

    This module performs BatchNorm + Activation for 3D convolutions in a manner that will remain backwards
    compatible with weights trained with separate bn, act.
    """
    def __init__(
            self,
            num_features,
            eps=1e-5,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
            apply_act=True,
            act_layer=nn.ReLU,
            act_kwargs=None,
            inplace=True,
            drop_layer=None,
            device=None,
            dtype=None,
    ):
        try:
            factory_kwargs = {'device': device, 'dtype': dtype}
            super(BatchNormAct3d, self).__init__(
                num_features,
                eps=eps,
                momentum=momentum,
                affine=affine,
                track_running_stats=track_running_stats,
                **factory_kwargs,
            )
        except TypeError:
            # NOTE for backwards compat with old PyTorch w/o factory device/dtype support
            super(BatchNormAct3d, self).__init__(
                num_features,
                eps=eps,
                momentum=momentum,
                affine=affine,
                track_running_stats=track_running_stats,
            )
        self.drop = drop_layer() if drop_layer is not None else nn.Identity()
        self.act = _create_act(act_layer, act_kwargs=act_kwargs, inplace=inplace, apply_act=apply_act)

    def forward(self, x):
        _assert(x.ndim == 5, f'expected 5D input (got {x.ndim}D input)')

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked.add_(1)  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        x = F.batch_norm(
            x,
            self.running_mean if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight,
            self.bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )
        x = self.drop(x)
        x = self.act(x)
        return x
