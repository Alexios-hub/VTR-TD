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
from torch import nn
from torch.utils.checkpoint import checkpoint
from functools import partial

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
        self.act = nn.ReLU()
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
    #     down = self.act(down)
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
        down = self.act(down)
        down = self.dropout(down)
        up = self.up_proj(down)
        if len(up.shape) == 4:
            up = up.transpose(1,3)

        output = original_mlp_x + up * self.scale
        return output

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

class Attention3D(nn.Module):
    """Multi-headed Self Attention module.

    Source modified from:
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    """
    fused_attn: torch.jit.Final[bool]

    def __init__(
            self,
            dim: int,
            head_dim: int = 8,
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
            norm_layer: nn.Module = nn.BatchNorm2d,
            proj_drop: float = 0.0,
            drop_path: float = 0.0,
            num_frames: int = 4,
            use_pos_emb = False,
            n_positions = 300000
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
            self.pos_emb = get_sinusoid_encoding_table(n_position=n_positions, d_hid=dim)
        self.use_pos_emb = use_pos_emb
        self.num_frames = num_frames



    def forward(self, x):
        if self.use_pos_emb:
             B_T, C, H, W = x.shape
             B = B_T // self.num_frames
             N = self.num_frames * H * W
             x = x.reshape(B,self.num_frames,C,H,W).transpose(1,2).flatten(2).transpose(1,2)#[B,N,C]
             x = x + self.pos_emb[:,:N].type_as(x).to(x.device).clone().detach()
             x = x.reshape(B*self.num_frames, H, W, C).permute(0,3,1,2)#[B_T,C,H,W]
        x = x + self.drop_path1(self.layer_scale_1(self.token_mixer(self.norm(x))))
        x = x + self.drop_path2(self.layer_scale_2(self.mlp(x)))
        return x
    

class AdaptAttention(nn.Module):
    def __init__(self, original_mlp, in_dim, mid_dim, dropout=0.0, s=0.1, use_pos_emb=False, n_positions = 300000, num_frames=4):
        super().__init__()
        self.original_mlp = original_mlp
        self.in_dim = in_dim
        self.mid_dim = mid_dim
        self.down_proj = nn.Linear(in_dim, mid_dim)
        self.up_proj = nn.Linear(mid_dim, in_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = s
        self.use_pos_emb = use_pos_emb
        self.encoder = AttentionBlock3D(dim=mid_dim,num_frames=num_frames,use_pos_emb=use_pos_emb,n_positions=n_positions)
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

        dims = [64,128,256,512]

        # clip_2d.visual.trunk.stem = nn.Sequential(
        #     AdaptAttention(original_mlp=clip_2d.visual.trunk.stem[0], in_dim=64,mid_dim=32, use_pos_emb=True, n_positions=num_frames*128*128, num_frames=self.num_frames),
        #     AdaptAttention(original_mlp=clip_2d.visual.trunk.stem[1], in_dim=64,mid_dim=32, num_frames=self.num_frames),
        #     AdaptAttention(original_mlp=clip_2d.visual.trunk.stem[2], in_dim=64,mid_dim=32, num_frames=self.num_frames)
        # )

        # for stage, dim in zip(clip_2d.visual.trunk.stages,dims):
        #     for block in stage.blocks:
        #         block.mlp = AdaptAttention(original_mlp=block.mlp,in_dim=dim,mid_dim=max(dim//4,32),num_frames=num_frames)

        # clip_2d.visual.trunk.stem[2] = AdaptAttention(original_mlp=clip_2d.visual.trunk.stem[2], in_dim=64,mid_dim=32,use_pos_emb=True,n_positions=num_frames*64*64, num_frames=self.num_frames)
        for stage, dim in zip(clip_2d.visual.trunk.stages,dims):
            if dim == 64:
                stage.blocks[len(stage.blocks)-1].mlp = AdaptAttention(original_mlp=stage.blocks[len(stage.blocks)-1].mlp,in_dim=dim,mid_dim=64,use_pos_emb=True,n_positions=num_frames*64*64,num_frames=num_frames)
            else:
                stage.blocks[len(stage.blocks)-1].mlp = AdaptAttention(original_mlp=stage.blocks[len(stage.blocks)-1].mlp,in_dim=dim,mid_dim=64,num_frames=num_frames)
        
        for block in clip_2d.text.transformer.resblocks:
            block.mlp = AdaptMLP(original_mlp=block.mlp,in_dim=512, mid_dim=64)
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
        # with torch.no_grad():
        out_dict = self.clip_2d(video, text)
        # if isinstance(out_dict['image_features'],tuple):
        #     out_dict.update({
        #         'image_features':out_dict['image_features'][0],#[B*T,512]
        #         'tokens':out_dict['image_features'][1]#[B*T,196,768]
        #     })
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
