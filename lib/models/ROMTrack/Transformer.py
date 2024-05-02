from functools import partial
from itertools import repeat
# from torch._six import container_abcs
import collections.abc as container_abcs

import torch
from torch import nn
import torch.nn.functional as F

import numpy as np

from einops import rearrange
from einops.layers.torch import Rearrange

from timm.models.layers import Mlp, DropPath

from lib.utils.misc import is_main_process


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]  # 768

        # for template
        num_patches_t = model.patch_embed_t.num_patches  # 8*8=64
        num_extra_tokens_t = model.pos_embed_t.shape[-2] - num_patches_t + 1  # need to drop cls_token
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens_t) ** 0.5)  # 14
        # height (== width) for the new position embedding
        new_size = int(num_patches_t ** 0.5)  # 8
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Template Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens_t]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens_t:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            # new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            new_pos_embed = pos_tokens
            checkpoint_model['pos_embed_t'] = new_pos_embed
        else:
            print("Template Position %dx%d to %dx%d, no need to interpolate" % (orig_size, orig_size, new_size, new_size))
            checkpoint_model['pos_embed_t'] = pos_embed_checkpoint[:, num_extra_tokens_t:]

        # for search
        num_patches_s = model.patch_embed_s.num_patches  # 16*16=256
        num_extra_tokens_s = model.pos_embed_s.shape[-2] - num_patches_s + 1  # need to drop cls_token
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens_t) ** 0.5)  # 14
        # height (== width) for the new position embedding
        new_size = int(num_patches_s ** 0.5)  # 16
        if orig_size != new_size:
            print("Search Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens_s]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens_s:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            # new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            new_pos_embed = pos_tokens
            checkpoint_model['pos_embed_s'] = new_pos_embed
        else:
            print("Search Position %dx%d to %dx%d, no need to interpolate" % (orig_size, orig_size, new_size, new_size))
            checkpoint_model['pos_embed_t'] = pos_embed_checkpoint[:, num_extra_tokens_s:]


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True,
                 stride=16, padding=0):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        # (input - kernel + 2 * padding) / stride + 1
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class EmptyPatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding without param
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True,
                 stride=16, padding=0):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten


class Attention(nn.Module):
    def __init__(self, t_size, s_size, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.t_size = t_size  # 64 for ROMTrack, 144 for ROMTrack-384
        self.s_size = s_size  # 256 for ROMTrack, 576 for ROMTrack-384
        self.mix_size = self.t_size + self.s_size

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # B, num_heads, N, C//num_heads
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        q_t, q_it, q_s = torch.split(q, [self.t_size, self.t_size, self.s_size], dim=2)
        k_t, k_it, k_s = torch.split(k, [self.t_size, self.t_size, self.s_size], dim=2)
        v_t, v_it, v_s = torch.split(v, [self.t_size, self.t_size, self.s_size], dim=2)

        # mix attention of t&s to t&it&s
        q_ts = torch.cat([q_t, q_s], dim=2)
        mix_attn = (q_ts @ k.transpose(-2, -1)) * self.scale
        mix_attn = mix_attn.softmax(dim=-1)
        mix_attn = self.attn_drop(mix_attn)

        ts = (mix_attn @ v).transpose(1, 2).reshape(B, self.mix_size, C)
        t, s = torch.split(ts, [self.t_size, self.s_size], dim=1)

        # inherent attention of it
        inh_attn = (q_it @ k_it.transpose(-2, -1)) * self.scale
        inh_attn = inh_attn.softmax(dim=-1)
        inh_attn = self.attn_drop(inh_attn)

        it = (inh_attn @ v_it).transpose(1, 2).reshape(B, self.t_size, C)

        # concat and proj
        x = torch.cat([t, it, s], dim=1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward_train_generate_variation_token(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # B, num_heads, N, C//num_heads
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        q_t, q_it, q_s = torch.split(q, [self.t_size, self.t_size, self.s_size], dim=2)
        k_t, k_it, k_s = torch.split(k, [self.t_size, self.t_size, self.s_size], dim=2)
        v_t, v_it, v_s = torch.split(v, [self.t_size, self.t_size, self.s_size], dim=2)

        self.k_vt = k_t
        self.v_vt = v_t
        self.k_it = k_it
        self.v_it = v_it

        # mix attention of t&s to t&it&s
        q_ts = torch.cat([q_t, q_s], dim=2)
        mix_attn = (q_ts @ k.transpose(-2, -1)) * self.scale
        mix_attn = mix_attn.softmax(dim=-1)
        mix_attn = self.attn_drop(mix_attn)

        ts = (mix_attn @ v).transpose(1, 2).reshape(B, self.mix_size, C)
        t, s = torch.split(ts, [self.t_size, self.s_size], dim=1)

        # inherent attention of it
        inh_attn = (q_it @ k_it.transpose(-2, -1)) * self.scale
        inh_attn = inh_attn.softmax(dim=-1)
        inh_attn = self.attn_drop(inh_attn)

        it = (inh_attn @ v_it).transpose(1, 2).reshape(B, self.t_size, C)

        # concat and proj
        x = torch.cat([t, it, s], dim=1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, (self.k_vt, self.v_vt, self.k_it, self.v_it)

    def forward_train_fuse_vt(self, x, vt_it):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # B, num_heads, N, C//num_heads
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        k_t, k_s = torch.split(k, [self.t_size, self.s_size], dim=2)
        v_t, v_s = torch.split(v, [self.t_size, self.s_size], dim=2)

        self.k_vt, self.v_vt, self.k_it, self.v_it = vt_it

        # mix attention of t&s to t&it&vt&s
        k = torch.cat([k_t, self.k_it, self.k_vt, k_s], dim=2)
        v = torch.cat([v_t, self.v_it, self.v_vt, v_s], dim=2)
        mix_attn = (q @ k.transpose(-2, -1)) * self.scale
        mix_attn = mix_attn.softmax(dim=-1)
        mix_attn = self.attn_drop(mix_attn)

        ts = (mix_attn @ v).transpose(1, 2).reshape(B, self.mix_size, C)

        x = self.proj(ts)
        x = self.proj_drop(x)
        return x

    def set_online(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # B, num_heads, N, C//num_heads
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
        self.k_vt = k
        self.v_vt = v
        self.k_it = k
        self.v_it = v

        # inherent attention of it
        inh_attn = (q @ k.transpose(-2, -1)) * self.scale
        inh_attn = inh_attn.softmax(dim=-1)
        inh_attn = self.attn_drop(inh_attn)

        it = (inh_attn @ v).transpose(1, 2).reshape(B, self.t_size, C)

        x = self.proj(it)
        x = self.proj_drop(x)
        return x
    
    def set_online_without_vt(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # B, num_heads, N, C//num_heads
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
        self.k_it = k
        self.v_it = v

        # inherent attention of it
        inh_attn = (q @ k.transpose(-2, -1)) * self.scale
        inh_attn = inh_attn.softmax(dim=-1)
        inh_attn = self.attn_drop(inh_attn)

        it = (inh_attn @ v).transpose(1, 2).reshape(B, self.t_size, C)

        x = self.proj(it)
        x = self.proj_drop(x)
        return x

    def forward_test(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # B, num_heads, N, C//num_heads
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        k_t, k_s = torch.split(k, [self.t_size, self.s_size], dim=2)
        v_t, v_s = torch.split(v, [self.t_size, self.s_size], dim=2)

        # mix attention of t&s to t&it&vt&s
        k = torch.cat([k_t, self.k_it, self.k_vt, k_s], dim=2)
        v = torch.cat([v_t, self.v_it, self.v_vt, v_s], dim=2)
        self.k_vt = k_t
        self.v_vt = v_t
        mix_attn = (q @ k.transpose(-2, -1)) * self.scale
        mix_attn = mix_attn.softmax(dim=-1)
        mix_attn = self.attn_drop(mix_attn)

        ts = (mix_attn @ v).transpose(1, 2).reshape(B, self.mix_size, C)

        x = self.proj(ts)
        x = self.proj_drop(x)
        return x

    def forward_test_without_vt(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # B, num_heads, N, C//num_heads
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        k_t, k_s = torch.split(k, [self.t_size, self.s_size], dim=2)
        v_t, v_s = torch.split(v, [self.t_size, self.s_size], dim=2)

        # mix attention of t&s to t&it&vt&s
        k = torch.cat([k_t, self.k_it, k_s], dim=2)
        v = torch.cat([v_t, self.v_it, v_s], dim=2)
        mix_attn = (q @ k.transpose(-2, -1)) * self.scale
        mix_attn = mix_attn.softmax(dim=-1)
        mix_attn = self.attn_drop(mix_attn)

        ts = (mix_attn @ v).transpose(1, 2).reshape(B, self.mix_size, C)

        x = self.proj(ts)
        x = self.proj_drop(x)
        return x

    def forward_profile(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # B, num_heads, N, C//num_heads
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        q_t, q_it, q_s = torch.split(q, [self.t_size, self.t_size, self.s_size], dim=2)
        k_t, k_it, k_s = torch.split(k, [self.t_size, self.t_size, self.s_size], dim=2)
        v_t, v_it, v_s = torch.split(v, [self.t_size, self.t_size, self.s_size], dim=2)

        # just use t as vt just for profile

        # mix attention of t&s to t&it&vt&s
        q = torch.cat([q_t, q_s], dim=2)
        k = torch.cat([k_t, k_it, k_t, k_s], dim=2)
        v = torch.cat([v_t, v_it, v_t, v_s], dim=2)
        mix_attn = (q @ k.transpose(-2, -1)) * self.scale
        mix_attn = mix_attn.softmax(dim=-1)
        mix_attn = self.attn_drop(mix_attn)

        ts = (mix_attn @ v).transpose(1, 2).reshape(B, self.mix_size, C)
        t, s = torch.split(ts, [self.t_size, self.s_size], dim=1)

        # inherent attention of it
        inh_attn = (q_it @ k_it.transpose(-2, -1)) * self.scale
        inh_attn = inh_attn.softmax(dim=-1)
        inh_attn = self.attn_drop(inh_attn)

        it = (inh_attn @ v_it).transpose(1, 2).reshape(B, self.t_size, C)

        # concat and proj
        x = torch.cat([t, it, s], dim=1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Module):

    def __init__(
            self, t_size, s_size, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(t_size, s_size, dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))

        return x

    def forward_train_generate_variation_token(self, x):
        tmp_x, vt_it = self.attn.forward_train_generate_variation_token(self.norm1(x))
        x = x + self.drop_path1(self.ls1(tmp_x))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))

        return x, vt_it

    def forward_train_fuse_vt(self, x, vt_it):
        x = x + self.drop_path1(self.ls1(self.attn.forward_train_fuse_vt(self.norm1(x), vt_it)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))

        return x

    def set_online(self, x):
        x = x + self.drop_path1(self.ls1(self.attn.set_online(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))

        return x

    def forward_test(self, x):
        x = x + self.drop_path1(self.ls1(self.attn.forward_test(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))

        return x

    def set_online_without_vt(self, x):
        x = x + self.drop_path1(self.ls1(self.attn.set_online_without_vt(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))

        return x

    def forward_test_without_vt(self, x):
        x = x + self.drop_path1(self.ls1(self.attn.forward_test_without_vt(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))

        return x

    def forward_profile(self, x):
        x = x + self.drop_path1(self.ls1(self.attn.forward_profile(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))

        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """

    def __init__(
            self, img_size_t=128, img_size_s=256, patch_size=16, in_chans=3, num_classes=1000, global_pool='token',
            embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, init_values=None,
            class_token=True, fc_norm=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., weight_init='',
            embed_layer=PatchEmbed, norm_layer=None, act_layer=None, block_fn=Block):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            global_pool (str): type of global pooling for final sequence (default: 'token')
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            init_values: (float): layer-scale init values
            class_token (bool): use class token
            fc_norm (Optional[bool]): pre-fc norm after pool, set if global_pool == 'avg' if None (default: None)
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            weight_init (str): weight init scheme
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            act_layer: (nn.Module): MLP activation layer
        """
        super().__init__()
        assert global_pool in ('', 'avg', 'token')
        assert class_token or global_pool != 'token'
        # use_fc_norm = global_pool == 'avg' if fc_norm is None else fc_norm
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1 if class_token else 0
        self.grad_checkpointing = False

        self.patch_embed_t = embed_layer(
            img_size=img_size_t, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, stride=patch_size)
        num_patches_t = self.patch_embed_t.num_patches

        self.patch_embed_s = EmptyPatchEmbed(
            img_size=img_size_s, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, stride=patch_size)
        num_patches_s = self.patch_embed_s.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if self.num_tokens > 0 else None
        self.pos_embed_t = nn.Parameter(torch.randn(1, num_patches_t + self.num_tokens, embed_dim) * .02)
        self.pos_embed_s = nn.Parameter(torch.randn(1, num_patches_s + self.num_tokens, embed_dim) * .02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        t_size = (img_size_t // patch_size) ** 2
        s_size = (img_size_s // patch_size) ** 2
        self.patch_size = patch_size
        self.depth = depth
        self.blocks = nn.Sequential(*[
            block_fn(
                t_size=t_size, s_size=s_size, dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, init_values=init_values,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        # self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()
        self.norm = norm_layer(embed_dim)

    def forward_features(self, template, inherent_template, search, curdepth):
        if curdepth == 0:
            template = self.patch_embed_t(template)
            template = self.pos_drop(template + self.pos_embed_t)
            inherent_template = self.patch_embed_t(inherent_template)
            inherent_template = self.pos_drop(inherent_template + self.pos_embed_t)
            search = self.patch_embed_t(search)
            search = self.pos_drop(search + self.pos_embed_s)

        x = torch.cat([template, inherent_template, search], dim=1)

        x = self.blocks[curdepth](x)

        if curdepth == (self.depth - 1):
            x = self.norm(x)

        return x

    def forward(self, template, inherent_template, search, curdepth):
        t_B, t_C, t_H, t_W = template.shape
        s_B, s_C, s_H, s_W = search.size()
        if curdepth == 0:
            t_H, t_W, s_H, s_W = t_H // self.patch_size, t_W // self.patch_size, s_H // self.patch_size, s_W // self.patch_size
        else:
            template = rearrange(template, 'b c h w -> b (h w) c').contiguous()
            inherent_template = rearrange(inherent_template, 'b c h w -> b (h w) c').contiguous()
            search = rearrange(search, 'b c h w -> b (h w) c').contiguous()

        x = self.forward_features(template, inherent_template, search, curdepth)
        template, inherent_template, search = torch.split(x, [t_H * t_W, t_H * t_W, s_H * s_W], dim=1)

        template = rearrange(template, 'b (h w) c -> b c h w', h=t_H, w=t_W).contiguous()
        inherent_template = rearrange(inherent_template, 'b (h w) c -> b c h w', h=t_H, w=t_W).contiguous()
        search = rearrange(search, 'b (h w) c -> b c h w', h=s_H, w=s_W).contiguous()

        return template, inherent_template, search

    def forward_train_generate_variation_token(self, template, inherent_template, search, curdepth):
        t_B, t_C, t_H, t_W = template.shape
        s_B, s_C, s_H, s_W = search.size()
        if curdepth == 0:
            t_H, t_W, s_H, s_W = t_H // self.patch_size, t_W // self.patch_size, s_H // self.patch_size, s_W // self.patch_size
        else:
            template = rearrange(template, 'b c h w -> b (h w) c').contiguous()
            inherent_template = rearrange(inherent_template, 'b c h w -> b (h w) c').contiguous()
            search = rearrange(search, 'b c h w -> b (h w) c').contiguous()

        if curdepth == 0:
            template = self.patch_embed_t(template)
            template = self.pos_drop(template + self.pos_embed_t)
            inherent_template = self.patch_embed_t(inherent_template)
            inherent_template = self.pos_drop(inherent_template + self.pos_embed_t)
            search = self.patch_embed_t(search)
            search = self.pos_drop(search + self.pos_embed_s)
        x = torch.cat([template, inherent_template, search], dim=1)
        x, vt_it = self.blocks[curdepth].forward_train_generate_variation_token(x)
        if curdepth == (self.depth - 1):
            x = self.norm(x)

        template, inherent_template, search = torch.split(x, [t_H * t_W, t_H * t_W, s_H * s_W], dim=1)
        template = rearrange(template, 'b (h w) c -> b c h w', h=t_H, w=t_W).contiguous()
        inherent_template = rearrange(inherent_template, 'b (h w) c -> b c h w', h=t_H, w=t_W).contiguous()
        search = rearrange(search, 'b (h w) c -> b c h w', h=s_H, w=s_W).contiguous()
        return template, inherent_template, search, vt_it

    def forward_train_fuse_vt(self, template, search, curdepth, vt_it):
        t_B, t_C, t_H, t_W = template.shape
        s_B, s_C, s_H, s_W = search.size()
        if curdepth == 0:
            t_H, t_W, s_H, s_W = t_H // self.patch_size, t_W // self.patch_size, s_H // self.patch_size, s_W // self.patch_size
        else:
            template = rearrange(template, 'b c h w -> b (h w) c').contiguous()
            search = rearrange(search, 'b c h w -> b (h w) c').contiguous()

        if curdepth == 0:
            template = self.patch_embed_t(template)
            template = self.pos_drop(template + self.pos_embed_t)
            search = self.patch_embed_t(search)
            search = self.pos_drop(search + self.pos_embed_s)
        x = torch.cat([template, search], dim=1)
        x = self.blocks[curdepth].forward_train_fuse_vt(x, vt_it)
        if curdepth == (self.depth - 1):
            x = self.norm(x)

        template, search = torch.split(x, [t_H * t_W, s_H * s_W], dim=1)
        template = rearrange(template, 'b (h w) c -> b c h w', h=t_H, w=t_W).contiguous()
        search = rearrange(search, 'b (h w) c -> b c h w', h=s_H, w=s_W).contiguous()
        return template, search
    
    def set_online(self, ini_it_vt, curdepth):
        t_B, t_C, t_H, t_W = ini_it_vt.shape
        if curdepth == 0:
            t_H, t_W = t_H // self.patch_size, t_W // self.patch_size
        else:
            ini_it_vt = rearrange(ini_it_vt, 'b c h w -> b (h w) c').contiguous()

        if curdepth == 0:
            ini_it_vt = self.patch_embed_t(ini_it_vt)
            ini_it_vt = self.pos_drop(ini_it_vt + self.pos_embed_t)
        x = self.blocks[curdepth].set_online(ini_it_vt)
        if curdepth == (self.depth - 1):
            x = self.norm(x)

        ini_it_vt = rearrange(x, 'b (h w) c -> b c h w', h=t_H, w=t_W).contiguous()
        return ini_it_vt

    def forward_test(self, template, search, curdepth):
        t_B, t_C, t_H, t_W = template.shape
        s_B, s_C, s_H, s_W = search.size()
        if curdepth == 0:
            t_H, t_W, s_H, s_W = t_H // self.patch_size, t_W // self.patch_size, s_H // self.patch_size, s_W // self.patch_size
        else:
            template = rearrange(template, 'b c h w -> b (h w) c').contiguous()
            search = rearrange(search, 'b c h w -> b (h w) c').contiguous()

        if curdepth == 0:
            template = self.patch_embed_t(template)
            template = self.pos_drop(template + self.pos_embed_t)
            search = self.patch_embed_t(search)
            search = self.pos_drop(search + self.pos_embed_s)
        x = torch.cat([template, search], dim=1)
        x = self.blocks[curdepth].forward_test(x)
        if curdepth == (self.depth - 1):
            x = self.norm(x)

        template, search = torch.split(x, [t_H * t_W, s_H * s_W], dim=1)
        template = rearrange(template, 'b (h w) c -> b c h w', h=t_H, w=t_W).contiguous()
        search = rearrange(search, 'b (h w) c -> b c h w', h=s_H, w=s_W).contiguous()
        return template, search
    
    def set_online_without_vt(self, ini_it, curdepth):
        t_B, t_C, t_H, t_W = ini_it.shape
        if curdepth == 0:
            t_H, t_W = t_H // self.patch_size, t_W // self.patch_size
        else:
            ini_it = rearrange(ini_it, 'b c h w -> b (h w) c').contiguous()

        if curdepth == 0:
            ini_it = self.patch_embed_t(ini_it)
            ini_it = self.pos_drop(ini_it + self.pos_embed_t)
        x = self.blocks[curdepth].set_online_without_vt(ini_it)
        if curdepth == (self.depth - 1):
            x = self.norm(x)

        ini_it = rearrange(x, 'b (h w) c -> b c h w', h=t_H, w=t_W).contiguous()
        return ini_it

    def forward_test_without_vt(self, template, search, curdepth):
        t_B, t_C, t_H, t_W = template.shape
        s_B, s_C, s_H, s_W = search.size()
        if curdepth == 0:
            t_H, t_W, s_H, s_W = t_H // self.patch_size, t_W // self.patch_size, s_H // self.patch_size, s_W // self.patch_size
        else:
            template = rearrange(template, 'b c h w -> b (h w) c').contiguous()
            search = rearrange(search, 'b c h w -> b (h w) c').contiguous()

        if curdepth == 0:
            template = self.patch_embed_t(template)
            template = self.pos_drop(template + self.pos_embed_t)
            search = self.patch_embed_t(search)
            search = self.pos_drop(search + self.pos_embed_s)
        x = torch.cat([template, search], dim=1)
        x = self.blocks[curdepth].forward_test_without_vt(x)
        if curdepth == (self.depth - 1):
            x = self.norm(x)

        template, search = torch.split(x, [t_H * t_W, s_H * s_W], dim=1)
        template = rearrange(template, 'b (h w) c -> b c h w', h=t_H, w=t_W).contiguous()
        search = rearrange(search, 'b (h w) c -> b c h w', h=s_H, w=s_W).contiguous()
        return template, search

    def forward_profile(self, template, inherent_template, search, curdepth):
        t_B, t_C, t_H, t_W = template.shape
        s_B, s_C, s_H, s_W = search.size()
        if curdepth == 0:
            t_H, t_W, s_H, s_W = t_H // self.patch_size, t_W // self.patch_size, s_H // self.patch_size, s_W // self.patch_size
        else:
            template = rearrange(template, 'b c h w -> b (h w) c').contiguous()
            inherent_template = rearrange(inherent_template, 'b c h w -> b (h w) c').contiguous()
            search = rearrange(search, 'b c h w -> b (h w) c').contiguous()

        if curdepth == 0:
            template = self.patch_embed_t(template)
            template = self.pos_drop(template + self.pos_embed_t)
            inherent_template = self.patch_embed_t(inherent_template)
            inherent_template = self.pos_drop(inherent_template + self.pos_embed_t)
            search = self.patch_embed_t(search)
            search = self.pos_drop(search + self.pos_embed_s)
        x = torch.cat([template, inherent_template, search], dim=1)
        x = self.blocks[curdepth].forward_profile(x)
        if curdepth == (self.depth - 1):
            x = self.norm(x)

        template, inherent_template, search = torch.split(x, [t_H * t_W, t_H * t_W, s_H * s_W], dim=1)
        template = rearrange(template, 'b (h w) c -> b c h w', h=t_H, w=t_W).contiguous()
        inherent_template = rearrange(inherent_template, 'b (h w) c -> b c h w', h=t_H, w=t_W).contiguous()
        search = rearrange(search, 'b (h w) c -> b c h w', h=s_H, w=s_W).contiguous()
        return template, inherent_template, search


# Load models
def load_checkpoint_tiny_22k_npz(cur_encoder, ckpt_path):
    try:
        ckpt = np.load(ckpt_path)
        filelist = ckpt.files
        tmpckpt = {}
        for item in filelist:
            tmpckpt[item] = torch.from_numpy(ckpt[item])
        ckpt = tmpckpt
        transfer_ckpt = {}
        # create pos_embed and copy from Transformer/posembed_input/pos_embedding
        transfer_ckpt['pos_embed'] = ckpt['Transformer/posembed_input/pos_embedding']
        # interpolate position embedding
        interpolate_pos_embed(cur_encoder, transfer_ckpt)
        # create patch_embed and copy from embedding
        transfer_ckpt['patch_embed_t.proj.weight'] = ckpt['embedding/kernel'].permute(3, 2, 0, 1)
        transfer_ckpt['patch_embed_t.proj.bias'] = ckpt['embedding/bias']
        # create norm and copy from encoder_norm
        transfer_ckpt['norm.weight'] = ckpt['Transformer/encoder_norm/scale']
        transfer_ckpt['norm.bias'] = ckpt['Transformer/encoder_norm/bias']
        # deal with name in layers
        for item in filelist:
            if 'Transformer/encoderblock_' not in item:
                continue
            curkey = item.replace('Transformer/encoderblock_', 'blocks/')
            ck = curkey.split('/')
            if ck[3] == 'out':
                k1 = 'blocks.' + ck[1] + '.attn.proj.'
                if ck[4] == 'bias':
                    transfer_ckpt[k1 + 'bias'] = ckpt[item]
                else:
                    transfer_ckpt[k1 + 'weight'] = ckpt[item].reshape(192, 192).t()
            elif ck[3] == 'query':
                wbq = 'Transformer/encoderblock_' + ck[1] + '/' + ck[2] + '/query/' + ck[4]
                wbk = 'Transformer/encoderblock_' + ck[1] + '/' + ck[2] + '/key/' + ck[4]
                wbv = 'Transformer/encoderblock_' + ck[1] + '/' + ck[2] + '/value/' + ck[4]
                kwb = 'blocks.' + ck[1] + '.attn.qkv.'
                if ck[4] == 'bias':
                    transfer_ckpt[kwb + 'bias'] = torch.cat([ckpt[wbq].reshape(192),
                                                             ckpt[wbk].reshape(192),
                                                             ckpt[wbv].reshape(192)], dim=0)
                else:
                    transfer_ckpt[kwb + 'weight'] = torch.cat([ckpt[wbq].reshape(192, 192).t(),
                                                               ckpt[wbk].reshape(192, 192).t(),
                                                               ckpt[wbv].reshape(192, 192).t()], dim=0)
            elif ck[3] != 'key' and ck[3] != 'value':
                if ck[2] == 'LayerNorm_0':
                    if ck[3] == 'bias':
                        transfer_ckpt['blocks.' + ck[1] + '.norm1.bias'] = ckpt[item]
                    else:
                        transfer_ckpt['blocks.' + ck[1] + '.norm1.weight'] = ckpt[item]
                elif ck[2] == 'LayerNorm_2':
                    if ck[3] == 'bias':
                        transfer_ckpt['blocks.' + ck[1] + '.norm2.bias'] = ckpt[item]
                    else:
                        transfer_ckpt['blocks.' + ck[1] + '.norm2.weight'] = ckpt[item]
                elif ck[3] == 'Dense_0':
                    if ck[4] == 'bias':
                        transfer_ckpt['blocks.' + ck[1] + '.mlp.fc1.bias'] = ckpt[item]
                    else:
                        transfer_ckpt['blocks.' + ck[1] + '.mlp.fc1.weight'] = ckpt[item].t()
                else:
                    if ck[4] == 'bias':
                        transfer_ckpt['blocks.' + ck[1] + '.mlp.fc2.bias'] = ckpt[item]
                    else:
                        transfer_ckpt['blocks.' + ck[1] + '.mlp.fc2.weight'] = ckpt[item].t()
        # load
        model_ckpt = cur_encoder.state_dict()
        state_ckpt = {k: v for k, v in transfer_ckpt.items() if k in model_ckpt.keys()}
        model_ckpt.update(state_ckpt)
        missing_keys, unexpected_keys = cur_encoder.load_state_dict(model_ckpt, strict=False)
        # print to check
        for k, v in cur_encoder.named_parameters():
            if k in transfer_ckpt.keys():
                if is_main_process():
                    print(k)
            else:
                if is_main_process():
                    print("# not in transfer_ckpt: " + k)
        if is_main_process():
            print("missing keys:", missing_keys)
            print("unexpected keys:", unexpected_keys)
            print("Loading pretrained 22k done.")
    except Exception as e:
        print("Warning: Pretrained 22k weights are not loaded")
        print(e.args)


def load_checkpoint_small_22k_npz(cur_encoder, ckpt_path):
    try:
        ckpt = np.load(ckpt_path)
        filelist = ckpt.files
        tmpckpt = {}
        for item in filelist:
            tmpckpt[item] = torch.from_numpy(ckpt[item])
        ckpt = tmpckpt
        transfer_ckpt = {}
        # create pos_embed and copy from Transformer/posembed_input/pos_embedding
        transfer_ckpt['pos_embed'] = ckpt['Transformer/posembed_input/pos_embedding']
        # interpolate position embedding
        interpolate_pos_embed(cur_encoder, transfer_ckpt)
        # create patch_embed and copy from embedding
        transfer_ckpt['patch_embed_t.proj.weight'] = ckpt['embedding/kernel'].permute(3, 2, 0, 1)
        transfer_ckpt['patch_embed_t.proj.bias'] = ckpt['embedding/bias']
        # create norm and copy from encoder_norm
        transfer_ckpt['norm.weight'] = ckpt['Transformer/encoder_norm/scale']
        transfer_ckpt['norm.bias'] = ckpt['Transformer/encoder_norm/bias']
        # deal with name in layers
        for item in filelist:
            if 'Transformer/encoderblock_' not in item:
                continue
            curkey = item.replace('Transformer/encoderblock_', 'blocks/')
            ck = curkey.split('/')
            if ck[3] == 'out':
                k1 = 'blocks.' + ck[1] + '.attn.proj.'
                if ck[4] == 'bias':
                    transfer_ckpt[k1 + 'bias'] = ckpt[item]
                else:
                    transfer_ckpt[k1 + 'weight'] = ckpt[item].reshape(384, 384).t()
            elif ck[3] == 'query':
                wbq = 'Transformer/encoderblock_' + ck[1] + '/' + ck[2] + '/query/' + ck[4]
                wbk = 'Transformer/encoderblock_' + ck[1] + '/' + ck[2] + '/key/' + ck[4]
                wbv = 'Transformer/encoderblock_' + ck[1] + '/' + ck[2] + '/value/' + ck[4]
                kwb = 'blocks.' + ck[1] + '.attn.qkv.'
                if ck[4] == 'bias':
                    transfer_ckpt[kwb + 'bias'] = torch.cat([ckpt[wbq].reshape(384),
                                                             ckpt[wbk].reshape(384),
                                                             ckpt[wbv].reshape(384)], dim=0)
                else:
                    transfer_ckpt[kwb + 'weight'] = torch.cat([ckpt[wbq].reshape(384, 384).t(),
                                                               ckpt[wbk].reshape(384, 384).t(),
                                                               ckpt[wbv].reshape(384, 384).t()], dim=0)
            elif ck[3] != 'key' and ck[3] != 'value':
                if ck[2] == 'LayerNorm_0':
                    if ck[3] == 'bias':
                        transfer_ckpt['blocks.' + ck[1] + '.norm1.bias'] = ckpt[item]
                    else:
                        transfer_ckpt['blocks.' + ck[1] + '.norm1.weight'] = ckpt[item]
                elif ck[2] == 'LayerNorm_2':
                    if ck[3] == 'bias':
                        transfer_ckpt['blocks.' + ck[1] + '.norm2.bias'] = ckpt[item]
                    else:
                        transfer_ckpt['blocks.' + ck[1] + '.norm2.weight'] = ckpt[item]
                elif ck[3] == 'Dense_0':
                    if ck[4] == 'bias':
                        transfer_ckpt['blocks.' + ck[1] + '.mlp.fc1.bias'] = ckpt[item]
                    else:
                        transfer_ckpt['blocks.' + ck[1] + '.mlp.fc1.weight'] = ckpt[item].t()
                else:
                    if ck[4] == 'bias':
                        transfer_ckpt['blocks.' + ck[1] + '.mlp.fc2.bias'] = ckpt[item]
                    else:
                        transfer_ckpt['blocks.' + ck[1] + '.mlp.fc2.weight'] = ckpt[item].t()
        # load
        model_ckpt = cur_encoder.state_dict()
        state_ckpt = {k: v for k, v in transfer_ckpt.items() if k in model_ckpt.keys()}
        model_ckpt.update(state_ckpt)
        missing_keys, unexpected_keys = cur_encoder.load_state_dict(model_ckpt, strict=False)
        # print to check
        for k, v in cur_encoder.named_parameters():
            if k in transfer_ckpt.keys():
                if is_main_process():
                    print(k)
            else:
                if is_main_process():
                    print("# not in transfer_ckpt: " + k)
        if is_main_process():
            print("missing keys:", missing_keys)
            print("unexpected keys:", unexpected_keys)
            print("Loading pretrained 22k small done.")
    except Exception as e:
        print("Warning: Pretrained 22k small weights are not loaded")
        print(e.args)


def load_checkpoint(cur_encoder, ckpt_path):
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        ckpt_type = "MAE"
        ckpt = ckpt['model']
        # interpolate position embedding
        interpolate_pos_embed(cur_encoder, ckpt)
        # copy patch_embed
        ckpt['patch_embed_t.proj.weight'] = ckpt['patch_embed.proj.weight']
        ckpt['patch_embed_t.proj.bias'] = ckpt['patch_embed.proj.bias']
        # load
        model_ckpt = cur_encoder.state_dict()
        state_ckpt = {k: v for k, v in ckpt.items() if k in model_ckpt.keys()}
        model_ckpt.update(state_ckpt)
        missing_keys, unexpected_keys = cur_encoder.load_state_dict(model_ckpt, strict=False)
        # print to check
        for k, v in cur_encoder.named_parameters():
            if k in ckpt.keys():
                if is_main_process():
                    print(k)
            else:
                if is_main_process():
                    print("# not in ckpt: " + k)
        if is_main_process():
            print("missing keys:", missing_keys)
            print("unexpected keys:", unexpected_keys)
            print("Loading pretrained {} done.".format(ckpt_type))
    except Exception as e:
        print("Warning: Pretrained weights are not loaded.")
        print(e.args)


def build_transformer_tiny(t_size, s_size):
    model = VisionTransformer(
        img_size_t=t_size, img_size_s=s_size, patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        class_token=False, drop_path_rate=0.1, norm_layer=partial(nn.LayerNorm, eps=1e-6), global_pool='')
    load_checkpoint_tiny_22k_npz(model, ckpt_path = 'pretrained/21k_1k/vit_tiny_22k_384.npz')

    return model


def build_transformer_small(t_size, s_size):
    model = VisionTransformer(
        img_size_t=t_size, img_size_s=s_size, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        class_token=False, drop_path_rate=0.1, norm_layer=partial(nn.LayerNorm, eps=1e-6), global_pool='')
    load_checkpoint_small_22k_npz(model, ckpt_path = 'pretrained/21k_1k/vit_small_22k_384.npz')

    return model


def build_transformer_base(t_size, s_size):
    model = VisionTransformer(
        img_size_t=t_size, img_size_s=s_size, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        class_token=False, drop_path_rate=0.1, norm_layer=partial(nn.LayerNorm, eps=1e-6), global_pool='')
    load_checkpoint(model, ckpt_path='pretrained/mae/mae_pretrain_vit_base.pth')

    return model


def build_transformer_large(t_size, s_size):
    model = VisionTransformer(
        img_size_t=t_size, img_size_s=s_size, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        class_token=False, drop_path_rate=0.1, norm_layer=partial(nn.LayerNorm, eps=1e-6), global_pool='')
    load_checkpoint(model, ckpt_path='pretrained/mae/mae_pretrain_vit_large.pth')

    return model

def build_transformer_huge(t_size, s_size):
    model = VisionTransformer(
        img_size_t=t_size, img_size_s=s_size, patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        class_token=False, drop_path_rate=0.1, norm_layer=partial(nn.LayerNorm, eps=1e-6), global_pool='')
    load_checkpoint(model, ckpt_path='pretrained/mae/mae_pretrain_vit_huge.pth')

    return model