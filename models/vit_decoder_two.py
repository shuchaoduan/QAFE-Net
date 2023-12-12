from functools import partial

import torch
import torch.nn as nn

from timm.models.layers import DropPath, trunc_normal_
import numpy as np


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x




class CrossAttention(nn.Module):
    def __init__(self, dim, out_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.out_dim = out_dim
        head_dim = out_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.k_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.v_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(out_dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, v):
        B, N, _ = q.shape
        C = self.out_dim
        k = v
        NK = k.size(1)

        q = self.q_map(q).view(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_map(k).view(B, NK, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_map(v).view(B, NK, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class DecoderBlock(nn.Module):
    def __init__(self, dim, num_heads, dim_q=None, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        dim_q = dim_q or dim
        self.img_dim = 5 
        self.norm_q = norm_layer(dim_q)
        self.norm_v = norm_layer(dim)
        self.attn = CrossAttention(
            dim, dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x_1 = x[:, :self.img_dim, :]
        x_2 = x[:, self.img_dim:, :]

        x_frame = x_1 + self.drop_path(self.attn(self.norm_q(x_1), self.norm_v(x_2)))
        x_frame = x_frame + self.drop_path(self.mlp(self.norm2(x_frame)))

        x_heatmap = x_2 + self.drop_path(self.attn(self.norm_q(x_2), self.norm_v(x_1)))
        x_heatmap = x_heatmap + self.drop_path(self.mlp(self.norm2(x_heatmap)))
        x = torch.cat((x_frame, x_heatmap), dim=1)

        return x


class decoder_fuser(nn.Module):
    def __init__(self, dim, num_heads, num_layers, drop_rate):
        super(decoder_fuser, self).__init__()
        model_list = []
        for i in range(num_layers):
            model_list.append(DecoderBlock(dim, num_heads))
        self.model = nn.ModuleList(model_list)
        self.pos_embed = nn.Parameter(torch.randn(1, 5, dim)) 
        self.pos_drop = nn.Dropout(drop_rate)
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm = norm_layer(dim)
        self.conv = nn.Conv1d(5*2, 5*2, 1)

    def forward(self, x_frame, x_heatmap):
        x_frame = self.pos_drop(x_frame+self.pos_embed)
        x_comb = torch.cat((x_frame, x_heatmap), dim=1)

        for _layer in self.model:
            x_comb = _layer(x_comb)
        x_comb = self.conv(x_comb) 

        return x_comb

if __name__=='__main__':
    x = torch.randn((2,5, 512))
    model = decoder_fuser(dim=512, num_heads=8, num_layers=3, drop_rate=0.)
    y = model(x,x)
    print(y)
