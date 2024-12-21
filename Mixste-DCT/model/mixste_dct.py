import math
import torch
import torch.nn as nn
from functools import partial
from timm.models.layers import DropPath
from einops import rearrange, repeat
import numpy as np
from model.dct_attention import Attention, DCTAttention
import torch.nn.functional as F

def xboost(x):
    # x.shape: b f j c
    b, f, j, c = x.shape
    y = torch.zeros_like(x, device='cuda' if torch.cuda.is_available() else 'cpu')
    y[:, :, 0] = x[:, :, 0]
    y[:, :, 1:4] = (x[:, :, 1:4]+x[:, :, 0:3])/2
    y[:, :, 4:7] = (x[:, :, 4:7]+x[:, :, (0, 4, 5)])/2
    y[:, :, 7:11] = (x[:, :, 7:11]+x[:, :, (0, 7, 8, 9)])/2
    y[:, :, 11:14] = (x[:, :, 11:14]+x[:, :, (8, 11, 12)])/2
    y[:, :, 14:17] = (x[:, :, 14:17]+x[:, :, (8, 14, 15)])/2
    return y


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


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_hidden_dim, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, dct_percentage=0.5):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = DCTAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, \
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, dct_percentage=dct_percentage)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
    def forward(self, x):
        if isinstance(self.attn, DCTAttention):
            x, shortx = self.attn(self.norm1(x))
            x = shortx + self.drop_path(x)
        elif isinstance(self.attn, Attention):
            x = x + self.drop_path(self.attn(self.norm1(x)))
        else:
            raise NotImplementedError
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()

        depth = 8
        embed_dim = args.channel
        mlp_hidden_dim = args.channel * 2

        self.recover_num = args.frames
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dct_percentage = args.dct_ratio

        drop_path_rate = 0.1
        drop_rate = 0.
        attn_drop_rate = 0.
        qkv_bias = True
        qk_scale = None
        num_heads = 8
        num_joints = args.n_joints

        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.Spatial_patch_to_embedding = nn.Linear(4, embed_dim)
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim))
        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, args.frames, embed_dim))

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.block_depth = depth
        self.index = [1, 3, 5, 7]
        self.STEblocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_hidden_dim=mlp_hidden_dim, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                dct_percentage=self.dct_percentage if i in self.index else 1)
            for i in range(depth)])

        self.TTEblocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_hidden_dim=mlp_hidden_dim, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                dct_percentage=self.dct_percentage if i in self.index else 1)
            for i in range(depth)])


        # self.freq_pos_embed = nn.ParameterList([])
        # framenum = args.frames
        # for i in range(len(self.index)):
        #     if i==0:
        #         self.freq_pos_embed.append(nn.Parameter(torch.zeros(1, framenum, embed_dim)))
        #     else:
        #         framenum = math.ceil(framenum * self.dct_percentage)
        #         self.freq_pos_embed.append(nn.Parameter(torch.zeros(1, framenum, embed_dim)))


        # self.freq_pos_embed = nn.ParameterList([nn.Parameter(
        #     torch.zeros(1, math.ceil(self.recover_num*math.pow(self.dct_percentage, i)), embed_dim)
        # ) for i in range(len(self.index))])


        self.Spatial_norm = norm_layer(embed_dim)
        self.Temporal_norm = norm_layer(embed_dim)

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim , 3),
        )

    def forward(self, x):
        b, f, n, c = x.shape
        x_ = xboost(x)
        x = torch.cat((x, x_), dim=-1)
        x = rearrange(x, 'b f n c  -> (b f) n c')
        x = self.Spatial_patch_to_embedding(x)
        x += self.Spatial_pos_embed
        x = self.pos_drop(x)
        x = self.STEblocks[0](x)
        x = self.Spatial_norm(x)

        x = rearrange(x, '(b f) n c -> (b n) f c', f=f)
        x += self.Temporal_pos_embed
        x = self.pos_drop(x)
        x = self.TTEblocks[0](x)
        x = self.Temporal_norm(x)
        x = rearrange(x, '(b n) f c -> b f n c', n=n)

        global_residual = list()
        global_residual.append(x)
        for i in range(1, self.block_depth):

            x = rearrange(x, 'b f n c -> (b f) n c')
            steblock = self.STEblocks[i]
            tteblock = self.TTEblocks[i]
            
            x = steblock(x)
            x = self.Spatial_norm(x)
            x = rearrange(x, '(b f) n c -> (b n) f c', b=b)
            # if i in self.index:
            #     x += self.freq_pos_embed[c]
            #     c += 1
            x = tteblock(x)
            x = self.Temporal_norm(x)
            x = rearrange(x, '(b n) f c -> b f n c', n=n)
            global_residual.append(x)


        x = global_residual[0]
        b, f, j, c = x.shape
        for global_x in global_residual[1:]:
            global_x = rearrange(global_x, 'b f n c -> (b n) c f')
            global_x = F.interpolate(global_x, size=(f), mode='nearest')
            global_x = rearrange(global_x, '(b n) c f -> b f n c', b=b)
            x = x + global_x

        x = self.head(x)

        x = x.view(b, -1, n, 3)

        return x

if __name__ == '__main__':
    import argparse

    args = argparse.ArgumentParser().parse_args()
    args.layers, args.channel, args.d_hid, args.frames = 8, 512, 1024, 243
    args.n_joints, args.out_joints = 17, 17
    args.dct_ratio = 0.7

    input_2d = torch.rand(1, args.frames, 17, 2)

    with torch.no_grad():
        model = Model(args)
        model.eval()

        model_params = 0
        for parameter in model.parameters():
            model_params += parameter.numel()
        print('INFO: Trainable parameter count:', model_params/ 1000000)

        print(input_2d.shape, 1)
        output = model(input_2d)
        print(output.shape, 2)

    from thop import profile
    from thop import clever_format
    macs, params = profile(model, inputs=(input_2d, ))
    print('macs: ', macs/1000000, 'params: ', params/1000000)
    macs, params = clever_format([macs*2, params], "%.3f")
    print('flops: ', macs, 'params: ', params)








