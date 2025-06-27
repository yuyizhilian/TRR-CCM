import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math

class RoiTransformer_EA(nn.Module):
    def __init__(self, num_token, channel, emb_dim, num_heads, mlp_dim, depth=2):
        super(RoiTransformer_EA, self).__init__()
        self.model = nn.ModuleList()
        for i in range(depth):
            self.model.append(ExternalAttention(num_token, channel))
        return

    def forward(self, bbox_feats):
        for ea in self.model:
            bbox_feats = ea(bbox_feats)

        return bbox_feats

class RoiTransformer_EA_MLPmixer(nn.Module):
    def __init__(self, num_token, channel, emb_dim, num_heads, mlp_dim, DS = 256, depth=2):
        super(RoiTransformer_EA_MLPmixer, self).__init__()
        self.EAs = nn.ModuleList()
        self.MLPmixer = nn.ModuleList()
        for i in range(depth):
            self.EAs.append(ExternalAttention(num_token, channel))
        for i in range(depth):
            self.MLPmixer.append(MixerBlock(7*7, 256, emb_dim, mlp_dim))
            # emb_dim == DS, mlp_dim == DC
        return

    def forward(self, bbox_feats, shape):
        b, num_sample, c, h, w = shape
        for ea,mlp in zip(self.EAs, self.MLPmixer):
            bbox_feats = ea(bbox_feats)
            bbox_feats = bbox_feats.reshape(b*num_sample, c, h * w)
            bbox_feats = bbox_feats.permute(0, 2, 1)
            bbox_feats = mlp(bbox_feats)
            bbox_feats = bbox_feats.permute(0, 2, 1)
            bbox_feats = bbox_feats.reshape(b, num_sample, -1)
        return bbox_feats




class ExternalAttention(nn.Module):
    def __init__(self, num_token, channel):
        super(ExternalAttention, self).__init__()
        self.M_k = nn.Linear(channel, num_token, bias=False)
        self.M_v = nn.Linear(num_token, channel, bias=False)
        return

    def forward(self, bbox_feats):
        b, l, c = bbox_feats.shape
        attn = self.M_k(bbox_feats)
        attn = F.softmax(attn, dim=1)
        attn = attn / torch.sum(attn, dim=2, keepdim=True)
        out = self.M_v(attn)

        return out + bbox_feats


class MixerBlock(nn.Module):
    def __init__(self,num_token, emb_dim, DS, mlp_dim):
        super(MixerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.token_mixing = MLPBlock(num_token, DS)
        self.channel_mixing = MLPBlock(emb_dim, mlp_dim)

    def forward(self, x):
        b, l, c = x.shape
        y = self.norm1(x)
        y = y.permute(0, 2, 1)
        y = self.token_mixing(y)
        y = y.permute(0, 2, 1)
        x = x + y
        y = self.norm2(x)
        x = x + self.channel_mixing(y)
        return x

class MLPBlock(nn.Module):
    def __init__(self,inchannel, hidden_channel):
        super(MLPBlock, self).__init__()
        self.fc1 = nn.Linear(inchannel, hidden_channel)
        self.fc2 = nn.Linear(hidden_channel, inchannel)

    def forward(self,x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

class MLPMixer(nn.Module):
    def __init__(self, num_token, emb_dim, DS, mlp_dim, depth = 1):
        super(MLPMixer, self).__init__()
        self.model = nn.ModuleList()
        for i in range(depth):
            self.model.append(MixerBlock(num_token, emb_dim, DS, mlp_dim))


    def forward(self,x):
        for mlp in self.model:
            x = mlp(x)
        return x


class RoiTransformer(nn.Module):
    def __init__(self, num_token, channel, emb_dim, num_heads, mlp_dim, depth = 2, mode = "transformer"):
        super(RoiTransformer,self).__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.stage_num = 4
        # Tokenization
        self.Trokenize = nn.Linear(channel, num_token)
        self.proj = nn.Linear(channel,emb_dim)

        self.Q_T2X = nn.Linear(channel,channel)
        self.K_T2X = nn.Linear(emb_dim,channel)
        self.deproj = nn.Linear(emb_dim, channel)

        # self.deproj = nn.Linear(emb_dim, channel, bias=False)
        # self.bn = nn.BatchNorm1d(channel)

        # self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = nn.Parameter(torch.empty(1, (num_token), emb_dim))
        torch.nn.init.normal_(self.pos_embedding, std=.02)  # initialized based on the paper
        self.dropout = nn.Dropout(0.1)

        if mode == 'transformer':
            self.transformer = Transformer(emb_dim,depth=depth,heads=num_heads, mlp_dim=mlp_dim,dropout=0.1)
        else:
            self.transformer = MLPMixer(num_token=num_token, emb_dim=emb_dim, DS=256, mlp_dim=mlp_dim, depth = depth)

        mu = torch.Tensor(1, emb_dim, num_token)
        mu.normal_(0, math.sqrt(2. / num_token))  # Init with Kaiming Norm.
        mu = self._l2norm(mu, dim=1)
        self.register_buffer('mu', mu)
        return

    def forward(self, bbox_feats):
        b, l, c = bbox_feats.shape
        shortcut = bbox_feats

        A = self.Trokenize(bbox_feats)  # (b,l,num_token)
        V = self.proj(bbox_feats) # (b,l,emb_dim)
        A = A.permute(0,2,1)
        A = F.softmax(A, dim=2)
        T = torch.bmm(A,V)

        # cls_tokens = self.cls_token.expand(b, -1, -1)
        # T = torch.cat((cls_tokens, T), dim=1)
        # T += self.pos_embedding
        T = self.dropout(T)
        T = self.transformer(T) # (b,num_token,emb_dim)

        X_rev = self.Q_T2X(bbox_feats)
        T_rev = self.K_T2X(T)
        attn = torch.bmm(X_rev, T_rev.permute(0,2,1))
        attn = F.softmax(attn, dim=-1)
        V_rev = self.deproj(T)
        res = torch.bmm(attn,V_rev)

        bbox_feats = shortcut + res
        return bbox_feats

    # cluster
    # def forward(self, bbox_feats, train=False):
    #     b, l, c = bbox_feats.shape
    #     shortcut = bbox_feats
    #
    #     x = self.proj(bbox_feats) # (b,l,emb_dim)
    #     mu = self.mu.repeat(b, 1, 1)  # b * emb_dim * num_token
    #     with torch.no_grad():
    #         for i in range(self.stage_num - 1):
    #             x_t = x
    #             z = torch.bmm(x_t, mu)  # b * l * num_token
    #             z = F.softmax(z, dim=2)  # b * l * num_token
    #             z_ = z / (1e-6 + z.sum(dim=1, keepdim=True))
    #             mu = torch.bmm(x.permute(0,2,1), z_)  # b * emb_dim * num_token
    #             mu = self._l2norm(mu, dim=1)
    #     x_t = x
    #     z = torch.bmm(x_t, mu)  # b * l * num_token
    #     z = F.softmax(z, dim=2)  # b * l * num_token
    #     z_ = z / (1e-6 + z.sum(dim=1, keepdim=True))
    #     mu = torch.bmm(x.permute(0,2,1), z_)  # b * emb_dim * num_token
    #     mu = self._l2norm(mu, dim=1)
    #
    #     T = mu.permute(0,2,1)
    #     T = self.dropout(T)
    #     T = self.transformer(T, None) # (b,l,emb_dim)
    #
    #     # z_t = z.permute(0, 2, 1)  # b * num_token * l
    #     # x = T.matmul(z_t)  # b * emb_dim * l
    #     x = z.matmul(T) # b * l * emb_dim
    #     x = F.relu(x, inplace=True)
    #
    #     x = self.deproj(x)
    #     x = x.view(b*l,c)
    #     x = self.bn(x)
    #     x = x.view(b, l,c)
    #
    #     bbox_feats = shortcut + x
    #     if train==True:
    #         return bbox_feats, mu.detach()
    #     else:
    #         return bbox_feats

    # def forward(self, bbox_feats, train=False):
    #     b, l, c = bbox_feats.shape
    #     shortcut = bbox_feats
    #
    #     x = self.proj(bbox_feats)  # (b,l,emb_dim)
    #     with torch.no_grad():
    #         mu = self.kmeans(x) # (b, emb_dim, num_token)
    #     A = torch.bmm(x,mu)
    #     A = A.permute(0,2,1)
    #     A = F.softmax(A, dim=2) # (b, num_token, l)
    #     T = torch.bmm(A,x)
    #
    #     T = self.dropout(T)
    #     T = self.transformer(T, None)  # (b,l,emb_dim)
    #
    #     X_rev = self.Q_T2X(bbox_feats)
    #     T_rev = self.K_T2X(T)
    #     attn = torch.bmm(X_rev, T_rev.permute(0,2,1))
    #     attn = F.softmax(attn, dim=-1)
    #     V_rev = self.deproj(T)
    #     res = torch.bmm(attn,V_rev)
    #
    #     bbox_feats = shortcut + res
    #     if train == True:
    #         return bbox_feats, mu.detach()
    #     else:
    #         return bbox_feats

    def kmeans(self, bbox_feats):
        b, l, c = bbox_feats.shape
        bbox_feats = bbox_feats.permute(0,2,1)
        bbox_feats = self._l2norm(bbox_feats, dim=1)
        mu = self.mu.repeat(b, 1, 1)
        for i in range(self.stage_num):
            dist_npl = (bbox_feats[...,None] - self.mu[:,:,None,:]).norm(dim=1)
            term,idx = dist_npl.min(dim=2)
            term = term.unsqueeze(2)
            mask_npl =  (dist_npl == term).float()
            mu = torch.bmm(bbox_feats,mask_npl)
            mu = mu / (mask_npl.sum(dim=1,keepdim=True) + 1e-6 )
            mu = self._l2norm(mu,dim=1)
        return mu

    def _l2norm(self, inp, dim):
        return inp / (1e-6 + inp.norm(dim=dim, keepdim=True))

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class LayerNormalize(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class MLP_Block(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.nn1 = nn.Linear(dim, hidden_dim)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.normal_(self.nn1.bias, std=1e-6)
        self.af1 = nn.GELU()
        self.do1 = nn.Dropout(dropout)
        self.nn2 = nn.Linear(hidden_dim, dim)
        torch.nn.init.xavier_uniform_(self.nn2.weight)
        torch.nn.init.normal_(self.nn2.bias, std=1e-6)
        self.do2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.nn1(x)
        x = self.af1(x)
        x = self.do1(x)
        x = self.nn2(x)
        x = self.do2(x)

        return x


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5  # 1/sqrt(dim)

        self.to_qkv = nn.Linear(dim, dim * 3, bias=True)  # Wq,Wk,Wv for each vector, thats why *3
        torch.nn.init.xavier_uniform_(self.to_qkv.weight)
        torch.nn.init.zeros_(self.to_qkv.bias)

        self.nn1 = nn.Linear(dim, dim)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.zeros_(self.nn1.bias)
        self.do1 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x)  # gets q = Q = Wq matmul x1, k = Wk mm x2, v = Wv mm x3
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=h)  # split into multi head attentions

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)  # follow the softmax,q,d,v equation in the paper

        out = torch.einsum('bhij,bhjd->bhid', attn, v)  # product of v times whatever inside softmax
        out = rearrange(out, 'b h n d -> b n (h d)')  # concat heads into one matrix, ready for next encoder block
        out = self.nn1(out)
        out = self.do1(out)
        return out


# class Transformer(nn.Module):
#     def __init__(self, dim, depth, heads, mlp_dim, dropout):
#         super().__init__()
#         self.layers = nn.ModuleList([])
#         for _ in range(depth):
#             self.layers.append(nn.ModuleList([
#                 Residual(LayerNormalize(dim, Attention(dim, heads=heads, dropout=dropout))),
#                 Residual(LayerNormalize(dim, MLP_Block(dim, mlp_dim, dropout=dropout)))
#             ]))
#
#     def forward(self, x, mask=None):
#         for attention, mlp in self.layers:
#             x = attention(x, mask=mask)  # go to attention
#             x = mlp(x)  # go to MLP_Block
#         return x

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(Attention(dim, heads=heads, dropout=dropout)),
                Residual(MLP_Block(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x, mask=None):
        for attention, mlp in self.layers:
            x = self.norm1(x)
            x = attention(x, mask=mask)  # go to attention
            x = self.norm2(x)
            x = mlp(x)  # go to MLP_Block
        return x

class BfMamba(nn.Module):
    def __init__(self, channel, depth=2):
        super(BfMamba, self).__init__()
        self.model=nn.ModuleList()
        for i in range(depth):
            self.model.append(Block(channel))
        return

    def forward(self, bbox_feats):
        for ea in self.model:
            bbox_feats = ea(bbox_feats)

        return bbox_feats
        

from mamba_ssm import Mamba

class Block(nn.Module):
    def __init__(self, channel):
        super(Block, self).__init__()
        self.dim = channel
        self.norm = nn.LayerNorm(channel)
        self.mamba = Mamba(
            d_model=channel, # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,    # Local convolution width
            expand=2,    # Block expansion factor
        )
        return

    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)

        return out