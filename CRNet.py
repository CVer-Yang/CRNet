"""
Codes of LinkNet based on https://github.com/snakers4/spacenet-three
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
import torch.nn.functional as F
import math
from functools import partial
import numpy as np
from torch import nn, einsum
from einops import rearrange, repeat
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nonlinearity = partial(F.relu,inplace=True)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class Dblock_more_dilate(nn.Module):
    def __init__(self,channel):
        super(Dblock_more_dilate, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        self.dilate5 = nn.Conv2d(channel, channel, kernel_size=3, dilation=16, padding=16)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()
                    
    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate2(dilate1_out))
        dilate3_out = nonlinearity(self.dilate3(dilate2_out))
        dilate4_out = nonlinearity(self.dilate4(dilate3_out))
        dilate5_out = nonlinearity(self.dilate5(dilate4_out))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out + dilate5_out
        return out

class Dblock(nn.Module):
    def __init__(self,channel):
        super(Dblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        #self.dilate5 = nn.Conv2d(channel, channel, kernel_size=3, dilation=16, padding=16)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()
                    
    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate2(dilate1_out))
        dilate3_out = nonlinearity(self.dilate3(dilate2_out))
        dilate4_out = nonlinearity(self.dilate4(dilate3_out))
        #dilate5_out = nonlinearity(self.dilate5(dilate4_out))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out# + dilate5_out
        return out

class GlobalContext(nn.Module):
    def __init__(self, in_channels):
        super(GlobalContext, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // 4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 4, in_channels, bias=False),
            nn.Sigmoid()
        )
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.global_avg_pool(x).view(b, c)
      
        y = self.fc(y).view(b, c, 1, 1)
        y = self.up(x*y)
      
        return y

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock,self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


import torch
import torch.nn as nn
import torch.nn.functional as F




class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class TransformerModel(nn.Module):
    def __init__(self, input_channels, embed_dim, num_heads, ff_dim, num_layers):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_channels, embed_dim)
        self.transformer_layers = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)]
        )
        self.output_linear = nn.Linear(embed_dim, input_channels//2)
        self.up = nn.Upsample(scale_factor=2)

    def forward(self, x):
        # Flatten the spatial dimensions and embed the channels
        b, c, h, w = x.shape
        x = x.view(b, c, h * w).permute(2, 0, 1)  # (N, B, C)
        x = self.embedding(x)  # (N, B, D)

        for layer in self.transformer_layers:
            x = layer(x)

        x = self.output_linear(x)  # (N, B, 256)

        # Reshape back to (B, 256, 64, 64)

        x = x.permute(1, 2, 0).view(b, c//2, h, w)
        x = self.up(x)

        return x




class Mesh_TransformerDecoderLayer(nn.Module):
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1,
                 layer_norm_eps=1e-5, batch_first=False, norm_first=False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Mesh_TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)


        self.activation = nn.ReLU()
        self.activation2 = nn.Softmax(dim=-1)

        self.fc_alpha1 = nn.Linear(d_model + d_model, d_model)
        self.fc_alpha2 = nn.Linear(d_model + d_model, d_model)
        self.fc_alpha3 = nn.Linear(d_model + d_model, d_model)
        self.sig = nn.Sigmoid()

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_alpha1.weight)
        nn.init.xavier_uniform_(self.fc_alpha2.weight)
        nn.init.xavier_uniform_(self.fc_alpha3.weight)
        nn.init.constant_(self.fc_alpha1.bias, 0)
        nn.init.constant_(self.fc_alpha2.bias, 0)
        nn.init.constant_(self.fc_alpha3.bias, 0)

    def forward(self, tgt,  tgt_mask=None,
                tgt_key_padding_mask = None):
        self_att_tgt = self.norm1(tgt + self._sa_block(tgt, tgt_mask, tgt_key_padding_mask))
        #x = self.norm2(x + self._ff_block(self_att_tgt))
        return self_att_tgt

    # self-attention block
    def _sa_block(self, x,
                  attn_mask, key_padding_mask):
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # multihead attention block

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        self.embedding_1D = nn.Embedding(16, int(d_model))

    def forward(self, x):
        # fixed
        x = x + self.pe[:x.size(0), :]
        # learnable
        x = x + self.embedding_1D(torch.arange(16, device=device).to(device)).unsqueeze(1).repeat(1,x.size(1),  1)
        return self.dropout(x)





class SpatialAttention(nn.Module):
    def __init__(self, in_dim1, in_dim2):
        super(SpatialAttention, self).__init__()
        self.linear = nn.Linear(in_dim1, in_dim2)


    def forward(self, x1, x2):
        # x4: [2, 1, 4]
        # x16: [2, 1, 16]

        # 使用卷积层将 x4 转换到 x16 的维度
        x1_transformed = self.linear(x1)  # [2, 1, 16]

        # 计算注意力权重
        attention_scores = torch.bmm(x2.transpose(1, 2), x1_transformed)  # [2, 16, 1] * [2, 1, 16] -> [2, 16, 16]
        attention_weights = F.softmax(attention_scores, dim=-1)  # [2, 16, 16]

        # 对 x16 进行加权
        weighted_x2 = torch.bmm(x2, attention_weights)  # [2, 1, 16] * [2, 16, 16] -> [2, 1, 16]

        return weighted_x2


class Relation_Refine(nn.Module):
    def __init__(self, in_channels1, in_channels2, stage, groups1=8, groups2=16):
        super(Relation_Refine, self).__init__()
        self.groups1 = groups1
        self.groups2 = groups2
        self.gap1 = nn.AdaptiveAvgPool2d(1)
        self.gap2 = nn.AdaptiveAvgPool2d(1)

        self.Conv1 = nn.Sequential(
            nn.Conv1d(in_channels1 // groups1, (in_channels1 // 2) // groups1, kernel_size=1),
            nn.ReLU()
        )

        self.Conv2 = nn.Sequential(
            nn.Conv1d(in_channels2 // groups1, in_channels2 // groups1, kernel_size=1),
            nn.ReLU()
        )

        self.conv1 = nn.Conv2d(in_channels1, 1, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels2, 1, kernel_size=1)
        self.up = nn.Upsample(scale_factor=2)

        self.convbnrelu = nn.Sequential(
            nn.Conv2d(in_channels2, in_channels2, kernel_size=1),
            nn.BatchNorm2d(in_channels2),
            nn.ReLU()
        )
        self.stage = stage
        if self.stage== 4:
            self.sa =SpatialAttention(4,16)
        elif self.stage ==3:
            self.sa =SpatialAttention(16,64)
        elif self.stage == 2:
            self.sa = SpatialAttention(64, 256)
        elif self.stage == 1:
            self.sa = SpatialAttention(256, 1024)

    def forward(self, x1, x2):
        # x1: (B, in_channels1, 32, 32)
        # x2: (B, in_channels2, 64, 64)
        B, C1, H1, W1 = x1.size()
        B, C2, H2, W2 = x2.size()

        # 分组语义提纯
        sem_refine_x2_list = []
        for i in range(self.groups1):
            x1_group = x1[:, i*(C1 // self.groups1):(i+1)*(C1 // self.groups1), :, :]
            x2_group = x2[:, i*(C2 // self.groups1):(i+1)*(C2 // self.groups1), :, :]
           

            gap1 = self.gap1(x1_group).view(x1_group.size(0), x1_group.size(1), -1)  # (B, C1 // groups, 1)
            gap2 = self.gap2(x2_group).view(x2_group.size(0), x2_group.size(1), -1)  # (B, C2 // groups, 1)

            mlp1 = self.Conv1(gap1)  # (B, C1 // (2*groups), 1)
            mlp2 = self.Conv2(gap2)  # (B, C2 // groups, 1)
            sem_matrix = torch.bmm(mlp1, mlp2.transpose(1, 2))  # (B, C1 // (2*groups), C2 // groups)

            channel_weight = torch.softmax(torch.bmm(sem_matrix, mlp2), dim=1).view(x2_group.size(0), x2_group.size(1), 1, 1)
            sem_refine_x2 = x2_group * channel_weight

            sem_refine_x2_list.append(sem_refine_x2)

        sem_refine_x2 = torch.cat(sem_refine_x2_list, dim=1)

        # 分组位置提纯
        spa_refine_x2_list = []
        split_size_h1 = H1 // self.groups2
        split_size_w1 = W1 // self.groups2
        
        split_size_h2 = H2 // self.groups2
        split_size_w2 = W2 // self.groups2
       
       
        for i in range(self.groups2):
            for j in range(self.groups2):
                x1_group = x1[:, :, i * split_size_h1:(i + 1) * split_size_h1, j * split_size_w1:(j + 1) * split_size_w1]
                x2_group = x2[:, :, i * split_size_h2:(i + 1) * split_size_h2, j * split_size_w2:(j + 1) * split_size_w2]
              
                B, C3, _, _ = x1_group.size()
                B, C4, _, _ = x2_group.size()
               
                pos1 = self.conv1(x1_group).view(x1_group.size(0), 1, -1)  # (B, 1, H1*W1)
               
                pos2 = self.conv2(x2_group).view(x2_group.size(0), 1, -1)  # (B, 1, H2*W2)
                spa_refine_x2 = self.sa(pos1,pos2)
               

                spa_refine_x2_list.append(spa_refine_x2)

        spa_refine_x2 = torch.cat(spa_refine_x2_list, dim=2).view(B,1,H2,W2)



        # 融合语义和位置提纯的特征
        
        fuse_feature = self.convbnrelu(sem_refine_x2 + spa_refine_x2)
        return fuse_feature


class Relation_Refine2(nn.Module):
    def __init__(self, in_channels1, in_channels2, stage, groups1=8, groups2=16):
        super(Relation_Refine2, self).__init__()
        self.groups1 = groups1
        self.groups2 = groups2
        self.gap1 = nn.AdaptiveAvgPool2d(1)
        self.gap2 = nn.AdaptiveAvgPool2d(1)

        self.Conv1 = nn.Sequential(
            nn.Conv1d(in_channels1 // groups1, (in_channels1 ) // groups1, kernel_size=1),
            nn.ReLU()
        )

        self.Conv2 = nn.Sequential(
            nn.Conv1d(in_channels2 // groups1, in_channels2 // groups1, kernel_size=1),
            nn.ReLU()
        )

        self.conv1 = nn.Conv2d(in_channels1, 1, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels2, 1, kernel_size=1)
        self.up = nn.Upsample(scale_factor=2)

        self.convbnrelu = nn.Sequential(
            nn.Conv2d(in_channels2, in_channels2, kernel_size=1),
            nn.BatchNorm2d(in_channels2),
            nn.ReLU()
        )
        self.stage = stage
        if self.stage == 4:
            self.sa = SpatialAttention(4, 16)
        elif self.stage == 3:
            self.sa = SpatialAttention(16, 64)
        elif self.stage == 2:
            self.sa = SpatialAttention(64, 256)
        elif self.stage == 1:
            self.sa = SpatialAttention(256, 1024)

    def forward(self, x1, x2):
        # x1: (B, in_channels1, 32, 32)
        # x2: (B, in_channels2, 64, 64)
        B, C1, H1, W1 = x1.size()
        B, C2, H2, W2 = x2.size()

        # 分组语义提纯
        sem_refine_x2_list = []
        for i in range(self.groups1):
            x1_group = x1[:, i * (C1 // self.groups1):(i + 1) * (C1 // self.groups1), :, :]
            x2_group = x2[:, i * (C2 // self.groups1):(i + 1) * (C2 // self.groups1), :, :]

            gap1 = self.gap1(x1_group).view(x1_group.size(0), x1_group.size(1), -1)  # (B, C1 // groups, 1)
            gap2 = self.gap2(x2_group).view(x2_group.size(0), x2_group.size(1), -1)  # (B, C2 // groups, 1)

            mlp1 = self.Conv1(gap1)  # (B, C1 // (2*groups), 1)
            mlp2 = self.Conv2(gap2)  # (B, C2 // groups, 1)
            sem_matrix = torch.bmm(mlp1, mlp2.transpose(1, 2))  # (B, C1 // (2*groups), C2 // groups)

            channel_weight = torch.softmax(torch.bmm(sem_matrix, mlp2), dim=1).view(x2_group.size(0), x2_group.size(1),
                                                                                    1, 1)
            sem_refine_x2 = x2_group * channel_weight

            sem_refine_x2_list.append(sem_refine_x2)

        sem_refine_x2 = torch.cat(sem_refine_x2_list, dim=1)

        # 分组位置提纯
        spa_refine_x2_list = []
        split_size_h1 = H1 // self.groups2
        split_size_w1 = W1 // self.groups2

        split_size_h2 = H2 // self.groups2
        split_size_w2 = W2 // self.groups2

        for i in range(self.groups2):
            for j in range(self.groups2):
                x1_group = x1[:, :, i * split_size_h1:(i + 1) * split_size_h1,
                           j * split_size_w1:(j + 1) * split_size_w1]
                x2_group = x2[:, :, i * split_size_h2:(i + 1) * split_size_h2,
                           j * split_size_w2:(j + 1) * split_size_w2]

                B, C3, _, _ = x1_group.size()
                B, C4, _, _ = x2_group.size()

                pos1 = self.conv1(x1_group).view(x1_group.size(0), 1, -1)  # (B, 1, H1*W1)

                pos2 = self.conv2(x2_group).view(x2_group.size(0), 1, -1)  # (B, 1, H2*W2)
                spa_refine_x2 = self.sa(pos1, pos2)

                spa_refine_x2_list.append(spa_refine_x2)

        spa_refine_x2 = torch.cat(spa_refine_x2_list, dim=2).view(B, 1, H2, W2)

        # 融合语义和位置提纯的特征
        
        fuse_feature = self.convbnrelu(sem_refine_x2 + spa_refine_x2)
        return fuse_feature


class Contextblock(nn.Module):
    def __init__(self, channel):
        super(Contextblock, self).__init__()

        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=5, padding=5)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=7, padding=7)
        self.swin2 = StageModule(in_channels=channel, hidden_dimension=channel, layers=2,
                                 num_heads=8, head_dim=32,
                                 window_size=2, relative_pos_embedding=True)
        self.swin4 = StageModule(in_channels=channel, hidden_dimension=channel, layers=2,
                                 num_heads=8, head_dim=32,
                                 window_size=4, relative_pos_embedding=True)
        self.swin8 = StageModule(in_channels=channel, hidden_dimension=channel, layers=2,
                                 num_heads=8, head_dim=32,
                                 window_size=8, relative_pos_embedding=True)
        self.swin16 = StageModule(in_channels=channel, hidden_dimension=channel, layers=2,
                                  num_heads=8, head_dim=32,
                                  window_size=16, relative_pos_embedding=True)

        self.crossfuse = inter_attn(f_dim=512)

        self.conv = nn.Conv2d(channel, channel, kernel_size=(1, 1), stride=1, padding=0)

        self.act1 = nn.Conv2d(channel, channel, kernel_size=1)
        self.act2 = nn.Conv2d(channel, channel, kernel_size=1)
        self.act3 = nn.Conv2d(channel, channel, kernel_size=1)
        self.act4 = nn.Conv2d(channel, channel, kernel_size=1)

        # self.interact = inter_attn(f_dim=512)

        # self.dilate5 = nn.Conv2d(channel, channel, kernel_size=3, dilation=16, padding=16)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):

        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate2(x))
        dilate3_out = nonlinearity(self.dilate3(x))
        dilate4_out = nonlinearity(self.dilate4(x))
        global_context = nonlinearity(self.swin4(x))
        fuse1 = self.act1(dilate1_out + self.crossfuse(dilate1_out,global_context))
        fuse2 = self.act2(dilate2_out + self.crossfuse(dilate2_out,global_context))
        fuse3 = self.act3(dilate3_out + self.crossfuse(dilate3_out,global_context))
        fuse4 = self.act4(dilate4_out + self.crossfuse(dilate4_out,global_context))
        fuse_contetx = self.conv(fuse1 + fuse2 + fuse3 + fuse4) + x

        return fuse_contetx


class HorizontalTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, num_layers):
        super(HorizontalTransformer, self).__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)
        ])

    def forward(self, x):
        B, C, H, W = x.size()
        x = x.permute(0, 2, 3, 1).reshape(B * H, W, C)  # (B*H, W, C)
        for layer in self.layers:
            x = layer(x)
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)
        return x

class VerticalTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, num_layers):
        super(VerticalTransformer, self).__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)
        ])

    def forward(self, x):
        B, C, H, W = x.size()
        x = x.permute(0, 3, 2, 1).reshape(B * W, H, C)  # (B*W, H, C)
        for layer in self.layers:
            x = layer(x)
        x = x.reshape(B, W, H, C).permute(0, 3, 2, 1)  # (B, C, H, W)
        return x



class Cross_MultiViewAtt(nn.Module):
    def __init__(self, in_channels, num_heads):
        super(Cross_MultiViewAtt, self).__init__()
        self.in_channels = in_channels
        self.HTransformer = HorizontalTransformer(embed_dim=in_channels, num_heads=num_heads, ff_dim=in_channels * 2, num_layers=1)
        self.VTransformer = VerticalTransformer(embed_dim=in_channels, num_heads=num_heads, ff_dim=in_channels, num_layers=1)
        self.fusion_conv1 = nn.Conv2d(2 * in_channels, in_channels, kernel_size=1)
        self.self_att = TransformerBlock(in_channels, num_heads, in_channels * 2)
        self.fusion_conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, feature_map):
        # Step 1: 获取上下文
        b = feature_map.size(0)
        Hcontext = self.HTransformer(feature_map)
        Vcontext = self.VTransformer(feature_map)
        fused_out = torch.cat([Hcontext, Vcontext], dim=1)
        global_map = self.fusion_conv1(fused_out)

        # Step 2: 切分特征图为16x16小块
        blocks = feature_map.unfold(1, 16, 16).unfold(2, 16, 16)  # shape: (B, C, 2, 2, 16, 16)
        blocks = blocks.contiguous().view(feature_map.size(0), self.in_channels, 2, 2, 16, 16)

        # Step 3: 交叉注意力计算
        attended_outputs = []
        for i in range(2 * 2):  # 现在有 2x2 的块
            block = blocks[:, :, i // 2, i % 2, :, :]  # (B, C, 16, 16)

            # 扩展维度以计算注意力
            block_q = block.view(feature_map.size(0), self.in_channels, -1).permute(0, 2, 1)  # (N, B, C)  # (B, C, 256)
            block_q = self.self_att(block_q)  # 进行自注意力
            feature_q = global_map.view(global_map.size(0), self.in_channels, -1)  # (B, C, 1024)

            # 计算注意力权重
            attention_weights = F.softmax(torch.matmul(block_q, feature_q), dim=-1)  # (B, 256, 1024)

            # 通过注意力权重加权原始特征图
            attended_feature = torch.matmul(attention_weights, feature_q.permute(0, 2, 1))  # (B, 1024, 256)
            attended_feature = attended_feature.view(feature_map.size(0), self.in_channels, 16, 16)  # (B, C, 16, 16)
            attended_outputs.append(attended_feature)

        # Step 4: 将每个小块的输出合并
        stacked_outputs = torch.stack(attended_outputs)  # 形状为 (4, 2, 512, 16, 16)
        reshaped_tensor = stacked_outputs.view(2, 2, b, 512, 16, 16)

        # 重新排列维度
        permuted_tensor = reshaped_tensor.permute(2, 3, 0, 4, 1, 5)  # (2, 2,2, 512, 16, 16)

        # 合并最后两个维度
        output = permuted_tensor.contiguous().view(b, 512, 32, 32)  # (2, 512, 32, 32)

        # Step 5: 融合输出
        output = self.fusion_conv2(global_map + output)
        return output



class Cross_ViewAtt(nn.Module):
    def __init__(self, in_channels, num_heads):
        super(Cross_ViewAtt, self).__init__()
        self.in_channels = in_channels
        self.HTransformer = HorizontalTransformer(embed_dim=in_channels, num_heads=num_heads, ff_dim=in_channels * 2, num_layers=1)
        self.VTransformer = VerticalTransformer(embed_dim=in_channels, num_heads=num_heads, ff_dim=in_channels, num_layers=1)
        self.fusion_conv1 = nn.Conv2d(2 * in_channels, in_channels, kernel_size=1)
        self.self_att = TransformerBlock(in_channels, num_heads, in_channels * 2)
        self.fusion_conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)


    def forward(self, feature_map):
        # Step 1: 获取上下文
        b = feature_map.size(0)
        Hcontext = self.HTransformer(feature_map)
        Vcontext = self.VTransformer(feature_map)
        fused_out = torch.cat([Hcontext, Vcontext], dim=1)
        global_map = self.fusion_conv1(fused_out)

        # Step 2: 切分特征图为16x16小块
        n, c, rows, cols = feature_map.size()

        # Step 2: 平分特征图为4个子特征图
        feature_map_1 = feature_map[:, :, :rows // 2, :cols // 2]
        feature_map_2 = feature_map[:, :, :rows // 2, cols // 2:]
        feature_map_3 = feature_map[:, :, rows // 2:, :cols // 2]
        feature_map_4 = feature_map[:, :, rows // 2:, cols // 2:]
        blocks = [feature_map_1,feature_map_2,feature_map_3,feature_map_4]

        # Step 3: 交叉注意力计算
        attended_outputs = []
        for i in range(2 * 2):  # 现在有 2x2 的块
            block = blocks[i]  # (B, C, 16, 16)

            # 扩展维度以计算注意力
            block_q = block.reshape(feature_map.size(0), self.in_channels, -1).permute(0, 2, 1)  # (N, B, C)  # (B, C, 256)
            block_q = self.self_att(block_q)  # 进行自注意力
            feature_q = global_map.reshape(global_map.size(0), self.in_channels, -1)  # (B, C, 1024)

            # 计算注意力权重
            attention_weights = F.softmax(torch.matmul(block_q, feature_q), dim=-1)  # (B, 256, 1024)

            # 通过注意力权重加权原始特征图
            attended_feature = torch.matmul(attention_weights, feature_q.permute(0, 2, 1))  # (B, 1024, 256)
            attended_feature = attended_feature.view(feature_map.size(0), self.in_channels, 16, 16)  # (B, C, 16, 16)
            attended_outputs.append(attended_feature)

        # Step 4: 将每个小块的输出合并
        stacked_outputs = torch.stack(attended_outputs)  # 形状为 (4, 2, 512, 16, 16)
        reshaped_tensor = stacked_outputs.reshape(2, 2, b, 512, 16, 16)

        # 重新排列维度
        permuted_tensor = reshaped_tensor.permute(2, 3, 0, 4, 1, 5)  # (2, 2,2, 512, 16, 16)

        # 合并最后两个维度
        output = permuted_tensor.contiguous().reshape(b, 512, 32, 32)  # (2, 512, 32, 32)

        # Step 5: 融合输出
        output = self.fusion_conv2(global_map + output)
        return output


class Cross_ViewAtt1(nn.Module):
    def __init__(self, in_channels, num_heads):
        super(Cross_ViewAtt1, self).__init__()
        self.in_channels = in_channels
        self.HTransformer = HorizontalTransformer(embed_dim=in_channels, num_heads=num_heads, ff_dim=in_channels * 2,
                                                  num_layers=1)
        self.VTransformer = VerticalTransformer(embed_dim=in_channels, num_heads=num_heads, ff_dim=in_channels,
                                                num_layers=1)
        self.fusion_conv1 = nn.Conv2d(2 * in_channels, in_channels, kernel_size=1)
        self.self_att = TransformerBlock(in_channels, num_heads, in_channels * 2)
        self.fusion_conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        self.layers = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, kernel_size=3,padding=1),
            nn.Conv2d(in_channels, in_channels, kernel_size=3,padding=1),
            nn.Conv2d(in_channels, in_channels, kernel_size=3,padding=1),
            nn.Conv2d(in_channels, in_channels, kernel_size=3,padding=1)
        ])


    def forward(self, feature_map):
        # Step 1: 获取上下文
        b = feature_map.size(0)
        Hcontext = self.HTransformer(feature_map)
        Vcontext = self.VTransformer(feature_map)
        fused_out = torch.cat([Hcontext, Vcontext], dim=1)
        global_map = self.fusion_conv1(fused_out)

        # Step 2: 切分特征图为16x16小块
        n, c, rows, cols = feature_map.size()

        # Step 2: 平分特征图为4个子特征图
        feature_map_1 = feature_map[:, :, :rows // 2, :cols // 2]
        feature_map_2 = feature_map[:, :, :rows // 2, cols // 2:]
        feature_map_3 = feature_map[:, :, rows // 2:, :cols // 2]
        feature_map_4 = feature_map[:, :, rows // 2:, cols // 2:]
        blocks = [feature_map_1, feature_map_2, feature_map_3, feature_map_4]

        # Step 3: 交叉注意力计算
        attended_outputs = []
        for i in range(2 * 2):  # 现在有 2x2 的块
            block = blocks[i]  # (B, C, 16, 16)
            attended_feature = self.layers[i](block)
            attended_outputs.append(attended_feature)

        # Step 4: 将每个小块的输出合并
        stacked_outputs = torch.stack(attended_outputs)  # 形状为 (4, 2, 512, 16, 16)
        reshaped_tensor = stacked_outputs.reshape(2, 2, b, 512, 16, 16)

        # 重新排列维度
        permuted_tensor = reshaped_tensor.permute(2, 3, 0, 4, 1, 5)  # (2, 2,2, 512, 16, 16)

        # 合并最后两个维度
        output = permuted_tensor.contiguous().reshape(b, 512, 32, 32)  # (2, 512, 32, 32)

        # Step 5: 融合输出
        output = self.fusion_conv2(global_map + output)
        return output





class CRNet(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(CRNet, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=False)
        resnet.load_state_dict(torch.load('./networks/resnet34.pth'))
        self.embed_dim=512
        self.vocab_size = 16
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        self.context =Cross_ViewAtt1( in_channels=512, num_heads=8)
        self.RR4 = Relation_Refine(512, 256,4)
        self.RR3 = Relation_Refine(256, 128,3)
        self.RR2 = Relation_Refine(128, 64,2)
        self.RR1 = Relation_Refine2(64, 64,1)
        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, image):

        # Encoder
        x = self.firstconv(image)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        e1 = self.firstmaxpool(x)
        e1 = self.encoder1(e1)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e4 = self.context(e4)

        # Decoder
        d4 = self.decoder4(e4) + self.RR4(e4, e3)
        d3 = self.decoder3(d4) + self.RR3(d4, e2)
        d2 = self.decoder2(d3) + self.RR2(d3,e1)
        d1 = self.decoder1(d2) + self.RR1(d2,x)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)

        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)



if __name__ == "__main__":
    model = CRNet(num_classes=1)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params / 1e6:.2f} M")