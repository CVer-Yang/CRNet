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


class Multi_brachDecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(Multi_brachDecoderBlock, self).__init__()

        # 空间特征分支（可分离卷积）
        self.spatial_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),  # 深度卷积
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),  # 点卷积
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, padding=1, groups=in_channels // 2),  # 再次深度卷积
            nn.Conv2d(in_channels // 2, in_channels // 4, kernel_size=1),  # 点卷积
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),

        )

        # 高频特征分支（基于边缘增强）
        self.high_freq_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels // 4, kernel_size=3, padding=1, bias=False, groups=in_channels // 4),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
        )

        # Sobel 卷积核用于显式提取边缘
        self.sobel_filter = nn.Conv2d(
            in_channels=in_channels // 4,
            out_channels=in_channels // 4,
            kernel_size=3,
            padding=1,
            groups=in_channels // 4,  # 按通道独立处理
            bias=False,
        )
        self.init_sobel_kernel()

        # 低频特征分支（全局信息）
        self.low_freq_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1, stride=2),  # 下采样
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 2, in_channels // 4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
        )

        # 门控机制（融合权重生成）
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels*3, in_channels, kernel_size=1),
            nn.Conv2d(in_channels, 3, kernel_size=1),
            nn.Softmax(dim=1),  # 输出权重归一化
        )

        # 融合模块
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels // 4 * 3, n_filters, kernel_size=1),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(n_filters, n_filters, 3, stride=2, padding=1,
                                              output_padding=1),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(inplace=True)
        )

    def init_sobel_kernel(self):
        """初始化 Sobel 卷积核"""
        # Sobel 核
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        sobel_kernels = torch.stack([sobel_x, sobel_y])  # (2, 3, 3)

        # 调整 Sobel 核以匹配输入通道数
        sobel_kernels = sobel_kernels.unsqueeze(1)  # (2, 1, 3, 3)
        sobel_kernels = sobel_kernels.repeat(self.sobel_filter.in_channels // 2, 1, 1,
                                             1)  # Repeat for each input channel

        with torch.no_grad():
            self.sobel_filter.weight = nn.Parameter(sobel_kernels)

    def forward(self, spatial_feat, high_freq_feat, low_freq_feat):
        # 空间特征处理
        spatial_out = self.spatial_branch(spatial_feat)

        # 高频特征处理
        high_freq_out = self.high_freq_branch(high_freq_feat)
        high_freq_out = self.sobel_filter(high_freq_out)  # 边缘增强

        # 低频特征处理
        low_freq_out = self.low_freq_branch(low_freq_feat)

        # 门控权重生成
        combined_features = torch.cat([spatial_feat, high_freq_feat, low_freq_feat], dim=1)
        weights = self.gate(combined_features)
        print(low_freq_out.shape)
        print(weights.shape)

        # 融合特征
        spatial_weighted = spatial_out * weights[:, 0:1, :, :]
        high_freq_weighted = high_freq_out * weights[:, 1:2, :, :]
        low_freq_weighted = low_freq_out * weights[:, 2:3, :, :]


        fused_features = torch.cat([spatial_weighted, high_freq_weighted, low_freq_weighted], dim=1)
        print(fused_features.shape)

        # 输出融合特征
        output = self.fusion(fused_features)
        return output


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

class AttenDecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(AttenDecoderBlock,self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        #self.global_context = GlobalContext(in_channels//4)

        self.deconv1 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 5, stride=2, padding=2, output_padding=1)
        self.norm3 = nn.BatchNorm2d(in_channels // 4)
        self.relu3 = nonlinearity

        self.deconv3 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 7, stride=2, padding=3, output_padding=1)
        self.norm4 = nn.BatchNorm2d(in_channels // 4)
        self.relu4 = nonlinearity


        self.conv5 = nn.Conv2d(in_channels // 4 *3, n_filters, 1)
        self.norm5 = nn.BatchNorm2d(n_filters)
        self.relu5 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x1 = self.deconv1(x)
        x1 = self.norm2(x1)
        x1 = self.relu2(x1)

        x2 = self.deconv2(x)
        x2 = self.norm3(x2)
        x2 = self.relu3(x2)

        x3 = self.deconv3(x)
        x3 = self.norm4(x3)
        x3 = self.relu4(x3)

        #x4 = self.global_context(x)

        x = self.conv5(torch.cat([x1,x2,x3],dim=1))
        x = self.norm5(x)
        x = self.relu5(x)
        return x

class DinkNet34_less_pool(nn.Module):
    def __init__(self, num_classes=1):
        super(DinkNet34_more_dilate, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        
        self.dblock = Dblock_more_dilate(256)

        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        
        #Center
        e3 = self.dblock(e3)

        # Decoder
        d3 = self.decoder3(e3) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        # Final Classification
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)
    
class DinkNet34(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(DinkNet34, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=False)
        resnet.load_state_dict(torch.load('./networks/resnet34.pth'))
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        
        self.dblock = Dblock(512)

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        
        # Center
        e4 = self.dblock(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)


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


class cm_align(nn.Module):
    def __init__(self, embed_dim=512, n_head=8):
        super(cm_align, self).__init__()
        self.dim = embed_dim
        self.head = n_head
        self.attn = nn.MultiheadAttention(self.dim, self.head)

    def forward(self,visual,text):
        batch_size, channels, height, width = visual.size()
        visual_reshaped = visual.view(batch_size, channels, height * width)  # (4, 512, 1024)
        query = visual_reshaped.permute(2, 0, 1)  # (sequence_length, batch_size, channels) => (1024, 4, 512)
        key = text.permute(1, 0, 2)  # (sequence_length, batch_size, channels) => (512, 4, 16)
        value = text.permute(1, 0, 2)  # (sequence_length, batch_size, channels) => (512, 4, 16)

        # 进行交叉注意力计算
        attn_output,_ = self.attn(query, key, value)  # (512, 4, 512)
        attn_output = attn_output.permute(1, 2, 0)  # 转回 (batch_size, channels, sequence_length) => (4, 512, 512)

        # Step 3: 将结果变回形状 (4, 512, 32, 32)
        attn_output_reshaped = attn_output.view(batch_size, channels, height, width)  # (4, 512, 32, 32)
        return  attn_output_reshaped

class ProjectionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ProjectionLayer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, input1):
        output1 = self.conv1(input1)
        output2 = self.conv2(input1)
        output3 = self.conv3(input1)
        return output1,output2,output3

class Feature_Fusion(nn.Module):
    def __init__(self, dim, num_heads=4):
        super(Feature_Fusion, self).__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"

        self.query_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.key_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.value_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.output_proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x1, x2, x3):
        batch_size, _,height, width = x1.size()

        # Project inputs
        query = self.query_proj(x1)
        key = self.key_proj(x2)
        value = self.value_proj(x2)

        # Reshape for multi-head attention
        query = query.view(batch_size, self.num_heads, self.head_dim, height * width)
        key = key.view(batch_size, self.num_heads, self.head_dim, height * width)
        value = value.view(batch_size, self.num_heads, self.head_dim, height * width)

        # Scaled dot-product attention
        attn_scores = torch.einsum("bnhd,bnhd->bnh", query, key) / (self.head_dim ** 0.5)
        attn_probs = F.softmax(attn_scores, dim=-1)

        attn_output = torch.einsum("bnh,bnhd->bnhd", attn_probs, value)
        attn_output = attn_output.view(batch_size, self.dim, height, width)

        # Project output
        output = self.output_proj(attn_output)+x3
        return output




class Enhance_MultiInstructDinkNet34(nn.Module):
    def __init__(self, vocab_size=16, num_classes=1, embed_dim=512, n_layers=1, n_head=8):
        super(Enhance_MultiInstructDinkNet34, self).__init__()

        filters = [64, 128, 256, 512]
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.dropout = 0.1
        resnet = models.resnet34(pretrained=False)
        resnet.load_state_dict(torch.load('./networks/resnet34.pth'))
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        self.projection = ProjectionLayer(in_channels=512, out_channels=128)
        self.Sim = similiar()

        self.conv1 = nn.Conv2d(128, 512, kernel_size=3, dilation=1, padding=1)
        self.conv2 = nn.Conv2d(512, 128, kernel_size=3, dilation=1, padding=1)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, dilation=1, padding=1)
        self.conv4 = nn.Conv2d(512, 256, kernel_size=3, dilation=1, padding=1)
        self.conv5 = nn.Conv2d(128, 512, kernel_size=3, dilation=1, padding=1)

        self.vocab_embedding = nn.Embedding(self.vocab_size, self.embed_dim)  # vocaburaly embedding
        self.position_encoding = PositionalEncoding(self.embed_dim)
        self.self_attn = nn.MultiheadAttention(embed_dim, n_head, dropout=self.dropout)
        self.FF = Feature_Fusion(dim=128, num_heads=8)

        self.cross3 = cm_align(embed_dim=512, n_head=8)
        self.cross2 = cm_align(embed_dim=512, n_head=8)
        self.cross1 = cm_align(embed_dim=512, n_head=8)
        self.ca3 = CoordAtt(inp=256, oup=256)
        self.ca2 = CoordAtt(inp=128, oup=128)
        self.ca1 = CoordAtt(inp=64, oup=64)

        self.decoder4 = AttenDecoderBlock(filters[3], filters[2])
        self.decoder3 = AttenDecoderBlock(filters[2], filters[1])
        self.decoder2 = AttenDecoderBlock(filters[1], filters[0])
        self.decoder1 = AttenDecoderBlock(filters[0], filters[0])

        self.Transdecoder4 =  TransformerModel(input_channels=512, embed_dim=512, num_heads=8, ff_dim=1024, num_layers=1)
        self.Transdecoder3 =  TransformerModel(input_channels=256, embed_dim=256, num_heads=8, ff_dim=1024, num_layers=1)

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, image, text):
        tgt = text.permute(1, 0)
        tgt_length = tgt.size(0)

        mask = (torch.triu(torch.ones(tgt_length, tgt_length)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        mask = mask.to(device)

        tgt_embedding = self.vocab_embedding(tgt)
        tgt_embedding = self.position_encoding(tgt_embedding)  # (length, batch, feature_dim)

        instruct_feature = tgt_embedding + self.self_attn(tgt_embedding, tgt_embedding, tgt_embedding)[0]
        # instruct_feature = self.transformer(tgt_embedding, tgt_mask=mask)  # (length, batch, feature_dim)
        instruct_feature = instruct_feature.permute(1, 0, 2)  # 4 9 1024
        # Encoder
        x = self.firstconv(image)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)

        e1 = self.encoder1(x)
        e2 = self.conv2(self.cross1(self.conv1(self.encoder2(e1)), instruct_feature))
        e3 = self.conv4(self.cross2(self.conv3(self.encoder3(e2)), instruct_feature))
        e4 = self.cross3(self.encoder4(e3), instruct_feature)
        feature1, feature2, feature3 = self.projection(e4)
        target_feature, supply_feature, background_feature = self.Sim(feature1, feature2, feature3, instruct_feature)
        target_feature = self.conv5(self.FF(target_feature, supply_feature, background_feature))

        # Decoder
        d4 = self.decoder4(target_feature+e4) + self.Transdecoder4(target_feature+e4) + e3
        d3 = self.decoder3(d4) +  self.Transdecoder3(d4)+ e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)

class MultiInstructDinkNet34(nn.Module):
    def __init__(self, vocab_size=16, num_classes=1, embed_dim=512, n_layers=1, n_head=8):
        super(MultiInstructDinkNet34, self).__init__()

        filters = [64, 128, 256, 512]
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.dropout = 0.1
        resnet = models.resnet34(pretrained=False)
        resnet.load_state_dict(torch.load('./networks/resnet34.pth'))
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        self.projection = ProjectionLayer(in_channels=512, out_channels=128)
        self.Sim = similiar()

        self.conv1 = nn.Conv2d(128, 512, kernel_size=3, dilation=1, padding=1)
        self.conv2 = nn.Conv2d(512, 128, kernel_size=3, dilation=1, padding=1)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, dilation=1, padding=1)
        self.conv4 = nn.Conv2d(512, 256, kernel_size=3, dilation=1, padding=1)
        self.conv5 = nn.Conv2d(128, 512, kernel_size=3, dilation=1, padding=1)

        self.vocab_embedding = nn.Embedding(self.vocab_size, self.embed_dim)  # vocaburaly embedding
        self.position_encoding = PositionalEncoding(self.embed_dim)
        self.self_attn = nn.MultiheadAttention(embed_dim, n_head, dropout=self.dropout)
        self.FF = Feature_Fusion(dim=128, num_heads=8)

        self.cross3 = cm_align(embed_dim=512, n_head=8)
        self.cross2 = cm_align(embed_dim=512, n_head=8)
        self.cross1 = cm_align(embed_dim=512, n_head=8)
        self.ca3 = CoordAtt(inp=256, oup=256)
        self.ca2 = CoordAtt(inp=128, oup=128)
        self.ca1 = CoordAtt(inp=64, oup=64)

        self.decoder4 = AttenDecoderBlock(filters[3], filters[2])
        self.decoder3 = AttenDecoderBlock(filters[2], filters[1])
        self.decoder2 = AttenDecoderBlock(filters[1], filters[0])
        self.decoder1 = AttenDecoderBlock(filters[0], filters[0])

        self.Transdecoder4 =  TransformerModel(input_channels=512, embed_dim=512, num_heads=8, ff_dim=1024, num_layers=1)
        self.Transdecoder3 =  TransformerModel(input_channels=256, embed_dim=256, num_heads=8, ff_dim=1024, num_layers=1)

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, image, text):
        tgt = text.permute(1, 0)
        tgt_length = tgt.size(0)

        mask = (torch.triu(torch.ones(tgt_length, tgt_length)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        mask = mask.to(device)

        tgt_embedding = self.vocab_embedding(tgt)
        tgt_embedding = self.position_encoding(tgt_embedding)  # (length, batch, feature_dim)

        instruct_feature = tgt_embedding + self.self_attn(tgt_embedding, tgt_embedding, tgt_embedding)[0]
        # instruct_feature = self.transformer(tgt_embedding, tgt_mask=mask)  # (length, batch, feature_dim)
        instruct_feature = instruct_feature.permute(1, 0, 2)  # 4 9 1024
        # Encoder
        x = self.firstconv(image)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)

        e1 = self.encoder1(x)
        e2 = self.conv2(self.cross1(self.conv1(self.encoder2(e1)), instruct_feature))
        e3 = self.conv4(self.cross2(self.conv3(self.encoder3(e2)), instruct_feature))
        e4 = self.cross3(self.encoder4(e3), instruct_feature)


        # Decoder
        d4 = self.decoder4(e4) + self.Transdecoder4(e4) + e3
        d3 = self.decoder3(d4) +  self.Transdecoder3(d4)+ e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)

class similiar(nn.Module):
    def __init__(self):
        super(similiar, self).__init__()
        self.proj_text_features = nn.Linear(512, 128)  
        
    def cosine_similarity(self,tensor1, tensor2):
        tensor1_norm = F.normalize(tensor1, p=2, dim=-1)
        tensor2_norm = F.normalize(tensor2, p=2, dim=-1)
       
        return torch.sum(tensor1_norm @ tensor2_norm, dim=-1)

    def forward(self,feature1,feature2,feature3,text_features):
        batch_size,c,_,_ = feature1.shape
        projected_feature1_flat = feature1.view(batch_size, 128, -1)
        projected_feature2_flat = feature2.view(batch_size, 128, -1)
        projected_feature3_flat = feature3.view(batch_size, 128, -1)
        projected_textfeatrue   = self.proj_text_features(text_features)

        # 计算相似性
        
        similarity1 = self.cosine_similarity(projected_textfeatrue,projected_feature1_flat)
        similarity2 = self.cosine_similarity(projected_textfeatrue,projected_feature2_flat)
        similarity3 = self.cosine_similarity(projected_textfeatrue,projected_feature3_flat)

        # 计算相似性平均值
        mean_similarity1 = similarity1.mean(dim=-1)
        mean_similarity2 = similarity2.mean(dim=-1)
        mean_similarity3 = similarity3.mean(dim=-1)

        # 获取相似性排序
        similarities = torch.stack([mean_similarity1, mean_similarity2, mean_similarity3], dim=1)
        sorted_similarities, indices = torch.sort(similarities, dim=1, descending=True)

        # 将相似性最大的和中间的特征图识别为道路和建筑特征图
        target_feature_map = torch.zeros_like(projected_feature1_flat)
        supply_feature_map = torch.zeros_like(projected_feature2_flat)

        for i in range(batch_size):
            target_feature_map[i] = projected_feature1_flat[i] if indices[i, 0] == 0 else projected_feature2_flat[i] if indices[
                                                                                                                i, 0] == 1 else \
                projected_feature3_flat[i]
            supply_feature_map[i] = projected_feature1_flat[i] if indices[i, 1] == 0 else projected_feature2_flat[i] if indices[
                                                                                                                    i, 1] == 1 else \
                projected_feature3_flat[i]

        # 最小相似性作为背景特征图
        background_feature_map = torch.zeros_like(projected_feature3_flat)

        for i in range(batch_size):
            background_feature_map[i] = projected_feature1_flat[i] if indices[i, 2] == 0 else projected_feature2_flat[i] if \
                indices[i, 2] == 1 else projected_feature3_flat[i]

        return target_feature_map.view(batch_size,c, 32, 32),supply_feature_map.view(batch_size,c, 32, 32),background_feature_map.view(batch_size,c, 32, 32)


class DinkNet50(nn.Module):
    def __init__(self, num_classes=1):
        super(DinkNet50, self).__init__()

        filters = [256, 512, 1024, 2048]
        resnet = models.resnet50(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        
        self.dblock = Dblock_more_dilate(2048)

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        
        # Center
        e4 = self.dblock(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)
    
class DinkNet101(nn.Module):
    def __init__(self, num_classes=1):
        super(DinkNet101, self).__init__()

        filters = [256, 512, 1024, 2048]
        resnet = models.resnet101(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.dblock = Dblock_more_dilate(2048)

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        
        # Center
        e4 = self.dblock(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)

class LinkNet34(nn.Module):
    def __init__(self, num_classes=1):
        super(LinkNet34, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=False)
        resnet.load_state_dict(torch.load('./networks/resnet34.pth'))
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)


class LanguageGate(nn.Module):
    def __init__(self, input_channels):
        super(LanguageGate, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, input_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(input_channels, input_channels, kernel_size=1)

    def forward(self, Fi):
        Si = torch.tanh(self.conv2(F.relu(self.conv1(Fi))))
        Ei = Fi * Si
        return Ei


class AttentionModule(nn.Module):
    def __init__(self, visual_channels, linguistic_channels):
        super(AttentionModule, self).__init__()
        self.Wvi = nn.Conv2d(visual_channels, visual_channels, kernel_size=1)
        self.Wvq = nn.Conv2d(visual_channels, visual_channels, kernel_size=1)
        self.Wli = nn.Conv1d(linguistic_channels, visual_channels, kernel_size=1)
        self.Wliv = nn.Conv1d(linguistic_channels, visual_channels, kernel_size=1)
        self.Wo = nn.Conv2d(visual_channels, visual_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Vi, L):
        # 计算 Vim 和 Viq
        Vim = self.Wvi(Vi)
        Viq = self.Wvq(Vi)

        # 计算 Lik
        Lik = self.Wli(L)


        # 矩阵乘法，计算 Gi
        Viq_flat = Viq.view(Viq.size(0), Viq.size(1), -1)  # 展平
        Gi = torch.matmul(Viq_flat.permute(0, 2, 1), Lik)  # 计算点积
        Gi = self.softmax(Gi)  # 应用softmax

        # 计算 Liv
        Liv = self.Wliv(L)

        # 矩阵乘法，计算 Si
        Si = torch.matmul(Gi, Liv.permute(0, 2, 1))  # 矩阵乘法
        Si = Si.view(Vi.size())  # 恢复原始形状

        # 计算 Fi
        Fi = Vim * Si
        Fi = self.Wo(Fi)

        return Fi


class VisionAttention(nn.Module):
    def __init__(self, visual_channels, linguistic_channels):
        super(VisionAttention, self).__init__()

        self.Wvq = nn.Conv2d(visual_channels, visual_channels, kernel_size=1)
        self.Wli = nn.Conv1d(linguistic_channels, visual_channels, kernel_size=1)
        self.Wliv = nn.Conv1d(linguistic_channels, visual_channels, kernel_size=1)
        self.Wo = nn.Conv2d(visual_channels, visual_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Vi, L):
        # 计算 Vim 和 Viq

        Viq = self.Wvq(Vi)

        # 计算 Lik
        Lik = self.Wli(L)


        # 矩阵乘法，计算 Gi
        Viq_flat = Viq.view(Viq.size(0), Viq.size(1), -1)  # 展平
        Gi = torch.matmul(Viq_flat.permute(0, 2, 1), Lik)  # 计算点积
        Gi = self.softmax(Gi)  # 应用softmax

        # 计算 Liv
        Liv = self.Wliv(L)

        # 矩阵乘法，计算 Si
        Si = torch.matmul(Gi, Liv.permute(0, 2, 1))  # 矩阵乘法
        Si = Si.view(Vi.size())  # 恢复原始形状

        # 计算 Fi
        Fi = Vi + Si
        Fi = self.Wo(Fi)

        return Fi

class LanguageAttention(nn.Module):
    def __init__(self, visual_channels, linguistic_channels):
        super(LanguageAttention, self).__init__()

        self.Wk = nn.Conv2d(visual_channels, visual_channels, kernel_size=1)
        self.Wv = nn.Conv2d(visual_channels, visual_channels, kernel_size=1)
        self.Wli = nn.Conv1d(linguistic_channels, visual_channels, kernel_size=1)
        self.Wl = nn.Conv1d(visual_channels, linguistic_channels, kernel_size=1)
        self.Wo = nn.Conv1d(linguistic_channels, linguistic_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Vi, L):
        # 计算 Vim 和 Viq

        Vik = self.Wk(Vi)
        Viv = self.Wv(Vi)

        # 计算 Lik
        Lik = self.Wli(L)

        # 矩阵乘法，计算 Gi
        Vik_flat = Vik.view(Vik.size(0), Vik.size(1), -1)  # 展平  n,c,hw
        Viv_flat = Viv.view(Viv.size(0), Viv.size(1), -1)  # 展平  n,c,hw
        Gi = torch.matmul(Lik.permute(0,2,1), Vik_flat)
        Gi = self.softmax(Gi)  # 应用softmax

        # 矩阵乘法，计算 Si
        Si = torch.matmul(Gi, Viv_flat.permute(0, 2, 1))  # 矩阵乘法
        Si = Si.permute(0,2,1)
        Si = self.Wl(Si)

        # 计算 Fi

        Fi = L + Si
        Fi = self.Wo(Fi)
        return Fi


class VLAttention(nn.Module):
    def __init__(self, visual_channels, linguistic_channels):
        super(VLAttention, self).__init__()
        self.Visionattention = VisionAttention(visual_channels,linguistic_channels)
        self.Langeattention  = LanguageAttention(visual_channels,linguistic_channels)
    def forward(self, Vi, L):
        Vi = self.Visionattention(Vi,L)
        Li = self.Langeattention(Vi,L)
        return Vi, Li

class LAVT_context(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(LAVT_context, self).__init__()

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
        self.LG1 = LanguageGate(64)
        self.LG2 = LanguageGate(128)
        self.LG3 = LanguageGate(256)

        self.PWAM1 = AttentionModule(64, 512)
        self.PWAM2 = AttentionModule(128, 512)
        self.PWAM3 = AttentionModule(256, 512)
        self.PWAM4 = AttentionModule(512, 512)
        self.vocab_embedding = nn.Embedding(self.vocab_size, self.embed_dim)  # vocaburaly embedding
        self.position_encoding = PositionalEncoding(self.embed_dim)
        self.self_attn = nn.MultiheadAttention(512, 8, dropout=0.1)
        self.context = Dblock_more_dilate(512)

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.RR4= Relation_Refine(512,256)
        self.RR3= Relation_Refine(256,128)
        self.RR2= Relation_Refine(128,64)
        self.RR1= Relation_Refine(64,64)

        #self.Transdecoder4 =  TransformerModel(input_channels=512, embed_dim=512, num_heads=8, ff_dim=1024, num_layers=1)
        #self.Transdecoder3 =  TransformerModel(input_channels=256, embed_dim=256, num_heads=8, ff_dim=1024, num_layers=1)

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, image, text):
        tgt = text.permute(1, 0)
        tgt_length = tgt.size(0)

        mask = (torch.triu(torch.ones(tgt_length, tgt_length)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        mask = mask.to(device)

        tgt_embedding = self.vocab_embedding(tgt)
        tgt_embedding = self.position_encoding(tgt_embedding)  # (length, batch, feature_dim)

        instruct_feature = tgt_embedding + self.self_attn(tgt_embedding, tgt_embedding, tgt_embedding)[0]
        # instruct_feature = self.transformer(tgt_embedding, tgt_mask=mask)  # (length, batch, feature_dim)
        instruct_feature = instruct_feature.permute(1, 2, 0)  # 4  512 9

        # Encoder
        x = self.firstconv(image)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e11 = self.LG1(self.PWAM1(e1,instruct_feature))
        e1 = e1+e11
        e2 = self.encoder2(e1)
        e22 = self.LG2(self.PWAM2(e2, instruct_feature))
        e2 = e2+e22
        e3 = self.encoder3(e2)
        e33 = self.LG3(self.PWAM3(e3, instruct_feature))
        e3 = e3+e33
        e4 = self.encoder4(e3)
        e4 = self.PWAM4(e4,instruct_feature)
        e4 = self.context(e4)


        # Decoder
        d4 = self.decoder4(e4) + self.RR4(e4,e33)
        d3 = self.decoder3(d4) + self.RR3(d4,e22)
        #d4 = self.decoder4(e4) + self.RR4(e4,e33)
        #d3 = self.decoder3(d4) + self.RR3(d4,e22)
        d2 = self.decoder2(d3) + e11
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)

class LAVT_context1(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(LAVT_context1, self).__init__()

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
        self.LG1 = LanguageGate(64)
        self.LG2 = LanguageGate(128)
        self.LG3 = LanguageGate(256)

        self.PWAM1 = AttentionModule(64, 512)
        self.PWAM2 = AttentionModule(128, 512)
        self.PWAM3 = AttentionModule(256, 512)
        self.PWAM4 = AttentionModule(512, 512)
        self.vocab_embedding = nn.Embedding(self.vocab_size, self.embed_dim)  # vocaburaly embedding
        self.position_encoding = PositionalEncoding(self.embed_dim)
        self.self_attn = nn.MultiheadAttention(512, 8, dropout=0.1)
        self.context =context()

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])



        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, image, text):
        tgt = text.permute(1, 0)
        tgt_length = tgt.size(0)

        mask = (torch.triu(torch.ones(tgt_length, tgt_length)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        mask = mask.to(device)

        tgt_embedding = self.vocab_embedding(tgt)
        tgt_embedding = self.position_encoding(tgt_embedding)  # (length, batch, feature_dim)

        instruct_feature = tgt_embedding + self.self_attn(tgt_embedding, tgt_embedding, tgt_embedding)[0]
        # instruct_feature = self.transformer(tgt_embedding, tgt_mask=mask)  # (length, batch, feature_dim)
        instruct_feature = instruct_feature.permute(1, 2, 0)  # 4  512 9

        # Encoder
        x = self.firstconv(image)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e11 = self.LG1(self.PWAM1(e1,instruct_feature))
        e1 = e1+e11
        e2 = self.encoder2(e1)
        e22 = self.LG2(self.PWAM2(e2, instruct_feature))
        e2 = e2+e22
        e3 = self.encoder3(e2)
        e33 = self.LG3(self.PWAM3(e3, instruct_feature))
        e3 = e3+e33
        e4 = self.encoder4(e3)
        e4 = self.PWAM4(e4,instruct_feature)
        e4 = self.context(e4,instruct_feature)


        # Decoder
        d4 = self.decoder4(e4) + e33
        d3 = self.decoder3(d4) + e22
        d2 = self.decoder2(d3) + e11
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)


class LAVT_context2(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(LAVT_context2, self).__init__()

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


        self.VL1 = VLAttention(64, 512)
        self.VL2 = VLAttention(128, 512)
        self.VL3 = VLAttention(256, 512)
        self.VL4 = VLAttention(512, 512)
        self.vocab_embedding = nn.Embedding(self.vocab_size, self.embed_dim)  # vocaburaly embedding
        self.position_encoding = PositionalEncoding(self.embed_dim)
        self.self_attn = nn.MultiheadAttention(512, 8, dropout=0.1)

        self.context = AxialContext(embed_dim=512, num_heads=8, num_points=4)
        self.RR4 = Relation_Refine(512, 256, 4)
        self.RR3 = Relation_Refine(256, 128, 3)
        self.RR2 = Relation_Refine(128, 64, 2)
        self.RR1 = Relation_Refine2(64, 64, 1)

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, image, text):
        tgt = text.permute(1, 0)
        tgt_length = tgt.size(0)

        mask = (torch.triu(torch.ones(tgt_length, tgt_length)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        mask = mask.to(device)

        tgt_embedding = self.vocab_embedding(tgt)
        tgt_embedding = self.position_encoding(tgt_embedding)  # (length, batch, feature_dim)

        instruct_feature = tgt_embedding + self.self_attn(tgt_embedding, tgt_embedding, tgt_embedding)[0]
        # instruct_feature = self.transformer(tgt_embedding, tgt_mask=mask)  # (length, batch, feature_dim)
        instruct_feature = instruct_feature.permute(1, 2, 0)  # 4  512 9

        # Encoder
        x = self.firstconv(image)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e11,l1 = self.VL1(e1,instruct_feature)
        e1 = e1+e11
        e2 = self.encoder2(e1)
        e22,l2 = self.VL2(e2, l1+instruct_feature)
        e2 = e2+e22
        e3 = self.encoder3(e2)
        e33,l3 = self.VL3(e3, l2+instruct_feature)
        e3 = e3+e33
        e4 = self.encoder4(e3)
        e44,l4 = self.VL4(e4, l3+instruct_feature)
        e4 = self.context(e4+e44)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)


class LAVT_context3(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(LAVT_context3, self).__init__()

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


        self.VL1 = VLAttention(64, 512)
        self.VL2 = VLAttention(128, 512)
        self.VL3 = VLAttention(256, 512)
        self.VL4 = VLAttention(512, 512)
        self.vocab_embedding = nn.Embedding(self.vocab_size, self.embed_dim)  # vocaburaly embedding
        self.position_encoding = PositionalEncoding(self.embed_dim)
        self.self_attn = nn.MultiheadAttention(512, 8, dropout=0.1)
        #self.context =context()

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, image, text):
        tgt = text.permute(1, 0)
        tgt_length = tgt.size(0)

        mask = (torch.triu(torch.ones(tgt_length, tgt_length)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        mask = mask.to(device)

        tgt_embedding = self.vocab_embedding(tgt)
        tgt_embedding = self.position_encoding(tgt_embedding)  # (length, batch, feature_dim)

        instruct_feature = tgt_embedding + self.self_attn(tgt_embedding, tgt_embedding, tgt_embedding)[0]
        # instruct_feature = self.transformer(tgt_embedding, tgt_mask=mask)  # (length, batch, feature_dim)
        instruct_feature = instruct_feature.permute(1, 2, 0)  # 4  512 9

        # Encoder
        x = self.firstconv(image)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e11,l1 = self.VL1(e1,instruct_feature)
        e1 = e1+e11
        e2 = self.encoder2(e1)
        e22,l2 = self.VL2(e2, l1+instruct_feature)
        e2 = e2+e22
        e3 = self.encoder3(e2)
        e33,l3 = self.VL3(e3, l2+instruct_feature)
        e3 = e3+e33
        e4 = self.encoder4(e3)
        e4,l4 = self.VL4(e4, l3+instruct_feature)
        #e4 = self.context(e4,l4+instruct_feature)

        # Decoder
        d4 = self.decoder4(e4) + e33
        d3 = self.decoder3(d4) + e22
        d2 = self.decoder2(d3) + e11
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)


class LAVT_context4(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(LAVT_context4, self).__init__()

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


        self.VL1 = VLAttention(64, 512)
        self.VL2 = VLAttention(128, 512)
        self.VL3 = VLAttention(256, 512)
        self.VL4 = VLAttention(512, 512)
        self.vocab_embedding = nn.Embedding(self.vocab_size, self.embed_dim)  # vocaburaly embedding
        self.position_encoding = PositionalEncoding(self.embed_dim)
        self.self_attn = nn.MultiheadAttention(512, 8, dropout=0.1)
        #self.context =context()
        self.context = AxialContext(embed_dim=512, num_heads=8, num_points=4)
        self.RR4 = Relation_Refine(512, 256, 4)
        self.RR3 = Relation_Refine(256, 128, 3)
        self.RR2 = Relation_Refine(128, 64, 2)
        self.RR1 = Relation_Refine2(64, 64, 1)

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, image, text):
        tgt = text.permute(1, 0)
        tgt_length = tgt.size(0)

        mask = (torch.triu(torch.ones(tgt_length, tgt_length)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        mask = mask.to(device)

        tgt_embedding = self.vocab_embedding(tgt)
        tgt_embedding = self.position_encoding(tgt_embedding)  # (length, batch, feature_dim)

        instruct_feature = tgt_embedding + self.self_attn(tgt_embedding, tgt_embedding, tgt_embedding)[0]
        # instruct_feature = self.transformer(tgt_embedding, tgt_mask=mask)  # (length, batch, feature_dim)
        instruct_feature = instruct_feature.permute(1, 2, 0)  # 4  512 9

        # Encoder
        x = self.firstconv(image)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        e1 = self.firstmaxpool(x)
        e1 = self.encoder1(e1)
        e11,l1 = self.VL1(e1,instruct_feature)
        e1 = e1+e11
        e2 = self.encoder2(e1)
        e22,l2 = self.VL2(e2, l1+instruct_feature)
        e2 = e2+e22
        e3 = self.encoder3(e2)
        e33,l3 = self.VL3(e3, l2+instruct_feature)
        e3 = e3+e33
        e4 = self.encoder4(e3)
        e44,l4 = self.VL4(e4, l3+instruct_feature)
        #e4 = self.context(e4,l4+instruct_feature)
        e4 = self.context(e4+e44)

        # Decoder
        d4 = self.decoder4(e4) + self.RR4(e4, e3)
        d3 = self.decoder3(d4) + self.RR3(d4, e2)
        d2 = self.decoder2(d3) + self.RR2(d3, e1)
        d1 = self.decoder1(d2) + self.RR1(d2, x)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)


class similiar_order(nn.Module):
    def __init__(self):
        super(similiar_order, self).__init__()
        self.proj_text_features = nn.Conv1d(in_channels=512, out_channels=256,kernel_size=1)

    def cosine_similarity(self, tensor1, tensor2):
        tensor1_norm = F.normalize(tensor1, p=2, dim=-1)
        tensor2_norm = F.normalize(tensor2, p=2, dim=-1)

        return torch.sum(tensor1_norm @ tensor2_norm, dim=-1)

    def forward(self, feature1, feature2, text_features):
        batch_size, c, _, _ = feature1.shape
        projected_feature1_flat = feature1.view(batch_size, 256, -1)
        projected_feature2_flat = feature2.view(batch_size, 256, -1)

        projected_textfeatrue = self.proj_text_features(text_features)
        projected_textfeatrue = projected_textfeatrue.permute(0,2, 1)
        # 计算相似性

        similarity1 = self.cosine_similarity(projected_textfeatrue, projected_feature1_flat)
        similarity2 = self.cosine_similarity(projected_textfeatrue, projected_feature2_flat)

        # 计算相似性平均值
        mean_similarity1 = similarity1.mean(dim=-1)
        mean_similarity2 = similarity2.mean(dim=-1)

        # 获取相似性排序
        similarities = torch.stack([mean_similarity1, mean_similarity2], dim=1)
        sorted_similarities, indices = torch.sort(similarities, dim=1, descending=True)

        # 将相似性最大的和中间的特征图识别为道路和建筑特征图
        target_feature_map = torch.zeros_like(projected_feature1_flat)
        supply_feature_map = torch.zeros_like(projected_feature2_flat)

        for i in range(batch_size):
            target_feature_map[i] = projected_feature1_flat[i] if indices[i, 0] == 0 else projected_feature2_flat[i]
            supply_feature_map[i] = projected_feature1_flat[i] if indices[i, 1] == 0 else projected_feature2_flat[i]
        return target_feature_map,supply_feature_map


class context(nn.Module):
    def __init__(self):
        super(context, self).__init__()
        self.proj1 = nn.Conv2d(in_channels=512, out_channels=256,kernel_size=1)
        self.proj2 = nn.Conv2d(in_channels=512, out_channels=256,kernel_size=1)
        self.order = similiar_order()
        self.cross_attention = nn.MultiheadAttention(embed_dim=256, num_heads=8)
        self.fc = nn.Conv1d(256, 512,kernel_size=1)

    def forward(self,x,text):
        batch_size, c, h, w = x.shape
        x1 = self.proj1(x)
        x2 = self.proj2(x)
   
        target, background = self.order(x1,x2,text)
            
        target = target.permute(2, 0, 1)  # (1024, 4, 256)
        background = background.permute(2, 0, 1)  # (1024, 4, 256)
      
        supply,_ = self.cross_attention(target,background,background)
        target = supply+target
        target = target.permute(1, 2, 0)
        target = self.fc(target)
        return target.view(batch_size,c,h,w)


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


from einops import rearrange, repeat


def weights_init(layer):
    classname = layer.__class__.__name__
    # print(classname)
    if classname.find('Conv2d') != -1:
        nn.init.xavier_uniform_(layer.weight.data)
    elif classname.find('Linear') != -1:
        nn.init.xavier_uniform_(layer.weight.data)
        if layer.bias is not None:
            nn.init.constant_(layer.bias.data, 0.0)


class CyclicShift(nn.Module):
    def __init__(self, displacement):
        super().__init__()
        self.displacement = displacement

    def forward(self, x):
        return torch.roll(x, shifts=(self.displacement, self.displacement), dims=(1, 2))


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


def create_mask(window_size, displacement, upper_lower, left_right):
    mask = torch.zeros(window_size ** 2, window_size ** 2)

    if upper_lower:
        mask[-displacement * window_size:, :-displacement * window_size] = float('-inf')
        mask[:-displacement * window_size, -displacement * window_size:] = float('-inf')

    if left_right:
        mask = rearrange(mask, '(h1 w1) (h2 w2) -> h1 w1 h2 w2', h1=window_size, h2=window_size)
        mask[:, -displacement:, :, :-displacement] = float('-inf')
        mask[:, :-displacement, :, -displacement:] = float('-inf')
        mask = rearrange(mask, 'h1 w1 h2 w2 -> (h1 w1) (h2 w2)')

    return mask


def get_relative_distances(window_size):
    indices = torch.tensor(np.array([[x, y] for x in range(window_size) for y in range(window_size)]))
    distances = indices[None, :, :] - indices[:, None, :]
    return distances


class WindowAttention(nn.Module):
    def __init__(self, dim, heads, head_dim, shifted, window_size, relative_pos_embedding):
        super().__init__()
        inner_dim = head_dim * heads

        self.heads = heads
        self.scale = head_dim ** -0.5
        self.window_size = window_size
        self.relative_pos_embedding = relative_pos_embedding
        self.shifted = shifted

        if self.shifted:
            displacement = window_size // 2
            self.cyclic_shift = CyclicShift(-displacement)
            self.cyclic_back_shift = CyclicShift(displacement)
            self.upper_lower_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement,
                                                             upper_lower=True, left_right=False), requires_grad=False)
            self.left_right_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement,
                                                            upper_lower=False, left_right=True), requires_grad=False)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        if self.relative_pos_embedding:
            self.relative_indices = get_relative_distances(window_size) + window_size - 1
            self.pos_embedding = nn.Parameter(torch.randn(2 * window_size - 1, 2 * window_size - 1))
        else:
            self.pos_embedding = nn.Parameter(torch.randn(window_size ** 2, window_size ** 2))

        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        if self.shifted:
            x = self.cyclic_shift(x)

        b, n_h, n_w, _, h = *x.shape, self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        nw_h = n_h // self.window_size
        nw_w = n_w // self.window_size

        q, k, v = map(
            lambda t: rearrange(t, 'b (nw_h w_h) (nw_w w_w) (h d) -> b h (nw_h nw_w) (w_h w_w) d',
                                h=h, w_h=self.window_size, w_w=self.window_size), qkv)

        dots = einsum('b h w i d, b h w j d -> b h w i j', q, k) * self.scale

        if self.relative_pos_embedding:
            dots += self.pos_embedding[self.relative_indices[:, :, 0], self.relative_indices[:, :, 1]]
        else:
            dots += self.pos_embedding

        if self.shifted:
            dots[:, :, -nw_w:] += self.upper_lower_mask
            dots[:, :, nw_w - 1::nw_w] += self.left_right_mask

        attn = dots.softmax(dim=-1)

        out = einsum('b h w i j, b h w j d -> b h w i d', attn, v)
        out = rearrange(out, 'b h (nw_h nw_w) (w_h w_w) d -> b (nw_h w_h) (nw_w w_w) (h d)',
                        h=h, w_h=self.window_size, w_w=self.window_size, nw_h=nw_h, nw_w=nw_w)
        out = self.to_out(out)

        if self.shifted:
            out = self.cyclic_back_shift(out)
        return out


class SwinBlock(nn.Module):
    def __init__(self, dim, heads, head_dim, mlp_dim, shifted, window_size, relative_pos_embedding):
        super().__init__()
        self.attention_block = Residual(PreNorm(dim, WindowAttention(dim=dim,
                                                                     heads=heads,
                                                                     head_dim=head_dim,
                                                                     shifted=shifted,
                                                                     window_size=window_size,
                                                                     relative_pos_embedding=relative_pos_embedding)))
        self.mlp_block = Residual(PreNorm(dim, FeedForward(dim=dim, hidden_dim=mlp_dim)))

    def forward(self, x):
        x = self.attention_block(x)
        x = self.mlp_block(x)
        return x


class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b, -1, h, w).permute(0, 2, 3, 1)
        x = self.linear(x)
        return x


class StageModule(nn.Module):
    def __init__(self, in_channels, hidden_dimension, layers, num_heads, head_dim, window_size,
                 relative_pos_embedding):
        super().__init__()
        assert layers % 2 == 0, 'Stage layers need to be divisible by 2 for regular and shifted block.'

        self.patch_partition = PatchMerging(in_channels=in_channels, out_channels=in_channels)

        self.layers = nn.ModuleList([])
        for _ in range(layers // 2):
            self.layers.append(nn.ModuleList([
                SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=False, window_size=window_size, relative_pos_embedding=relative_pos_embedding),
                SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=True, window_size=window_size, relative_pos_embedding=relative_pos_embedding),
            ]))

    def forward(self, x):
        x = self.patch_partition(x)
        for regular_block, shifted_block in self.layers:
            x = regular_block(x)
            x = shifted_block(x)
        return x.permute(0, 3, 1, 2)


class inter_attn(nn.Module):
    def __init__(self, f_dim, n_heads=8, d_q=None, d_v=None, dropout=0.3):
        super().__init__()

        self.build_inter_attn(f_dim, n_heads, d_q, d_v, dropout)

        for m in self.modules():
            weights_init(m)

    def build_inter_attn(self, f_dim, n_heads=4, d_q=None, d_v=None, dropout=0.1):
        if d_q is None:
            d_q = f_dim // n_heads
        if d_v is None:
            d_v = f_dim // n_heads

        self.n_heads = n_heads
        self.d_q = d_q
        self.d_v = d_v
        self.norm = d_q ** 0.5
        self.f_dim = f_dim

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.w_qs = nn.Linear(f_dim, n_heads * d_q)
        self.w_ks = nn.Linear(f_dim, n_heads * d_q)
        self.w_vs = nn.Linear(f_dim, n_heads * d_v)
        self.fc = nn.Linear(n_heads * d_v, f_dim)

        self.layer_norm1 = nn.LayerNorm(f_dim, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(f_dim, eps=1e-6)

    def inter_attn(self, Local, Fuse, mask_L2R=None, mask_R2L=None):
        b, c, h, w = Local.shape
        Lf = Local.view(b, c, -1).permute(0, 2, 1)
        Rf = Fuse.view(b, c, -1).permute(0, 2, 1)
        BS, V, fdim = Lf.shape
        assert fdim == self.f_dim
        BS, V, fdim = Rf.shape
        assert fdim == self.f_dim

        Lf2 = self.layer_norm1(Lf)
        Rf2 = self.layer_norm2(Rf)

        Lq = self.w_qs(Lf2).view(BS, V, self.n_heads, self.d_q).transpose(1, 2)  # BS x h x V x q
        Lk = self.w_ks(Lf2).view(BS, V, self.n_heads, self.d_q).transpose(1, 2)  # BS x h x V x q
        Lv = self.w_vs(Lf2).view(BS, V, self.n_heads, self.d_v).transpose(1, 2)  # BS x h x V x v

        Rq = self.w_qs(Rf2).view(BS, V, self.n_heads, self.d_q).transpose(1, 2)  # BS x h x V x q
        Rk = self.w_ks(Rf2).view(BS, V, self.n_heads, self.d_q).transpose(1, 2)  # BS x h x V x q
        Rv = self.w_vs(Rf2).view(BS, V, self.n_heads, self.d_v).transpose(1, 2)  # BS x h x V x v

        attn_R2L = torch.matmul(Lq, Rk.transpose(-1, -2)) / self.norm  # bs, h, V, V

        if mask_R2L is not None:
            attn_R2L = attn_R2L.masked_fill(mask_R2L == 0, -1e9)

        attn_R2L = F.softmax(attn_R2L, dim=-1)  # bs, h, V, V

        attn_R2L = self.dropout1(attn_R2L)

        feat_R2L = torch.matmul(attn_R2L, Rv).transpose(1, 2).contiguous().view(BS, V, -1)

        feat_R2L = self.dropout2(self.fc(feat_R2L))

        Lf = Lf + feat_R2L

        Local = Lf.view(Local.size(0), Local.size(2), Local.size(3), Local.size(1)).permute(0, 3, 1, 2)

        return Local

    def forward(self, Lf, Rf, mask_L2R=None, mask_R2L=None):

        Lf = self.inter_attn(Lf, Rf, mask_L2R, mask_R2L)

        return Lf

class ENet(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(ENet, self).__init__()

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
        self.context =Contextblock(channel=512)
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


class Instruct_ENet(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(Instruct_ENet, self).__init__()

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


        self.Instruct1 = VisionAttention(64, 512)
        self.Instruct2 = VisionAttention(128, 512)
        self.Instruct3 = VisionAttention(256, 512)
        self.Instruct4 = VisionAttention(512, 512)

        self.context = Contextblock(channel=512)
        self.RR4 = Relation_Refine(512, 256, 4)
        self.RR3 = Relation_Refine(256, 128, 3)
        self.RR2 = Relation_Refine(128, 64, 2)
        self.RR1 = Relation_Refine2(64, 64, 1)

        self.vocab_embedding = nn.Embedding(self.vocab_size, self.embed_dim)  # vocaburaly embedding
        self.position_encoding = PositionalEncoding(self.embed_dim)
        self.self_attn = nn.MultiheadAttention(512, 8, dropout=0.1)


        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

        self.conv5 = nn.Conv2d(in_channels=filters[3], out_channels=filters[0],
                               kernel_size=3,
                               stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=filters[2], out_channels=filters[0],
                               kernel_size=3,
                               stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=filters[1], out_channels=filters[0],
                               kernel_size=3,
                               stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=filters[0], out_channels=filters[0],
                               kernel_size=3,
                               stride=1, padding=1)

        self.hd4_d0 = nn.Upsample(scale_factor=16)
        self.hd3_d0 = nn.Upsample(scale_factor=8)
        self.hd2_d0 = nn.Upsample(scale_factor=4)
        self.hd1_d0 = nn.Upsample(scale_factor=2)

    def forward(self, image, text):
        tgt = text.permute(1, 0)
        tgt_length = tgt.size(0)

        mask = (torch.triu(torch.ones(tgt_length, tgt_length)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        mask = mask.to(device)

        tgt_embedding = self.vocab_embedding(tgt)
        tgt_embedding = self.position_encoding(tgt_embedding)  # (length, batch, feature_dim)

        instruct_feature = tgt_embedding + self.self_attn(tgt_embedding, tgt_embedding, tgt_embedding)[0]
        # instruct_feature = self.transformer(tgt_embedding, tgt_mask=mask)  # (length, batch, feature_dim)
        instruct_feature = instruct_feature.permute(1, 2, 0)  # 4  512 9

        # Encoder
        x = self.firstconv(image)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        e1 = self.firstmaxpool(x)
        e1 = self.encoder1(e1)

        e11 = self.Instruct1(e1,instruct_feature)
        e1 = e1+e11
        e2 = self.encoder2(e1)
        e22 = self.Instruct2(e2, instruct_feature)
        e2 = e2+e22
        e3 = self.encoder3(e2)
        e33 = self.Instruct3(e3, instruct_feature)
        e3 = e3+e33
        e4 = self.encoder4(e3)
        e4 = self.context(e4)
        e44 = self.Instruct4(e4,instruct_feature)+e4


        # Decoder
        d4 = self.decoder4(e44) + self.RR4(e4, e3)
        d3 = self.decoder3(d4) + self.RR3(d4, e2)
        d2 = self.decoder2(d3) + self.RR2(d3, e1)
        d1 = self.decoder1(d2) + self.RR1(d2, x)

        x_final = d1 + self.conv5(self.hd4_d0(e44)) + self.conv4(self.hd3_d0(d4)) + self.conv3(
            self.hd2_d0(d3)) + self.conv2(self.hd1_d0(d2))

        out = self.finaldeconv1(x_final)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)

class RNet(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(RNet, self).__init__()

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




class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention
        attn_output, _ = self.attention(x, x, x)
        out1 = self.norm1(x + self.dropout(attn_output))

        # Feed-forward
        ff_output = self.ff(out1)
        out2 = self.norm2(out1 + self.dropout(ff_output))
        return out2

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


class WaveFormer(nn.Module):
    def __init__(self, embed_dim):
        super(WaveFormer, self).__init__()
        # 小波变换的低通和高通滤波器
        self.low_pass = nn.Parameter(torch.tensor([[[[0.5, 0.5], [0.5, 0.5]]]]), requires_grad=False)
        self.high_pass = nn.Parameter(torch.tensor([[[[0.5, -0.5], [0.5, -0.5]]]]), requires_grad=False)

        # 卷积层用于对分解后的频率信息进行变换
        self.conv_low = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.conv_high = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)

    def forward(self, x):
        # 对输入进行小波变换分解
        low_freq, high_freq = self.dwt(x)

        # 对低频和高频成分分别进行卷积处理
        low_freq = self.conv_low(low_freq)
        high_freq = self.conv_high(high_freq)

        # 确保低频和高频特征图的分辨率是原始特征图的一半
        low_freq = F.interpolate(low_freq, size=(x.size(-2), x.size(-1)), mode='nearest')
        high_freq = F.interpolate(high_freq, size=(x.size(-2), x.size(-1)), mode='nearest')

        return low_freq, high_freq

    def dwt(self, x):
        # 对输入进行二维小波变换分解
        # 使用低通和高通滤波器
        b, c, h, w = x.shape

        # 扩展输入以适应滤波器
        x = F.pad(x, (1, 1, 1, 1), mode='reflect')

        # 低频成分
        low = F.conv2d(x, self.low_pass.repeat(c, 1, 1, 1), groups=c)
        low = low[:, :, ::2, ::2]

        # 高频成分
        high = F.conv2d(x, self.high_pass.repeat(c, 1, 1, 1), groups=c)
        high = high[:, :, ::2, ::2]

        return low, high
import torch.fft as fft  # 用于傅里叶变换
class FrequencyEnhancement(nn.Module):
    def __init__(self, embed_dim):
        super(FrequencyEnhancement, self).__init__()
        # 卷积层用于对频率信息进行变换
        self.conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)

    def forward(self, x):
        # 进行2D傅里叶变换，提取频率信息
        x_freq = fft.fft2(x, dim=(-2, -1))  # 对最后两个维度(H, W)进行2D傅里叶变换
        x_freq = torch.abs(x_freq)  # 取幅值
        x_freq = self.conv(x_freq)  # 对频域信息进行卷积处理

        return x_freq


class HorizontalTransformer1(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, num_layers):
        super(HorizontalTransformer1, self).__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)
        ])
        self.freq_enhancement = WaveFormer(embed_dim)
        self.fusion_conv = nn.Conv2d(embed_dim * 3, embed_dim, kernel_size=1)  # 用于融合的卷积层

    def forward(self, x):
        B, C, H, W = x.size()

        # 提取频率增强特征
        low_frequency, high_frequency = self.freq_enhancement(x)

        # 对输入进行维度变换，适应Transformer操作
        low_frequency = low_frequency.permute(0, 2, 3, 1).reshape(B * H, W, C)  # (B*H, W, C)
        for layer in self.layers:
            low_frequency = layer(low_frequency)
        low_frequency = low_frequency.reshape(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)

        # 对输入进行维度变换，适应Transformer操作
        high_frequency = high_frequency.permute(0, 2, 3, 1).reshape(B * H, W, C)  # (B*H, W, C)
        for layer in self.layers:
            high_frequency = layer(high_frequency)
        high_frequency = high_frequency.reshape(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)

        # 将频域特征与Transformer输出融合
        fused_features = torch.cat([x, low_frequency, high_frequency], dim=1)
        # 通过卷积层融合拼接后的特征
        x = self.fusion_conv(fused_features)
        return x


class VerticalTransformer1(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, num_layers):
        super(VerticalTransformer1, self).__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)
        ])
        self.freq_enhancement = WaveFormer(embed_dim)
        self.fusion_conv = nn.Conv2d(embed_dim * 3, embed_dim, kernel_size=1)  # 用于融合的卷积层

    def forward(self, x):
        B, C, H, W = x.size()

        # 提取频率增强特征
        low_frequency, high_frequency = self.freq_enhancement(x)

        # 对输入进行维度变换，适应Transformer操作
        low_frequency = low_frequency.permute(0, 3, 2, 1).reshape(B * W, H, C)  # (B*W, H, C)
        for layer in self.layers:
            low_frequency = layer(low_frequency)
        low_frequency = low_frequency.reshape(B, W, H, C).permute(0, 3, 2, 1)  # (B, C, H, W)
        # 对输入进行维度变换，适应Transformer操作
        high_frequency = high_frequency.permute(0, 3, 2, 1).reshape(B * W, H, C)  # (B*W, H, C)
        for layer in self.layers:
            high_frequency = layer(high_frequency)
        high_frequency = high_frequency.reshape(B, W, H, C).permute(0, 3, 2, 1)  # (B, C, H, W)
        # 将频域特征与Transformer输出融合
        fused_features = torch.cat([x, low_frequency, high_frequency], dim=1)
        # 通过卷积层融合拼接后的特征
        x = self.fusion_conv(fused_features)
        return x

class AxialContext(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(AxialContext, self).__init__()
        self.HTransformer = HorizontalTransformer1(embed_dim=embed_dim, num_heads=num_heads, ff_dim=embed_dim*2,  num_layers=1)
        self.VTransformer = VerticalTransformer1(embed_dim=embed_dim, num_heads=num_heads, ff_dim=embed_dim, num_layers=1)
        self.fusion_conv = nn.Conv2d(3 * embed_dim, embed_dim, kernel_size=1)

    def forward(self,x):

        Hcontext = self.HTransformer(x)
        Vcontext = self.VTransformer(x)
        fused_out = torch.cat([Hcontext, Vcontext, x], dim=1)
        fused_out = self.fusion_conv(fused_out)
        return fused_out

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

class ARRNet(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(ARRNet, self).__init__()

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
        #self.context =AxialContext( embed_dim=512, num_heads=8, num_points=4)
        self.context =AxialContext( embed_dim=512, num_heads=8)
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




class ARRNet_context(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(ARRNet_context, self).__init__()

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
        self.context = AxialContext( embed_dim=512, num_heads=8, num_points=4)

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
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2) + x

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)

        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)


class LAVT_context5(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(LAVT_context5, self).__init__()

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
        self.LG1 = LanguageGate(64)
        self.LG2 = LanguageGate(128)
        self.LG3 = LanguageGate(256)

        self.PWAM1 = AttentionModule(64, 512)
        self.PWAM2 = AttentionModule(128, 512)
        self.PWAM3 = AttentionModule(256, 512)
        self.PWAM4 = AttentionModule(512, 512)

        self.context = AxialContext(embed_dim=512, num_heads=8, num_points=4)
        self.RR4 = Relation_Refine(512, 256, 4)
        self.RR3 = Relation_Refine(256, 128, 3)
        self.RR2 = Relation_Refine(128, 64, 2)
        self.RR1 = Relation_Refine2(64, 64, 1)

        self.vocab_embedding = nn.Embedding(self.vocab_size, self.embed_dim)  # vocaburaly embedding
        self.position_encoding = PositionalEncoding(self.embed_dim)
        self.self_attn = nn.MultiheadAttention(512, 8, dropout=0.1)


        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])


        #self.Transdecoder4 =  TransformerModel(input_channels=512, embed_dim=512, num_heads=8, ff_dim=1024, num_layers=1)
        #self.Transdecoder3 =  TransformerModel(input_channels=256, embed_dim=256, num_heads=8, ff_dim=1024, num_layers=1)

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, image, text):
        tgt = text.permute(1, 0)
        tgt_length = tgt.size(0)

        mask = (torch.triu(torch.ones(tgt_length, tgt_length)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        mask = mask.to(device)

        tgt_embedding = self.vocab_embedding(tgt)
        tgt_embedding = self.position_encoding(tgt_embedding)  # (length, batch, feature_dim)

        instruct_feature = tgt_embedding + self.self_attn(tgt_embedding, tgt_embedding, tgt_embedding)[0]
        # instruct_feature = self.transformer(tgt_embedding, tgt_mask=mask)  # (length, batch, feature_dim)
        instruct_feature = instruct_feature.permute(1, 2, 0)  # 4  512 9

        # Encoder
        x = self.firstconv(image)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        e1 = self.firstmaxpool(x)
        e1 = self.encoder1(e1)
        e11 = self.LG1(self.PWAM1(e1,instruct_feature))
        e1 = e1+e11
        e2 = self.encoder2(e1)
        e22 = self.LG2(self.PWAM2(e2, instruct_feature))
        e2 = e2+e22
        e3 = self.encoder3(e2)
        e33 = self.LG3(self.PWAM3(e3, instruct_feature))
        e3 = e3+e33
        e4 = self.encoder4(e3)
        e4 = self.PWAM4(e4,instruct_feature)
        e4 = self.context(e4)

        # Decoder
        d4 = self.decoder4(e4) + self.RR4(e4, e3)
        d3 = self.decoder3(d4) + self.RR3(d4, e2)
        d2 = self.decoder2(d3) + self.RR2(d3, e1)
        d1 = self.decoder1(d2) + self.RR1(d2, x)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)

class LAVT_context6(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(LAVT_context6, self).__init__()

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


        self.Instruct1 = AttentionModule(64, 512)
        self.Instruct2 = AttentionModule(128, 512)
        self.Instruct3 = AttentionModule(256, 512)
        self.Instruct4 = AttentionModule(512, 512)

        self.context = AxialContext(embed_dim=512, num_heads=8, num_points=4)
        self.RR4 = Relation_Refine(512, 256, 4)
        self.RR3 = Relation_Refine(256, 128, 3)
        self.RR2 = Relation_Refine(128, 64, 2)
        self.RR1 = Relation_Refine2(64, 64, 1)

        self.vocab_embedding = nn.Embedding(self.vocab_size, self.embed_dim)  # vocaburaly embedding
        self.position_encoding = PositionalEncoding(self.embed_dim)
        self.self_attn = nn.MultiheadAttention(512, 8, dropout=0.1)


        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])


        #self.Transdecoder4 =  TransformerModel(input_channels=512, embed_dim=512, num_heads=8, ff_dim=1024, num_layers=1)
        #self.Transdecoder3 =  TransformerModel(input_channels=256, embed_dim=256, num_heads=8, ff_dim=1024, num_layers=1)

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, image, text):
        tgt = text.permute(1, 0)
        tgt_length = tgt.size(0)

        mask = (torch.triu(torch.ones(tgt_length, tgt_length)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        mask = mask.to(device)

        tgt_embedding = self.vocab_embedding(tgt)
        tgt_embedding = self.position_encoding(tgt_embedding)  # (length, batch, feature_dim)

        instruct_feature = tgt_embedding + self.self_attn(tgt_embedding, tgt_embedding, tgt_embedding)[0]
        # instruct_feature = self.transformer(tgt_embedding, tgt_mask=mask)  # (length, batch, feature_dim)
        instruct_feature = instruct_feature.permute(1, 2, 0)  # 4  512 9

        # Encoder
        x = self.firstconv(image)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        e1 = self.firstmaxpool(x)
        e1 = self.encoder1(e1)
        e11 = self.Instruct1(e1,instruct_feature)
        e1 = e1+e11
        e2 = self.encoder2(e1)
        e22 = self.Instruct2(e2, instruct_feature)
        e2 = e2+e22
        e3 = self.encoder3(e2)
        e33 = self.Instruct3(e3, instruct_feature)
        e3 = e3+e33
        e4 = self.encoder4(e3)
        e4 = self.Instruct4(e4,instruct_feature)
        e4 = self.context(e4)

        # Decoder
        d4 = self.decoder4(e4) + self.RR4(e4, e3)
        d3 = self.decoder3(d4) + self.RR3(d4, e2)
        d2 = self.decoder2(d3) + self.RR2(d3, e1)
        d1 = self.decoder1(d2) + self.RR1(d2, x)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)

class ARNet(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(ARNet, self).__init__()

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
        self.context =AxialContext( embed_dim=512, num_heads=8, num_points=4)
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



class Axial_Mamba_Context(nn.Module):
    def __init__(self, embed_dim):
        super(Axial_Mamba_Context, self).__init__()
        self.HorizontalMamba = HorizontalMamba(embed_dim=embed_dim, num_layers=1)
        self.VerticalMamba = VerticalMamba(embed_dim=embed_dim,  num_layers=1)
        self.fusion_conv = nn.Conv2d(3 * embed_dim, embed_dim, kernel_size=1)
    def forward(self,x):
        Hcontext = self.HorizontalMamba(x)
        Vcontext = self.VerticalMamba(x)
        fused_out = torch.cat([Hcontext, Vcontext, x], dim=1)
        fused_out = self.fusion_conv(fused_out)
        return fused_out


class FeatureEnhancer(nn.Module):
    def __init__(self, channels, kernel_size=3):
        """
        FeatureEnhancer模块用于高频特征增强，通过高通滤波器提取边缘和细节信息。

        参数:
            channels (int): 特征图的通道数
            kernel_size (int): 高通滤波器的卷积核尺寸，默认为3
        """
        super(FeatureEnhancer, self).__init__()
        self.channels = channels
        self.kernel_size = kernel_size

        # 初始化高通滤波卷积核
        self.high_pass_filter = nn.Parameter(self._create_high_pass_filter(), requires_grad=False)

    def _create_high_pass_filter(self):
        """
        创建高通滤波器卷积核。使用拉普拉斯算子，以提取高频信息。

        返回:
            torch.Tensor: 适配通道数的高通滤波卷积核
        """
        # 3x3 拉普拉斯高通滤波器
        base_filter = torch.tensor([[[[0, -1, 0],
                                      [-1, 4, -1],
                                      [0, -1, 0]]]], dtype=torch.float32)
        # 将过滤器扩展到每个通道独立使用
        high_pass_filter = base_filter.expand(self.channels, 1, self.kernel_size, self.kernel_size)
        return high_pass_filter

    def forward(self, x):
        """
        前向传播，应用高通滤波增强特征。

        参数:
            x (torch.Tensor): 输入特征图，尺寸为 (batch_size, channels, height, width)

        返回:
            torch.Tensor: 高频增强后的特征图
        """
        # 提取高频特征
        high_freq_features = F.conv2d(x, self.high_pass_filter, padding=1, groups=self.channels)

        # 增强特征：原始特征 + 高频特征
        enhanced_features = x + high_freq_features

        return enhanced_features

class Cross_Context(nn.Module):
    def __init__(self, embed_dim=512,num_layers=1):
        super(Cross_Context, self).__init__()
        self.Low_layers = nn.ModuleList([
            SS2D(d_model=embed_dim, dropout=0.1, d_state=16) for _ in range(num_layers)
        ])
        self.high_layers = nn.ModuleList([
            SS2D(d_model=embed_dim, dropout=0.1, d_state=16) for _ in range(num_layers)
        ])
        self.fusion_conv = nn.Conv2d(3 * embed_dim, embed_dim, kernel_size=1)
        self.avg =nn.AdaptiveAvgPool2d((8,8))
        self.up = nn.Upsample(scale_factor=4,  mode='nearest')
        self.high_enhance = FeatureEnhancer(channels=512)


    def forward(self, feature_map):

        """
        输入是一个特征图列表，每个特征图切片成 `32x32` 的子特征图。
        """
        low_frequency = self.up(self.avg(feature_map))
        B1, C1, H1, W1 = low_frequency.size()
        x = low_frequency.permute(0, 3, 2, 1)
        for layer in self.Low_layers:
            low = layer(x)
        low_feature = low.reshape(B1, H1, W1, C1).permute(0, 3, 2, 1)  # (B, C, H, W)
       

        B2, C2, H2, W2 = feature_map.size()
        feature_map = self.high_enhance(feature_map)
        high = feature_map.permute(0, 3, 2, 1)
        for layer in self.high_layers:
            high = layer(high)
        high_feature = high.reshape(B2, H2, W2, C2).permute(0, 3, 2, 1)  # (B, C, H, W)
        context = self.fusion_conv(torch.cat([low_feature,high_feature,feature_map],dim=1))
        return context


class Mamba_Context(nn.Module):
    def __init__(self, embed_dim=512, num_layers=1):
        super(Mamba_Context, self).__init__()

        self.high_layers = nn.ModuleList([
            SS2D(d_model=embed_dim, dropout=0.1, d_state=16) for _ in range(num_layers)
        ])
        self.fusion_conv = nn.Conv2d(2 * embed_dim, embed_dim, kernel_size=1)
        self.avg = nn.AdaptiveAvgPool2d((8, 8))
        self.up = nn.Upsample(scale_factor=4, mode='nearest')
        self.high_enhance = FeatureEnhancer(channels=512)

    def forward(self, feature_map):

        B, C, H, W = feature_map.size()
        feature_map = self.high_enhance(feature_map)
        high = feature_map.permute(0, 3, 2, 1)
        for layer in self.high_layers:
            high = layer(high)
        high_feature = high.reshape(B, H, W, C).permute(0, 3, 2, 1)  # (B, C, H, W)
        context = self.fusion_conv(torch.cat([ high_feature, feature_map], dim=1))
        return context


class MRNet(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(MRNet, self).__init__()

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
        self.context = Mamba_Context( embed_dim=512)
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

        e4= self.context(e4)

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



class LAVT_context7(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(LAVT_context7, self).__init__()

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

        self.Instruct1 = AttentionModule(64, 512)
        self.Instruct2 = AttentionModule(128, 512)
        self.Instruct3 = AttentionModule(256, 512)
        self.Instruct4 = AttentionModule(512, 512)

        self.context = Cross_ViewAtt( in_channels=512, num_heads=8)

        self.vocab_embedding = nn.Embedding(self.vocab_size, self.embed_dim)  # vocaburaly embedding
        self.position_encoding = PositionalEncoding(self.embed_dim)
        self.self_attn = nn.MultiheadAttention(512, 8, dropout=0.1)


        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])


        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, image, text):
        tgt = text.permute(1, 0)
        tgt_length = tgt.size(0)

        mask = (torch.triu(torch.ones(tgt_length, tgt_length)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        mask = mask.to(device)

        tgt_embedding = self.vocab_embedding(tgt)
        tgt_embedding = self.position_encoding(tgt_embedding)  # (length, batch, feature_dim)

        instruct_feature = tgt_embedding + self.self_attn(tgt_embedding, tgt_embedding, tgt_embedding)[0]
        # instruct_feature = self.transformer(tgt_embedding, tgt_mask=mask)  # (length, batch, feature_dim)
        instruct_feature = instruct_feature.permute(1, 2, 0)  # 4  512 9

        # Encoder
        x = self.firstconv(image)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        e1 = self.firstmaxpool(x)
        e1 = self.encoder1(e1)
        e11 = self.Instruct1(e1,instruct_feature)
        e1 = e1+e11
        e2 = self.encoder2(e1)
        e22 = self.Instruct2(e2, instruct_feature)
        e2 = e2+e22
        e3 = self.encoder3(e2)
        e33 = self.Instruct3(e3, instruct_feature)
        e3 = e3+e33
        e4 = self.encoder4(e3)
        e4 = self.Instruct4(e4,instruct_feature)
        e4 = self.context(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2) + x

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)


class GaussianUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, scale_factor=2):
        super(GaussianUpsample, self).__init__()
        self.scale_factor = scale_factor
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)

        # 初始化高斯核参数
        self.sigma_x = nn.Parameter(torch.ones(1))  # 可学习的 x 方向标准差
        self.sigma_y = nn.Parameter(torch.ones(1))  # 可学习的 y 方向标准差
        self.opacity = nn.Parameter(torch.ones(1))  # 可学习的透明度

    def forward(self, x):
        batch_size, channels, height, width = x.shape

        # 生成高斯核
        kernel = self._generate_gaussian_kernel()
        kernel = kernel.to(x.device)  # 确保在同一个设备上

        # 上采样
        upsampled = F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')

        # 逐通道应用高斯核
        gaussian_out = []
        for i in range(channels):
            channel_out = F.conv2d(upsampled[:, i:i + 1, :, :], kernel, padding=self.kernel_size // 2)
            gaussian_out.append(channel_out)

        # 将所有通道的结果拼接回一个张量
        gaussian_out = torch.cat(gaussian_out, dim=1)

        # 应用卷积
        out = self.conv(gaussian_out)
        return out

    def _generate_gaussian_kernel(self):
        """生成高斯核"""
        # 确保所有张量都在与输入张量相同的设备上
        device = self.sigma_x.device  # 获取参数所在的设备

        ax = torch.arange(-self.kernel_size // 2 + 1, self.kernel_size // 2 + 1, device=device).float()
        xx, yy = torch.meshgrid(ax, ax)
        kernel = torch.exp(-(xx ** 2 / (2 * self.sigma_x ** 2) + (yy ** 2 / (2 * self.sigma_y ** 2))))
        kernel = kernel / (2 * torch.pi * self.sigma_x * self.sigma_y)  # 归一化
        kernel = kernel * self.opacity  # 应用透明度
        kernel = kernel.view(1, 1, self.kernel_size, self.kernel_size)  # 调整形状
        return kernel


class MRRNet8(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(MRRNet8, self).__init__()

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
        self.context = AxialContext(embed_dim=512, num_heads=8)


        self.decoder4 = DecoderBlock(filters[3] // 2, filters[2] // 2)
        self.gaussian_upsample4 = GaussianUpsample(filters[3] // 2, filters[2] // 2, kernel_size=3, scale_factor=2)
        self.decoder3 = DecoderBlock(filters[2] // 2, filters[1] // 2)
        self.gaussian_upsample3 = GaussianUpsample(filters[2] // 2, filters[1] // 2, kernel_size=3, scale_factor=2)
        self.decoder2 = DecoderBlock(filters[1] // 2, filters[0] // 2)
        self.gaussian_upsample2 = GaussianUpsample(filters[1] // 2, filters[0] // 2, kernel_size=3, scale_factor=2)
        self.decoder1 = DecoderBlock(filters[0] // 2, filters[0] // 2)
        self.gaussian_upsample1 = GaussianUpsample(filters[0] // 2, filters[0] // 2, kernel_size=3, scale_factor=2)

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)
    def split_and_upsample(self, x, gaussian_upsample, decoder_block, skip_connection):
        """
        将特征图通道拆分，分别进行高斯核上采样和普通解码器操作，然后拼接。
        :param x: 输入特征图
        :param gaussian_upsample: 高斯核上采样模块
        :param decoder_block: 普通解码器模块
        :param skip_connection: 跳跃连接特征图
        :return: 拼接后的特征图
        """
        C = x.size(1)
        x_gaussian = x[:, :C // 2, :, :]  # 高斯核部分
        x_decoder = x[:, C // 2:, :, :]  # 普通解码器部分

        # 高斯核上采样
        gaussian_out = gaussian_upsample(x_gaussian)
        # 普通解码器操作
        decoder_out = decoder_block(x_decoder)

        # 拼接通道
        out = torch.cat([gaussian_out, decoder_out], dim=1)
        return out + skip_connection  # 加上跳跃连接

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
        d4 = self.split_and_upsample(e4, self.gaussian_upsample4, self.decoder4, e3)
        # Decoder3
        d3 = self.split_and_upsample(d4, self.gaussian_upsample3, self.decoder3, e2)
        # Decoder2
        d2 = self.split_and_upsample(d3, self.gaussian_upsample2, self.decoder2, e1)
        # Decoder1
        d1 = self.split_and_upsample(d2, self.gaussian_upsample1, self.decoder1, x)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)
        return F.sigmoid(out)



class LAVT_context8(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(LAVT_context8, self).__init__()

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

        self.Instruct1 = AttentionModule(64, 512)
        self.Instruct2 = AttentionModule(128, 512)
        self.Instruct3 = AttentionModule(256, 512)
        self.Instruct4 = AttentionModule(512, 512)

        self.context = AxialContext(embed_dim=512, num_heads=8)

        self.vocab_embedding = nn.Embedding(self.vocab_size, self.embed_dim)  # vocaburaly embedding
        self.position_encoding = PositionalEncoding(self.embed_dim)
        self.self_attn = nn.MultiheadAttention(512, 8, dropout=0.1)


        self.decoder4 = DecoderBlock(filters[3]//2, filters[2]//2)
        self.gaussian_upsample4 = GaussianUpsample(filters[3]//2, filters[2]//2, kernel_size=3, scale_factor=2)
        self.decoder3 = DecoderBlock(filters[2]//2, filters[1]//2)
        self.gaussian_upsample3 = GaussianUpsample(filters[2]//2, filters[1]//2, kernel_size=3, scale_factor=2)
        self.decoder2 = DecoderBlock(filters[1]//2, filters[0]//2)
        self.gaussian_upsample2 = GaussianUpsample(filters[1]//2, filters[0]//2, kernel_size=3, scale_factor=2)
        self.decoder1 = DecoderBlock(filters[0]//2, filters[0]//2)
        self.gaussian_upsample1 = GaussianUpsample(filters[0]//2, filters[0]//2, kernel_size=3, scale_factor=2)


        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, image, text):
        tgt = text.permute(1, 0)
        tgt_length = tgt.size(0)

        mask = (torch.triu(torch.ones(tgt_length, tgt_length)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        mask = mask.to(device)

        tgt_embedding = self.vocab_embedding(tgt)
        tgt_embedding = self.position_encoding(tgt_embedding)  # (length, batch, feature_dim)

        instruct_feature = tgt_embedding + self.self_attn(tgt_embedding, tgt_embedding, tgt_embedding)[0]
        # instruct_feature = self.transformer(tgt_embedding, tgt_mask=mask)  # (length, batch, feature_dim)
        instruct_feature = instruct_feature.permute(1, 2, 0)  # 4  512 9

        # Encoder
        x = self.firstconv(image)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        e1 = self.firstmaxpool(x)
        e1 = self.encoder1(e1)
        e11 = self.Instruct1(e1,instruct_feature)
        e1 = e1+e11
        e2 = self.encoder2(e1)
        e22 = self.Instruct2(e2, instruct_feature)
        e2 = e2+e22
        e3 = self.encoder3(e2)
        e33 = self.Instruct3(e3, instruct_feature)
        e3 = e3+e33
        e4 = self.encoder4(e3)
        e4 = self.Instruct4(e4,instruct_feature)
        e4 = self.context(e4)

        # Decoder
        d4 = self.split_and_upsample(e4, self.gaussian_upsample4, self.decoder4, e3)
        # Decoder3
        d3 = self.split_and_upsample(d4, self.gaussian_upsample3, self.decoder3, e2)
        # Decoder2
        d2 = self.split_and_upsample(d3, self.gaussian_upsample2, self.decoder2, e1)
        # Decoder1
        d1 = self.split_and_upsample(d2, self.gaussian_upsample1, self.decoder1, x)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)

    def split_and_upsample(self, x, gaussian_upsample, decoder_block, skip_connection):
        """
        将特征图通道拆分，分别进行高斯核上采样和普通解码器操作，然后拼接。
        :param x: 输入特征图
        :param gaussian_upsample: 高斯核上采样模块
        :param decoder_block: 普通解码器模块
        :param skip_connection: 跳跃连接特征图
        :return: 拼接后的特征图
        """
        C = x.size(1)
        x_gaussian = x[:, :C // 2, :, :]  # 高斯核部分
        x_decoder = x[:, C // 2:, :, :]  # 普通解码器部分

        # 高斯核上采样
        gaussian_out = gaussian_upsample(x_gaussian)
        # 普通解码器操作
        decoder_out = decoder_block(x_decoder)

        # 拼接通道
        out = torch.cat([gaussian_out, decoder_out], dim=1)
        return out + skip_connection  # 加上跳跃连接



class Instruct_DLinkNet(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(Instruct_DLinkNet, self).__init__()

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


        self.Instruct1 = VisionAttention(64, 512)
        self.Instruct2 = VisionAttention(128, 512)
        self.Instruct3 = VisionAttention(256, 512)
        self.Instruct4 = VisionAttention(512, 512)

        self.context =  Dblock(512)

        self.vocab_embedding = nn.Embedding(self.vocab_size, self.embed_dim)  # vocaburaly embedding
        self.position_encoding = PositionalEncoding(self.embed_dim)
        self.self_attn = nn.MultiheadAttention(512, 8, dropout=0.1)


        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])


        #self.Transdecoder4 =  TransformerModel(input_channels=512, embed_dim=512, num_heads=8, ff_dim=1024, num_layers=1)
        #self.Transdecoder3 =  TransformerModel(input_channels=256, embed_dim=256, num_heads=8, ff_dim=1024, num_layers=1)

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, image, text):
        tgt = text.permute(1, 0)
        tgt_length = tgt.size(0)

        mask = (torch.triu(torch.ones(tgt_length, tgt_length)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        mask = mask.to(device)

        tgt_embedding = self.vocab_embedding(tgt)
        tgt_embedding = self.position_encoding(tgt_embedding)  # (length, batch, feature_dim)

        instruct_feature = tgt_embedding + self.self_attn(tgt_embedding, tgt_embedding, tgt_embedding)[0]
        # instruct_feature = self.transformer(tgt_embedding, tgt_mask=mask)  # (length, batch, feature_dim)
        instruct_feature = instruct_feature.permute(1, 2, 0)  # 4  512 9

        # Encoder
        x = self.firstconv(image)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        e1 = self.firstmaxpool(x)
        e1 = self.encoder1(e1)

        e11 = self.Instruct1(e1,instruct_feature)
        e1 = e1+e11
        e2 = self.encoder2(e1)
        e22 = self.Instruct2(e2, instruct_feature)
        e2 = e2+e22
        e3 = self.encoder3(e2)
        e33 = self.Instruct3(e3, instruct_feature)
        e3 = e3+e33
        e4 = self.encoder4(e3)
        e4 = self.context(e4)
        e44 = self.Instruct4(e4,instruct_feature)+e4


        # Decoder
        d4 = self.decoder4(e44) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2) + x

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)



class Instruct_RBNet(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(Instruct_RBNet, self).__init__()

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


        self.Instruct1 = VisionAttention(64, 512)
        self.Instruct2 = VisionAttention(128, 512)
        self.Instruct3 = VisionAttention(256, 512)
        self.Instruct4 = VisionAttention(512, 512)

        self.context = Cross_MultiViewAtt( in_channels=512, num_heads=8)
        self.RR4 = Relation_Refine(512, 256, 4)
        self.RR3 = Relation_Refine(256, 128, 3)
        self.RR2 = Relation_Refine(128, 64, 2)
        self.RR1 = Relation_Refine2(64, 64, 1)

        self.vocab_embedding = nn.Embedding(self.vocab_size, self.embed_dim)  # vocaburaly embedding
        self.position_encoding = PositionalEncoding(self.embed_dim)
        self.self_attn = nn.MultiheadAttention(512, 8, dropout=0.1)


        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])


        #self.Transdecoder4 =  TransformerModel(input_channels=512, embed_dim=512, num_heads=8, ff_dim=1024, num_layers=1)
        #self.Transdecoder3 =  TransformerModel(input_channels=256, embed_dim=256, num_heads=8, ff_dim=1024, num_layers=1)

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, image, text):
        tgt = text.permute(1, 0)
        tgt_length = tgt.size(0)

        mask = (torch.triu(torch.ones(tgt_length, tgt_length)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        mask = mask.to(device)

        tgt_embedding = self.vocab_embedding(tgt)
        tgt_embedding = self.position_encoding(tgt_embedding)  # (length, batch, feature_dim)

        instruct_feature = tgt_embedding + self.self_attn(tgt_embedding, tgt_embedding, tgt_embedding)[0]
        # instruct_feature = self.transformer(tgt_embedding, tgt_mask=mask)  # (length, batch, feature_dim)
        instruct_feature = instruct_feature.permute(1, 2, 0)  # 4  512 9

        # Encoder
        x = self.firstconv(image)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        e1 = self.firstmaxpool(x)
        e1 = self.encoder1(e1)

        e11 = self.Instruct1(e1,instruct_feature)
        e1 = e1+e11
        e2 = self.encoder2(e1)
        e22 = self.Instruct2(e2, instruct_feature)
        e2 = e2+e22
        e3 = self.encoder3(e2)
        e33 = self.Instruct3(e3, instruct_feature)
        e3 = e3+e33
        e4 = self.encoder4(e3)
        e4 = self.context(e4)
        e44 = self.Instruct4(e4,instruct_feature)+e4


        # Decoder
        d4 = self.decoder4(e44) + self.RR4(e4, e3)
        d3 = self.decoder3(d4) + self.RR3(d4, e2)
        d2 = self.decoder2(d3) + self.RR2(d3, e1)
        d1 = self.decoder1(d2) + self.RR1(d2, x)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)

class MRRNet(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(MRRNet, self).__init__()

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
        self.context =Cross_MultiViewAtt( in_channels=512, num_heads=8)
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
        self.context =Cross_ViewAtt( in_channels=512, num_heads=8)
        self.RR4 = Relation_Refine(512, 256,4)
        self.RR3 = Relation_Refine(256, 128,3)
        self.RR2 = Relation_Refine(128, 64,2)
        self.RR1 = Relation_Refine2(64, 64,1)
        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.conv5 = nn.Conv2d(in_channels=filters[3], out_channels=filters[0],
                               kernel_size=3,
                               stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=filters[2], out_channels=filters[0],
                               kernel_size=3,
                               stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=filters[1], out_channels=filters[0],
                               kernel_size=3,
                               stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=filters[0], out_channels=filters[0],
                               kernel_size=3,
                               stride=1, padding=1)

        self.hd4_d0 = nn.Upsample(scale_factor=16)
        self.hd3_d0 = nn.Upsample(scale_factor=8)
        self.hd2_d0 = nn.Upsample(scale_factor=4)
        self.hd1_d0 = nn.Upsample(scale_factor=2)

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
        x_final = d1 + self.conv5(self.hd4_d0(e4)) + self.conv4(self.hd3_d0(d4)) + self.conv3(
            self.hd2_d0(d3)) + self.conv2(self.hd1_d0(d2))

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)

        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)



class CRNet_2(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(CRNet_2, self).__init__()

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
        self.context =Cross_ViewAtt( in_channels=512, num_heads=2)
        self.RR4 = Relation_Refine(512, 256,4)
        self.RR3 = Relation_Refine(256, 128,3)
        self.RR2 = Relation_Refine(128, 64,2)
        self.RR1 = Relation_Refine2(64, 64,1)
        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.conv5 = nn.Conv2d(in_channels=filters[3], out_channels=filters[0],
                               kernel_size=3,
                               stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=filters[2], out_channels=filters[0],
                               kernel_size=3,
                               stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=filters[1], out_channels=filters[0],
                               kernel_size=3,
                               stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=filters[0], out_channels=filters[0],
                               kernel_size=3,
                               stride=1, padding=1)

        self.hd4_d0 = nn.Upsample(scale_factor=16)
        self.hd3_d0 = nn.Upsample(scale_factor=8)
        self.hd2_d0 = nn.Upsample(scale_factor=4)
        self.hd1_d0 = nn.Upsample(scale_factor=2)

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
        x_final = d1 + self.conv5(self.hd4_d0(e4)) + self.conv4(self.hd3_d0(d4)) + self.conv3(
            self.hd2_d0(d3)) + self.conv2(self.hd1_d0(d2))

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)

        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)


class CRNet_4(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(CRNet_4, self).__init__()

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
        self.context =Cross_ViewAtt( in_channels=512, num_heads=4)
        self.RR4 = Relation_Refine(512, 256,4)
        self.RR3 = Relation_Refine(256, 128,3)
        self.RR2 = Relation_Refine(128, 64,2)
        self.RR1 = Relation_Refine2(64, 64,1)
        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.conv5 = nn.Conv2d(in_channels=filters[3], out_channels=filters[0],
                               kernel_size=3,
                               stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=filters[2], out_channels=filters[0],
                               kernel_size=3,
                               stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=filters[1], out_channels=filters[0],
                               kernel_size=3,
                               stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=filters[0], out_channels=filters[0],
                               kernel_size=3,
                               stride=1, padding=1)

        self.hd4_d0 = nn.Upsample(scale_factor=16)
        self.hd3_d0 = nn.Upsample(scale_factor=8)
        self.hd2_d0 = nn.Upsample(scale_factor=4)
        self.hd1_d0 = nn.Upsample(scale_factor=2)

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
        x_final = d1 + self.conv5(self.hd4_d0(e4)) + self.conv4(self.hd3_d0(d4)) + self.conv3(
            self.hd2_d0(d3)) + self.conv2(self.hd1_d0(d2))

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)

        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)

class Frequency_CRNet(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(Frequency_CRNet, self).__init__()
        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=False)
        resnet.load_state_dict(torch.load('./networks/resnet34.pth'))

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # Frequency domain branch
        self.fusion_conv1 = GatedFeatureFusion(in_channels=64, reduction=16)
        self.fusion_conv2 = GatedFeatureFusion(in_channels=128, reduction=16)
        self.fusion_conv3 = GatedFeatureFusion(in_channels=256, reduction=16)
        self.fusion_conv4 = GatedFeatureFusion(in_channels=512, reduction=16)

        self.context = Cross_ViewAtt1(in_channels=512, num_heads=8)
        self.RR4 = Relation_Refine(512, 256, 4)
        self.RR3 = Relation_Refine(256, 128, 3)
        self.RR2 = Relation_Refine(128, 64, 2)
        self.RR1 = Relation_Refine2(64, 64, 1)

        # Decoders with Frequency Sub-branches
        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])


        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def frequency_branch1(self, x):
        """
        使用 Sobel 算子提取高频和低频
        Args:
            x: 输入张量 (B, C, H, W)
        Returns:
            high_freq: 高频分量 (边缘)
            low_freq: 低频分量 (平滑)
        """
        # 确保输入张量在正确的设备上
        device = x.device

        # Sobel 卷积核
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device).view(1, 1, 3,
                                                                                                              3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=device).view(1, 1, 3,
                                                                                                              3)
        sobel_x = sobel_x.repeat(x.shape[1], 1, 1, 1)  # 每个通道共享
        sobel_y = sobel_y.repeat(x.shape[1], 1, 1, 1)

        # Apply Sobel 算子
        grad_x = F.conv2d(x, sobel_x, padding=1, groups=x.shape[1])
        grad_y = F.conv2d(x, sobel_y, padding=1, groups=x.shape[1])

        # 增加 eps 防止数值问题
        eps = 1e-6
        high_freq = torch.sqrt(grad_x ** 2 + grad_y ** 2 + eps)  # 边缘强度

        # 通过平滑滤波获取低频分量
        low_freq = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)

        return high_freq, low_freq

    def frequency_branch2(self, x):
        """
        使用 Sobel 算子提取高频和低频
        Args:
            x: 输入张量 (B, C, H, W)
        Returns:
            high_freq: 高频分量 (边缘)
            low_freq: 低频分量 (平滑)
        """
        # 确保输入张量在正确的设备上
        device = x.device

        # Sobel 卷积核
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device).view(1, 1, 3,
                                                                                                              3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=device).view(1, 1, 3,
                                                                                                              3)
        sobel_x = sobel_x.repeat(x.shape[1], 1, 1, 1)  # 每个通道共享
        sobel_y = sobel_y.repeat(x.shape[1], 1, 1, 1)

        # Apply Sobel 算子
        grad_x = F.conv2d(x, sobel_x, padding=1, groups=x.shape[1])
        grad_y = F.conv2d(x, sobel_y, padding=1, groups=x.shape[1])

        # 增加 eps 防止数值问题
        eps = 1e-6
        high_freq = torch.sqrt(grad_x ** 2 + grad_y ** 2 + eps)  # 边缘强度

        # 通过平滑滤波获取低频分量
        low_freq = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)

        return high_freq, low_freq

    def frequency_branch3(self, x):
        """
        使用 Sobel 算子提取高频和低频
        Args:
            x: 输入张量 (B, C, H, W)
        Returns:
            high_freq: 高频分量 (边缘)
            low_freq: 低频分量 (平滑)
        """
        # 确保输入张量在正确的设备上
        device = x.device

        # Sobel 卷积核
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device).view(1, 1, 3,
                                                                                                              3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=device).view(1, 1, 3,
                                                                                                              3)
        sobel_x = sobel_x.repeat(x.shape[1], 1, 1, 1)  # 每个通道共享
        sobel_y = sobel_y.repeat(x.shape[1], 1, 1, 1)

        # Apply Sobel 算子
        grad_x = F.conv2d(x, sobel_x, padding=1, groups=x.shape[1])
        grad_y = F.conv2d(x, sobel_y, padding=1, groups=x.shape[1])

        # 增加 eps 防止数值问题
        eps = 1e-6
        high_freq = torch.sqrt(grad_x ** 2 + grad_y ** 2 + eps)  # 边缘强度

        # 通过平滑滤波获取低频分量
        low_freq = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)

        return high_freq, low_freq

    def frequency_branch4(self, x):
        """
        使用 Sobel 算子提取高频和低频
        Args:
            x: 输入张量 (B, C, H, W)
        Returns:
            high_freq: 高频分量 (边缘)
            low_freq: 低频分量 (平滑)
        """
        # 确保输入张量在正确的设备上
        device = x.device

        # Sobel 卷积核
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device).view(1, 1, 3,
                                                                                                              3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=device).view(1, 1, 3,
                                                                                                              3)
        sobel_x = sobel_x.repeat(x.shape[1], 1, 1, 1)  # 每个通道共享
        sobel_y = sobel_y.repeat(x.shape[1], 1, 1, 1)

        # Apply Sobel 算子
        grad_x = F.conv2d(x, sobel_x, padding=1, groups=x.shape[1])
        grad_y = F.conv2d(x, sobel_y, padding=1, groups=x.shape[1])

        # 增加 eps 防止数值问题
        eps = 1e-6
        high_freq = torch.sqrt(grad_x ** 2 + grad_y ** 2 + eps)  # 边缘强度

        # 通过平滑滤波获取低频分量
        low_freq = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)

        return high_freq, low_freq

    def forward(self, image):
        # Frequency branch

        # Spatial encoder
        x = self.firstconv(image)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        e1 = self.firstmaxpool(x)
        e1 = self.encoder1(e1)
        high_freq1, low_freq1= self.frequency_branch1(e1)
        e1 = self.fusion_conv1(e1,high_freq1,low_freq1)

        e2 = self.encoder2(e1)
        high_freq2, low_freq2 = self.frequency_branch2(e2)
        e2 = self.fusion_conv2(e2, high_freq2, low_freq2)

        e3 = self.encoder3(e2)
        high_freq3, low_freq3 = self.frequency_branch3(e3)
        e3 = self.fusion_conv3(e3, high_freq3, low_freq3)

        e4 = self.encoder4(e3)
        high_freq4, low_freq4 = self.frequency_branch4(e4)
        e4 = self.fusion_conv4(self.context(e4), high_freq4, low_freq4)

        # Decoder
        d4 = self.decoder4(e4) + self.RR4(e4, e3)
        d3 = self.decoder3(d4) + self.RR3(d4, e2)
        d2 = self.decoder2(d3) + self.RR2(d3, e1)
        d1 = self.decoder1(d2) + self.RR1(d2, x)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)

        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)


class CRNet_context(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(CRNet_context, self).__init__()

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
        #self.context =Cross_ViewAtt1( in_channels=512, num_heads=8)
        self.RR4 = Relation_Refine(512, 256,4)
        self.RR3 = Relation_Refine(256, 128,3)
        self.RR2 = Relation_Refine(128, 64,2)
        self.RR1 = Relation_Refine2(64, 64,1)
        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.conv5 = nn.Conv2d(in_channels=filters[3], out_channels=filters[0],
                               kernel_size=3,
                               stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=filters[2], out_channels=filters[0],
                               kernel_size=3,
                               stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=filters[1], out_channels=filters[0],
                               kernel_size=3,
                               stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=filters[0], out_channels=filters[0],
                               kernel_size=3,
                               stride=1, padding=1)

        self.hd4_d0 = nn.Upsample(scale_factor=16)
        self.hd3_d0 = nn.Upsample(scale_factor=8)
        self.hd2_d0 = nn.Upsample(scale_factor=4)
        self.hd1_d0 = nn.Upsample(scale_factor=2)

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
        #e4 = self.context(e4)

        # Decoder
        d4 = self.decoder4(e4) + self.RR4(e4, e3)
        d3 = self.decoder3(d4) + self.RR3(d4, e2)
        d2 = self.decoder2(d3) + self.RR2(d3,e1)
        d1 = self.decoder1(d2) + self.RR1(d2,x)
        x_final = d1 + self.conv5(self.hd4_d0(e4)) + self.conv4(self.hd3_d0(d4)) + self.conv3(
            self.hd2_d0(d3)) + self.conv2(self.hd1_d0(d2))

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)

        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)


class CRNet_RR(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(CRNet_RR, self).__init__()

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
        #self.RR4 = Relation_Refine(512, 256,4)
        #self.RR3 = Relation_Refine(256, 128,3)
        #self.RR2 = Relation_Refine(128, 64,2)
        #self.RR1 = Relation_Refine2(64, 64,1)
        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.conv5 = nn.Conv2d(in_channels=filters[3], out_channels=filters[0],
                               kernel_size=3,
                               stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=filters[2], out_channels=filters[0],
                               kernel_size=3,
                               stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=filters[1], out_channels=filters[0],
                               kernel_size=3,
                               stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=filters[0], out_channels=filters[0],
                               kernel_size=3,
                               stride=1, padding=1)

        self.hd4_d0 = nn.Upsample(scale_factor=16)
        self.hd3_d0 = nn.Upsample(scale_factor=8)
        self.hd2_d0 = nn.Upsample(scale_factor=4)
        self.hd1_d0 = nn.Upsample(scale_factor=2)

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
        #e4 = self.context(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2) + x
        x_final = d1 + self.conv5(self.hd4_d0(e4)) + self.conv4(self.hd3_d0(d4)) + self.conv3(
            self.hd2_d0(d3)) + self.conv2(self.hd1_d0(d2))

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)

        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)


class Instruct_CRNet(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(Instruct_CRNet, self).__init__()

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


        self.Instruct1 = VisionAttention(64, 512)
        self.Instruct2 = VisionAttention(128, 512)
        self.Instruct3 = VisionAttention(256, 512)
        self.Instruct4 = VisionAttention(512, 512)

        self.context = Cross_ViewAtt(in_channels=512, num_heads=8)
        self.RR4 = Relation_Refine(512, 256, 4)
        self.RR3 = Relation_Refine(256, 128, 3)
        self.RR2 = Relation_Refine(128, 64, 2)
        self.RR1 = Relation_Refine2(64, 64, 1)

        self.vocab_embedding = nn.Embedding(self.vocab_size, self.embed_dim)  # vocaburaly embedding
        self.position_encoding = PositionalEncoding(self.embed_dim)
        self.self_attn = nn.MultiheadAttention(512, 8, dropout=0.1)


        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])


        #self.Transdecoder4 =  TransformerModel(input_channels=512, embed_dim=512, num_heads=8, ff_dim=1024, num_layers=1)
        #self.Transdecoder3 =  TransformerModel(input_channels=256, embed_dim=256, num_heads=8, ff_dim=1024, num_layers=1)

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, image, text):
        tgt = text.permute(1, 0)
        tgt_length = tgt.size(0)

        mask = (torch.triu(torch.ones(tgt_length, tgt_length)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        mask = mask.to(device)

        tgt_embedding = self.vocab_embedding(tgt)
        tgt_embedding = self.position_encoding(tgt_embedding)  # (length, batch, feature_dim)

        instruct_feature = tgt_embedding + self.self_attn(tgt_embedding, tgt_embedding, tgt_embedding)[0]
        # instruct_feature = self.transformer(tgt_embedding, tgt_mask=mask)  # (length, batch, feature_dim)
        instruct_feature = instruct_feature.permute(1, 2, 0)  # 4  512 9

        # Encoder
        x = self.firstconv(image)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        e1 = self.firstmaxpool(x)
        e1 = self.encoder1(e1)

        e11 = self.Instruct1(e1,instruct_feature)
        e1 = e1+e11
        e2 = self.encoder2(e1)
        e22 = self.Instruct2(e2, instruct_feature)
        e2 = e2+e22
        e3 = self.encoder3(e2)
        e33 = self.Instruct3(e3, instruct_feature)
        e3 = e3+e33
        e4 = self.encoder4(e3)
        e4 = self.context(e4)
        e44 = self.Instruct4(e4,instruct_feature)+e4


        # Decoder
        d4 = self.decoder4(e44) + self.RR4(e4, e3)
        d3 = self.decoder3(d4) + self.RR3(d4, e2)
        d2 = self.decoder2(d3) + self.RR2(d3, e1)
        d1 = self.decoder1(d2) + self.RR1(d2, x)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)



class Instruct_CRNet_a(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(Instruct_CRNet_a, self).__init__()

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


        self.Instruct1 = VisionAttention(64, 512)
        self.Instruct2 = VisionAttention(128, 512)
        self.Instruct3 = VisionAttention(256, 512)
        self.Instruct4 = VisionAttention(512, 512)

        self.context = Cross_ViewAtt1(in_channels=512, num_heads=8)
        self.RR4 = Relation_Refine(512, 256, 4)
        self.RR3 = Relation_Refine(256, 128, 3)
        self.RR2 = Relation_Refine(128, 64, 2)
        self.RR1 = Relation_Refine2(64, 64, 1)

        self.vocab_embedding = nn.Embedding(self.vocab_size, self.embed_dim)  # vocaburaly embedding
        self.position_encoding = PositionalEncoding(self.embed_dim)
        self.self_attn = nn.MultiheadAttention(512, 8, dropout=0.1)


        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])


        #self.Transdecoder4 =  TransformerModel(input_channels=512, embed_dim=512, num_heads=8, ff_dim=1024, num_layers=1)
        #self.Transdecoder3 =  TransformerModel(input_channels=256, embed_dim=256, num_heads=8, ff_dim=1024, num_layers=1)

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, image, text):
        tgt = text.permute(1, 0)
        tgt_length = tgt.size(0)

        mask = (torch.triu(torch.ones(tgt_length, tgt_length)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        mask = mask.to(device)

        tgt_embedding = self.vocab_embedding(tgt)
        tgt_embedding = self.position_encoding(tgt_embedding)  # (length, batch, feature_dim)

        instruct_feature = tgt_embedding + self.self_attn(tgt_embedding, tgt_embedding, tgt_embedding)[0]
        # instruct_feature = self.transformer(tgt_embedding, tgt_mask=mask)  # (length, batch, feature_dim)
        instruct_feature = instruct_feature.permute(1, 2, 0)  # 4  512 9

        # Encoder
        x = self.firstconv(image)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        e1 = self.firstmaxpool(x)
        e1 = self.encoder1(e1)

        e11 = self.Instruct1(e1,instruct_feature)
        e1 = e1+e11
        e2 = self.encoder2(e1)
        e22 = self.Instruct2(e2, instruct_feature)
        e2 = e2+e22
        e3 = self.encoder3(e2)
        e33 = self.Instruct3(e3, instruct_feature)
        e3 = e3+e33
        e4 = self.encoder4(e3)
        e4 = self.context(e4)
        e44 = self.Instruct4(e4,instruct_feature)+e4


        # Decoder
        d4 = self.decoder4(e44) + self.RR4(e4, e3)
        d3 = self.decoder3(d4) + self.RR3(d4, e2)
        d2 = self.decoder2(d3) + self.RR2(d3, e1)
        d1 = self.decoder1(d2) + self.RR1(d2, x)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)

class CRNet1(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(CRNet1, self).__init__()

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



class CNet1(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(CNet1, self).__init__()

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
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2) + x

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)

        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)



class GatedFeatureFusion(nn.Module):
    def __init__(self, in_channels, reduction=16):
        """
        门控机制融合空间域、高频、低频特征
        :param in_channels: 每个输入特征图的通道数
        :param reduction: 用于注意力权重生成的通道缩减系数
        """
        super(GatedFeatureFusion, self).__init__()

        # 通道注意力生成器
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),  # 降维
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, 3, kernel_size=1, bias=False),  # 为三种特征生成权重
            nn.Sigmoid()  # 将权重限制到 [0, 1]
        )

        # 特征融合后的卷积
        self.fusion_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, F_s, F_h, F_l):
        """
        :param F_s: 空间域特征 (batch, channels, H, W)
        :param F_h: 高频特征 (batch, channels, H, W)
        :param F_l: 低频特征 (batch, channels, H, W)
        """
        # 叠加输入特征
        combined_features = F_s + F_h + F_l

        # 生成权重 (batch, 3, 1, 1)
        weights = self.channel_attention(combined_features)
        w_s, w_h, w_l = torch.split(weights, 1, dim=1)  # 分割权重

        # 对每个特征加权
        F_s_weighted = F_s * w_s
        F_h_weighted = F_h * w_h
        F_l_weighted = F_l * w_l

        # 融合加权特征
        fused_features = F_s_weighted + F_h_weighted + F_l_weighted

        # 融合后的卷积层
        output = self.relu(self.fusion_conv(fused_features))

        return output


class Frequency_resnet(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(Frequency_resnet, self).__init__()
        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=False)
        resnet.load_state_dict(torch.load('./networks/resnet34.pth'))

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # Frequency domain branch
        self.fusion_conv1 = GatedFeatureFusion(in_channels=64, reduction=16)
        self.fusion_conv2 = GatedFeatureFusion(in_channels=128, reduction=16)
        self.fusion_conv3 = GatedFeatureFusion(in_channels=256, reduction=16)
        self.fusion_conv4 = GatedFeatureFusion(in_channels=512, reduction=16)

        # Decoders with Frequency Sub-branches
        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])


        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def frequency_branch1(self, x, inchannels=64,outchannels=64):
        # Apply FFT
        #Conv = nn.Conv2d(kernel_size=3, in_channels=inchannels, out_channels=outchannels, stride=2,padding=1).to('cuda')
        #x = Conv(x)
        fft= torch.fft.fft2(x.float()).real
        fft_abs = torch.abs(fft)  # Magnitude spectrum

        # Split into low-frequency and high-frequency components
        low_freq = F.interpolate(fft_abs, scale_factor=0.5, mode='bilinear', align_corners=False)
        high_freq = fft_abs - F.interpolate(low_freq, size=fft_abs.shape[-2:], mode='bilinear', align_corners=False)
        low_freq = F.interpolate(low_freq, size=fft_abs.shape[-2:], mode='bilinear', align_corners=False)

        return high_freq, low_freq
    def frequency_branch2(self, x, inchannels=128,outchannels=128):
        # Apply FFT

        fft= torch.fft.fft2(x.float()).real
        fft_abs = torch.abs(fft)  # Magnitude spectrum

        # Split into low-frequency and high-frequency components
        low_freq = F.interpolate(fft_abs, scale_factor=0.5, mode='bilinear', align_corners=False)
        high_freq = fft_abs - F.interpolate(low_freq, size=fft_abs.shape[-2:], mode='bilinear', align_corners=False)
        low_freq = F.interpolate(low_freq, size=fft_abs.shape[-2:], mode='bilinear', align_corners=False)

        return high_freq, low_freq
    def frequency_branch3(self, x, inchannels=256,outchannels=256):
        # Apply FFT

        fft= torch.fft.fft2(x.float()).real
        fft_abs = torch.abs(fft)  # Magnitude spectrum

        # Split into low-frequency and high-frequency components
        low_freq = F.interpolate(fft_abs, scale_factor=0.5, mode='bilinear', align_corners=False)
        high_freq = fft_abs - F.interpolate(low_freq, size=fft_abs.shape[-2:], mode='bilinear', align_corners=False)
        low_freq = F.interpolate(low_freq, size=fft_abs.shape[-2:], mode='bilinear', align_corners=False)

        return high_freq, low_freq
    def frequency_branch4(self, x, inchannels=512,outchannels=512):
        # Apply FFT

        fft= torch.fft.fft2(x.float()).real
        fft_abs = torch.abs(fft)  # Magnitude spectrum

        # Split into low-frequency and high-frequency components
        low_freq = F.interpolate(fft_abs, scale_factor=0.5, mode='bilinear', align_corners=False)
        high_freq = fft_abs - F.interpolate(low_freq, size=fft_abs.shape[-2:], mode='bilinear', align_corners=False)
        low_freq = F.interpolate(low_freq, size=fft_abs.shape[-2:], mode='bilinear', align_corners=False)

        return high_freq, low_freq


    def forward(self, image):
        # Frequency branch

        # Spatial encoder
        x = self.firstconv(image)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        e1 = self.firstmaxpool(x)
        e1 = self.encoder1(e1)
        high_freq1, low_freq1= self.frequency_branch1(e1)
        e1 = self.fusion_conv1(e1,high_freq1,low_freq1)

        e2 = self.encoder2(e1)
        high_freq2, low_freq2 = self.frequency_branch2(e2)
        e2 = self.fusion_conv2(e2, high_freq2, low_freq2)

        e3 = self.encoder3(e2)
        high_freq3, low_freq3 = self.frequency_branch3(e3)
        e3 = self.fusion_conv3(e3, high_freq3, low_freq3)

        e4 = self.encoder4(e3)
        high_freq4, low_freq4 = self.frequency_branch4(e4)
        e4 = self.fusion_conv4(e4, high_freq4, low_freq4)

        # Frequency decoder

        #d4 = self.decoder4(e4,high_freq4,low_freq4) + e3
        #d3 = self.decoder3(d4,high_freq3,low_freq3) + e2
        #d2 = self.decoder2(d3,high_freq2,low_freq2) + e1
        #d1 = self.decoder1(d2,high_freq1,low_freq1) + x
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2) + x

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)

        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)


class FARNet(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(FARNet, self).__init__()

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
        self.context =AxialContext( embed_dim=512, num_heads=8, num_points=4)
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


class Frequency_ARNet(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(Frequency_ARNet, self).__init__()
        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=False)
        resnet.load_state_dict(torch.load('./networks/resnet34.pth'))

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4


        # Frequency domain branch
        self.fusion_conv1 = GatedFeatureFusion(in_channels=64, reduction=16)
        self.fusion_conv2 = GatedFeatureFusion(in_channels=128, reduction=16)
        self.fusion_conv3 = GatedFeatureFusion(in_channels=256, reduction=16)
        self.fusion_conv4 = GatedFeatureFusion(in_channels=512, reduction=16)

        self.context = AxialContext(embed_dim=512, num_heads=8)
        self.RR4 = Relation_Refine(512, 256, 4)
        self.RR3 = Relation_Refine(256, 128, 3)
        self.RR2 = Relation_Refine(128, 64, 2)
        self.RR1 = Relation_Refine2(64, 64, 1)
        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        # Decoders with Frequency Sub-branches
        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])


        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)


    def frequency_branch1(self, x):
        # Apply FFT
        #Conv = nn.Conv2d(kernel_size=3, in_channels=inchannels, out_channels=outchannels, stride=2,padding=1).to('cuda')
        #x = Conv(x)
        fft= torch.fft.fft2(x.float()).real


        # Split into low-frequency and high-frequency components
        low_freq = F.interpolate(fft, scale_factor=0.5, mode='bilinear', align_corners=False)
        high_freq = fft - F.interpolate(low_freq, size=fft.shape[-2:], mode='bilinear', align_corners=False)
        low_freq = F.interpolate(low_freq, size=fft.shape[-2:], mode='bilinear', align_corners=False)

        return high_freq, low_freq
    def frequency_branch2(self, x, inchannels=128,outchannels=128):
        # Apply FFT

        fft= torch.fft.fft2(x.float()).real

        # Split into low-frequency and high-frequency components
        low_freq = F.interpolate(fft, scale_factor=0.5, mode='bilinear', align_corners=False)
        high_freq = fft - F.interpolate(low_freq, size=fft.shape[-2:], mode='bilinear', align_corners=False)
        low_freq = F.interpolate(low_freq, size=fft.shape[-2:], mode='bilinear', align_corners=False)

        return high_freq, low_freq
    def frequency_branch3(self, x, inchannels=256,outchannels=256):
        # Apply FFT

        fft= torch.fft.fft2(x.float()).real


        # Split into low-frequency and high-frequency components
        low_freq = F.interpolate(fft, scale_factor=0.5, mode='bilinear', align_corners=False)
        high_freq = fft - F.interpolate(low_freq, size=fft.shape[-2:], mode='bilinear', align_corners=False)
        low_freq = F.interpolate(low_freq, size=fft.shape[-2:], mode='bilinear', align_corners=False)

        return high_freq, low_freq
    def frequency_branch4(self, x, inchannels=512,outchannels=512):
        # Apply FFT

        fft= torch.fft.fft2(x.float()).real


        # Split into low-frequency and high-frequency components
        low_freq = F.interpolate(fft, scale_factor=0.5, mode='bilinear', align_corners=False)
        high_freq = fft - F.interpolate(low_freq, size=fft.shape[-2:], mode='bilinear', align_corners=False)
        low_freq = F.interpolate(low_freq, size=fft.shape[-2:], mode='bilinear', align_corners=False)

        return high_freq, low_freq


    def forward(self, image):
        # Frequency branch

        # Spatial encoder
        x = self.firstconv(image)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        e1 = self.firstmaxpool(x)
        e1 = self.encoder1(e1)
        #high_freq1, low_freq1= self.frequency_split(e1)
        high_freq1 = self.extract_high_frequency(e1)
        low_freq1 = self.extract_low_frequency(e1)
        e1 = self.fusion_conv1(e1,high_freq1,low_freq1)

        e2 = self.encoder2(e1)
        high_freq2 = self.extract_high_frequency(e2)
        low_freq2 = self.extract_low_frequency(e2)
        e2 = self.fusion_conv2(e2, high_freq2, low_freq2)

        e3 = self.encoder3(e2)
        high_freq3 = self.extract_high_frequency(e3)
        low_freq3 = self.extract_low_frequency(e3)
        e3 = self.fusion_conv3(e3, high_freq3, low_freq3)

        e4 = self.encoder4(e3)
        high_freq4 = self.extract_high_frequency(e4)
        low_freq4 = self.extract_low_frequency(e4)
        e4 = self.fusion_conv4(e4, high_freq4, low_freq4)
        e4 = self.context(e4)

        # Frequency decoder

        #d4 = self.decoder4(e4,high_freq4,low_freq4) + e3
        #d3 = self.decoder3(d4,high_freq3,low_freq3) + e2
        #d2 = self.decoder2(d3,high_freq2,low_freq2) + e1
        #d1 = self.decoder1(d2,high_freq1,low_freq1) + x
        d4 = self.decoder4(e4) + self.RR4(e4, e3)
        d3 = self.decoder3(d4) + self.RR3(d4, e2)
        d2 = self.decoder2(d3) + self.RR2(d3, e1)
        d1 = self.decoder1(d2) + self.RR1(d2, x)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)

        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)

    def extract_high_frequency(self, x):
        residual = x
        dev = x.device
        cutoff = 30
        # global cnt  # 在函数的最开始声明 global cnt
        batch, channels, rows, cols = x.shape

        highpass_filter = torch.ones(rows, cols, dtype=torch.float32, device=dev)

        center_row, center_col = rows // 2, cols // 2
        highpass_filter[center_row - cutoff:center_row + cutoff, center_col - cutoff:center_col + cutoff] = 0

        highpass_filter = highpass_filter.unsqueeze(0).expand(channels, -1, -1)

        fft_image = torch.fft.fft2(x)  # [C, H, W]
        fft_image = torch.fft.fftshift(fft_image)
        highpass_fft_image = fft_image * highpass_filter

        highpass_image = torch.fft.ifft2(torch.fft.ifftshift(highpass_fft_image)).real
        highpass_image = highpass_image.permute(0, 2, 3, 1)

        high_freq = highpass_image.permute(0, 3, 1, 2)

        #lowpass_image = self.extract_low_frequency(residual)

        # lowpass_image = self.avg(residual)
        # lowpass_image = F.interpolate(lowpass_image, scale_factor=2)

        # out1 = residual * highpass_image
        # out2 = residual * lowpass_image

        return high_freq

    def extract_low_frequency(self,x, cutoff=0.25):
        """
        提取特征图的低频信息
        :param feature_map: 输入的特征图，形状为 [N, C, H, W]
        :param cutoff: 截止频率，范围 [0, 1]，控制低频范围，0.25 表示 1/4 的频域大小
        :return: 低频分量的特征图，形状与输入一致
        """
        # 获取输入形状
        N, C, H, W = x.shape

        # 计算 2D 傅里叶变换（从时域到频域）
        fft = torch.fft.fft2(x.float(), dim=(-2, -1))  # 对最后两个维度 (H, W) 进行傅里叶变换
        fft_shifted = torch.fft.fftshift(fft)  # 将低频部分移动到频谱中心

        # 构造低通滤波器
        Y, X = torch.meshgrid(torch.linspace(-0.5, 0.5, H), torch.linspace(-0.5, 0.5, W))
        distance = torch.sqrt(X ** 2 + Y ** 2).to(x.device)  # 计算频率距离
        filter_mask = (distance <= cutoff).float()  # 截止频率以内为 1，其它为 0
        filter_mask = filter_mask.unsqueeze(0).unsqueeze(0)  # 扩展到 [1, 1, H, W]

        # 应用低通滤波器
        low_freq_fft = fft_shifted * filter_mask

        # 逆傅里叶变换（从频域返回时域）
        low_freq_fft = torch.fft.ifftshift(low_freq_fft)  # 逆移位
        low_freq = torch.fft.ifft2(low_freq_fft, dim=(-2, -1)).real  # 取实部作为输出

        return low_freq
