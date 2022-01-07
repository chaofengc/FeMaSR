import torch
from torch import nn as nn
from torch.nn import functional as F

from .vgg_arch import VGGFeatureExtractor

import numpy as np
import random
import math

class AttnBlock(nn.Module):
    def __init__(self, in_channels, norm_type='gn'):
        super().__init__()
        self.in_channels = in_channels

        self.norm = NormLayer(in_channels, norm_type)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)


    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_



class NormLayer(nn.Module):
    """Normalization Layers.
    ------------
    # Arguments
        - channels: input channels, for batch norm and instance norm.
        - input_size: input shape without batch size, for layer norm.
    """
    def __init__(self, channels, norm_type='bn'):
        super(NormLayer, self).__init__()
        norm_type = norm_type.lower()
        self.norm_type = norm_type
        self.channels = channels
        if norm_type == 'bn':
            self.norm = nn.BatchNorm2d(channels, affine=True)
        elif norm_type == 'in':
            self.norm = nn.InstanceNorm2d(channels, affine=False)
        elif norm_type == 'gn':
            self.norm = nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6, affine=True)
        elif norm_type == 'none':
            self.norm = lambda x: x*1.0
        else:
            assert 1==0, 'Norm type {} not support.'.format(norm_type)

    def forward(self, x):
        return self.norm(x)


class ActLayer(nn.Module):
    """activation layer.
    ------------
    # Arguments
        - relu type: type of relu layer, candidates are
            - ReLU
            - LeakyReLU: default relu slope 0.2
            - PRelu 
            - SELU
            - none: direct pass
    """
    def __init__(self, channels, relu_type='leakyrelu'):
        super(ActLayer, self).__init__()
        relu_type = relu_type.lower()
        if relu_type == 'relu':
            self.func = nn.ReLU(True)
        elif relu_type == 'leakyrelu':
            self.func = nn.LeakyReLU(0.2, inplace=True)
        elif relu_type == 'prelu':
            self.func = nn.PReLU(channels)
        elif relu_type == 'none':
            self.func = lambda x: x*1.0
        elif relu_type == 'silu':
            self.func = nn.SiLU(True)
        elif relu_type == 'gelu':
            self.func = nn.GELU()
        else:
            assert 1==0, 'activation type {} not support.'.format(relu_type)

    def forward(self, x):
        return self.func(x)


class EqualConv2d(nn.Module):
    """Equalized Linear as StyleGAN2.
    Args:
        in_channels (int): Channel number of the input.
        out_channels (int): Channel number of the output.
        kernel_size (int): Size of the convolving kernel.
        stride (int): Stride of the convolution. Default: 1
        padding (int): Zero-padding added to both sides of the input.
            Default: 0.
        bias (bool): If ``True``, adds a learnable bias to the output.
            Default: ``True``.
        bias_init_val (float): Bias initialized value. Default: 0.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, bias_init_val=0):
        super(EqualConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.scale = 1 / math.sqrt(in_channels * kernel_size**2)

        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels).fill_(bias_init_val))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        out = F.conv2d(
            x,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out


class ResBlock(nn.Module):
    """
    Use preactivation version of residual block, the same as taming
    """
    def __init__(self, in_channel, out_channel, norm_type='gn', act_type='leakyrelu'):
        super(ResBlock, self).__init__()

        self.conv = nn.Sequential(
            NormLayer(in_channel, norm_type),
            ActLayer(in_channel, act_type),
            nn.Conv2d(in_channel, out_channel, 3, stride=1, padding=1),
            NormLayer(out_channel, norm_type),
            ActLayer(out_channel, act_type),
            nn.Conv2d(out_channel, out_channel, 3, stride=1, padding=1),
        )

    def forward(self, input):
        res = self.conv(input)
        out = res + input
        return out


class ConvUpLayer(nn.Module):
    """Conv Up Layer. Bilinear upsample + Conv.
    Args:
        in_channels (int): Channel number of the input.
        out_channels (int): Channel number of the output.
        kernel_size (int): Size of the convolving kernel.
        stride (int): Stride of the convolution. Default: 1
        padding (int): Zero-padding added to both sides of the input.
            Default: 0.
        bias (bool): If ``True``, adds a learnable bias to the output.
            Default: ``True``.
        bias_init_val (float): Bias initialized value. Default: 0.
        activate (bool): Whether use activateion. Default: True.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 bias=True,
                 bias_init_val=0,
                 activate=True):
        super(ConvUpLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.scale = 1 / math.sqrt(in_channels * kernel_size**2)

        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))

        if bias and not activate:
            self.bias = nn.Parameter(torch.zeros(out_channels).fill_(bias_init_val))
        else:
            self.register_parameter('bias', None)

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 1, 1)

        # activation
        if activate:
            if bias:
                self.activation = ActLayer(out_channels, 'leakyrelu')
            else:
                self.activation = ActLayer(out_channels, 'leakyrelu')
        else:
            self.activation = None

    def forward(self, x):
        # bilinear upsample
        out = F.interpolate(x, scale_factor=2, mode='nearest')
        # conv
        out = F.conv2d(
            out,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )
        # activation
        if self.activation is not None:
            out = self.activation(out)
        return out


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


class AdaINFusionBlock(nn.Module):
    """Defines fusion network."""

    def __init__(self, layer_name='relu4_1'):
        super().__init__()
        self.layer_name = layer_name
        self.vgg = VGGFeatureExtractor([layer_name])

        self.dec = nn.Sequential(
                nn.Conv2d(512, 256, 3, 1, 1),
                nn.LeakyReLU(0.2, True),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(256, 256, 3, 1, 1),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(256, 256, 3, 1, 1),
                nn.LeakyReLU(0.2, True),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(256, 128, 3, 1, 1),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(128, 128, 3, 1, 1),
                nn.LeakyReLU(0.2, True),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(128, 64, 3, 1, 1),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(64, 64, 3, 1, 1),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(64, 3, 3, 1, 1),
                )
        
    def forward(self, x):
        content, style = x.chunk(2, dim=1)
        content_feat = self.vgg(content)[self.layer_name]
        style_feat = self.vgg(style)[self.layer_name]

        feat = adaptive_instance_normalization(content_feat, style_feat)
        out = self.dec(feat)
        return out


class SimSFT(nn.Module):
    def __init__(self, in_channel, sft_out_channel):
        super().__init__()

        self.condition_scale = nn.Conv2d(in_channel, sft_out_channel, 3, 1, 1)
        self.condition_shift = nn.Conv2d(in_channel, sft_out_channel, 3, 1, 1)
            
    def forward(self, x):
        scale = self.condition_scale(x)
        shift = self.condition_shift(x)
        return scale, shift


class DualAttnGate(nn.Module):
    def __init__(self, in_channel1, in_channel2, out_channel=None, reduction=32):
        super().__init__()
        
        if out_channel is None:
            out_channel = in_channel2

        self.spatial_attn = nn.Sequential(
                nn.Conv2d(in_channel1, 1, 3, 1, 1),
                nn.Sigmoid(),
                )

        self.channel_attn = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channel1, reduction, 1, 1, 0),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(reduction, in_channel2, 1, 1, 0),
                nn.Sigmoid(),
                )

        self.combine_conv = nn.Conv2d(in_channel2 * 2, out_channel, 1, 1, 0)
            
    def forward(self, x1, x2):
        #  spatial_w = self.spatial_attn(
                #  torch.cat((x1.mean(dim=1, keepdim=True), x1.max(dim=1, keepdim=True)[0]), dim=1))
        spatial_w = self.spatial_attn(x1)
        channel_w = self.channel_attn(x1)
        out = x2 * spatial_w + x2 * channel_w
        return out

