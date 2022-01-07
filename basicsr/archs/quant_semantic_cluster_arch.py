import torch
from torch import nn as nn
from torch.nn import functional as F

import numpy as np
import math

from .quant_utils import *
from .vgg_arch import VGGFeatureExtractor


class VectorQuantizer(nn.Module):
    """
    see https://github.com/MishaLaskin/vqvae/blob/d761a999e2267766400dc646d82d3ac3657771d4/models/quantizer.py
    ____________________________________________
    Discretization bottleneck part of the VQ-VAE.
    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    _____________________________________________
    """

    def __init__(self, n_e, e_dim, beta=1, LQ_stage=False, dist_func='l2', margin=0.40, s=5.0):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.LQ_stage = LQ_stage
        self.margin = margin
        self.s = s
        #  init_gain = 1.0
        #  init_gain = 0.1
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        #  self.embedding.weight.data.uniform_(-1.0 / self.n_e * init_gain, 1.0 / self.n_e * init_gain) 
        
        # self.avgpool = torch.nn.AvgPool2d(kernel_size=(4, padding=0)
        self.dist_func = dist_func
        self.cls_criterion = torch.nn.CrossEntropyLoss()

    def forward(self, z, gt_indices=None):
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vector that is the index of the closest embedding vector e_j
        z (continuous) -> z_q (discrete)
        z.shape = (batch, channel, height, width)
        quantization pipeline:
            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)
        """
        #  print(self.embedding.weight.abs().mean())
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)

        codebook = self.embedding.weight
        #  codebook = torch.load('./useful_code.pth').squeeze(1).to(z)
        #  codebook = torch.load('./useful_code_semantic.pth').squeeze(1).to(z)
        self.dist_func = 'l2' 
        if self.dist_func == 'cosine':
            d = F.linear(F.normalize(z_flattened), F.normalize(codebook))
            d = 1 - d
        elif self.dist_func == 'l2': 
            d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
                torch.sum(codebook**2, dim=1) - 2 * \
                torch.matmul(z_flattened, codebook.t()) 

        #  d = F.linear(F.normalize(z_flattened), F.normalize(self.embedding.weight))
        #  d = 1 - d
        #  print('use cos distance')

        #  d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            #  torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            #  torch.matmul(z_flattened, self.embedding.weight.t()) 

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], codebook.shape[0]).to(z)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        cls_loss = z.sum() * 0 
        if gt_indices is not None:
            min_encoding_indices = gt_indices.reshape_as(min_encoding_indices)
            min_encodings = torch.zeros(
                min_encoding_indices.shape[0], codebook.shape[0]).to(z)
            min_encodings.scatter_(1, min_encoding_indices, 1)
            
        # dtype min encodings: torch.float32
        # min_encodings shape: torch.Size([2048, 512])
        # min_encoding_indices.shape: torch.Size([2048, 1])

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, codebook)
        z_q = z_q.view(z.shape)
        
        # compute loss for embedding
        # loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
        #     torch.mean((z_q - z.detach()) ** 2)
        #  e_latent_loss = torch.mean((z_q.detach() - z)**2)
        #  q_latent_loss = torch.mean((z_q - z.detach())**2)  # TODO: replace this term with the moving update strategy

        e_latent_loss = torch.mean((z_q.detach() - z)**2)
        q_latent_loss = torch.mean((z_q - z.detach())**2)  # TODO: replace this term with the moving update strategy

        #  if not self.LQ_stage:  # if HQ stage
            #  codebook_loss = q_latent_loss + self.beta * e_latent_loss
        #  else:
            #  codebook_loss = self.beta * e_latent_loss
        codebook_loss = q_latent_loss
        # preserve gradients
        #  z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q, codebook_loss, cls_loss, min_encoding_indices.reshape(z_q.shape[0], 1, z_q.shape[2], z_q.shape[3])

    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        min_encodings = torch.zeros(indices.shape[0], self.n_e).to(indices)
        min_encodings.scatter_(1, indices[:,None], 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings.float(), self.embedding.weight)

        if shape is not None:
            z_q = z_q.view(shape)

            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q
    
    def get_codebook(self):
        return self.embedding.weight


class EMACluster(nn.Module):
    def __init__(self, n_e, e_dim, LQ_stage=False, use_codebook_cls=False, dist_func='l2', margin=0.40, s=5.0):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.LQ_stage = LQ_stage
        self.use_codebook_cls = use_codebook_cls
        self.margin = margin
        self.s = s
        init_gain = 1
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e * init_gain, 1.0 / self.n_e * init_gain) 

        self.decay = 0.99
        self.eps = 1e-8
        self.register_buffer("cluster_size", torch.zeros(n_e))
        self.register_buffer("embed_avg", self.embedding.weight.data.clone())

        # self.avgpool = torch.nn.AvgPool2d(kernel_size=(4, padding=0)
        self.dist_func = dist_func
        self.cls_criterion = torch.nn.CrossEntropyLoss()

    def forward(self, z):
        #  print(self.embedding.weight.abs().mean())
        # reshape z -> (batch, height, width, channel) and flatten
        batch, channel, height, width = z.shape
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
                torch.sum(self.embedding.weight**2, dim=1) - 2 * \
                torch.matmul(z_flattened, self.embedding.weight.t()) 

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e).to(z)
        min_encodings.scatter_(1, min_encoding_indices, 1)
                    
        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight)

        if self.training: 
            embed_onehot_sum = min_encodings.sum(0)
            embed_sum = z_flattened.transpose(0, 1) @ min_encodings
            embed_sum = embed_sum.transpose(0, 1)

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_e * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
            self.embedding.weight.data.copy_(embed_normalized)

        z_q = z_q.view(z.shape)
        codebook_loss = torch.mean((z_q - z) ** 2).detach()

        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        cls_loss = 0
        
        return z_q, codebook_loss, cls_loss, min_encoding_indices


class MultiScaleVGGCluster(nn.Module):
    def __init__(self,
                 *,
                 in_channel=3,
                 channel=128,
                 n_res_block=2,
                 n_res_channel=64,
                 beta=0.25,
                 codebook_dist_func='l2',
                 codebook_margin=0.40,
                 codebook_norm_feature=5.0,
                 codebook_params=None,
                 gt_resolution=256,
                 LQ_stage=False,
                 norm_type='gn',
                 act_type='leakyrelu',
                 enc_quant_fusion_type='concat',
                 content_encoder='rrdb',
                 content_encoder_path='',
                 fusion_type = 'unet',
                 with_semantic=True,
                 **ignore_kwargs
                 ):
        super().__init__()

        #  codebook_params = np.array(codebook_params)

        #  self.codebook_scale = codebook_params[:, 0]
        #  codebook_emb_num = codebook_params[:, 1] 
        #  codebook_emb_dim = codebook_params[:, 2]

        self.in_channel = in_channel
        self.beta = beta
        self.gt_res = gt_resolution
        self.LQ_stage = LQ_stage
        self.enc_quant_fusion_type = enc_quant_fusion_type
        self.content_encoder = content_encoder.lower()
        self.with_semantic = with_semantic

        channel_query_dict = {
                #  8: 256,
                16: 256,
                32: 256,
                64: 256,
                128: 128,
                256: 64,
                512: 32,
                }
        
        codebook_params = [
                [512, 256],
                [1024, 512],
                [1024, 512]
                ]
        #  self.vgg_feat_list = ['conv3_4', 'conv4_4', 'conv5_4']
        self.vgg_feat_list = ['relu3_4', 'relu4_4', 'relu5_4']
        self.vgg_feat_extractor = VGGFeatureExtractor(self.vgg_feat_list) 

        self.ema_cluster_group = nn.ModuleList()
        for num, dim in codebook_params:
            #  self.ema_cluster_group.append(EMACluster(num, dim))
            self.ema_cluster_group.append(VectorQuantizer(num, dim))

        #  for p in self.ema_cluster_group.parameters():
            #  p.requires_grad = False

        self.param = nn.Conv2d(3, 3, 1, 1, 1)
        
    def encode_and_decode(self, input, gt_indices=None):

        with torch.no_grad():
            vgg_feats = self.vgg_feat_extractor(input)
        vgg_feats = [vgg_feats[x] for x in self.vgg_feat_list]

        codebook_loss_list = []
        cls_loss_list = []
        indices_list = []

        #  from torchvision.utils import save_image
        #  save_feat = vgg_feats[1].mean(dim=1, keepdim=True)
        #  save_image(save_feat, 'test_vgg_feature.jpg', normalize=True)
        #  save_image(input, 'test_input.jpg', normalize=True)
        #  exit()

        for idx, vggf in enumerate(vgg_feats):
            z_quant, loss, cls_loss, indices = self.ema_cluster_group[idx](vggf)
            
            loss += 0 * self.param(input).mean() 
            codebook_loss_list.append(loss)
            cls_loss_list.append(cls_loss)
            indices_list.append(indices)
  
        out_imgs = [input] * 3

        return out_imgs, sum(codebook_loss_list), sum(cls_loss_list), indices_list 

    def get_quant_feat(self, input):
        with torch.no_grad():
            vgg_feats = self.vgg_feat_extractor(input)
        vgg_feats = [vgg_feats[x] for x in self.vgg_feat_list]

        z_quant_list = []
        for idx, vggf in enumerate(vgg_feats):
            z_quant, loss, cls_loss, indices = self.ema_cluster_group[idx](vggf)
            z_quant_list.append(z_quant)
        return z_quant_list
    
    def forward(self, input, gt_indices=None):
        # if input is LQ, upsample it to GT size first 
        # TODO: to modify the upsample justification
        # scale_factor = self.gt_res // input.shape[2]
        #  if self.LQ_stage:
            #  input = F.interpolate(input, scale_factor=2, mode='bilinear')
        #  input = F.interpolate(input, scale_factor=2, mode='bilinear')
        
        # if input.shape[2] > self.gt_res and gt_indices is None and self.LQ_stage:
        #     input = F.interpolate(input, scale_factor=2, mode='bilinear')

        # in HQ stage, or LQ test stage, no GT indices needed.
        dec, codebook_loss, cls_loss, indices = self.encode_and_decode(input) 

        return dec, codebook_loss, cls_loss, indices


if __name__ == '__main__':

    codebook_params = [
            [16, 1024, 512],
            [32, 512, 512],
            [64, 256, 256],
            ]