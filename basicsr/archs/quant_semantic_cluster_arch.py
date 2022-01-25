import torch
from torch import nn as nn
from torch.nn import functional as F

import numpy as np
import math

from .quant_utils import *
from .vgg_arch import VGGFeatureExtractor
from basicsr.utils.registry import ARCH_REGISTRY


class VectorQuantizer(nn.Module):
    """
    see https://github.com/MishaLaskin/vqvae/blob/d761a999e2267766400dc646d82d3ac3657771d4/models/quantizer.py

    Discretization bottleneck part of the VQ-VAE.
    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    """

    def __init__(self, n_e, e_dim, LQ_stage=False, dist_func='l2'):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.LQ_stage = LQ_stage
        self.dist_func = dist_func

        self.embedding = nn.Embedding(self.n_e, self.e_dim)

    def forward(self, z, gt_indices=None):
        """
        Args:
            z: input features to be quantized, z (continuous) -> z_q (discrete)
               z.shape = (batch, channel, height, width)
            gt_indices: feature map of given indices, used for visualization. 
        """
        #  print(self.embedding.weight.abs().mean())
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)

        codebook = self.embedding.weight

        if self.dist_func == 'cosine':
            d = F.linear(F.normalize(z_flattened), F.normalize(codebook))
            d = 1 - d
        elif self.dist_func == 'l2': 
            d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
                torch.sum(codebook**2, dim=1) - 2 * \
                torch.matmul(z_flattened, codebook.t()) 

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], codebook.shape[0]).to(z)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        if gt_indices is not None:
            min_encoding_indices = gt_indices.reshape_as(min_encoding_indices)
            min_encodings = torch.zeros(
                min_encoding_indices.shape[0], codebook.shape[0]).to(z)
            min_encodings.scatter_(1, min_encoding_indices, 1)
            
        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, codebook)
        z_q = z_q.view(z.shape)
        
        q_latent_loss = torch.mean((z_q - z.detach())**2)  

        codebook_loss = q_latent_loss

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q, codebook_loss, 0, min_encoding_indices.reshape(z_q.shape[0], 1, z_q.shape[2], z_q.shape[3])
    

@ARCH_REGISTRY.register()
class MultiScaleVGGCluster(nn.Module):
    def __init__(self,
                 *,
                 in_channel=3,
                 channel=128,
                 n_res_block=2,
                 n_res_channel=64,
                 codebook_dist_func='l2',
                 codebook_params=None,
                 gt_resolution=256,
                 LQ_stage=False,
                 norm_type='gn',
                 act_type='leakyrelu',
                 use_semantic_loss=False,
                 **ignore_kwargs
                 ):
        super().__init__()

        self.in_channel = in_channel
        self.gt_res = gt_resolution
        self.LQ_stage = LQ_stage
        self.use_semantic_loss = use_semantic_loss

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

        self.vgg_feat_list = ['relu3_4', 'relu4_4', 'relu5_4']
        self.vgg_feat_extractor = VGGFeatureExtractor(self.vgg_feat_list) 

        self.ema_cluster_group = nn.ModuleList()
        for num, dim in codebook_params:
            self.ema_cluster_group.append(VectorQuantizer(num, dim))
        
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
        with torch.no_grad():
            vgg_feats = self.vgg_feat_extractor(input)
        vgg_feats = [vgg_feats[x] for x in self.vgg_feat_list]

        codebook_loss_list = []
        cls_loss_list = []
        indices_list = []
 
        for idx, vggf in enumerate(vgg_feats):
            z_quant, loss, cls_loss, indices = self.ema_cluster_group[idx](vggf)
            
            codebook_loss_list.append(loss)
            cls_loss_list.append(cls_loss)
            indices_list.append(indices)
  
        out_imgs = [input] * 3

        return out_imgs, sum(codebook_loss_list), sum(cls_loss_list), indices_list 


