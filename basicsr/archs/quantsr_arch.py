import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np
import math

from .rrdbnet_arch import RRDBNet
from .quant_semantic_cluster_arch import MultiScaleVGGCluster 

from .quant_utils import *

from basicsr.utils.registry import ARCH_REGISTRY


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

    def __init__(self, n_e, e_dim, beta, LQ_stage=False, dist_func='l2', margin=0.40, s=5.0):
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
        #  z_q = z_q + codebook.mean() # gradient propagation to avoid code collapse
        z_q = z_q.view(z.shape)
        
        # compute loss for embedding
        # loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
        #     torch.mean((z_q - z.detach()) ** 2)
        #  e_latent_loss = torch.mean((z_q.detach() - z)**2)
        #  q_latent_loss = torch.mean((z_q - z.detach())**2)  # TODO: replace this term with the moving update strategy

        e_latent_loss = torch.mean((z_q.detach() - z)**2)
        q_latent_loss = torch.mean((z_q - z.detach())**2)  # TODO: replace this term with the moving update strategy

        if not self.LQ_stage:  # if HQ stage
            codebook_loss = q_latent_loss + self.beta * e_latent_loss
        else:
            codebook_loss = self.beta * e_latent_loss
        # preserve gradients
        z_q = z + (z_q - z).detach()

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


class CombineQuantBlock(nn.Module):
    """
    Use preactivation version of residual block, the same as taming
    """
    def __init__(self, in_ch1, in_ch2, out_channel):
        super().__init__()
        self.conv = nn.Conv2d(in_ch1 + in_ch2, out_channel, 3, 1, 1)

    def forward(self, input1, input2=None):
        if input2 is not None:
            input2 = torch.nn.functional.interpolate(input2, input1.shape[2:])
            input = torch.cat((input1, input2), dim=1)
        else:
            input = input1
        out = self.conv(input)
        return out


class MultiScaleEncoder(nn.Module):
    def __init__(self, in_channel, max_depth, n_res_block=2, input_res=256, channel_query_dict=None, norm_type='gn', act_type='leakyrelu', with_attn=True):
        super().__init__()

        ksz = 3

        self.in_conv = nn.Conv2d(in_channel, channel_query_dict[input_res], 4, padding=1)

        self.blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        res = input_res
        for i in range(max_depth):
            in_ch, out_ch = channel_query_dict[res], channel_query_dict[res // 2] 
            tmp_down_block = [
                    nn.Conv2d(in_ch, out_ch, ksz, stride=2, padding=1),
                    ResBlock(out_ch, out_ch, norm_type, act_type),
                    ResBlock(out_ch, out_ch, norm_type, act_type),
                    ]
            self.blocks.append(nn.Sequential(*tmp_down_block))
            res = res // 2

        self.with_attn = with_attn
        if with_attn: 
            self.attn_block = nn.Sequential(
                AttnBlock(out_ch, norm_type),
                ResBlock(out_ch, out_ch, norm_type, act_type),
                )
        
    def forward(self, input):
        outputs = [] 
        x = self.in_conv(input)
        
        for m in self.blocks:
            x = m(x)
            outputs.append(x)
        if self.with_attn:
            outputs[-1] = self.attn_block(outputs[-1])

        return outputs


class DecoderBlock(nn.Module):
    def __init__(
        self, in_channel, out_channel, n_res_block=2, norm_type='gn', act_type='leakyrelu'
    ):
        super().__init__()

        self.block = []
        self.block += [
                        nn.Upsample(scale_factor=2),
                        nn.Conv2d(in_channel, out_channel, 3, stride=1, padding=1),
                        ResBlock(out_channel, out_channel, norm_type, act_type),
                        ResBlock(out_channel, out_channel, norm_type, act_type),
                    ]

        self.block = nn.Sequential(*self.block)

    def forward(self, input):
        return self.block(input)


@ARCH_REGISTRY.register()
class QuanTexSRNet(nn.Module):
    def __init__(self,
                 *,
                 in_channel=3,
                 channel=128,
                 n_res_block=2,
                 n_res_channel=64,
                 beta=0.25,
                 codebook_dist_func='l2',
                 codebook_params=None,
                 gt_resolution=256,
                 LQ_stage=True,
                 norm_type='gn',
                 act_type='silu',
                 enc_quant_fusion_type='concat',
                 content_encoder='rrdb',
                 content_encoder_path=None,
                 use_quantize=True,
                 scale_factor=4,
                 with_attn=False,
                 use_semantic_loss=False,
                 semantic_cluster_path='',
                 **ignore_kwargs
                 ):
        super().__init__()

        codebook_params = np.array(codebook_params)

        self.codebook_scale = codebook_params[:, 0]
        codebook_emb_num = codebook_params[:, 1] 
        codebook_emb_dim = codebook_params[:, 2]

        self.use_quantize = use_quantize 
        self.in_channel = in_channel
        self.beta = beta
        self.gt_res = gt_resolution
        self.LQ_stage = LQ_stage
        self.enc_quant_fusion_type = enc_quant_fusion_type

        channel_query_dict = {
                8: 256,
                16: 256,
                32: 256,
                64: 256,
                128: 128,
                256: 64,
                512: 32,
                }
            
        # build modules from the bottom to the top
        self.max_depth = int(np.log2(gt_resolution // self.codebook_scale[0]))
        self.multiscale_encoder = MultiScaleEncoder(in_channel, self.max_depth, 2, self.gt_res, channel_query_dict, norm_type, act_type, with_attn)

        self.quantize_group = nn.ModuleList()
        self.before_quant_group = nn.ModuleList()
        self.after_quant_group = nn.ModuleList()
        self.decoder_group = nn.ModuleList()
        self.sft_fusion_group = nn.ModuleList()

        for i in range(self.max_depth):
            res = gt_resolution // 2 ** self.max_depth * 2 ** i
            in_ch, out_ch = channel_query_dict[res], channel_query_dict[res * 2] 
            self.decoder_group.append(DecoderBlock(in_ch, out_ch, 2, norm_type, act_type))
            self.sft_fusion_group.append(SimSFT(in_ch, in_ch))

        self.out_conv = nn.Conv2d(out_ch, 3, 3, 1, 1)  

        for scale in range(0, codebook_params.shape[0]):
            quantize = VectorQuantizer(codebook_emb_num[scale], codebook_emb_dim[scale], beta=0.25, 
                                       LQ_stage=self.LQ_stage, 
                                       dist_func=codebook_dist_func,
                                       )
            self.quantize_group.append(quantize)

            scale_in_ch = channel_query_dict[self.codebook_scale[scale]] 
            if scale == 0:
                quant_conv_in_ch = scale_in_ch 
                comb_quant_in_ch1 = codebook_emb_dim[scale] 
                comb_quant_in_ch2 = 0 
            else:
                if enc_quant_fusion_type == 'concat':
                    quant_conv_in_ch = scale_in_ch * 2
                elif enc_quant_fusion_type == 'upconcat':
                    quant_conv_in_ch = scale_in_ch + 256 
                elif enc_quant_fusion_type == 'weighted_sum':
                    quant_conv_in_ch = scale_in_ch 

                comb_quant_in_ch1 = codebook_emb_dim[scale - 1]
                comb_quant_in_ch2 = codebook_emb_dim[scale]
            self.before_quant_group.append(nn.Conv2d(quant_conv_in_ch, codebook_emb_dim[scale], 1))
            self.after_quant_group.append(CombineQuantBlock(comb_quant_in_ch1, comb_quant_in_ch2, scale_in_ch))
        
        if LQ_stage:
            self.content_model = RRDBNet(3, 3, scale=scale_factor) 
            if content_encoder_path is not None:
                self.content_model.load_state_dict(torch.load(content_encoder_path)['params_ema'])

        self.use_semantic_loss = use_semantic_loss
        if use_semantic_loss: 
            self.semantic_cluster = MultiScaleVGGCluster() 
            self.semantic_cluster.load_state_dict(torch.load(semantic_cluster_path)['params'])

    def encode_and_decode(self, input, gt_img=None, gt_indices=None):
        pre_input = input

        codebook_loss_list = []
        cls_loss_list = []
        indices_list = []

        enc_feats = self.multiscale_encoder(input.detach())
        enc_feats = enc_feats[::-1]

        if self.use_semantic_loss:
            with torch.no_grad():
                semantic_z_quant_list = self.semantic_cluster.get_quant_feat(input)
                semantic_z_quant_dict = {}
                for zq in semantic_z_quant_list:
                    semantic_z_quant_dict[zq.shape[-1]] = zq

        quant_idx = 0
        prev_dec_feat = None
        prev_quant_feat = None
        x = enc_feats[0]
        for i in range(self.max_depth):
            cur_res = self.gt_res // 2 ** self.max_depth * 2 ** i
            if cur_res in self.codebook_scale: # needs to perform quantize
                if prev_dec_feat is not None:
                    before_quant_feat = torch.cat((enc_feats[i], prev_dec_feat), dim=1)
                else:
                    before_quant_feat = enc_feats[i]
                feat_to_quant = self.before_quant_group[quant_idx](before_quant_feat)
                if gt_indices is not None:
                    z_quant, loss, cls_loss, indices = self.quantize_group[quant_idx](feat_to_quant, gt_indices[quant_idx])
                else:
                    z_quant, loss, cls_loss, indices = self.quantize_group[quant_idx](feat_to_quant)

                if not self.use_quantize:
                    z_quant = feat_to_quant

                if self.use_semantic_loss:
                    cls_loss = torch.mean((z_quant - semantic_z_quant_dict[cur_res].detach()) ** 2)

                after_quant_feat = self.after_quant_group[quant_idx](z_quant, prev_quant_feat)

                codebook_loss_list.append(loss)
                cls_loss_list.append(cls_loss)
                indices_list.append(indices)

                quant_idx += 1
                prev_quant_feat = z_quant 
                x = after_quant_feat
            else:
                if self.LQ_stage:
                    scale, shift = self.sft_fusion_group[i](enc_feats[i])
                    #  x = x + x * scale + shift 
                    # gated fusion
                    x = x * torch.sigmoid(scale)

            x = self.decoder_group[i](x)
            prev_dec_feat = x
                
        out_img = self.out_conv(x)

        out_imgs = [pre_input, out_img.clone().detach(), out_img]

        return out_imgs, sum(codebook_loss_list), sum(cls_loss_list), indices_list 

    def test(self, input):

        input = self.content_model(input)
        dec, codebook_loss, cls_loss, indices = self.encode_and_decode(input) 

        return dec[-1]
    
    def forward(self, input, gt_indices=None, gt_img=None):
        # if input is LQ, upsample it to GT size first 
        # TODO: to modify the upsample justification

        if self.LQ_stage:
            input = self.content_model(input)

        if gt_indices is not None:
            # in LQ training stage, need to parse GT indices for classification supervise. 
            dec, codebook_loss, cls_loss, indices = self.encode_and_decode(input, gt_img, gt_indices)
        else:
            # in HQ stage, or LQ test stage, no GT indices needed.
            dec, codebook_loss, cls_loss, indices = self.encode_and_decode(input, gt_img) 

        return dec, codebook_loss, cls_loss, indices


if __name__ == '__main__':

    codebook_params = [
            [16, 1024, 256],
            #  [32, 512, 128],
            #  [64, 256, 64],
            ]

    models = [
        #  MultiScaleVQVAENew(codebook_scale=[16]).cuda(),
        #  MultiScaleVQVAENew(codebook_scale=[32]).cuda(),
        #  MultiScaleVQVAENew(codebook_scale=[64]).cuda(),
        #  MultiScaleVQVAENew(codebook_scale=[16, 64]).cuda(),
        #  MultiScaleVQVAEV1(LQ_stage=True, codebook_params=codebook_params,
            #  content_encoder='rcan',
            #  content_encoder_path='/root/experiments/BasicSR/experiments/pretrained_models/models_ECCV2018RCAN//RCAN_BIX2.pt',
            #  fusion_type='adain',
            #  ).cuda(),
        MultiScaleVQVAEV2(LQ_stage=True, codebook_params=codebook_params,
            content_encoder='rcan',
            content_encoder_path='/root/experiments/BasicSR/experiments/pretrained_models/models_ECCV2018RCAN//RCAN_BIX2.pt',
            fusion_type='spade',
            ).cuda(),

        ]
    input = torch.randn(2, 3, 128, 128).cuda()

    for model in models:
        out_img, codebook_loss, cls_loss, indices = model(input)
        #  weight = torch.load('/root/experiments/ckpt_bk/101_1_scale_HQ_stage_1024_codebook_largeDataset/models/net_g_110000.pth')
        #  model.load_state_dict(weight['params'])
        print([x.shape for x in out_img])
        print(codebook_loss.shape, cls_loss.shape)
        print([x.shape for x in indices])


