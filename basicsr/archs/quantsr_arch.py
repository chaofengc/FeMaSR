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

    def __init__(self, n_e, e_dim, LQ_stage=False, dist_func='l2'):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.LQ_stage = LQ_stage

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        #  self.embedding.weight.data.uniform_(-1.0 / self.n_e * init_gain, 1.0 / self.n_e * init_gain) 
        
        self.dist_func = dist_func

    def forward(self, z, gt_indices=None):
        """
        Args:
            z: input features to be quantized, z (continuous) -> z_q (discrete)
               z.shape = (batch, channel, height, width)
            gt_indices: feature map of given indices, used for visualization. 
        """
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

        cls_loss = z.sum() * 0 
        if gt_indices is not None:
            min_encoding_indices = gt_indices.reshape_as(min_encoding_indices)
            min_encodings = torch.zeros(
                min_encoding_indices.shape[0], codebook.shape[0]).to(z)
            min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, codebook)
        z_q = z_q.view(z.shape)
        
        e_latent_loss = torch.mean((z_q.detach() - z)**2)
        q_latent_loss = torch.mean((z_q - z.detach())**2)  

        if self.LQ_stage:  
            codebook_loss = e_latent_loss
        else:
            codebook_loss = q_latent_loss + e_latent_loss

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q, codebook_loss, cls_loss, min_encoding_indices.reshape(z_q.shape[0], 1, z_q.shape[2], z_q.shape[3])
    

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
                 content_encoder_path=None,
                 use_quantize=True,
                 scale_factor=4,
                 with_attn=False,
                 use_semantic_loss=False,
                 semantic_cluster_path=None,
                 **ignore_kwargs
                 ):
        super().__init__()

        codebook_params = np.array(codebook_params)

        self.codebook_scale = codebook_params[:, 0]
        codebook_emb_num = codebook_params[:, 1] 
        codebook_emb_dim = codebook_params[:, 2]

        self.use_quantize = use_quantize 
        self.in_channel = in_channel
        self.gt_res = gt_resolution
        self.LQ_stage = LQ_stage
        self.scale_factor = scale_factor

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
            quantize = VectorQuantizer(codebook_emb_num[scale], codebook_emb_dim[scale], 
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
            self.semantic_cluster.load_state_dict(torch.load(semantic_cluster_path)['params'], strict=False)

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
                    # rsft fusion
                    x = x + x * scale + shift 
                    # gated fusion
                    #  x = x * torch.sigmoid(scale)

            x = self.decoder_group[i](x)
            prev_dec_feat = x
                
        out_img = self.out_conv(x)

        out_imgs = [pre_input, out_img]

        return out_imgs, sum(codebook_loss_list), sum(cls_loss_list), indices_list 

    @torch.no_grad()
    def test_tile(self, input, tile_size=256, tile_pad=10):
        """It will first crop input images to tiles, and then process each tile.
        Finally, all the processed tiles are merged into one images.
        Modified from: https://github.com/xinntao/Real-ESRGAN/blob/master/realesrgan/utils.py
        """
        batch, channel, height, width = input.shape
        output_height = height * self.scale_factor
        output_width = width * self.scale_factor
        output_shape = (batch, channel, output_height, output_width)

        # start with black image
        output = input.new_zeros(output_shape)
        tiles_x = math.ceil(width / tile_size)
        tiles_y = math.ceil(height / tile_size)

        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * tile_size
                ofs_y = y * tile_size
                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + tile_size, width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + tile_size, height)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - tile_pad, 0)
                input_end_x_pad = min(input_end_x + tile_pad, width)
                input_start_y_pad = max(input_start_y - tile_pad, 0)
                input_end_y_pad = min(input_end_y + tile_pad, height)

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = y * tiles_x + x + 1
                input_tile = input[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

                # upscale tile
                output_tile = self.test(input_tile)

                # output tile area on total image
                output_start_x = input_start_x * self.scale_factor
                output_end_x = input_end_x * self.scale_factor
                output_start_y = input_start_y * self.scale_factor
                output_end_y = input_end_y * self.scale_factor

                # output tile area without padding
                output_start_x_tile = (input_start_x - input_start_x_pad) * self.scale_factor
                output_end_x_tile = output_start_x_tile + input_tile_width * self.scale_factor
                output_start_y_tile = (input_start_y - input_start_y_pad) * self.scale_factor
                output_end_y_tile = output_start_y_tile + input_tile_height * self.scale_factor

                # put tile into output image
                output[:, :, output_start_y:output_end_y,
                            output_start_x:output_end_x] = output_tile[:, :, output_start_y_tile:output_end_y_tile,
                                                                       output_start_x_tile:output_end_x_tile]
        return output

    @torch.no_grad()
    def test(self, input):

        input = self.content_model(input)
        dec, codebook_loss, cls_loss, indices = self.encode_and_decode(input) 

        return dec[-1]
    
    def forward(self, input, gt_indices=None, gt_img=None):

        if self.LQ_stage:
            input = self.content_model(input)

        if gt_indices is not None:
            # in LQ training stage, need to parse GT indices for classification supervise. 
            dec, codebook_loss, cls_loss, indices = self.encode_and_decode(input, gt_img, gt_indices)
        else:
            # in HQ stage, or LQ test stage, no GT indices needed.
            dec, codebook_loss, cls_loss, indices = self.encode_and_decode(input, gt_img) 

        return dec, codebook_loss, cls_loss, indices



