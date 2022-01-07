import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel
from .. import lpips
import copy


def calculate_lpips(img1, img2, loss_fn):
    img1 = lpips.im2tensor(img1) # RGB image from [-1,1]
    img2 = lpips.im2tensor(img2)
    img1 = img1.cuda()
    img2 = img2.cuda()

    # Compute distance
    lpips_score = loss_fn.forward(img1,img2)
    return lpips_score.item()


@MODEL_REGISTRY.register()
class QuanTexSRGANModel(BaseModel):
    def __init__(self, opt):
        super(QuanTexSRGANModel, self).__init__(opt)

         # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        # self.print_network(self.net_g)

        self.lpips_fn = lpips.LPIPS(net='alex',version='0.1')
        self.lpips_fn = self.model_to_device(self.lpips_fn)

        self.has_gt_model = False
        # load pre-trained HQ ckpt, frozen parts of network and finetune
        self.LQ_stage = self.opt['network_g'].get('LQ_stage', None) 
        self.latent_size = self.opt['network_g'].get('latent_size')
        if self.LQ_stage:
            hq_ckpt_path = self.opt['network_g'].get('hq_path', None)
            if hq_ckpt_path is not None:
                self.has_gt_model = True
                # load hq checkpoint for HQ net, frozen the whole model
                self.net_hq = build_network(opt['network_g'])
                self.net_hq = self.model_to_device(self.net_hq)
                # self.print_network(self.net_hq)
                self.load_network(self.net_hq, hq_ckpt_path,
                            self.opt['path']['strict_load'])  
                if isinstance(self.net_hq, torch.nn.DataParallel):
                    self.net_hq.module.dist_func = 'l2'
                    self.net_hq.module.LQ_stage = False 
                else:
                    self.net_hq.dist_func = 'l2'
                    self.net_hq.LQ_stage = False 

                for module in self.net_hq.modules():
                    for p in module.parameters():
                        p.requires_grad = False

                # load hq checkpoint for LQ net initialization, frozen the codebook and the decoder  
                self.load_network(self.net_g, hq_ckpt_path,
                            self.opt['path']['strict_load']) 
                #  frozen_module_keywords = ['quantize', 'decoder', 'before_quant_group', 'after_quant_group', 'out_conv']
            frozen_module_keywords = self.opt['network_g'].get('frozen_module_keywords', None) 
            if frozen_module_keywords is not None:
                for name, module in self.net_g.named_modules():
                    for fkw in frozen_module_keywords:
                        if fkw in name:
                            for p in module.parameters():
                                p.requires_grad = False
                            break
        self.has_gt_model = False # to test dataset without GT
        # load pretrained models
        load_path = self.opt['path'].get('pretrain_model_g', None)
        resume_module_keywords = self.opt['network_g'].get('resume_module_keywords_g', None)
        if load_path is not None:
            if resume_module_keywords is None:
                self.load_network(self.net_g, load_path, self.opt['path']['strict_load'])
            else:
                # only reload weights in resume module keywords
                logger = get_root_logger()
                logger.warning(f'Only reload weights in {resume_module_keywords} from {load_path}')
                net = self.get_bare_model(self.net_g)
                w = torch.load(load_path)['params']
                new_w = OrderedDict()
                for wk in w.keys():
                    for rkw in resume_module_keywords:
                        if rkw in wk:
                            new_w[wk] = w[wk]
                net.load_state_dict(new_w, strict=False)

        if self.is_train:
            self.init_training_settings()
            self.use_dis = (self.opt['train']['gan_opt']['loss_weight'] != 0) 
        
        self.net_g_best = copy.deepcopy(self.net_g)
        self.net_d_best = copy.deepcopy(self.net_d)

    def init_training_settings(self):
        train_opt = self.opt['train']
        self.net_g.train()
        if hasattr(self, 'net_hq'):
            self.net_hq.eval()

        # define network net_d
        self.net_d = build_network(self.opt['network_d'])
        self.net_d = self.model_to_device(self.net_d)
        # self.print_network(self.net_d)
        # load pretrained d models
        load_path = self.opt['path'].get('pretrain_model_d', None)
        # print(load_path)
        if load_path is not None:
            self.load_network(self.net_d, load_path, self.opt['path'].get('strict_load_d', True))
            
        self.net_d.train()
    
        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get('gan_opt'):
            self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device)

        self.net_d_iters = train_opt.get('net_d_iters', 1)
        self.net_d_init_iters = train_opt.get('net_d_init_iters', 0)

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()
    
    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            optim_params.append(v)
            if not v.requires_grad:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(optim_params,
                                                **train_opt['optim_g'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

        # optimizer d
        optim_type = train_opt['optim_d'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_d = torch.optim.Adam(self.net_d.parameters(),
                                                **train_opt['optim_d'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_d)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        train_opt = self.opt['train']

        for p in self.net_d.parameters():
            p.requires_grad = False
        self.optimizer_g.zero_grad()

        self.use_codebook_cls = self.opt['network_g'].get('use_codebook_cls', None) 
        if self.LQ_stage is True:
            if self.use_codebook_cls:
                with torch.no_grad():
                    self.gt_rec, _, _, gt_indices = self.net_hq(self.gt)
                    self.gt_indices = gt_indices
                self.output, l_ae, l_cls, _ = self.net_g(self.lq, gt_indices)
            else:
                outputs, l_ae, l_cls, _ = self.net_g(self.lq, gt_img=self.gt) 
                self.output_l1 = outputs[0]
                self.output_vqgan = outputs[1]
                self.output = outputs[-1]
        else:
            outputs, l_ae, l_cls, _ = self.net_g(self.lq) 
            self.output = outputs[-1]
            self.output_l1 = outputs[0]
            self.output_vqgan = outputs[1]

        l_g_total = 0
        loss_dict = OrderedDict()

        # ========================= Content loss ==========================
        # pixel loss
        #  if self.cri_pix and self.LQ_stage:
            #  l_pix = self.cri_pix(self.output_l1, self.gt)
            #  l_g_total += l_pix
            #  loss_dict['l_pix_rrdb'] = l_pix

        # ========================= Texture loss ==========================
        # pixel loss
        #  if self.cri_pix:
            #  l_pix = self.cri_pix(self.output_vqgan, self.gt)
            #  l_g_total += l_pix
            #  loss_dict['l_pix_vqgan'] = l_pix
        # perceptual loss
        #  if self.cri_perceptual:
            #  l_percep, l_style = self.cri_perceptual(self.output_vqgan, self.gt)
            #  if l_percep is not None:
                #  l_g_total += l_percep.mean()
                #  loss_dict['l_percep_vqgan'] = l_percep.mean()
            #  if l_style is not None:
                #  l_g_total += l_style
                #  loss_dict['l_style'] = l_style

        # ========================= Fusion loss ==========================
        # codebook loss
        if train_opt.get('codebook_opt'):
            l_ae *= train_opt['codebook_opt']['loss_weight'] 
            l_g_total += l_ae.mean()
            loss_dict['l_ae'] = l_ae.mean()

        # cls loss, only for LQ stage!
        if train_opt.get('cls_opt'):
            l_cls *= train_opt['cls_opt']['loss_weight'] 
            l_cls = l_cls.mean()
            l_g_total += l_cls
            loss_dict['l_cls'] = l_cls

        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_g_total += l_pix
            loss_dict['l_pix'] = l_pix
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_g_total += l_percep.mean()
                loss_dict['l_percep'] = l_percep.mean()
            if l_style is not None:
                l_g_total += l_style
                loss_dict['l_style'] = l_style
        # ========================= Texture branch ===============================
        
        # gan loss
        if self.use_dis and current_iter > train_opt['net_d_init_iters']:
            fake_g_pred = self.net_d(self.output)
            l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan

        l_g_total.mean().backward()
        self.optimizer_g.step()

        # optimize net_d
        self.fixed_disc = self.opt['network_g']['fixed_disc']
        if not self.fixed_disc and self.use_dis and current_iter > train_opt['net_d_init_iters']:
            for p in self.net_d.parameters():
                p.requires_grad = True
            self.optimizer_d.zero_grad()
            # real
            real_d_pred = self.net_d(self.gt)
            l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
            loss_dict['l_d_real'] = l_d_real
            loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
            l_d_real.backward()
            # fake
            fake_d_pred = self.net_d(self.output.detach())
            l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
            loss_dict['l_d_fake'] = l_d_fake
            loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
            l_d_fake.backward()
            self.optimizer_d.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)
        
    def test(self):
        self.net_g.eval()
        net_g = self.get_bare_model(self.net_g)
        org_use_sloss = net_g.use_semantic_loss 
        net_g.use_semantic_loss = False
        min_size = 8000 * 8000
        with torch.no_grad():
            #  lq_input = torch.nn.functional.interpolate(self.lq, scale_factor=1/2)
            lq_input = self.lq
            _, _, h, w = lq_input.shape
            if h*w < min_size:
                outputs, _, _, self.test_out_indices = self.net_g(lq_input)
                self.output = outputs[-1] 
                self.output_l1 = outputs[0]
            else:
                self.output = self.forward_chop(lq_input)
        self.net_g.train()
        net_g.use_semantic_loss = org_use_sloss 

        #  if self.has_gt_model:
            #  self.net_hq.eval()
            #  with torch.no_grad():
                #  self.gt_rec, _, _, self.gt_indices = self.net_hq(self.gt)

    def forward_chop(self, lq_input):
        div_sz = 512
        shave = 16

        lq_inputs = [
                lq_input[..., 0:div_sz+shave, 0:div_sz+shave],
                lq_input[..., 0:div_sz+shave, div_sz-shave:],
                lq_input[..., div_sz-shave:, 0:div_sz+shave],
                lq_input[..., div_sz-shave:, div_sz-shave:],
                ]

        chop_outputs = []
        for lq in lq_inputs:
            outputs, _, _, self.test_out_indices = self.net_g(lq)
            output = outputs[-1] 
            chop_outputs.append(output)
        
        scale = self.opt['scale']
        
        up_half = torch.cat((chop_outputs[0][..., 0:div_sz*scale, 0:div_sz*scale],
            chop_outputs[1][..., 0:div_sz*scale, shave*scale:]), dim=3)
        down_half = torch.cat((chop_outputs[2][..., shave*scale:, 0:div_sz*scale],
            chop_outputs[3][..., shave*scale:, shave*scale:]), dim=3)
        y = torch.cat((up_half, down_half), dim=2)

        return y 
    
    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, save_as_dir=None):
        logger = get_root_logger()
        logger.info('Only support single GPU validation.')
        self.nondist_validation(dataloader, current_iter, tb_logger, save_img, save_as_dir)

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img, save_as_dir):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
        pbar = tqdm(total=len(dataloader), unit='image')

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)

            # zero self.metric_results
            self.metric_results = {metric: 0 for metric in self.metric_results}
            self.key_metric = self.opt['val'].get('key_metric') 

        metric_data = dict()
        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()
            
            sr_img = tensor2img(self.output)
            metric_data['img'] = sr_img
            if hasattr(self, 'output_l1'):
                output_l1_img = tensor2img(self.output_l1) 
            if hasattr(self, 'gt'):
                gt_img = tensor2img(self.gt)
                metric_data['img2'] = gt_img

            # tentative for out of GPU memory
            # del self.lq
            # del self.output
            # torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], 'image_results',
                                             f'{current_iter}', 
                                             f'{img_name}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}_{self.opt["name"]}.png')
                if save_as_dir:
                    save_as_img_path = osp.join(save_as_dir, f'{img_name}.png')
                    imwrite(sr_img, save_as_img_path)
                    if hasattr(self, 'output_l1'):
                        save_as_img_l1_path = osp.join(save_as_dir + '_l1', f'{img_name}.png')
                        imwrite(output_l1_img, save_as_img_l1_path)
                imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    if name == 'lpips':
                        self.metric_results[name] += calculate_lpips(metric_data['img'], metric_data['img2'], self.lpips_fn)
                    else:
                        self.metric_results[name] += calculate_metric(metric_data, opt_)

            pbar.update(1)
            pbar.set_description(f'Test {img_name}')

        pbar.close()
            
        if with_metrics:
            # calculate average metric
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
            
            if self.key_metric is not None:
                # If the best metric is updated, update and save best model
                to_update = self._update_best_metric_result(dataset_name, self.key_metric, self.metric_results[self.key_metric], current_iter)
            
                if to_update:
                    for name, opt_ in self.opt['val']['metrics'].items():
                        self._update_metric_result(dataset_name, name, self.metric_results[name], current_iter)
                    self.copy_model(self.net_g, self.net_g_best)
                    self.copy_model(self.net_d, self.net_d_best)
                    self.save_network(self.net_g, 'net_g_best', '')
                    self.save_network(self.net_d, 'net_d_best', '')
            else:
                # update each metric separately 
                updated = []
                for name, opt_ in self.opt['val']['metrics'].items():
                    tmp_updated = self._update_best_metric_result(dataset_name, name, self.metric_results[name], current_iter)
                    updated.append(tmp_updated)
                # save best model if any metric is updated 
                if sum(updated): 
                    self.copy_model(self.net_g, self.net_g_best)
                    self.copy_model(self.net_d, self.net_d_best)
                    self.save_network(self.net_g, 'net_g_best', '')
                    self.save_network(self.net_d, 'net_d_best', '')

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
            
    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)
    
    def vis_single_code(self, up_factor=2):
        net_g = self.get_bare_model(self.net_g)
        org_use_sloss = net_g.use_semantic_loss 
        net_g.module.use_semantic_loss = False
        with torch.no_grad():
            code_idx = torch.arange(1024)
            input_res = 16 * 16 // self.latent_size * up_factor
            input_tensor = torch.randn(code_idx.shape[0], 3, input_res, input_res).cuda()
            code_idx = code_idx.repeat_interleave(up_factor**2)
            outputs, _, _, _ = self.net_g(input_tensor, gt_indices=[code_idx.cuda()])
            output_img = outputs[-1]
            output_img = tvu.make_grid(output_img, nrow=32)
        net_g.module.use_semantic_loss = org_use_sloss 
        return output_img.unsqueeze(0)

    def get_current_visuals(self):
        vis_samples = 16
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()[:vis_samples]
        out_dict['result'] = self.output.detach().cpu()[:vis_samples]
        if self.LQ_stage:
            out_dict['result_l1'] = self.output_l1.detach().cpu()[:vis_samples]
        out_dict['result_vqgan'] = self.output_vqgan.detach().cpu()[:vis_samples]
        if not self.LQ_stage:
            out_dict['codebook'] = self.vis_single_code()
        if hasattr(self, 'gt_rec'):
            out_dict['gt_rec'] = self.gt_rec.detach().cpu()[:vis_samples]
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()[:vis_samples]
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_network(self.net_d, 'net_d', current_iter)
        self.save_training_state(epoch, current_iter)
