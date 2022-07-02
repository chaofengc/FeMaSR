from itertools import count
from tokenize import PlainToken
import torch
import torchvision.transforms as tf
from torchvision.utils import save_image

import numpy as np
import os
import random
from tqdm import tqdm
import cv2
from matplotlib import pyplot as plt
import seaborn as sns

from basicsr.utils.misc import set_random_seed
from basicsr.utils import img2tensor, tensor2img, imwrite 
from basicsr.archs.femasr_arch import FeMaSRNet 


def reconstruct_ost(model, data_dir, save_dir, maxnum=100):

    texture_classes = list(os.listdir(data_dir))
    texture_classes.remove('manga109')
    code_idx_dict = {}
    for tc in texture_classes:
        img_name_list = os.listdir(os.path.join(data_dir, tc))
        random.shuffle(img_name_list)
        tmp_code_idx_list = []
        for img_name in tqdm(img_name_list[:maxnum]):
            img_path = os.path.join(data_dir, tc, img_name)

            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            img_tensor = img2tensor(img).to(device) / 255.
            img_tensor = img_tensor.unsqueeze(0)

            rec, _, _, indices = model(img_tensor) 
            indices = indices[0]

            save_path = os.path.join(save_dir, tc, img_name)
            if not os.path.exists(os.path.join(save_dir, tc)):
                os.makedirs(os.path.join(save_dir, tc), exist_ok=True)
            imwrite(tensor2img(rec), save_path)

            save_org_dir = save_dir.replace('rec', 'org')
            save_org_path = os.path.join(save_org_dir, tc, img_name)
            if not os.path.exists(os.path.join(save_org_dir, tc)):
                os.makedirs(os.path.join(save_org_dir, tc), exist_ok=True)
            imwrite(tensor2img(img_tensor), save_org_path)

            tmp_code_idx_list.append(indices)
        code_idx_dict[tc] = tmp_code_idx_list
    
    torch.save(code_idx_dict, './tmp_code_vis/code_idx_dict.pth')


def vis_hrp(model, code_list_path, save_dir, samples_each_class=16):
    code_idx_dict = torch.load(code_list_path)
    classes = list(code_idx_dict.keys())
    
    latent_size = 8 
    color_palette = sns.color_palette()
    for idx, (key, value) in enumerate(code_idx_dict.items()):
        all_idx = torch.cat([x.flatten() for x in value]) 

        plt.figure(figsize=(16, 8)) 
        sns.histplot(all_idx.cpu().numpy(), color=color_palette[idx])
        plt.xlabel(key, fontsize=30)
        plt.ylabel('Count', fontsize=30)
        plt.savefig(f'./tmp_code_vis/code_stat/code_index_bincount_{key}.pdf')

        counts = all_idx.bincount()
        dist = counts / sum(counts)
        dist = dist.cpu().numpy()

        vis_tex_samples = []
        for sid in range(32): 
            vis_tex_map = np.random.choice(np.arange(dist.shape[0]), latent_size ** 2, p=dist)
            vis_tex_map = torch.from_numpy(vis_tex_map).to(all_idx)
            vis_tex_map = vis_tex_map.reshape(1, 1, latent_size, latent_size)
            vis_tex_img = model.decode_indices(vis_tex_map)
            vis_tex_samples.append(vis_tex_img)
        vis_tex_samples = torch.cat(vis_tex_samples, dim=0)
        save_image(vis_tex_samples, f'./tmp_code_vis/tmp_tex_vis/{key}.jpg', normalize=True, nrow=16)

if __name__ == '__main__':
    # set random seeds to ensure reproducibility
    set_random_seed(123)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    
    # set up the model
    weight_path = './experiments/pretrained_models/QuanTexSR/pretrain_semantic_vqgan_net_g_latest.pth'
    vqgan = FeMaSRNet(codebook_params=[[32, 1024, 512]], LQ_stage=False).to(device)
    vqgan.load_state_dict(torch.load(weight_path)['params'], strict=False)
    vqgan.eval()

    reconstruct_ost(vqgan, '../datasets/SR_OST_datasets/OutdoorSceneTrain_v2/', './tmp_code_vis/ost_rec', maxnum=1000) 
    vis_hrp(vqgan, './tmp_code_vis/code_idx_dict.pth', './tmp_code_vis/')
