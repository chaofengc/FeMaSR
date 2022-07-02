from PIL import Image

import torch
import torchvision.transforms as tf
from torchvision.utils import save_image
from vqgan_vis_arch import MultiScaleVQVAESemanticHQ

import numpy as np
import os
import random
from tqdm import tqdm
from kmeans_pytorch import kmeans
import cv2

# set random seeds to ensure reproducibility
seed = 123
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


color_table = np.array([
        [153, 153, 153],  # 0, background
        [0, 255, 255],    # 1, sky
        [109, 158, 235],  # 2, water
        [183, 225, 205],  # 3, grass
        [153, 0, 255],    # 4, mountain
        [17, 85, 204],    # 5, building
        [106, 168, 79],   # 6, plant
        [224, 102, 102],  # 7, animal
        [255, 255, 255],  # 8/255, void
        [153, 153, 153],  # 0, background
        [0, 255, 255],    # 1, sky
        #  [109, 158, 235],  # 2, water
        #  [183, 225, 205],  # 3, grass
        #  [153, 0, 255],    # 4, mountain
        #  [17, 85, 204],    # 5, building
        #  [106, 168, 79],   # 6, plant
        #  [224, 102, 102],  # 7, animal
        #  [153, 153, 153],  # 0, background
        #  [0, 255, 255],    # 1, sky
        #  [109, 158, 235],  # 2, water
        #  [183, 225, 205],  # 3, grass
        #  [153, 0, 255],    # 4, mountain
        #  [17, 85, 204],    # 5, building
        #  [106, 168, 79],   # 6, plant
        #  [224, 102, 102],  # 7, animal
    ])


def index_to_color(index_map):
    """shape: (H, W)"""
    color_map = np.zeros((index_map.shape[0], index_map.shape[1], 3)) 
    for i in range(color_table.shape[0]):
        color_map[index_map == i] = color_table[i]
    return color_map


def cluster_codebook(model, num_clusters, codebook=None):
    if codebook is None:
        codebook = model.quantize_group[0].embedding.weight
    else:
        codebook = codebook.squeeze()
    cluster_ids_x, cluster_centers = kmeans(X=codebook, num_clusters=num_clusters, distance='euclidean', iter_limit=500, device=torch.device('cuda'))
    #  cluster_ids_x, cluster_centers = kmeans(X=codebook, num_clusters=num_clusters, distance='cosine', iter_limit=500, device=torch.device('cuda'))
    return cluster_ids_x, cluster_centers

def vis_single_code(model, code_idx, up_factor=1):
    input_tensor = torch.randn(code_idx.shape[0], 3, 16*up_factor, 16*up_factor).cuda()
    code_idx = code_idx.repeat_interleave(up_factor**2)
    outputs, _, _, _ = model(input_tensor, gt_indices=[code_idx.cuda()])
    output_img = outputs[-1]
    return output_img.clamp(0, 1)

def vis_tiled_single_code(model, code_idx):
    input_tensor = torch.randn(code_idx.shape[0], 3, 256, 256).cuda()
    outputs, _, _, _ = model(input_tensor, gt_indices=[code_idx.repeat(16*16).cuda()])
    output_img = outputs[-1]
    return output_img

def vis_cluster_code(cluster_ids_x, model):
    num_clusters = torch.max(cluster_ids_x)
    new_order = []
    for i in range(num_clusters + 1):
        tmp_idx = torch.nonzero(cluster_ids_x == i)
        new_order.append(tmp_idx)        
        print(f'cluster {i}, shape', tmp_idx.shape)
    new_order = torch.cat(new_order)
    vis_img = vis_single_code(model, new_order)
    return new_order, vis_img

def get_useful_code(cluster_ids_x, model):
    num_clusters = torch.max(cluster_ids_x)
    new_order = []
    for i in range(num_clusters + 1):
        #  if not i in [24]:
        if i in [6]:
            tmp_idx = torch.nonzero(cluster_ids_x == i)
            new_order.append(tmp_idx)        
            print(f'cluster {i}, shape', tmp_idx.shape)
    new_order = torch.cat(new_order)
    codebook = model.quantize_group[0].embedding.weight
    new_codebook = codebook[new_order] 
    torch.save(new_codebook, 'useful_code_semantic.pth')
    print(new_codebook.shape)

def vis_random_codes(code_num, sample_num, model):
    output_imgs = []
    all_idx = torch.arange(16)
    for i in range(sample_num):
        selected_idx = all_idx[torch.randint(all_idx.shape[0], (code_num,))]
        #  tmp_idx = torch.randperm(16)
        #  selected_idx = all_idx[tmp_idx[:code_num]]
        print(code_num, i, selected_idx.squeeze().numpy())
        sampled_code_idx = selected_idx[torch.randint(selected_idx.shape[0], (16*16,))]
        input_tensor = torch.randn(1, 3, 256, 256).cuda()
        outputs, _, _, _ = model(input_tensor, gt_indices=[sampled_code_idx.cuda()])
        output_img = outputs[-1]
        output_imgs.append(output_img)
    output_img = torch.cat(output_imgs, dim=0)
    return output_img


def vis_seq_codes(code_num, sample_num, model):
    output_imgs = []
    all_idx = torch.arange(16)
    for i in range(sample_num):
        selected_idx = torch.arange(code_num)
        sampled_code_idx = selected_idx.repeat(256//(i+1) + 1)[:256] 
        input_tensor = torch.randn(1, 3, 256, 256).cuda()
        outputs, _, _, _ = model(input_tensor, gt_indices=[sampled_code_idx.cuda()])
        output_img = outputs[-1]
        output_imgs.append(output_img)
    output_img = torch.cat(output_imgs, dim=0)
    return output_img


def vis_cluster_samples(cluster_ids_x, cluster_id, model, sample_num=1):
    selected_idx = torch.nonzero(cluster_ids_x == cluster_id)
    print(cluster_id, selected_idx.squeeze().cpu().numpy())
    output_imgs = []
    for i in range(sample_num):
        sampled_code_idx = selected_idx[torch.randint(selected_idx.shape[0], (16*16,))]
        input_tensor = torch.randn(1, 3, 256, 256).cuda()
        outputs, _, _, _ = model(input_tensor, gt_indices=[sampled_code_idx.cuda()])
        output_img = outputs[-1]
        output_imgs.append(output_img)
    output_img = torch.cat(output_imgs, dim=0)
    return output_img


def vis_given_code_nums(code_num, model, sample_num=8):
    output_imgs = []
    all_idx = torch.arange(19)
    for i in range(sample_num):
        if len(code_num) > 0:
            selected_idx = code_num
        else:
            selected_idx = all_idx[torch.randint(all_idx.shape[0], (code_num,))]
        print(i, selected_idx)
        sampled_code_idx = selected_idx[torch.randint(selected_idx.shape[0], (16*16,))]
        input_tensor = torch.randn(1, 3, 256, 256).cuda()
        outputs, _, _, _ = model(input_tensor, gt_indices=[sampled_code_idx.cuda()])
        output_img = outputs[-1]
        output_imgs.append(output_img)
    output_img = torch.cat(output_imgs, dim=0)
    return output_img

def vis_given_code_list(code_list, model, sample_num=8):
    output_imgs = []
    for i in range(sample_num):
        selected_idx = torch.tensor(code_list) 
        sampled_code_idx = selected_idx[torch.randint(selected_idx.shape[0], (16*16,))]
        input_tensor = torch.randn(1, 3, 256, 256).cuda()
        outputs, _, _, _ = model(input_tensor, gt_indices=[sampled_code_idx.cuda()])
        output_img = outputs[-1]
        output_imgs.append(output_img)
    output_img = torch.cat(output_imgs, dim=0)
    return output_img

def read_img_tensor(img_path):
    img = Image.open(img_path)
    img_tensor = tf.functional.to_tensor(img)
    return img_tensor.unsqueeze(0).cuda()

viscode = False 
semantic = False 
#  semantic = False 
save_suffix = 'semantic' if semantic else 'nosemantic'

if semantic:
    weight_path = './experiments/0001_VQGAN_SemanticGuide_RRDBFuse_HQ_stage/models/net_g_290000.pth'
    model = MultiScaleVQVAESemanticHQ(with_semantic=True).cuda() 
else:
    #  weight_path = './experiments/101_1_scale_HQ_stage_1024_codebook_largeDataset/models/net_g_110000.pth'
    weight_path = './experiments/0005_VQGAN_MultiscaleNoSemantic_NoAttn_RRDBFuse_HQ_stage/models/net_g_latest.pth'
    model = MultiScaleVQVAESemanticHQ(act_type='gelu', codebook_params=[[16, 1024, 512]]).cuda() 

model.load_state_dict(torch.load(weight_path)['params'], strict=True)

#  img = vis_given_code_nums(torch.tensor([0, 8, 15, 5, 6, 7]), model, 8)
#  save_image(img, f'../tmp_visdir/vis_test_given_id_{save_suffix}.png', nrow=8)
#  exit()
#  num_clusters = 10 
#  cluster_ids_x, cluster_centers = cluster_codebook(model, num_clusters)
#  new_order, vis_img = vis_cluster_code(cluster_ids_x, model)
#  save_image(vis_img, f'vis_clustered_codes_{save_suffix}.png', nrow=32)
#  for i in range(num_clusters):
#  for i in range(1):
    #  tmp_img = vis_cluster_samples(cluster_ids_x, i, model)
    #  save_image(tmp_img, f'../tmp_visdir/vis_sample_cluster{i}_{save_suffix}.png')

save_root = './tmp_vis/'

#  up_factor = 1
#  code_vis = vis_single_code(model, torch.arange(1024), up_factor)
#  save_image(code_vis, f'{save_root}/vis_useful_codes_{up_factor}_{save_suffix}.png', nrow=32)
#  exit()

#  for up_factor in range(1, 9):
    #  code_vis = vis_single_code(model, torch.arange(19), up_factor)
    #  code_vis = torch.nn.functional.interpolate(code_vis, (256, 256), mode='bicubic')
    #  save_image(code_vis, f'../tmp_visdir/vis_useful_codes_{up_factor}_{save_suffix}.png', nrow=8)

#  vis_imgs = []
#  sample_num = 16 
#  for i in range(4, 8):
    #  code_num = i + 1
    #  tmp_img = vis_random_codes(code_num, sample_num, model)
    #  vis_imgs.append(tmp_img)
#  save_image(torch.cat(vis_imgs, dim=0), f'{save_root}/vis_random_sample_codes_{save_suffix}.png', nrow=sample_num)

#  semantic_indexes = [
        #  [1, 2, 5, 7, 13, 14],
        #  [2, 8, 7, 10, 12, 13, 14],
        #  [3, 5, 7, 14],
        #  [1, 4, 6, 9, 10, 12],
        #  [1, 3, 4, 6, 13, 15]
        #  ]
#  for sl in semantic_indexes:
    #  print(' '.join([chr(x+ord('a')) for x in sl]))
#  exit()
#  sample_num = 8
#  for sidx, idx_list in enumerate(semantic_indexes):
    #  tmp_img = vis_given_code_list(idx_list, model, sample_num=sample_num)
    #  save_image(tmp_img, f'{save_root}/vis_semantic_codes_sample_{sidx}.png', nrow=sample_num)

#  vis_imgs = []
#  sample_num = 1 
#  for i in range(8):
    #  tmp_img = vis_seq_codes(i+1, sample_num, model)
    #  vis_imgs.append(tmp_img)
#  save_image(torch.cat(vis_imgs, dim=0), f'../tmp_visdir/vis_seq_tiled_codes_{save_suffix}.png', nrow=sample_num)

#  get_useful_code(cluster_ids_x, model)

## ================= visualize semantic code center ==============
#  num_clusters = 8 
#  cluster_ids_x, cluster_centers = cluster_codebook(model, num_clusters, codebook=torch.load('./useful_code_semantic.pth'))
#  vis_imgs = []
#  for i in range(num_clusters):
    #  out_img = vis_cluster_samples(cluster_ids_x, i, model, sample_num=8)
    #  vis_imgs.append(out_img)
#  save_image(torch.cat(vis_imgs, dim=0), f'../tmp_visdir/vis_random_sample_center_{save_suffix}.png', nrow=8)
## ================= visualize semantic code center ==============

## ================= visualize code index =======================
#  img_root = '../datasets/test_real_datasets_mod16/OutdoorSceneTest300/lrx4/'
#  for img_name in tqdm(os.listdir(img_root)):
    #  img_path = os.path.join(img_root, img_name)
    #  img_tensor = read_img_tensor(img_path)
    #  outputs, _, _, _ = model(img_tensor)
    #  output_img = outputs[-1]
    #  save_image(output_img, f'../tmp_visdir/rec/vis_vqgan_rec_{img_name}')

#  img_path = '../tmp_visdir/OST_001.png'
#  gt_label = cv2.imread(img_path)[:, :, 0]
#  color_gt_label = index_to_color(gt_label) 
#  cv2.imwrite('../tmp_visdir/OST_001_color.png', color_gt_label.astype(np.uint8))

#  img_path = '../datasets/test_real_datasets_mod16/OutdoorSceneTest300/lrx4/OST_001.png'
#  num_clusters = 4 
#  cluster_ids_x, cluster_centers = cluster_codebook(model, num_clusters, codebook=torch.load('./useful_code_semantic.pth'))

#  img_root = '../datasets/test_real_datasets_mod16/OutdoorSceneTest300/lrx4/'
#  for img_name in tqdm(os.listdir(img_root)):
    #  img_path = os.path.join(img_root, img_name)
    #  img_tensor = read_img_tensor(img_path)
    #  outputs, _, _, match_indices = model(img_tensor)
    #  match_indice = match_indices[0].squeeze() 

    #  clustered_index = cluster_ids_x[match_indice.view(-1)].reshape_as(match_indice)
    #  clustered_index = clustered_index.cpu().numpy().astype(np.uint8)
    #  clustered_index = cv2.resize(clustered_index, None, fx=16, fy=16, interpolation=cv2.INTER_CUBIC)
    #  colored_clustered_index = index_to_color(clustered_index.astype(np.int))
    #  cv2.imwrite(f'../tmp_visdir/rec_index_maps/{img_name}', colored_clustered_index.astype(np.uint8))
## ================= visualize code index =======================
#  exit()

img_path = '../datasets/test_datasets/Set5/gt_mod16/baby.png'
img_tensor = read_img_tensor(img_path)
if viscode:
    img_tensor = torch.nn.functional.interpolate(img_tensor, (16, 16))
else:
    img_tensor = torch.nn.functional.interpolate(img_tensor, (256, 256))

with torch.no_grad():
    if viscode:
        save_image(output_img, f'tmp_visdir/vis_codebook_{save_suffix}.png', nrow=32)
    else:
        outputs, _, _, _ = model(img_tensor)
        output_img = outputs[-1]
        #  save_image(output_img, f'vis_vqgan_rec_fullcode_{save_suffix}.png')
        save_image(output_img, f'./tmp_vis/vis_vqgan_rec_usefulcode_{save_suffix}.png')
