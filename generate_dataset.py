import os
import cv2
import numpy as np
import random
from tqdm import tqdm
from multiprocessing import Pool

from basicsr.data.bsrgan_util import degradation_bsrgan

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, max_dataset_size=float("inf"), followlinks=True):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir, followlinks=followlinks)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]

def degrade_img(hr_path, save_path):
    img_gt = cv2.imread(hr_path).astype(np.float32) / 255.
    img_gt = img_gt[:, :, [2, 1, 0]] # BGR to RGB
    img_lq, img_gt = degradation_bsrgan(img_gt, sf=scale, use_crop=False)
    img_lq = (img_lq[:, :, [2, 1, 0]] * 255).astype(np.uint8)
    print(f'Save {save_path}')
    cv2.imwrite(save_path, img_lq)


seed = 123
random.seed(seed)
np.random.seed(seed)

# scale = 2
scale = 4
hr_img_list = make_dataset('../datasets/HQ_sub')
pool = Pool(processes=40)

#  hr_img_list = ['../datasets/HQ_sub_samename/DIV8K_train_HR_sub/div8k_1383_s021.png']

#  scale = 2
#  hr_img_list = ['../datasets/HQ_sub_samename/DIV8K_train_HR_sub/div8k_0903_s056.png']

#  scale = 4
#  hr_img_list = make_dataset('../datasets/LQ_sub_samename_X4')

for hr_path in hr_img_list:
    save_path = hr_path.replace('HQ_sub', f'LQ_sub_X{scale}')
    save_path = save_path.replace('HR', 'LR')
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    pool.apply_async(degrade_img(hr_path, save_path))

pool.close()
pool.join()

