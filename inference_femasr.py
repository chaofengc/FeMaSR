import argparse
import cv2
import glob
import os
from tqdm import tqdm
import torch
import torch.nn.functional as F

from basicsr.utils import img2tensor, tensor2img, imwrite 
from basicsr.data.transforms import mod_crop
from basicsr.archs.femasr_arch import FeMaSRNet 


def main():
    """Inference demo for FeMaSR 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='inputs', help='Input image or folder')
    parser.add_argument('-w', '--weight', type=str, default='', help='path for model weights')
    parser.add_argument('-o', '--output', type=str, default='results', help='Output folder')
    parser.add_argument('-s', '--outscale', type=float, default=4, help='The final upsampling scale of the image')
    parser.add_argument('--suffix', type=str, default='', help='Suffix of the restored image')
    parser.add_argument('--mod_scale', type=int, default=16, help='Pre padding size to be divisible by 16')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    
    # set up the model
    qsr_model = FeMaSRNet(codebook_params=[[32, 1024, 512]], LQ_stage=True).to(device)
    qsr_model.load_state_dict(torch.load(args.weight)['params'], strict=False)
    qsr_model.eval()
    
    os.makedirs(args.output, exist_ok=True)
    if os.path.isfile(args.input):
        paths = [args.input]
    else:
        paths = sorted(glob.glob(os.path.join(args.input, '*')))

    pbar = tqdm(total=len(paths), unit='image')
    for idx, path in enumerate(paths):
        img_name = os.path.basename(path)
        pbar.set_description(f'Test {img_name}')

        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img_tensor = img2tensor(img).to(device) / 255.
        img_tensor = img_tensor.unsqueeze(0)

        max_size = 600 * 600
        h, w = img_tensor.shape[2:]
        if h * w < max_size: 
            output = qsr_model.test(img_tensor)
        else:
            output = qsr_model.test_tile(img_tensor)
        output_img = tensor2img(output)

        save_path = os.path.join(args.output, f'{img_name}')
        imwrite(output_img, save_path)
        pbar.update(1)
    pbar.close()


if __name__ == '__main__':
    main()
