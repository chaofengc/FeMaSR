# QuanTexSR 

This is the official PyTorch codes for the paper "Blind Image Super Resolution with Semantic-Aware Quantized Texture Prior"

## Dependencies and Installation

- Ubuntu >= 18.04
- CUDA >= 11.0 
- Other required packages in `requirements.txt`
```
# git clone this repository
git clone https://github.com/chaofengc/QuanTexSR.git
cd QuanTexSR
# create new anaconda env and install dependencies
pip3 install -r requirements.txt
python setup.py develop
```

## Quick Inference

Download pretrained model from [BaiduNetdisk](), extract code `qtsr` (only provide x4 model now). Test the model with the following script
```
python inference_quantexsr.py -w ./path/to/model/weight -i ./path/to/test/image[or folder]
```

## Train the model

### Pretrain of semantic codebook

### Pretrain of semantic-aware VQGAN

### Train SR model 

## Acknowledgement

This project is based on [BasicSR](https://github.com/xinntao/BasicSR).
