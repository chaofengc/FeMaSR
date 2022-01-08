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

# create new anaconda env
conda create -n quantexsr python=3.8
source activate quantexsr

# install python dependencies
pip3 install -r requirements.txt
python setup.py develop
```

## Quick Inference

Download pretrained model from [BaiduNetdisk](), extract code `qtsr` (only provide x4 model now). Test the model with the following script
```
python inference_quantexsr.py -w ./path/to/model/weight -i ./path/to/test/image[or folder]
```

## Train the model


### Train SR model

```
python basicsr/train.py -opt options/train_QuanTexSR_LQ_stage.yml
```

### Model pretrain

In case you want to pretrain your own VQGAN prior, we also provide the training scripts.

#### Pretrain semantic codebook

The semantic-aware codebook is obtained with VGG19 features using a mini-batch version of K-means, optimized with Adam. This script will give three levels of codebooks from `relu3_4`, `relu4_4` and `relu5_4` features. We use `relu4_4` for this project.

```
python basicsr/train.py -opt options/train_QuanTexSR_semantic_cluster_stage.yml
```

#### Pretrain of semantic-aware VQGAN

```
python basicsr/train.py -opt options/train_QuanTexSR_HQ_pretrain_stage.yml
```

## Acknowledgement

This project is based on [BasicSR](https://github.com/xinntao/BasicSR).
