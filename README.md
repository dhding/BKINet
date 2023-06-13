# Bilateral Knowledge Interaction Network for Referring Image Segmentation

## Environment preparation
```bash
conda create -n BKINet python=3.6
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install -r requirements.txt
```

## Dataset Preparation

### 1. Download the COCO train2014 to BKINet/ln_data/images.
```bash
wget https://pjreddie.com/media/files/train2014.zip
```

### 2. Download the RefCOCO, RefCOCO+, RefCOCOg to BKINet/ln_data.
```bash
cd ln_data
wget https://bvisionweb1.cs.unc.edu/licheng/referit/data/refclef.zip
wget https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip
wget https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip
wget https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip

```

### 3. Run dataset/data_process.py to generate the annotations.
```bash
cd dataset
python data_process.py --data_root ../ln_data --output_dir ../ln_data --dataset [refcoco/refcoco+/refcocog] --split unc --generate_mask
```

### 4. Process annotations to generate *.pth.
```
cd dataset
python datascript.py
```
In line 25, 26, 27, the 'input_txt' 'dataset' 'split' should be modified.

## Training
```bash
CUDA_VISIBLE_DEVICES = 0 python train_model.py --dataset [refcoco/refcoco+/refcocog]
```

##  Testing
```bash
CUDA_VISIBLE_DEVICES = 0 python test_model.py --dataset [refcoco/refcoco+/refcocog] --resume saved_models/modelname.pth.tar
```

## Acknowledgement
Thanks for a lot of codes from [onestage_grounding](https://github.com/zyang-ur/onestage_grounding) , [VLT](https://github.com/henghuiding/Vision-Language-Transformer), [CLIP](https://github.com/openai/CLIP), [K-Net](https://github.com/ZwwWayne/K-Net) , [DETR](https://github.com/facebookresearch/detr) and the framework of CLIP using for backbone pretraining.