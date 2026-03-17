# NTIRE2026-ALN White Lighting Submission

This repository contains solution of Team GeoNorm for the NTIRE 2026 Ambient Light Normalization (ALN) White Lighting challenge.

---

## Method Overview

Our solution follows a geometry-aware pipeline for ambient lighting normalization:

1. **IFBlend** for initial illumination normalization (frequency-aware preprocessing)
2. **Depth Anything V2** for monocular depth estimation
3. **Depth-to-normal conversion** to extract surface normal maps
4. **PromptNorm** for geometry-guided image restoration

By combining frequency-domain preprocessing and geometry-aware restoration, the model aims to improve illumination consistency while preserving structural details.

---
## Installation
```bash
git clone https://github.com/bslee8711/NTIRE2026-GeoNorm.git
cd NTIRE2026-GeoNorm
pip install -r requirements.txt
```

---
## Code Base and Dependencies

### [PromptNorm](https://github.com/davidserra9/promptnorm)

We use a modified version of [PromptNorm](https://github.com/davidserra9/promptnorm)
We adapted and extended the code for our NTIRE 2026 submission, including geometry-guided inputs and modified training/inference pipelines.

### [IFBlend](https://github.com/fvasluianu97/IFBlend)

We include a modified version of [IFBlend](https://github.com/fvasluianu97/IFBlend) directly in this repository.
We adapted IFBlend for integration into our pipeline.


### [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)
Please clone the official repository and place it in the project root directory 
(same level as `PromptNorm` and `IFBlend`):
```bash
git clone https://github.com/DepthAnything/Depth-Anything-V2.git
```

Note:  
Pretrained weights are **not included**.  
Please download them from the official IFBlend repository and place them according to their instructions.

## Running
After cloning Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) and preparing the dataset into training and validation sets as follows:

```text
project_root/
├── data/
│   ├─ train/
│	│	├── in/
│	│	└── gt/
│	└── valid/
│		├── in/
│		└──  gt/
├── IFBlend/
├── Depth-Anything-V2/
├── promptnorm/
└── requirements.txt
```
### 1. IFBlend Preprocessing
Follow the IFBlend naming convention for input/output directories.
```bash
cd IFBlend/
python inference.py --data_src /path/to/input/ --ckp_dir checkpoints --res_dir /path/to/output/dir --load_from IFBlend_ambient6k
```
After inference:
rename the output directory to `out`
place it alongside `in/` and `gt/`


### 2. Depth map generation
We use the ViT-Large model provided by [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2):
```bash
cd Depth-Anything-V2/
python run.py --img-path /path/to/input/ --outdir /path/to/depth/ --encoder vitl --pred-only
```
Rename output directory to `depth` and place it alongside: `in/`, `gt/`, and `out/'

### 3. Normal map generation
Create a directory named `normal` in the same location
```bash
cd promptnorm
python utils/depth2normal.py
```
After finishing all preprocessing steps, the directory structure should be:
```text
project_root/
├── data/
│	├── train/
│	├────── in/
│	│		└── 0_in.png
│	├────── out/
│	│		└── 0_in_out.png
│	├────── gt/
│	│		└── 0_gt.png
│	├────── normal/
│	│		└── 0_normal.png
│	├────── depth/
│	│		└── 0_in.png
│	│
│	└── valid/
│		├── in/
│		├── gt/
│		├── out/
│		├── normal/
│		└──depth/
├── IFBlend/
├── Depth-Anything-V2/
├── promptnorm/
└── requirements.txt
```
### 4. PromptNorm training
Training (single GPU):
```bash
cd promptnorm
python train.py --cuda 0 --num_gpus 1 --epochs 30 --batch_size 1 --patch_size 512 --num_workers 8 --train_input_dir /path/to/out/ --train_normals_dir /path/to/normal/ --train_target_dir /path/to/gt/ --test_input_dir /path/to/out(test)/ --test_normals_dir /path/to/normal(test)/ --test_target_dir /path/to/gt(test)/ --ckpt_dir /path/to/checkpoint/output/
```
You may adjust training settings in `options.py` according to your environment and dataset configuration.

### 5. Inference
For inference, apply the same preprocessing steps used during training (IFBlend, depth estimation, and normal map generation) to the test set.
Then run:
```bash
cd promptnorm
python inference.py --cuda 0 --test_input_dir /path/to/out/ --test_normals_dir /path/to/normal/ --pretrained_ckpt_path /path/to/ckpt/ --output_path /path/to/output/
```
