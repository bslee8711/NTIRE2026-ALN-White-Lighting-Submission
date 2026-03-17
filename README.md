# NTIRE2026-ALN White Lighting Submission

This repository is for NTIRE2026 Ambient Light Normalization (ALN) White Lighting submission.

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
### PromptNorm

We use a modified version of [PromptNorm](https://github.com/davidserra9/promptnorm)
We adapted and extended the code for our NTIRE 2026 submission, including geometry-guided inputs and modified training/inference pipelines.

### IFBlend

We include a modified version of [IFBlend](https://github.com/fvasluianu97/IFBlend) directly in this repository.
We adapted IFBlend for integration into our pipeline.

Note:  
Pretrained weights are **not included**.  
Please download them from the official IFBlend repository and place them according to their instructions.

--- 
## External Repositories Setup

```markdown
We do not redistribute pretrained weights from external repositories.
Please obtain them directly from the official sources.
```

This project builds upon and modifies existing implementations from prior works.  
Please clone the following repositories and place them in your working directory:


### 2. [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)
Please follow the official repository instructions to:
	•	download pretrained depth estimation models
	•	place them in the correct locations as described in their README

## Running
After cloning Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2), download Challenge dataset and split it into trainset and validation set. 

```text
project_root/
├── data/
	├── train/
		├── in/
		├── gt/
	├── valid/
		├── in/
		├── gt/
├── IFBlend/
├── Depth-Anything-V2/
├── promptnorm/
└── requirements.txt
```
### IFBlend output generation
Change the Folder name following IFBlend official repository.
```bash
cd IFBlend/
python inference.py --data_src /path/to/input/ --ckp_dir checkpoints --res_dir /path/to/output/dir --load_from IFBlend_ambient6k
```
Then change the output file to out and place it in the same place as in/ and gt/

### Depth map generation
We used provided ViT-Large model to generate depth map in [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2).
```bash
cd Depth-Anything-V2/
python run.py --img-path /path/to/input/ --outdir /path/to/depth/ --encoder vitl --pred-only
```
Name the depth folder 'depth' and place it in the same place as in/, gt/, out/ 

### Normal map generation
Create a folder name 'normal' in the same place as in/, gt/, out/, depth.
```bash
cd promptnorm
python utils/depth2normal.py
```
After finishing all of these processes above, your folder will look like this:
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
### PromptNorm training
Training on single GPU
```bash
cd promptnorm
python train.py 
```
Modify the configuration of `options.py`.
