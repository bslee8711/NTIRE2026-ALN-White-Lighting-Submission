# NTIRE2026-ALN-White-Lighting-Submission

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
git clone !!!!!!!!!!!!!!!
cd !!!!!!!!!!
pip install -r requirements.txt
```

---
### PromptNorm

We use a modified version of [PromptNorm](https://github.com/davidserra9/promptnorm)
We adapted and extended the code for our NTIRE 2026 submission, including geometry-guided inputs and modified training/inference pipelines.

--- 
## External Repositories Setup

```markdown
We do not redistribute pretrained weights from external repositories.
Please obtain them directly from the official sources.
```

This project builds upon and modifies existing implementations from prior works.  
Please clone the following repositories and place them in your working directory:

### 1. [IFBlend](https://github.com/fvasluianu97/IFBlend)
Please follow the instructions in the IFBlend repository to:
	•	download pretrained weights
	•	place the weights in the appropriate directory as specified in their README

### 2. [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)
Please follow the official repository instructions to:
	•	download pretrained depth estimation models
	•	place them in the correct locations as described in their README
