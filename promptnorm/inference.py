import os
import time
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import lightning.pytorch as pl

from utils.aln_dataset_inference import ALNDatasetGeom
from model import PromptNorm
from options import options as opt


class PromptNormModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = PromptNorm(decoder=True)
        self.loss_fn = nn.L1Loss()  

    def forward(self, x, normal):
        return self.net(x, normal)

    def tile_forward(self, img, normal, tile=512, overlap=128):
        """
        img:    (B,3,H,W)
        normal: (B,3,H,W)
        return: (B,3,H,W)
        """
        B, C, H, W = img.shape
        stride = tile - overlap

        out = torch.zeros((B, C, H, W), device=img.device, dtype=img.dtype)
        wgt = torch.zeros((B, 1, H, W), device=img.device, dtype=img.dtype)

        wy = torch.hann_window(tile, device=img.device, dtype=img.dtype).view(1, 1, tile, 1)
        wx = torch.hann_window(tile, device=img.device, dtype=img.dtype).view(1, 1, 1, tile)
        win = (wy * wx).clamp(min=1e-3)

        for y0 in range(0, H, stride):
            for x0 in range(0, W, stride):
                y1 = min(y0 + tile, H)
                x1 = min(x0 + tile, W)
                y0c = max(y1 - tile, 0)
                x0c = max(x1 - tile, 0)

                it = img[:, :, y0c:y1, x0c:x1]
                nt = normal[:, :, y0c:y1, x0c:x1]

                pad_h = tile - (y1 - y0c)
                pad_w = tile - (x1 - x0c)
                if pad_h > 0 or pad_w > 0:
                    it = F.pad(it, (0, pad_w, 0, pad_h), mode="reflect")
                    nt = F.pad(nt, (0, pad_w, 0, pad_h), mode="reflect")

                ot = self.net(it, nt)

                ot = ot[:, :, : (y1 - y0c), : (x1 - x0c)]
                wt = win[:, :, : (y1 - y0c), : (x1 - x0c)]

                out[:, :, y0c:y1, x0c:x1] += ot * wt
                wgt[:, :, y0c:y1, x0c:x1] += wt

        return out / (wgt + 1e-8)


def load_only_net_from_ckpt(model: PromptNormModel, ckpt_path: str):

    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    net_sd = {}
    for k, v in sd.items():
        if k.startswith("net."):
            net_sd[k[len("net."):]] = v

    if len(net_sd) == 0:
        net_sd = sd

    missing, unexpected = model.net.load_state_dict(net_sd, strict=False)
    if missing:
        print("[load] Missing keys (first 20):", missing[:20])
    if unexpected:
        print("[load] Unexpected keys (first 20):", unexpected[:20])


def save_tensor_as_png01(output: torch.Tensor, save_path: str):
    """
    output: (1,3,H,W) or (3,H,W), expected in [0,1]
    """
    if output.dim() == 4:
        output = output[0]
    output = output.detach().float().cpu().clamp(0, 1)
    arr = (output.permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(arr).convert("RGB").save(save_path)


def main():
    print("Options")
    print(opt)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.cuda)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    valset = ALNDatasetGeom(
        input_folder=opt.test_input_dir,
        geom_folder=opt.test_normals_dir,
        target_folder=None,
        normal_suffix="_normal.png"
    )

    valloader = DataLoader(
        valset,
        batch_size=1,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
        num_workers=opt.num_workers
    )

    model = PromptNormModel()
    load_only_net_from_ckpt(model, opt.pretrained_ckpt_path)
    model.eval().to(device)

    os.makedirs(opt.output_path, exist_ok=True)

    times = []

    for batch in tqdm(valloader, desc="Inference"):
        ([name, _], input_img, normal_img, _) = batch

        input_img = input_img.to(device, non_blocking=True)
        normal_img = normal_img.to(device, non_blocking=True)

        start = time.time()
        with torch.no_grad():
            output = model.tile_forward(input_img, normal_img, tile=512, overlap=96)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        times.append(time.time() - start)

        fname = name[0] if isinstance(name, (list, tuple)) else name
        fname = str(fname)

        # eliminate IFBlend suffix
        fname = fname.replace("_out.png", ".png")

        if not fname.lower().endswith(".png"):
            fname += ".png"

        save_path = os.path.join(opt.output_path, fname)
        save_tensor_as_png01(output, save_path)

    print(f"Average inference time per image: {float(np.mean(times)):.4f} s")
    print(f"Saved outputs to: {opt.output_path}")


if __name__ == "__main__":
    main()