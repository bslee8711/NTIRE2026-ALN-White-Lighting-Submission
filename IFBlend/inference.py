import argparse
import os
from glob import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from PIL import Image
from tqdm import tqdm

from utils import load_checkpoint
from utils_model import get_model


class ImageOnlyDataset(Dataset):
    def __init__(self, data_src):
        super().__init__()
        self.data_src = data_src
        self.to_tensor = ToTensor()

        exts = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"]
        self.image_paths = []
        for ext in exts:
            self.image_paths.extend(glob(os.path.join(self.data_src, ext)))
        self.image_paths = sorted(self.image_paths)

        print(f"Found {len(self.image_paths)} input images")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        img = self.to_tensor(img)
        name = os.path.basename(path)  
        return name, img


class TiledModel(nn.Module):
    def __init__(self, model, tile_size=512, overlap=64, use_amp=True):
        super().__init__()
        self.model = model
        self.tile_size = tile_size
        self.overlap = overlap
        self.use_amp = use_amp

    def forward(self, x):
        # x: [B, C, H, W]
        b, c, h, w = x.shape

        if h <= self.tile_size and w <= self.tile_size:
            if self.use_amp and x.is_cuda:
                with torch.amp.autocast("cuda"):
                    return self.model(x)
            return self.model(x)

        stride = self.tile_size - self.overlap
        assert stride > 0, "overlap must be smaller than tile_size"

        outputs = []
        for bi in range(b):
            img = x[bi:bi + 1]  # [1, C, H, W]

            out_sum = None
            weight_sum = None

            ys = list(range(0, max(h - self.tile_size, 0) + 1, stride))
            xs = list(range(0, max(w - self.tile_size, 0) + 1, stride))

            if len(ys) == 0 or ys[-1] != h - self.tile_size:
                ys.append(max(h - self.tile_size, 0))
            if len(xs) == 0 or xs[-1] != w - self.tile_size:
                xs.append(max(w - self.tile_size, 0))

            for y in ys:
                for x0 in xs:
                    tile = img[:, :, y:y + self.tile_size, x0:x0 + self.tile_size]

                    pad_h = self.tile_size - tile.shape[2]
                    pad_w = self.tile_size - tile.shape[3]
                    if pad_h > 0 or pad_w > 0:
                        tile = F.pad(tile, (0, pad_w, 0, pad_h), mode="reflect")

                    if self.use_amp and tile.is_cuda:
                        with torch.amp.autocast("cuda"):
                            pred_tile = self.model(tile)
                    else:
                        pred_tile = self.model(tile)

                    pred_tile = pred_tile[:, :, :min(self.tile_size, h - y), :min(self.tile_size, w - x0)]

                    if out_sum is None:
                        out_channels = pred_tile.shape[1]
                        out_sum = torch.zeros((1, out_channels, h, w), device=pred_tile.device, dtype=pred_tile.dtype)
                        weight_sum = torch.zeros((1, 1, h, w), device=pred_tile.device, dtype=pred_tile.dtype)

                    out_sum[:, :, y:y + pred_tile.shape[2], x0:x0 + pred_tile.shape[3]] += pred_tile
                    weight_sum[:, :, y:y + pred_tile.shape[2], x0:x0 + pred_tile.shape[3]] += 1.0

                    del tile, pred_tile

            out = out_sum / weight_sum.clamp(min=1e-8)
            outputs.append(out)

        return torch.cat(outputs, dim=0)


def save_tensor_as_png(tensor, save_path):
    """
    tensor: [1, C, H, W] in [0,1]
    """
    tensor = tensor[0].detach().float().cpu().clamp(0, 1)
    arr = (tensor.permute(1, 2, 0).numpy() * 255.0 + 0.5).astype("uint8")
    Image.fromarray(arr).convert("RGB").save(save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="ifblend", help="Name of the tested model")
    parser.add_argument("--data_src", required=True, help="Folder containing input test images only")
    parser.add_argument("--res_dir", default="./final-results", help="Path for output dir")
    parser.add_argument("--ckp_dir", default="./checkpoints", help="Checkpoint root dir")
    parser.add_argument("--load_from", default="IFBlend_ambient6k", help="Checkpoint experiment name")
    parser.add_argument("--tile_size", type=int, default=3000, help="Tile size for tiled inference")
    parser.add_argument("--overlap", type=int, default=32, help="Overlap between tiles")
    parser.add_argument("--use_amp", action="store_true", help="Use mixed precision during inference")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--save_suffix", type=str, default="_out", help="Suffix added before .png")
    opt = parser.parse_args()

    print(opt)

    dataset = ImageOnlyDataset(opt.data_src)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True
    )

    model_net = get_model(opt.model_name)
    if torch.cuda.device_count() >= 1:
        model_net = torch.nn.DataParallel(model_net)

    optimizer = torch.optim.Adam(model_net.parameters(), lr=0.0002)
    scheduler = MultiStepLR(optimizer, milestones=[], gamma=0.6)

    if torch.cuda.is_available():
        model_net = model_net.cuda()

    load_model_checkpoint = "{}/{}/best/checkpoint.pt".format(opt.ckp_dir, opt.load_from)
    model_net, _, _ = load_checkpoint(load_model_checkpoint, model_net, optimizer, scheduler)
    model_net.eval()

    eval_model = TiledModel(
        model=model_net,
        tile_size=opt.tile_size,
        overlap=opt.overlap,
        use_amp=opt.use_amp
    )
    eval_model.eval()

    out_path = "{}/{}/".format(opt.res_dir, opt.load_from)
    os.makedirs(out_path, exist_ok=True)

    with torch.no_grad():
        for names, imgs in tqdm(dataloader, desc="IFBlend inference"):
            imgs = imgs.cuda(non_blocking=True) if torch.cuda.is_available() else imgs

            preds = eval_model(imgs)

            name = names[0]  # batch_size=1
            stem, ext = os.path.splitext(name)
            save_name = f"{stem}{opt.save_suffix}.png"
            save_path = os.path.join(out_path, save_name)

            save_tensor_as_png(preds, save_path)

    print(f"Saved outputs to: {out_path}")
