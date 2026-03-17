import random
from glob import glob
import subprocess
from tqdm import tqdm
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.aln_dataset import ALNDatasetGeom
from model import PromptNorm
from utils.schedulers import LinearWarmupCosineAnnealingLR
from options import options as opt
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger,TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from utils.val_utils import compute_psnr_ssim
from lpips import LPIPS
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM

class PromptNormModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        #self.net = PromptIR(decoder=True)
        self.net = PromptNorm(decoder=True)
        self.l1_loss  = nn.L1Loss()
        self.lpips_loss = LPIPS(net="vgg").requires_grad_(False)
        self.ssim_loss = SSIM()
        self.lpips_lambda = 0.1
        self.ssim_lambda = 0.2
    
    def forward(self,x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        ([clean_name, de_id], degrad_patch, depth_patch, clean_patch) = batch
        restored = self.net(degrad_patch, depth_patch)

        # Compute losses
        l1_loss = self.l1_loss(restored,clean_patch)
        lpips_loss = self.lpips_loss(restored,clean_patch)
        ssim_loss = 1 - self.ssim_loss(restored,clean_patch)
        total_loss = l1_loss + self.lpips_lambda * lpips_loss + self.ssim_lambda * ssim_loss

        # Logging to TensorBoard (if installed) by default
        self.log("l1_loss", l1_loss, sync_dist=True)
        self.log("lpips_loss", lpips_loss, sync_dist=True)
        self.log("ssim_loss", ssim_loss, sync_dist=True)
        self.log("total_loss", total_loss, sync_dist=True)

        return total_loss
    
    def tile_forward(self, img, normal, tile=384, overlap=96):
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
    
    def validation_step(self, batch, batch_idx):
        # validation_step defines the validation loop.
        ([clean_name, de_id], degrad_patch, depth_patch, target_patch) = batch

        with torch.no_grad():
            restored = self.tile_forward(degrad_patch, depth_patch, tile=512, overlap=96)
            psnr_i, ssim_i, _ = compute_psnr_ssim(restored, target_patch)

            self.log("val_psnr", psnr_i, prog_bar=True, on_epoch=True, sync_dist=True)
            self.log("val_ssim", ssim_i, prog_bar=True, on_epoch=True, sync_dist=True)


    def lr_scheduler_step(self,scheduler,metric):
        scheduler.step(self.current_epoch)
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-4)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer, warmup_epochs=15, max_epochs=150)

        return [optimizer],[scheduler]

def main():
    print("Options")
    print(opt)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.cuda)
    
    logger = TensorBoardLogger(save_dir="logs/")
    
    checkpoint_callback = ModelCheckpoint(dirpath = opt.ckpt_dir,every_n_epochs = 1,save_top_k=-1)
    best_checkpoint_callback = ModelCheckpoint(dirpath = opt.ckpt_dir,
                                               monitor="val_psnr",
                                               mode="max",
                                               save_top_k = 1,
                                               filename="best-epoch{epoch:02d}-psnr{val_psnr:.3f}")

    trainset = ALNDatasetGeom(input_folder=opt.train_input_dir,
                               geom_folder=opt.train_normals_dir,
                               target_folder=opt.train_target_dir,
                               resize_width_to=None,
                               patch_size=opt.patch_size)
    
    
    testset = ALNDatasetGeom(input_folder=opt.test_input_dir, geom_folder=opt.test_normals_dir, target_folder=opt.test_target_dir)
    
    trainloader = DataLoader(trainset, batch_size=opt.batch_size, pin_memory=True, shuffle=True,
                             drop_last=True, num_workers=opt.num_workers)
    testloader = DataLoader(testset, batch_size=1, pin_memory=True, shuffle=False, num_workers=opt.num_workers)
    model = PromptNormModel()
    
    

    num_params = sum(p.numel() for p in model.net.parameters())
    print(f"Params: {num_params/1e6:.2f}M")
    
    trainer = pl.Trainer(
        max_epochs=opt.epochs,
        accelerator="gpu",
        devices=opt.num_gpus,
        strategy="ddp_find_unused_parameters_true",
        logger=logger,
        callbacks=[checkpoint_callback, best_checkpoint_callback],
        check_val_every_n_epoch=3,
        num_sanity_val_steps=2,
        gradient_clip_val=1.0
        )
    trainer.fit(model=model, train_dataloaders=trainloader, val_dataloaders=testloader)


if __name__ == '__main__':
    main()


