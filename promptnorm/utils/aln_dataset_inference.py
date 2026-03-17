from PIL import Image
import os
from glob import glob
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torchvision.transforms import functional as TF
from torchvision import transforms
import random
import torch 

class ALNDatasetGeom(Dataset):
    def __init__(self, input_folder, geom_folder, target_folder=None,
                 resize_width_to=None, patch_size=None, filter_of_images=None,
                 normal_suffix="_normal.png"):
        super().__init__()
        self.input_folder = input_folder
        self.geom_folder = geom_folder
        self.target_folder = target_folder
        self.resize_width_to = resize_width_to
        self.patch_size = patch_size
        self.filter_of_images = filter_of_images
        self.normal_suffix = normal_suffix

        self.has_target = (target_folder is not None) and (str(target_folder).strip() != "")
        self._init_paths()
        self.toTensor = ToTensor()

    def _base_name(self, filename):
        if filename.endswith("_in_out.png"):
            return filename[:-len("_in_out.png")]
        if filename.endswith("_in.png"):
            return filename[:-len("_in.png")]
        if filename.endswith("_out.png"):
            return filename[:-len("_out.png")]
        return os.path.splitext(filename)[0]

    def _init_paths(self):
        self.image_paths = []

        for input_path in sorted(glob(os.path.join(self.input_folder, "*"))):
            if os.path.isdir(input_path):
                continue

            if self.filter_of_images is not None:
                if not any(["/"+str(filt)+"_in" in input_path for filt in self.filter_of_images]):
                    continue

            filename = os.path.basename(input_path)
            stem = self._base_name(filename)

            geom_path = os.path.join(self.geom_folder, stem + self.normal_suffix)
            if not os.path.exists(geom_path):
                raise FileNotFoundError(f"Geom/Normal file not found: {geom_path}")

            if self.has_target:
                if filename.endswith("_in.png"):
                    target_name = filename.replace("_in.png", "_gt.png")
                else:
                    target_name = filename
                target_path = os.path.join(self.target_folder, target_name)

                if not os.path.exists(target_path):
                    raise FileNotFoundError(f"Target file not found: {target_path}")
            else:
                target_path = None

            self.image_paths.append({"input": input_path, "geom": geom_path, "target": target_path})

        print(f"Found {len(self.image_paths)} items (has_target={self.has_target})")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        input_path = self.image_paths[idx]["input"]
        geom_path = self.image_paths[idx]["geom"]
        target_path = self.image_paths[idx]["target"]

        input_img = self.toTensor(Image.open(input_path).convert("RGB"))
        normal_img = self.toTensor(Image.open(geom_path).convert("RGB"))

        if self.has_target:
            target_img = self.toTensor(Image.open(target_path).convert("RGB"))
        else:
            target_img = torch.zeros_like(input_img)

        if self.resize_width_to is not None:
            new_h = int((input_img.shape[1] * self.resize_width_to) / input_img.shape[2])
            input_img = TF.resize(input_img, (new_h, self.resize_width_to))
            normal_img = TF.resize(normal_img, (new_h, self.resize_width_to))
            target_img = TF.resize(target_img, (new_h, self.resize_width_to))

            if random.random() > 0.5:
                input_img = TF.hflip(input_img)
                normal_img = TF.hflip(normal_img)
                target_img = TF.hflip(target_img)

        if self.patch_size is not None:
            i, j, h, w = transforms.RandomCrop.get_params(input_img, output_size=(self.patch_size, self.patch_size))
            input_img = TF.crop(input_img, i, j, h, w)
            normal_img = TF.crop(normal_img, i, j, h, w)
            target_img = TF.crop(target_img, i, j, h, w)

        name = os.path.basename(input_path)  # "0001.png"
        return [name, 0], input_img, normal_img, target_img