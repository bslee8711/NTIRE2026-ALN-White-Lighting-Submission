from PIL import Image
import os
from glob import glob
from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage, Compose, RandomCrop, ToTensor

from torchvision import transforms
from torchvision.transforms import functional as TF
import random

class ALNDatasetGeom(Dataset):
    def __init__(self, input_folder, target_folder, geom_folder, resize_width_to=None, patch_size=None, filter_of_images=None):
        super(ALNDatasetGeom, self).__init__()
        self.input_folder = input_folder
        self.geom_folder = geom_folder
        self.target_folder = target_folder
        self.resize_width_to = resize_width_to
        self.patch_size = patch_size
        self.filter_of_images = filter_of_images

        self._init_paths()
    
        self.toTensor = ToTensor()

    def _init_paths(self):
        self.image_paths = []
        for input_path in glob(self.input_folder + '/*'):
            if self.filter_of_images is not None:
                if not any(["/"+str(filt)+"_in" in input_path for filt in self.filter_of_images]):
                    continue
            filename = os.path.basename(input_path)      
            
            img_id = filename.split('_')[0]             

            target_path = os.path.join(self.target_folder, f"{img_id}_gt.png")
            geom_path = os.path.join(self.geom_folder, f"{img_id}_normal.png")
            #######################################################
            # target_path = self.target_folder + '/' + input_path.split('/')[-1].replace('_out.png', '_gt.png')
            # geom_path = self.geom_folder + '/' + input_path.split('/')[-1].replace('_out.png', '_normal.png')

            # Ensure target file exists
            if os.path.exists(target_path):
                self.image_paths.append({'input': input_path, 'target': target_path, 'geom': geom_path})
            else:
                raise FileNotFoundError(f"Target file not found: {target_path}")

        print(f"Found {len(self.image_paths)} image pairs")

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        input_path, geom_path, target_path = self.image_paths[idx]['input'], self.image_paths[idx]['geom'], self.image_paths[idx]['target']

        input_img = self.toTensor(Image.open(input_path).convert('RGB'))
        normal_img = self.toTensor(Image.open(geom_path).convert('RGB'))
        target_img = self.toTensor(Image.open(target_path).convert('RGB'))

        if self.resize_width_to is not None:
            input_img = TF.resize(input_img, (int((input_img.shape[1]*self.resize_width_to)/input_img.shape[2]), self.resize_width_to))
            normal_img = TF.resize(normal_img, (int((normal_img.shape[1]*self.resize_width_to)/normal_img.shape[2]), self.resize_width_to))
            target_img = TF.resize(target_img, (int((target_img.shape[1]*self.resize_width_to)/target_img.shape[2]), self.resize_width_to))            

        if self.patch_size is not None:
            if random.random() > 0.5:
                input_img = TF.hflip(input_img)
                normal_img = TF.hflip(normal_img)
                target_img = TF.hflip(target_img)
            # crop input and target images with the same random crop
            i, j, h, w = transforms.RandomCrop.get_params(input_img, output_size=(self.patch_size, self.patch_size))
            input_img = TF.crop(input_img, i, j, h, w)
            normal_img = TF.crop(normal_img, i, j, h, w)
            target_img = TF.crop(target_img, i, j, h, w)

        return [input_path.split('/')[-1].split('_')[0], 0], input_img, normal_img, target_img
    