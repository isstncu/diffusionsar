import os
import torch
import numpy as np
import torchvision
import torch.utils.data
import random
import glob
import imageio
from natsort import natsorted
from torch.utils.data import DataLoader
import utils

class data:
    def __init__(self, config):
        self.config = config

    def get_loaders(self, parse_patches=True):
        print("=> training mydata ...")
        train_path= os.path.join(self.config.data.data_dir, 'train')
        eval_path = os.path.join(self.config.data.data_dir, 'eval')

        train_set = Dataset(train_path, patch_size=self.config.data.patch_size, n=self.config.training.patch_n, 
                            parse_patches=self.config.data.parse_patches)
        val_set = Dataset(eval_path, patch_size=self.config.data.patch_size, n=1, 
                            parse_patches=self.config.data.parse_patches)
        
        train_loader = DataLoader(train_set, batch_size=self.config.training.batch_size, 
                                    shuffle=True, num_workers=self.config.data.num_workers, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=110, shuffle=True, num_workers=self.config.data.num_workers)
        return train_loader, val_loader

def mean_std_normalize(patch, eps=1e-8):
    patch = np.asarray(patch, dtype=np.float32)

    mean = np.mean(patch)
    std = np.std(patch)

    normalized_patch = (patch - mean) / max(std, eps)

    return normalized_patch

class Dataset(torch.utils.data.Dataset):
    def __init__(self, train_dir, patch_size, n, parse_patches=True):
        super().__init__()
        self.root_dir = train_dir
        self.extension = '.tif'
        self.input_dir = os.path.join(self.root_dir, 'noisy')
        self.output_dir = os.path.join(self.root_dir, 'gt')
        self.patch_size = patch_size
        self.transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        
        self.n = n
        self.parse_patches = parse_patches

        self.input_lists, self.output_lists, self.filename_lists = self._get_pair_path() 

    def _get_pair_path(self):
        names_input = natsorted(glob.glob(os.path.join(self.input_dir, '*' + self.extension)))
        names_output = natsorted(glob.glob(os.path.join(self.output_dir, '*' + self.extension)))
        filename_lists = []
        for name in names_output:
            filename = os.path.basename(name)
            filename_lists.append(str(filename))
        return names_input, names_output, filename_lists 

    @staticmethod
    def get_params(img, output_size, n):
        _, w, h = img.shape
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i_list = [random.randint(0, h - th) for _ in range(n)]
        j_list = [random.randint(0, w - tw) for _ in range(n)]
        return i_list, j_list, th, tw
    @staticmethod
    def n_random_crops(img, x, y, h, w):
        crops = []
        for i in range(len(x)):
            new_crop = img[:, y[i]:y[i]+w, x[i]:x[i]+h]
            crops.append(new_crop)
        return tuple(crops)

    def get_images(self, index):
        # input_image = self.transform((imageio.v3.imread(self.input_lists[index])).astype(np.float32))#01
        # output_image = self.transform((imageio.v3.imread(self.output_lists[index])).astype(np.float32))#01
        
##################preprocess
        input_image = self.transform((imageio.v3.imread(self.input_lists[index])).astype(np.float32))
        output_image = (imageio.v3.imread(self.output_lists[index])).astype(np.float32)
        
        noise, lambda_= utils.gamma_noise_v2(output_image, Look=1)
        output_image = utils.x0_process_v3(output_image, lambda_) #x0_process_v2 or x0_process_v3
        
        output_image = self.transform(output_image.astype(np.float32))
        noise = self.transform(noise.astype(np.float32))
#######################

        #filename = os.path.splitext(os.path.basename(self.filename_lists[index]))[0]

        if self.parse_patches:
            i, j, h, w = self.get_params(input_image, (self.patch_size, self.patch_size), self.n)

            input_image = self.n_random_crops(input_image, i, j, h, w)
            output_image = self.n_random_crops(output_image, i, j, h, w)
            noise = self.n_random_crops(noise, i, j, h, w)

            #outputs = [torch.cat([input_image[i], output_image[i]], dim=0)for i in range(self.n)]
            outputs = [torch.cat([input_image[i], output_image[i], noise[i]], dim=0)for i in range(self.n)]
            return torch.stack(outputs, dim=0),lambda_

        else:
            return torch.cat([input_image, output_image], dim=0)

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_lists)

class Dataset_test(torch.utils.data.Dataset):
    def __init__(self, train_dir):
        super().__init__()
        self.root_dir = train_dir
        self.extension = '.tif'
        self.input_dir = os.path.join(self.root_dir, 'noisy_true')
        self.transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        self.input_lists, self.filename_lists = self._get_pair_path()

    def _get_pair_path(self):
        names_input = natsorted(glob.glob(os.path.join(self.input_dir, '*' + self.extension)))
        filename_lists = []
        for name in names_input:
            filename = os.path.basename(name)
            filename_lists.append(str(filename))

        return names_input, filename_lists

    def get_images(self, index):
        # input_image = self.transform((imageio.v3.imread(self.input_lists[index])).astype(np.float32)) #01
        
        #preprocess
        input_image = (imageio.v3.imread(self.input_lists[index])).astype(np.float32)
        noise, lambda_= utils.gamma_noise_v2(input_image, Look=1)
        input_img0 = utils.x0_process_v3(input_image, lambda_)
        input_img0 = self.transform(input_img0.astype(np.float32))
        
        noise = self.transform(noise.astype(np.float32))
        input_image = self.transform(input_image)
        
        filename = os.path.splitext(os.path.basename(self.filename_lists[index]))[0]

        return [input_image,input_img0], filename ,lambda_  

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_lists)

