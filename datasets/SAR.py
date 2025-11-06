import os
from os import listdir
from os.path import isfile
import imageio
import torch
import numpy as np
import torchvision
import torch.utils.data
import re
import random
from PIL import Image
from torchvision import transforms


class SAR:
    def __init__(self, config):
        self.config = config
        # self.transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        # Get Probability
        p_h = random.randint(0, 1)
        p_v = random.randint(0, 1)

        # Data Augmentation
        if config.diffusion.mode == "train":
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                #transforms.RandomHorizontalFlip(p=p_h),
                #transforms.RandomVerticalFlip(p=p_v),
            ])

        elif config.diffusion.mode == "valid":
            self.transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])


    def get_loaders(self, parse_patches=True):
        if parse_patches == True:
            print("=> training {} set...".format(self.config.training.name))
        else:
            print("=> evaluating SAR set...")
        train_dataset = SARDataset(dir=os.path.join(self.config.data.data_dir, 'data', 'train'),
                                        n=self.config.training.patch_n,
                                        patch_size=self.config.data.image_size,
                                        transforms=self.transforms,
                                        filelist=None,
                                        parse_patches=parse_patches)
        val_dataset = SARDataset(dir=os.path.join(self.config.data.data_dir, 'data', 'val'),
                                      n=self.config.training.patch_n, # 16
                                      patch_size=self.config.data.image_size,# 64
                                      transforms=self.transforms,
                                      filelist=self.config.sampling.filelist,
                                      parse_patches=parse_patches)
        if not parse_patches:
            self.config.training.batch_size = 1
            self.config.sampling.batch_size = 1

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.training.batch_size,
                                                   shuffle=True, num_workers=self.config.data.num_workers,
                                                   pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.config.sampling.batch_size,
                                                 shuffle=False, num_workers=self.config.data.num_workers,
                                                 pin_memory=True)

        return train_loader, val_loader
        # return val_loader


class SARDataset(torch.utils.data.Dataset):
    def __init__(self, dir, patch_size, n, transforms, filelist=None, parse_patches=True):
        super().__init__()

        if filelist is None: #train on Simulated SAR
            input_names, gt_names = [], []

            # SAR train
            sar_inputs = os.path.join(dir, 'input')
            images = [f for f in listdir(sar_inputs) if isfile(os.path.join(sar_inputs, f))]
            input_names += [os.path.join(sar_inputs, i) for i in images]
            gt_names += [os.path.join(os.path.join(dir, 'gt'), i) for i in images]
            x = list(enumerate(input_names))
            random.shuffle(x)
            indices, input_names = zip(*x)
            gt_names = [gt_names[idx] for idx in indices]
        elif filelist == "syn_val": #val on Simulated SAR
            input_names, gt_names = [], []
            # SAR val
            sar_inputs = os.path.join(dir, 'input')
            images = [f for f in listdir(sar_inputs) if isfile(os.path.join(sar_inputs, f))]
            input_names += [os.path.join(sar_inputs, i) for i in images]
            gt_names += [os.path.join(os.path.join(dir, 'gt'), i) for i in images]
            x = list(enumerate(input_names))
            random.shuffle(x)
            indices, input_names = zip(*x)
            gt_names = [gt_names[idx] for idx in indices]
        elif filelist =="syn_test": #test Simulated SAR
            # test_list = os.path.join("scratch/data/syn_ture", filelist)
            # with open(test_list) as f:
            #     contents = f.readlines()
            #     input_names = [os.path.join("scratch/data/syn_ture/L1", i.strip()) for i in contents]
            #     gt_names = input_names
            sar_inputs = "scratch/data/syn_test/L1" #L1,L2,L4
            images = [f for f in listdir(sar_inputs) if isfile(os.path.join(sar_inputs, f))]
            input_names += [os.path.join(sar_inputs, i) for i in images]
            gt_names = input_names
        elif filelist == "real_sar_test.txt": #test real SAR
            test_list = os.path.join("scratch/data/SAR_test", filelist)
            with open(test_list) as f:
                contents = f.readlines()
                input_names = [os.path.join("scratch/data/SAR_test", i.strip()) for i in contents]
                gt_names = input_names
        else:
            print("None file")
            
        self.input_names = input_names
        self.gt_names = gt_names
        self.patch_size = patch_size 
        self.transforms = transforms # ToTensor
        self.n = n # 4
        self.parse_patches = parse_patches # True
    
    # @staticmethod
    # def get_params(img, output_size,n):
    #     _, w, h = img.shape  # 256x256
    #     x, y = output_size  # 128x128
    #     return x, y, h, w
    # @staticmethod
    # def n_random_crops(img, x, y, h, w):
    #     crops = []
    #     for i in range(int(w / y)):  # 2
    #         for j in range(int(h / x)):  # 2
    #             new_crop = img[:, j * y: (j + 1) * y, i * x: (i + 1) * x]
    #             crops.append(new_crop)
    #     return tuple(crops)
        #############
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
            new_crop = img[:, y[i]:y[i] + w, x[i]:x[i] + h]
            crops.append(new_crop)
        return tuple(crops)

    def get_images(self, index):
        input_name = self.input_names[index]
        gt_name = self.gt_names[index]
        img_id = re.split('/', input_name)[-1][:-4]
        input_img = imageio.v3.imread(os.path.join(self.dir, input_name)) if self.dir else imageio.v3.imread(input_name)
        gt_img = imageio.v3.imread(os.path.join(self.dir, gt_name)) if self.dir else imageio.v3.imread(gt_name)

        #preprocess
        noise, lambda_= utils.gamma_noise_v2(input_img, Look=1)
        gt_img = utils.x0_process_v3(gt_img, lambda_) #x0_process_v2 or x0_process_v3
        input_img0 = utils.x0_process_v3(input_img, lambda_)
        
        input_img = input_img.astype(np.float32)
        gt_img = gt_img.astype(np.float32)
        input_img0 = input_img0.astype(np.float32)

        if self.parse_patches:
            x, y, h, w = self.get_params(input_img, (self.patch_size, self.patch_size), 4)
            input_img = self.n_random_crops(input_img, x, y, h, w) # 128x128
            gt_img = self.n_random_crops(gt_img, x, y, h, w) # 128x128

            gamma_noise = self.n_random_crops(gt_img, x, y, h, w)
            outputs = [torch.cat(
                    [self.transforms(input_img[i]), self.transforms(gt_img[i])],self.transforms(gamma_noise[i])], dim=0)
                        for i in range(self.n)]
            return torch.stack(outputs, dim=0), img_id
        else:
            #Start denoising from gamma noise.
            return torch.cat([self.transforms(input_img), self.transforms(noise)], dim=0), img_id, lambda_
            #Direct denoising from the original SAR image
            #return torch.cat([self.transforms(input_img), self.transforms(input_img0)], dim=0), img_id, lambda_

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)

