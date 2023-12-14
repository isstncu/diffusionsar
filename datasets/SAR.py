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

        if filelist is None:
            sar_dir = dir
            input_names, gt_names = [], []

            # SAR train
            # sar_inputs = os.path.join(sar_dir, 'input_4L')
            sar_inputs = os.path.join(sar_dir, 'input')
            images = [f for f in listdir(sar_inputs) if isfile(os.path.join(sar_inputs, f))]
            # assert len(images) == 2000
            assert len(images) == 400
            input_names += [os.path.join(sar_inputs, i) for i in images]
            gt_names += [os.path.join(os.path.join(sar_dir, 'gt'), i) for i in images]

            x = list(enumerate(input_names))
            random.shuffle(x)
            indices, input_names = zip(*x)
            gt_names = [gt_names[idx] for idx in indices]
            self.dir = None
        elif filelist == "syn_val.txt":
            self.dir = None
            input_names, gt_names = [], []
            # SAR val
            # sar_inputs = os.path.join(dir, 'input_4L')
            sar_inputs = os.path.join(dir, 'input')
            images = [f for f in listdir(sar_inputs) if isfile(os.path.join(sar_inputs, f))]
            assert len(images) == 55
            # assert len(images) == 400
            input_names += [os.path.join(sar_inputs, i) for i in images]
            gt_names += [os.path.join(os.path.join(dir, 'gt'), i) for i in images]

            x = list(enumerate(input_names))
            random.shuffle(x)
            indices, input_names = zip(*x)
            gt_names = [gt_names[idx] for idx in indices]
            # train_list = os.path.join(dir, filelist)
            # with open(train_list) as f:
            #     contents = f.readlines()
            #     input_names = [i.strip() for i in contents]
            #     gt_names = [i.strip().replace('input', 'gt') for i in input_names] #
        elif filelist == "real_sar_test.txt":
            self.dir = None
            test_list = os.path.join("scratch/data/synthesis_v2/SAR_test", filelist)
            with open(test_list) as f:
                contents = f.readlines()
                input_names = [os.path.join("scratch/data/synthesis_v2/SAR_test", i.strip()) for i in contents]
                gt_names = input_names
        elif filelist =="test.txt":
            self.dir = None
            test_list = os.path.join("scratch/data/synthesis_v2/syn_ture", filelist)
            with open(test_list) as f:
                contents = f.readlines()
                input_names = [os.path.join("scratch/data/synthesis_v2/syn_ture/L1", i.strip()) for i in contents]
                gt_names = input_names

        self.input_names = input_names
        self.gt_names = gt_names
        self.patch_size = patch_size # 64
        self.transforms = transforms # ToTensor
        self.n = n # 4
        self.parse_patches = parse_patches # True

    # @staticmethod
    # def get_params(img, output_size):
    #     w, h = img.shape  # 256x256 将img.size->img.shape，之前是PIL格式
    #     x, y = output_size  # 64x64
    #     return x, y, h, w

    # @staticmethod
    # def n_random_crops(img, x, y, h, w):
    #     crops = []
    #     for i in range(int(h / x)):  # 4
    #         for j in range(int(w / y)):  # 4
    #             new_crop = img[i * x: (i + 1) * x, j * y: (j + 1) * y]
    #             crops.append(new_crop)
    #     return tuple(crops)
        #############随机取块
    @staticmethod
    def get_params(img, output_size, n):
        w, h = img.shape  # 将img.size->img.shape，之前是PIL格式1964,1264
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i_list = [random.randint(0, h - th) for _ in range(n)]  # 1200-x
        j_list = [random.randint(0, w - tw) for _ in range(n)]  # 1900-y
        return i_list, j_list, th, tw

    @staticmethod
    def n_random_crops(img, x, y, h, w):
        crops = []
        for i in range(len(x)):
            new_crop = img[y[i]:y[i] + w,
                        x[i]:x[i] + h]  # img.crop((y[i], x[i], y[i] + w, x[i] + h))#PIL库的crop()函数用于裁剪图片
            crops.append(new_crop)
        return tuple(crops)

    def get_images(self, index):
        input_name = self.input_names[index]
        gt_name = self.gt_names[index]
        img_id = re.split('/', input_name)[-1][:-4]
        # 将输入读取改为sar图像tif格式，使用imageio
        input_img = imageio.v3.imread(os.path.join(self.dir, input_name)) if self.dir else imageio.v3.imread(input_name)
        gt_img = imageio.v3.imread(os.path.join(self.dir, gt_name)) if self.dir else imageio.v3.imread(gt_name)
        # gt_img = Image.open(os.path.join(self.dir, gt_name)) if self.dir else Image.open(gt_name)
        # gt_img = gt_img.convert('L')
        # 作为float32输入
        # gt_img = np.array(gt_img)
        input_img = input_img.astype(np.float32)
        gt_img = gt_img.astype(np.float32)

        if self.parse_patches:
            x, y, h, w = self.get_params(input_img, (self.patch_size, self.patch_size), 4)
            input_img = self.n_random_crops(input_img, x, y, h, w) # 36 64x64
            gt_img = self.n_random_crops(gt_img, x, y, h, w) # 36 64x64
            outputs = [torch.cat(
                    [self.transforms(input_img[i]), self.transforms(gt_img[i])], dim=0)
                        for i in range(self.n)]
            return torch.stack(outputs, dim=0), img_id
        else:
            # Resizing images to multiples of 16 for whole-image restoration
            wd_new, ht_new = input_img.shape
            if ht_new > wd_new and ht_new > 1024:
                wd_new = int(np.ceil(wd_new * 1024 / ht_new))
                ht_new = 1024
            elif ht_new <= wd_new and wd_new > 1024:
                ht_new = int(np.ceil(ht_new * 1024 / wd_new))
                wd_new = 1024
            wd_new = int(16 * np.ceil(wd_new / 16.0))
            ht_new = int(16 * np.ceil(ht_new / 16.0))
            input_img.resize((wd_new, ht_new))
            gt_img.resize((wd_new, ht_new))
            # input_img = np.pad(input_img, pad_width=((64, 64), (64, 64)), mode='constant') # (256+64,256+64)
            # gt_img = np.pad(gt_img, pad_width=((64, 64), (64, 64)), mode='constant') # (256+64,256+64)
            # print(input_img.shape)

            return torch.cat([self.transforms(input_img), self.transforms(gt_img)], dim=0), img_id

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)
