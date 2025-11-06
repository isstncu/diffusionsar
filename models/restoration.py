import pandas as pd
import torch
import torch.nn as nn
from utils import metrics
import utils
import torchvision
import os

# def inverse_data_transform(X):
#     #return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)
#     return (X + 1.0) / 2.0

class DiffusiveRestoration:
    def __init__(self, diffusion, args, config):
        super(DiffusiveRestoration, self).__init__()
        self.args = args
        self.config = config
        self.diffusion = diffusion

        if os.path.isfile(args.resume):
            self.diffusion.load_ddm_ckpt(args.resume, ema=True)
            self.diffusion.model.eval()
        else:
            print('Pre-trained diffusion model path is missing!')

    def restore(self, val_loader, r=None):
        validation = self.config.training.name
        # 单张图
        image_folder = os.path.join('./result','ckpts', self.config.training.version, 'test') 
        with torch.no_grad():
            for i, (x, y) in enumerate(val_loader):
                y = y[-1]
                print(f"starting processing from image {y}")
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                x_cond = x[:, :1, :, :].to(self.diffusion.device)
                xt = x[:, 1:, :, :].to(self.diffusion.device)
                #x_output = self.diffusive_restoration(x_cond, r=r)
                x_output = self.diffusive_restoration(x_cond, xt, r=r)
                # x_output = inverse_data_transform(x_output) # torch.Size([1,1,img_size,img_size])
                utils.logging.save_image_v2(x_output, lambda_, os.path.join(image_folder, f"{y}_ddpm_best.tif"))
                #utils.logging.save_image(x_output, os.path.join(image_folder, f"{y}_ddpm_best.tif"))
    
    def diffusive_restoration(self, x_cond, x, r=None):
        p_size = self.config.data.image_size
        h_list, w_list = self.overlapping_grid_indices(x_cond, output_size=p_size, r=r)
        corners = [(i, j) for i in h_list for j in w_list]
        #x = torch.randn(x_cond.size(), device=self.diffusion.device)
        x_output = self.diffusion.sample_image(x_cond, x, patch_locs=corners, patch_size=p_size)
        return x_output

    def overlapping_grid_indices(self, x_cond, output_size, r=None):
        _, c, h, w = x_cond.shape
        r = 16 if r is None else r
        h_list = [i for i in range(0, h - output_size + 1, r)]
        w_list = [i for i in range(0, w - output_size + 1, r)]
        return h_list, w_list

