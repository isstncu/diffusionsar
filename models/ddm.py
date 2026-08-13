import datetime
from math import log as ln
import os
import time
import logging
import sys
import utils
import numpy as np
import csv
import pandas as pd
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import utils.metrics
from timeit import default_timer as timer
from models.unet import DiffusionUNet
from utils.loss import discretized_gaussian_log_likelihood, Loss

# This script is adapted from the following repositories
# https://github.com/ermongroup/ddim
# https://github.com/bahjat-kawar/ddrm

import torch

class EMAHelper(object):
    def __init__(self, mu=0.9999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        if isinstance(module, nn.DataParallel):
            inner_module = module.module
            module_copy = type(inner_module)(inner_module.config).to(inner_module.config.device)
            module_copy.load_state_dict(inner_module.state_dict())
            module_copy = nn.DataParallel(module_copy)
        else:
            module_copy = type(module)(module.config).to(module.config.device)
            module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2)
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, 1000, dtype=np.float64)
        #betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    elif beta_schedule == "cosine":
        u = np.linspace(0.0, 1.0, num_diffusion_timesteps, dtype=np.float64)      
        c = 1.0 - np.cos(u * np.pi / 2.0)
        c = (c - c.min()) / (c.max() - c.min())
        betas = beta_start + c * (beta_end - beta_start)
    else:
        raise NotImplementedError(beta_schedule)
    betas = betas[:num_diffusion_timesteps]
    #sample = 1000 // num_diffusion_timesteps+1
    #betas = betas[:1000:sample]
    print(betas.shape)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

#######################
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
def get_logger(logpath=None):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        if logpath is not None:
            file_handler = logging.FileHandler(logpath)
            file_handler.setFormatter(logging.Formatter('%(message)s'))
            logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)
    return logger
def log_csv(filepath, values, header=None, multirows=False):
    empty = False
    if not filepath.exists():
        filepath.touch()
        empty = True

    with open(filepath, 'a') as file:
        writer = csv.writer(file)
        if empty and header:
            writer.writerow(header)
        if multirows:
            writer.writerows(values)
        else:
            writer.writerow(values)

####################
class DenoisingDiffusion(object):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
        self.device = config.device
        self.logger = get_logger()
        self.history = args.loss_dir / 'history.csv'
        
        self.model = DiffusionUNet(config)
        self.model.to(self.device)
        self.model = torch.nn.DataParallel(self.model)

        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f'There are {n_params} trainable parameters for model.')

        self.ema_helper = EMAHelper()
        self.ema_helper.register(self.model)

        self.L1 = nn.L1Loss()
        self.MSE = nn.MSELoss()

        self.optimizer = utils.optimize.get_optimizer(self.config, self.model.parameters())
        self.start_epoch, self.step = 0, 0
        self.NAT = 1. / ln(2.0)#KL

        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )

        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

#####IDDPM-functions-losses
    def extract(self, arr, timesteps, broadcast_shape):
        """
        Extract values from a 1-D numpy array for a batch of indices.

        :param arr: the 1-D tensor.
        :param timesteps: a tensor of indices into the array to extract.
        :param broadcast_shape: a larger shape of K dimensions with the batch
                                dimension equal to the length of timesteps.
        :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
        """
        res = arr[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        res = res.expand(broadcast_shape)
        return res

    def unnormalize_to_zero_to_one(self, t):
        return (t + 1) * 0.5

    def normal_kl(self, mean1, logvar1, mean2, logvar2):
        """
        KL divergence between normal distributions parameterized by mean and log-variance.
        """
        return 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2) + ((mean1 - mean2) ** 2) * torch.exp(
            -logvar2))

    def meanflat(self, x):
        return x.mean(dim=list(range(1, len(x.shape))))

    def p_losses(self, model, x, t, noise, betas, channel,clip_denoised=True):

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        assert alphas_cumprod_prev.shape == (self.num_timesteps,) #
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

        # q_sample
        x0 = x[:, channel:channel+1, :, :]
        x_t = (self.extract(sqrt_alphas_cumprod, t, x0.shape) * x0
               + self.extract(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)

        # model output
        model_output = model(torch.cat([x[:, :channel, :, :], x_t], dim=1),t.float())

        # calculating kl loss for learned variance (interpolation)
        # q_posterior: Calculate the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0)
        posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
        posterior_mean_coef2 = (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        posterior_log_variance_clipped = torch.log(posterior_variance.clamp(min=1e-20))
        # Posterior mean and variance 
        true_mean = (self.extract(posterior_mean_coef1, t, x_t.shape) * x0
                     + self.extract(posterior_mean_coef2, t, x_t.shape) * x_t)
        true_log_variance_clipped = self.extract(posterior_log_variance_clipped, t, x_t.shape)

        # p_mean_variance: The incoming value is x_t, which predicts the mean and variance at time t-1
        sqrt_recip_alphas_cumprod = torch.sqrt(1. / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / alphas_cumprod - 1)
        pred_noise, var_interp_frac_unnormalized = model_output.chunk(2, dim=1)

        # predict_start_from_noise
        pred_x_start = (self.extract(sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                   - self.extract(sqrt_recipm1_alphas_cumprod, t, x_t.shape) * pred_noise)
        if clip_denoised:
            pred_x_start.clamp_(-1., 1.)
        var_interp_frac = self.unnormalize_to_zero_to_one(var_interp_frac_unnormalized)
        max_log = self.extract(torch.log(betas), t, x_t.shape)
        min_log = self.extract(posterior_log_variance_clipped, t, x_t.shape)
        model_mean = (self.extract(posterior_mean_coef1, t, x_t.shape) * pred_x_start
                      + self.extract(posterior_mean_coef2, t, x_t.shape) * x_t)
        model_log_variance = var_interp_frac * max_log + (1 - var_interp_frac) * min_log

        # kl loss with detached model predicted mean, for stability reasons as in paper
        detached_model_mean = model_mean.detach()
        kl = self.normal_kl(true_mean, true_log_variance_clipped, detached_model_mean, model_log_variance)
        kl = self.meanflat(kl) * self.NAT
        decoder_nll = -discretized_gaussian_log_likelihood(
            x0, means=detached_model_mean, log_scales=0.5 * model_log_variance
        )
        assert decoder_nll.shape == x0.shape
        decoder_nll = self.meanflat(decoder_nll) * self.NAT
        # at the first timestep return the decoder NLL, otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        kl_losses = torch.where(t == 0, decoder_nll, kl).mean(dim=0)
        # simple loss - predicting noise, x0, or x_prev
        kl_loss_weight = 10000
        mse_losses = (noise - pred_noise).square().sum(dim=(1, 2, 3)).mean(dim=0)
        #print('MSE:{0:.6f}, KL:{1:.6f}'.format(mse_losses, kl_losses * kl_loss_weight))
        return mse_losses + kl_loss_weight * kl_losses, pred_noise
###########################################
    def load_ddm_ckpt(self, load_path, ema=False):
        checkpoint = utils.logging.load_checkpoint(load_path, None)
        self.start_epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.ema_helper.load_state_dict(checkpoint['ema_helper'])
        if ema:
            self.ema_helper.ema(self.model)
        print("=> loaded checkpoint '{}' (epoch {})".format(load_path, checkpoint['epoch']))

    def train_on_epoch(self, epoch, train_loader):
        self.model.train()
        epg_loss = AverageMeter()
        epg_error = AverageMeter()
        data_start = time.time()
        data_time = 0
        
        batches = len(train_loader)
        #for i, x in enumerate(train_loader):
        for i, (x,lambda_) in enumerate(train_loader):
            x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
            n = x.size(0)
            data_time += time.time() - data_start

            sar_channel = 1
            x = x.to(self.device)
            
            # e = torch.randn_like(x[:, sar_channel:sar_channel+1, :, :]) #gaussian noise
            e = x[:,2:,:,:] #gamma noise
            
            b = self.betas

            # antithetic sampling对偶抽样法
            t = torch.randint(low=0, high=self.num_timesteps, size=(n // 2 + 1,)).to(self.device)
            t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
            
            loss,output = self.p_losses(self.model, x, t, e, b,sar_channel)
            
            # a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
            # x0=x[:, sar_channel:sar_channel+1, :, :]
            # x_t = x0 * a.sqrt() + e * (1.0 - a).sqrt()
            # output = self.model(torch.cat([x[:, :sar_channel, :, :], x_t], dim=1), t.float())
            # x0_pred = (x_t - output * (1.0 - a).sqrt()) / a.sqrt()
            # loss = self.MSE(output, e)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.ema_helper.update(self.model)
            
            epg_loss.update(loss.item())
            mse = F.mse_loss(output.detach(), e).item()
            epg_error.update(mse)
            data_start = time.time()
            if i % 100 == 0:
                print(f"Epoch[{epoch} {i}/{batches}], loss: {loss.item()}, data time: {data_time / (i+1)}")
        return epg_loss.avg, epg_error.avg
    @torch.no_grad()
    def test_on_epoch(self, epoch, val_loader):
        self.model.eval()
        epoch_error = AverageMeter()

        image_folder = os.path.join(self.args.image_folder, self.config.data.dataset + str(self.config.data.patch_size))
        print(f"Processing a single batch of validation images at epoch: {epoch}")
        #for i, x in enumerate(val_loader):
        for i, (x,lambda_) in enumerate(val_loader):
            x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
            break
        n = x.size(0)
        sar_channel = 1
        out_channel = 1
        x_cond = x[:, :sar_channel, :, :].to(self.device)
        target = x[:, sar_channel:sar_channel+1, :, :].to(self.device)
        
        #x = torch.randn(n, out_channel, self.config.data.patch_size, self.config.data.patch_size, device=self.device) #gaussian noise
        x = x[:, 2:, :, :].to(self.device) #gamma noise
        
        x = self.sample_image(x_cond, x).to(self.device)
        
        # DG loss + TV loss
        loss_function = Loss(self.device, 10) # DG + TV
        DG_loss, TV_loss, val_error = loss_function(x_cond, x, target)
        #MSE
        #val_error = F.mse_loss(x, target)
        epoch_error.update(val_error.item())

        ssim, psnr = 0, 0
        for i in range(n):
            #print(x[i].squeeze(0).shape)
            cal_ssim = utils.metrics.calculate_ssim(x[i], target[i])
            cal_psnr = utils.metrics.calculate_psnr(x[i], target[i])
            if cal_ssim > ssim:
                ssim = cal_ssim
            if cal_psnr > psnr:
                psnr = cal_psnr
            #utils.logging.save_image(x[i], os.path.join(image_folder, str(epoch), f"{i}.tif")) #gaussian noise
            utils.logging.save_image_v2(x[i],lambda_[i], os.path.join(image_folder, str(epoch), f"{i}.tif")) #gamma noise

        # for i in range(n):
            # utils.logging.save_image(x[i],noise_log[i], os.path.join(image_folder, str(epoch), f"{i}.tif"))
        return epoch_error.avg, ssim, psnr

    def train(self, DATASET):
        cudnn.benchmark = True
        train_loader, val_loader = DATASET.get_loaders()
        
        least_error = float('inf')

        if os.path.isfile(self.args.resume):
            self.load_ddm_ckpt(self.args.resume)
        if self.history.exists():
            df = pd.read_csv(self.history)
            least_error = df['val_loss'].min()
            val_loss = least_error

        for epoch in range(self.start_epoch, self.config.training.n_epochs):
            print('Epoch:', epoch)
            t_start = timer()
            train_loss, train_error = self.train_on_epoch(epoch, train_loader)
            
            n_e = epoch+1
            if n_e >= 1000:
                validation_epoch = 5
            else:
                validation_epoch = 10
            if n_e % validation_epoch == 0 or epoch == 0 :
                val_loss,val_ssim, val_psnr = self.test_on_epoch(epoch, val_loader)
                #val_loss = self.test_on_epoch(epoch, val_loader)
            t_end = timer()
            self.logger.info(f'Epoch[{epoch}] '
                             f'train_loss: {train_loss:.6f} - '
                             f'train_error: {train_error:.6f} - '
                             f'val_loss: {val_loss:.6f} - '
                             f'val_ssim: {val_ssim:.6f} - '
                             f'val_psnr: {val_psnr:.6f} - '
                             f'Time: {t_end - t_start}s')
            csv_header = ['epoch', 'train_loss', 'train_error', 'val_loss','val_ssim','val_psnr']
            csv_values = [epoch, train_loss, train_error, val_loss,val_ssim,val_psnr]
            log_csv(self.history, csv_values, header=csv_header)

            if val_loss < least_error:
                least_error = val_loss
                utils.logging.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'ema_helper': self.ema_helper.state_dict(),
                    'params': self.args,
                    'config': self.config
                }, filename=os.path.join(self.args.image_folder, 'ckpts', self.config.data.dataset + '_ddpm' + str(epoch)))
                
            elif epoch % 25 == 0:
                utils.logging.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'ema_helper': self.ema_helper.state_dict(),
                    'params': self.args,
                    'config': self.config
                }, filename=os.path.join(self.args.image_folder, 'ckpts', self.config.data.dataset + '_ddpm' + str(epoch)))

    # ddim
    def sample_image(self, x_cond, x, last=True, patch_locs=None, patch_size=None):
        skip = self.config.diffusion.num_diffusion_timesteps // self.args.sampling_timesteps # 1000 // 25 = 40
        seq = range(0, self.config.diffusion.num_diffusion_timesteps, skip) # range(0,1000,40)
        if patch_locs is not None:
            xs = utils.sampling.generalized_steps_overlapping(x, x_cond, seq, self.model, self.betas, eta=0.,
                                                              corners=patch_locs, p_size=patch_size)
        else:
            xs = utils.sampling.generalized_steps(x, x_cond, seq, self.model, self.betas, eta=0.)
        if last:
            xs = xs[0][-1]
        return xs
