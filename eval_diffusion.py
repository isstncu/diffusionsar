import argparse
import os
import random
import socket
import yaml
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import models
import datasets
import utils
from models import DenoisingDiffusion, DiffusiveRestoration
from datasets.data import Dataset_test
from torch.utils.data import DataLoader
from pathlib import Path
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def parse_args_and_config():
    parser = argparse.ArgumentParser(description='Restoring Weather with Patch-Based Denoising Diffusion Models')
    parser.add_argument("--config", type=str, default='mydata.yml',
                        help="Path to the config file")
    parser.add_argument('--resume', default='results/ckpts/data_ddpmxxx.pth.tar', type=str,
                        help='Path for the diffusion model checkpoint to load for evaluation')
    parser.add_argument('--loss_dir', type=Path, default=('results/'),
                        help='train dataset path')
    parser.add_argument("--grid_r", type=int, default=16,
                        help="Grid cell width r that defines the overlap between patches")
    parser.add_argument("--sampling_timesteps", type=int, default=25, #25
                        help="Number of implicit sampling steps")
    parser.add_argument("--image_folder", default='results/', type=str, 
                        help="Location to save restored images")
    parser.add_argument('--seed', default=61, type=int, metavar='N',
                        help='Seed for initializing training (default: 61)')
    args = parser.parse_args()

    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()

    # setup device to run
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device: {}".format(device))
    config.device = device

    if torch.cuda.is_available():
        print('Note: Currently supports evaluations (restoration) when run only on a single GPU!')

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    # data loading
    print("=> using dataset '{}'".format(config.data.dataset))
    test_dir = os.path.join(config.data.data_dir, 'test')
    test_set = Dataset_test(test_dir)
    test_loader = DataLoader(test_set, batch_size=config.sampling.batch_size, num_workers=config.data.num_workers)
    image_dirs = iter([p for p in Path(test_dir).iterdir() if p.is_dir()])

    # create model
    print("=> creating denoising-diffusion model with wrapper...")
    diffusion = DenoisingDiffusion(args, config)
    model = DiffusiveRestoration(diffusion, args, config)
    model.restore(test_loader, image_dirs, r=args.grid_r)


if __name__ == '__main__':
    main()
