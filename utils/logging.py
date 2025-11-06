import imageio
import numpy
import numpy as np
import torch
import os

import utils


def save_image(img, file_directory):
    if not os.path.exists(os.path.dirname(file_directory)):
        os.makedirs(os.path.dirname(file_directory))
    img = torch.clamp(img, 0.0, 1.0)
    img = img.reshape(img.shape[-2], img.shape[-1])
    img = img.squeeze().cpu().numpy()
    img = (img*255).astype(numpy.uint8)
    imageio.v3.imwrite(file_directory, img)

def save_image_v2(img, lambda_, file_directory):
    if not os.path.exists(os.path.dirname(file_directory)):
        os.makedirs(os.path.dirname(file_directory))
    img = torch.clamp(img, 0.0, 1.0)
    # print(img.shape)
    img = img.reshape(img.shape[-2], img.shape[-1])
    img = img.squeeze().cpu().numpy()
    img = utils.inverse_x0_process_v3(img,lambda_) #inverse_x0_process_v2 or inverse_x0_process_v3
    imageio.v3.imwrite(file_directory, img)

def save_checkpoint(state, filename):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    torch.save(state, filename + '.pth.tar')


def load_checkpoint(path, device):
    if device is None:
        return torch.load(path)
    else:
        return torch.load(path, map_location=device)
