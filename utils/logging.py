import imageio
import numpy
import numpy as np
import torch
import os


def save_image(img, file_directory):
    if not os.path.exists(os.path.dirname(file_directory)):
        os.makedirs(os.path.dirname(file_directory))
    img = torch.clamp(img, 0.0, 1.0)
    img = img.reshape(img.shape[-2], img.shape[-1])
    img = img.squeeze().cpu().numpy()
    img = (img*255).astype(numpy.uint8)
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
