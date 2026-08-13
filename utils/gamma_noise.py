import numpy
import numpy as np
import torch
import torchvision
from scipy import stats
from sklearn import preprocessing


def gamma_noise(x0, Look=1):
    L = Look
    size = x0.shape
    gamma_noise = np.random.gamma(L, 1/L, size)
    log_gamma_noise = np.log(gamma_noise)
    log_gamma_noise = log_gamma_noise.reshape(-1)
    transformed_data, lambda_ = stats.yeojohnson(log_gamma_noise)
    norm_transformed_data = preprocessing.scale(transformed_data)#归一化
    norm_transformed_data = norm_transformed_data.reshape(x0.shape)

    return norm_transformed_data.astype(np.float32)

def gamma_noise_v2(x0,Look=1):
    L = Look
    size = x0.shape
    gamma_noise = np.random.gamma(L, 1/L, size)
    log_gamma_noise = np.log(gamma_noise)
    log_gamma_noise = log_gamma_noise.reshape(-1)
    transformed_data, lambda_ = stats.yeojohnson(log_gamma_noise)
    norm_transformed_data = transformed_data.reshape(x0.shape)

    return norm_transformed_data.astype(np.float32), lambda_

from scipy.special import inv_boxcox
def inverse_yeojohnson(output, lambda_):
    # 反变换
    #if lambda_ != 0:
        #inverse_data = ((output * lambda_ + 1) ** (1 / lambda_)) - 1
    #else:
        #inverse_data = np.exp(output) - 1
    inverse_data = inv_boxcox(output, lambda_)-1
    return inverse_data

np_max=255 #or 10000 #Test on real SAR data: setting the maximum value to 10000 with t=80 gives better results.
def x0_process(x0):
    img = (x0 * np_max)
    img[img == 0] = 1
    img = np.log(img) / np.log(np_max)
    return img.astype(np.float32)

def x0_process_v2(x0, lambda_):
    img = (x0 * np_max)
    #img = x0 #The UC dataset does not need to be multiplied by 255
    img[img == 0] = 1
    log_img = np.log(img)
    img = stats.yeojohnson(log_img, lambda_)
    im_max = stats.yeojohnson(np.log(np_max), lambda_)
    img = img / im_max
    return img.astype(np.float32)

def x0_process_v3(x0, lambda_):
    img = (x0 * np_max)
    # img = x0 #The UC dataset does not need to be multiplied by 255
    img[img == 0] = 1
    log_img = np.log(img)/np.log(np_max)
    img = stats.yeojohnson(log_img, lambda_)
    im_max = stats.yeojohnson(1, lambda_)
    img = img / im_max
    return img.astype(np.float32)

def inverse_x0_process(output):
    output = output * np.log(np_max)
    output = np.exp(output)
    return output

def inverse_x0_process_v2(output0, lambda_):
    output = np.clip(output0, 0, 1)
    im_max = stats.yeojohnson(np.log(np_max), lambda_)
    output = output * im_max
    output = inverse_yeojohnson(output, lambda_)
    output = np.exp(output)
    output = output/np_max
    return output

def inverse_x0_process_v3(output0, lambda_):
    output = np.clip(output0, 0, 1)
    im_max = stats.yeojohnson(1, lambda_)
    output = output * im_max
    output = inverse_yeojohnson(output, lambda_)
    output = np.clip(output, 0, 1)
    output = np.exp(output * np.log(np_max))
    output = output/np_max
    return output
