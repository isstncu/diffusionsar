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
    norm_transformed_data = preprocessing.scale(transformed_data)
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
#
from scipy.special import inv_boxcox
def inverse_yeojohnson(output, lambda_):
    # 反变换
    #if lambda_ != 0:
        #inverse_data = ((output * lambda_ + 1) ** (1 / lambda_)) - 1
    #else:
        #inverse_data = np.exp(output) - 1
    inverse_data = inv_boxcox(output, lambda_)-1
    return inverse_data

def x0_process(x0):
    img = (x0 * 255).astype(np.uint8)
    img[img == 0] = 1
    img = np.log(img) / np.log(255.0)
    return img.astype(np.float32)

def x0_process_v2(x0, lambda_):
    img = (x0 * 255).astype(np.uint8)
    #img = x0 #The UC dataset does not need to be multiplied by 255
    img[img == 0] = 1
    log_img = np.log(img)
    img = stats.yeojohnson(log_img, lambda_)
    im_max = stats.yeojohnson(np.log(255), lambda_)
    img = img / im_max
    return img.astype(np.float32)

def x0_process_v3(x0, lambda_):
    img = (x0 * 255).astype(np.uint8)
    # img = x0 #The UC dataset does not need to be multiplied by 255
    img[img == 0] = 1
    log_img = np.log(img)/np.log(255)
    img = stats.yeojohnson(log_img, lambda_)
    im_max = stats.yeojohnson(1, lambda_)
    img = img / im_max
    return img.astype(np.float32)

def inverse_x0_process(output):
    output = output * np.log(255.0)
    output = np.exp(output).astype(np.uint8)
    return output

def inverse_x0_process_v2(output0, lambda_):
    output = np.clip(output0, 0, 1)
    im_max = stats.yeojohnson(np.log(255), lambda_)
    output = output * im_max
    output = inverse_yeojohnson(output, lambda_)
    output = np.exp(output)
    output = output.astype(np.uint8)
    return output

def inverse_x0_process_v3(output0, lambda_):
    output = np.clip(output0, 0, 1)
    im_max = stats.yeojohnson(1, lambda_)
    output = output * im_max
    output = inverse_yeojohnson(output, lambda_)
    output = np.clip(output, 0, 1)
    output = np.exp(output * np.log(255))
    output = output.astype(np.uint8)
    return output

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)
    if beta_schedule == "quad":
        betas = (np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2)
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

def generalized_gamma(x0,theta_0=0.001):
    b = get_beta_schedule(
        beta_schedule='linear',
        beta_start=0.0001,
        beta_end=0.02,
        num_diffusion_timesteps=1000,
    )
    b = torch.from_numpy(b).float()
    t = torch.randint(low=0, high=b.shape[0], size=(x0.size(0) // 2 + 1,))
    a = (1 - b).cumprod(dim=0)
    k = (b / a) / theta_0 ** 2
    theta = (a.sqrt() * theta_0).index_select(0, t).view(-1, 1, 1, 1).to(x0.device)
    k_bar = k.cumsum(dim=0).index_select(0, t).view(-1, 1, 1, 1).to(x0.device)
    a = a.index_select(0, t).view(-1, 1, 1, 1).to(x0.device)
    concentration = torch.ones(x0.size()).to(x0.device) * k_bar
    rates = torch.ones(x0.size()).to(x0.device) * theta
    m = torch.distributions.Gamma(concentration, 1 / rates)
    e = m.sample().to(x0.device)
    e = e - concentration * rates
    e = e / (1.0 - a).sqrt()

    return e
