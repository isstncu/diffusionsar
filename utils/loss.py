from math import sqrt, pi

import torch
import torch.nn as nn

class Loss(nn.Module) :
    def __init__(self, device, lambda_tv):
        # Inheritance
        super(Loss, self).__init__()

        # Initialize Device
        self._device_ = device

        # Initialize Loss Weight
        self._lambda_tv_ = lambda_tv

        # Create Loss Instance
        self._loss_function_ = nn.MSELoss()
        self._dg_loss_ = DGLoss(self._loss_function_, self._device_)
        self._tv_loss_ = TVLoss()

    def forward(self, inputs, preds, targets):
        # Pixel Loss
        dg_loss = self._dg_loss_(inputs, preds, targets)

        # TV Loss
        tv_loss = self._tv_loss_(preds)

        return dg_loss, tv_loss, dg_loss + self._lambda_tv_ * tv_loss + self._loss_function_(preds, targets)
        # return dg_loss, tv_loss, self._loss_function_(preds, targets)

class DGLoss(nn.Module) :
    def __init__(self, loss_function, device):
        # Inheritance
        super(DGLoss, self).__init__()

        # Initialize Device
        self._device_ = device

        # Initialize Loss Function
        self._loss_function_ = loss_function

    def forward(self, inputs, preds, targets):
        # Get Loss
        loss_denominator = self._loss_function_(preds, targets)
        loss_numerator = self._loss_function_(inputs, targets)

        # Calculate DG Loss
        dg_loss = torch.log10(loss_denominator / loss_numerator) # torch.log10(loss_denominator / loss_numerator)
        #dg_loss = loss_denominator / loss_numerator

        return dg_loss

class TVLoss(nn.Module) :
    def __init__(self) :
        # Inheritance
        super(TVLoss, self).__init__()

    def forward(self, x) :
        # Initialize Variables
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]

        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])

        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()

        return 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t) :
        return t.size()[1] * t.size()[2] * t.size()[3]

def log(t, eps = 1e-15):
    return torch.log(t.clamp(min = eps))

def approx_standard_normal_cdf(x):
    return 0.5 * (1.0 + torch.tanh(sqrt(2.0 / pi) * (x + 0.044715 * (x ** 3))))

def discretized_gaussian_log_likelihood(x, *, means, log_scales, thres=0.999):
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1. / 255.)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1. / 255.)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = log(cdf_plus)
    log_one_minus_cdf_min = log(1. - cdf_min)
    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(x < -thres,
                            log_cdf_plus,
                            torch.where(x > thres,
                                        log_one_minus_cdf_min,
                                        log(cdf_delta)))
    return log_probs
