import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from scipy.integrate import quad
from Config_doc.logger import get_logger
from torch.nn.modules.loss import _Loss


logger = get_logger('loss')

def Azimuthal_Integration(input_tensor, phi_range):
    # print(input_tensor)
    n,m = input_tensor.shape[-2:]
    max_radial = m//2
    # print(max_radial)

    ai_values = []
    # print(ai_values)

    for k in range(max_radial):
        omega_k = k * (2 * np.pi / max_radial)
        # print('ok',omega_k)

        # cos_term = lambda phi: torch.tensor(omega_k * np.cos(phi))
        # sin_term = lambda phi: torch.tensor(omega_k * np.sin(phi))
        # print(torch.square(input_tensor * cos_term) + torch.square(sin_term))
        # def norm_squared(phi):
        #     cosine_term = input_tensor.float() * torch.tensor(omega_k * np.cos(phi))
        #     sine_term = torch.tensor(omega_k * np.sin(phi))
        #
        #     norm_squared = torch.sum(torch.square(cosine_term) + torch.square(sine_term))
        #
        #     return norm_squared

        norm_squared = lambda phi: torch.sum(torch.square(input_tensor.float() * torch.tensor(omega_k * np.cos(phi))) + torch.square(torch.tensor(omega_k * np.sin(phi))))
        # print(norm_squared.type)

        result = quad(norm_squared, phi_range[0], phi_range[1])
        ai_values.append(result)

    # print(ai_values)
    return ai_values


# class SpectralLoss():
#     def __init__(self):
#         pass
#
#     def forward(self, generated_image):
#         spectrum = torch.fft.fft2(generated_image)
#         spectrum = torch.fft.fftshift(spectrum)
#         ai_values = Azimuthal_Integration(spectrum, (0, 2 * np.pi))
#
#         ai = torch.tensor(ai_values)
#         normalized_ai = F.normalize(ai, dim=0)
#         # print(normalized_ai)
#
#         return normalized_ai

# #
# class SpectralLoss(nn.Module):
#     def __init__(self):
#         super(SpectralLoss, self).__init__()
#
#     def forward(self, generated_image):
def SpectralLoss(generated_image):
        spectrum = torch.fft.fft2(generated_image)

        spectrum = torch.fft.fftshift(spectrum)
        ai_values = Azimuthal_Integration(spectrum, (0, 2 * np.pi))

        ai = torch.tensor(ai_values)
        normalized_ai = F.normalize(ai, dim=0)
        return normalized_ai

