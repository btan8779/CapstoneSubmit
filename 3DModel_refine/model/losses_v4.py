import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from scipy.integrate import quad
from Config_doc.logger import get_logger
from torch.nn.modules.loss import _Loss


logger = get_logger('loss')

# class SpectralLoss():
#     def __init__(self,generated_image):
#         self.generated_image = generated_image
#
#     def forward(self):
#         spectrum = torch.fft.fftn(self.generated_image).view(-1)
#         spectrum = F.normalize(spectrum, dim=0)
#         # spectrum = torch.fft.fftshift(spectrum)
#         spectrum = torch.abs(spectrum)
#         # spectrum = F.normalize(spectrum, dim= (3,4))
#
#         return spectrum
#
# class SpectralLoss():
#     def __init__(self,generated_image):
#         self.generated_image = generated_image

def SpectralLoss(generated_image):
        spectrum = torch.fft.fftn(generated_image)
        spectrum = torch.fft.fftshift(spectrum)
        # spectrum = spectrum.view(spectrum.shape[0],-1)
        spectrum = F.normalize(spectrum, dim=0)
        spectrum = torch.abs(spectrum)
        epsilon = 1e-5
        # Adjust the tensor to be in the range (epsilon, 1 - epsilon)
        spectrum = (1 - 2 * epsilon) * spectrum + epsilon

        # spectrum = F.normalize(spectrum, dim= )

        return spectrum