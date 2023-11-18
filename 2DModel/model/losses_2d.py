import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from scipy.integrate import quad
from Config_doc.logger import get_logger
from torch.nn.modules.loss import _Loss


logger = get_logger('loss')






def SpectralLoss(generated_image):
        spectrum = torch.fft.fft2(generated_image)
        spectrum = torch.fft.fftshift(spectrum)
        # spectrum = torch.rfft(generated_image,signal_ndim=3,normalized=True)
        # print(spectrum)
        # print(spectrum.shape)


        # spectrum = fft(generated_image)
        #
        #         # torch.fft.fft2)

        # print(spectrum.shape)
        # spectrum = spectrum.view(spectrum.shape[0], -1)
        # spectrum = spectrum[:,:,:,:,0].view(spectrum[:,:,:,:,0].shape[0], -1)
        # # print(spectrum.shape)
        # spectrum = F.normalize(spectrum, dim=0)
        # # print(spectrum.shape)
        # spectrum = torch.fft.fftshift(spectrum)
        spectrum = torch.abs(spectrum)
        spectrum = F.normalize(spectrum, dim=0)
        # epsilon = 1e-5
        # # Adjust the tensor to be in the range (epsilon, 1 - epsilon)
        # spectrum = (1 - 2 * epsilon) * spectrum + epsilon

        # # print(spectrum.shape)
        # # Set a small epsilon
        # epsilon = 1e-5
        #
        # # Adjust the tensor to be in the range (epsilon, 1 - epsilon)
        # spectrum = (1 - 2 * epsilon) * spectrum + epsilon
        # spectrum = torch.fft.fftshift(spectrum)
        # ai_values = Azimuthal_Integration(spectrum, (0, 2 * np.pi))
        #
        # ai = torch.tensor(ai_values)
        # normalized_ai = F.normalize(ai, dim=0)
        return spectrum