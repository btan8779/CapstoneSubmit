import torch
import torch.nn.functional as F
from torchvision import transforms
import argparse
import torch.nn as nn
from torch.autograd import Variable
# from scipy.integrate import quad
from Config_doc.logger import get_logger
from torch.nn.modules.loss import _Loss
import numpy as np

from torch_radon import radon

logger = get_logger('loss')


def min_max_normalizae(tensor,dim = 2):
    min_val = torch.min(tensor,dim=dim,keepdim=True).values
    max_val = torch.max(tensor, dim=dim, keepdim=True).values
    eps = 1e-5
    normalized_tensor = 0.001*eps + (1-2*eps)* (tensor-min_val)/(max_val-min_val+eps)

    return normalized_tensor


def SpectralLoss(generated_images):
    batch_size = generated_images.shape[0]
    image_size = generated_images.shape[-2:][0]
    device = generated_images.device

    # Create a tensor to store the spectrum of each image in the batch
    spectra = torch.zeros((batch_size,*generated_images.shape[-2:])).to(device)  # You will need to define sinogram_shape based on the output of the radon function
    # spectra = np.zeros((batch_size, *generated_images.shape[-2:]))

    n_angles = image_size
    angles = np.pi * torch.arange(n_angles, dtype=torch.float32) / n_angles
    # angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)


    # Loop through each image in the batch
    for i in range(batch_size):
        image = generated_images[i].squeeze(0)
        # print(image_size)
        sinogram = radon(image,angles)
        # radon_fanbeam = RadonFanbeam(image_size, fanbeam_angles, source_distance=n_angles, det_distance=n_angles,
        #                              det_spacing=2.5)
        # with torch.no_grad():
        # print(image.shape)
        # with torch.no_grad():
        #     sinogram = radon_fanbeam.forward(image)


        spectra[i] = sinogram

    # print(spectra)

    spectrum = min_max_normalizae(spectra,dim = 0)
    # del spectra,fanbeam_angles,sinogram,radon_fanbeam,image

    return spectrum.reshape(-1)





# def SpectralLoss(generated_images):
#     batch_size = generated_images.shape[0]
#     image_size = generated_images.shape[-2:]
#
#     # Create a tensor to store the spectrum of each image in the batch
#     spectra = torch.zeros((batch_size,
#                            *image_size))  # You will need to define sinogram_shape based on the output of the radon function
#
#     theta = np.linspace(0., 180., max(image_size), endpoint=False)
#
#     # Loop through each image in the batch
#     for i in range(batch_size):
#         image = generated_images[i].squeeze(
#             0).detach().cpu().numpy()  # Remove channel dimension and convert tensor to numpy array
#         sinogram = radon(image, theta=theta)
#
#         # Convert sinogram back to a tensor and normalize
#         sinogram = torch.tensor(sinogram).float()
#         spectrum = F.normalize(sinogram, dim=0)
#
#         # Store the spectrum
#         spectra[i] = sinogram
#     # print(spectra)
#     # print(spectra.shape)
#
#     return spectra
