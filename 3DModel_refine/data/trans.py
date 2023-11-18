from monai.transforms import Compose, NormalizeIntensityd,Resized,AddChanneld,ToTensord, Orientationd, Spacingd, CastToTyped,CastToType,AddChannel,ToTensor
import torch
import torchvision.transforms as transforms
from Config_doc.logger import get_logger
logger = get_logger('transform')

# def transform_3d_image():
#         transform = Compose([AddChanneld(keys=['raw', 'label']),
#                              # NormalizeIntensityd(keys=['raw']),
#                              ToTensord(keys=['raw', 'label'])
#                         #      Resized(keys=['raw', 'label'], spatial_size=(32, 256, 256)),
#                         #      NormalizeIntensityd(keys=['raw','label']),
#                              # Orientationd(("raw", "label"), axcodes="LAS"),
#                              # Spacingd(("raw", "label"), pixdim=(2, 2, 3),
#                              #          mode=("bilinear", "nearest")),
#                             #  ScaleIntensityRanged("raw", a_min=0, a_max=15, b_min=0.0, b_max=1.0, clip=False),
#                             #  ScaleIntensityRanged("raw", a_min=-100, a_max=250, b_min=0.0, b_max=1.0, clip=False),
#                             #  RandAffined(('raw', 'label'), prob=0.15, rotate_range=(0.05, 0.05),
#                             #              # 3 parameters control the transform on 3 dimensions
#                             #              scale_range=(0.1, 0.1), mode=("bilinear", "nearest")),
#                             #  RandGaussianNoised(('raw', 'label'), prob=0.15, std=0.01),
#                             #  CastToTyped(("raw", "label"), dtype=(torch.float64, torch.float64))
#                              ])
#         return transform
#

def transform_3d_image():
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts PIL Image or numpy.ndarray to tensor.
        # transforms.Lambda(lambda x: x.type(torch.float64))  # Cast to float32.
    ])
    return transform
def transform_image():
        transform = Compose([AddChannel(),
                             # NormalizeIntensityd(keys=['raw']),
                             ToTensor(),
                            #  Resized(keys=['raw', 'label'], spatial_size=(32, 256, 256)),
                            #  NormalizeIntensityd(keys=['raw','label']),
                             # Orientationd(("raw", "label"), axcodes="LAS"),
                             # Spacingd(("raw", "label"), pixdim=(2, 2, 3),
                             #          mode=("bilinear", "nearest")),
                            #  ScaleIntensityRanged("raw", a_min=0, a_max=15, b_min=0.0, b_max=1.0, clip=False),
                            #  ScaleIntensityRanged("raw", a_min=-100, a_max=250, b_min=0.0, b_max=1.0, clip=False),
                            #  RandAffined(('raw', 'label'), prob=0.15, rotate_range=(0.05, 0.05),
                            #              # 3 parameters control the transform on 3 dimensions
                            #              scale_range=(0.1, 0.1), mode=("bilinear", "nearest")),
                            #  RandGaussianNoised(('raw', 'label'), prob=0.15, std=0.01),
                            #  CastToType(dtype=(torch.float32))
                             ])
        return transform


    # else:
    #     transform = Compose([AddChanneld(keys=['raw', 'label']),  # Add channel dimension to raw and label
    #                          ToTensord(keys=['raw', 'label']),
    #                          Resized(keys=['raw', 'label'], spatial_size=(128, 128)),
    #                          NormalizeIntensityd(keys=['raw']),
    #                          Orientationd(("raw", "label"), axcodes="LAS"),
    #                          Spacingd(("raw", "label"), pixdim=(2, 2, 3),
    #                                   mode=("bilinear", "nearest")),
    #                         #  ScaleIntensityRanged("raw", a_min=0, a_max=15, b_min=0.0, b_max=1.0, clip=False),
    #                         #  ScaleIntensityRanged("raw", a_min=-100, a_max=250, b_min=0.0, b_max=1.0, clip=False),
    #                          CastToTyped(("raw", "label"), dtype=(torch.float32, torch.float32))])

       