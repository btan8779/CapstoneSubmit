# from monai.transforms import Compose, Resized,AddChanneld,ToTensord, Orientationd, Spacingd, CastToTyped
from Config_doc.logger import get_logger
import numpy as np
from monai.transforms import Compose, NormalizeIntensityd,Resized,AddChanneld,ToTensord, Orientationd, Spacingd, CastToTyped
from torchvision.transforms import ToTensor,Resize,Compose,ToPILImage
import importlib
import torch

# WARN: use fixed random state for reproducibility; if you want to randomize on each run seed with `time.time()` e.g.
GLOBAL_RANDOM_STATE = np.random.RandomState(47)

logger = get_logger('Transform')

def transform_2d_image():
    transform = Compose([AddChanneld(keys=['raw', 'label']),
                        # NormalizeIntensityd(keys=['raw']),
                         ToTensord(keys=['raw', 'label']),
                         Resized(keys=['raw', 'label'], spatial_size=(128, 128)),
                         NormalizeIntensityd(keys=['raw']),
                         Orientationd(("raw", "label"), axcodes="LAS"),
                         Spacingd(("raw", "label"), pixdim=(2, 2, 3), mode=("bilinear", "nearest")),
                        #  ScaleIntensityRanged("raw", a_min=0, a_max=15, b_min=0.0, b_max=1.0, clip=False),
                        #  ScaleIntensityRanged("raw", a_min=-100, a_max=250, b_min=0.0, b_max=1.0, clip=False),
                        #  RandAffined(('raw', 'label'), prob=0.15, rotate_range=(0.05, 0.05),
                        #              # 3 parameters control the transform on 3 dimensions
                        #              scale_range=(0.1, 0.1), mode=("bilinear", "nearest")),
                        #  RandGaussianNoised(('raw', 'label'), prob=0.15, std=0.01),
                        CastToTyped(("raw", "label"), dtype=(torch.float32, torch.float32))])
    return transform

class ToTensor_:
    def __init__(self,name,random_state=None ):
        
        # self.random_state = random_state
        # self.name = name
        # super(ToTensor, self).__init__()
        pass

    def __call__(self, name,random_state=None):
        return ToTensor()
    
class ToPILImage_:
    def __init__(self,name,random_state=None):
        # super(AddChannel, self).__init__()
        pass

    def __call__(self,name,random_state=None):
        return ToPILImage()

class Resize_:
    def __init__(self,name,size,random_state=None):
        logger.info(f'REsize:,{size},{name},{random_state}')
        self.size = size
        self.name = name
        self.random_state = random_state
        # super(Resize, self).__init__()
        # pass

    def __call__(self,image): #,name,size,random_state=None):
        return Resize(self.size)


class Transformer:
    def __init__(self, phase_config):
        self.phase_config = phase_config
        # self.config_base = base_config
        self.seed = GLOBAL_RANDOM_STATE.randint(10000000)

    def transform(self):
        return self._create_transform(self.phase_config)


    @staticmethod
    def _transformer_class(class_name):
        m = importlib.import_module('data.transforms')
        clazz = getattr(m, class_name)
        return clazz

    def _create_transform(self, name):
        for i in name:
            assert i in self.phase_config, f'Could not find {i} transform'
        return Compose([
            self._create_augmentation(c) for c in self.phase_config
        ])

    def _create_augmentation(self, c):
        config = {}
        config.update(c)
        config['random_state'] = np.random.RandomState(self.seed)
        aug_class = self._transformer_class(config['name'])
        # print('**config:',**config)
        # print(aug_class(**config))
        return aug_class(**config)
