import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,0"
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'


# os.environ['CUDA_VISIBLE_DEVICES'] = '1,0'
from Config_doc.logger import get_logger
from Config_doc.config import load_config
import random
import torch
from model.trainer import create_ARGAN_3d_trainer, create_DRFARGAN_3d_trainer, create_DRFARGAN_3d_trainer_total_back, create_DRFARGAN_3d_no_residual_trainer_total_back,create_ARGAN_3d_trainer_residual_refine
# import os



'''Create training set up logger'''

logger = get_logger('TrainingSetup')

'''Set up the training process'''

def main():
    config = load_config()
    # log the config information
    logger.info(config)

    manual_seed = config.get('manual_seed', None)
    if manual_seed is not None:
        logger.info(f'Seed the Random Number Generator for all devices with {manual_seed}')
        logger.warning('Using CuDNN deterministic setting. This may slow down the training!')
        random.seed(manual_seed)
        torch.manual_seed(manual_seed)
        # see https://pytorch.org/docs/stable/notes/randomness.html
        # Use deterministic algorithms for convolution operations, which makes the training process deterministic
        torch.backends.cudnn.deterministic = True
    # create trainer
    # trainer = create_DRFARGAN_3d_trainer(config)
    trainer = create_ARGAN_3d_trainer_residual_refine(config)
    # trainer = create_ARGAN_3d_trainer(config)
    # trainer = create_ARGAN_3d_trainer(config)
    # trainer = create_joint_trainer(config)
    # trainer = create_trainer(config)
    # Start training
    trainer.fit()
    return config

if __name__ == '__main__':
    main()