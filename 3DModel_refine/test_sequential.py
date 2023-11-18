import importlib
import os

import torch
import torch.nn as nn

from data.dataloader import get_test_loaders
from Config_doc.logger import get_logger
from Config_doc.config import load_config
from model.Model import get_model, define_G

# import glob
# import h5py

# from data.transforms import Transformer
# from data.trans import transform_2d_image
# import numpy as np
# import torch
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

logger = get_logger('Test')

def load_checkpoint(checkpoint_path, model, optimizer=None,
                    model_key='model_state_dict', optimizer_key='optimizer_state_dict'):
    """Loads model and training parameters from a given checkpoint_path
    If optimizer is provided, loads optimizer's state_dict of as well.

    Args:
        checkpoint_path (dict): path to the checkpoint to be loaded
        model (dict)(torch.nn.Module): model into which the parameters are to be copied
        optimizer (torch.optim.Optimizer) optional: optimizer instance into
            which the parameters are to be copied

    Returns:
        state
    """
    if isinstance(checkpoint_path, dict) and isinstance(model,dict):
        state = {}
        for drf,path in checkpoint_path.items():
            print(path)
            if not os.path.exists(path):
                raise IOError(f"Checkpoint '{checkpoint_path}' does not exist")

            state[drf] = torch.load(path, map_location='cpu')
            model[drf].load_state_dict(state[drf][model_key])
            if optimizer is not None:
                optimizer.load_state_dict(state[optimizer_key])
    else:
        if not os.path.exists(checkpoint_path):
            raise IOError(f"Checkpoint '{checkpoint_path}' does not exist")

        state = torch.load(checkpoint_path, map_location='cpu')

        model.load_state_dict(state[model_key])

        if optimizer is not None:
            optimizer.load_state_dict(state[optimizer_key])



    return state

# def _get_predictor(model, output_dir, config):
#     predictor_config = config.get('predictor', {})
#     class_name = predictor_config.get('name', 'StandardPredictor')

#     # m = importlib.import_module('unet3d.predictor')
#     m = importlib.import_module('unet3d.predictor')
#     predictor_class = getattr(m, class_name)

#     return predictor_class(model, output_dir, config, **predictor_config)


# def main():
#     # Load configuration
#     config = load_config()

#     # Create the model
#     model = get_model(config['model'])

#     # Load model state
#     model_path = config['model_path']
#     logger.info(f'Loading model from {model_path}...')
#     utils.load_checkpoint(model_path, model)
#     # use DataParallel if more than 1 GPU available
#     device = config['device']
#     if torch.cuda.device_count() > 1 and not device.type == 'cpu':
#         model = nn.DataParallel(model)
#         logger.info(f'Using {torch.cuda.device_count()} GPUs for prediction')

#     logger.info(f"Sending the model to '{device}'")
#     model = model.to(device)

#     output_dir = config['loaders'].get('output_dir', None)
#     if output_dir is not None:
#         os.makedirs(output_dir, exist_ok=True)
#         logger.info(f'Saving predictions to: {output_dir}')

#     # create predictor instance
#     predictor = _get_predictor(model, output_dir, config)

#     for test_loader in get_test_loaders(config):
#         # run the model prediction on the test_loader and save the results in the output_dir
#         predictor(test_loader)

def _get_predictor(refine_model, model, output_dir, config):
    predictor_config = config.get('predictor', {})
    class_name = predictor_config.get('name', 'StandardPredictor')

    # m = importlib.import_module('unet3d.predictor')
    m = importlib.import_module('model.predic')
    print('get m >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    predictor_class = getattr(m, class_name)

    return predictor_class(refine_model, model, output_dir, config, **predictor_config)


def main_sequential():
    # Load configuration
    config = load_config()

    # Create the model
    # model = get_model(config['model'])
    # model = get_model(config['model'])
    # drf_list = ['drf_50', 'drf_20', 'drf_10', 'drf_4', 'drf_2', 'Full_dose']
    drf_list = ['drf_20',  'drf_4',  'Full_dose']
    generate_models = {drf: define_G(**config['model']) for drf in drf_list}
    refine_models = {drf: get_model(config['refine_model']) for drf in drf_list}

    # model = define_G(**config['model'])
    # refine_model = get_model(config['refine_model'])
    # refine_model_train = get_model(config['refine_model'])

    # Load model state
    model_folder = config['model_folder']
    drf_paths = {drf: os.path.join(model_folder, drf) for drf in drf_list}
    # print(model_folder)
    # generate_model_paths = {drf: (drf_paths[drf]+'/best_checkpoint.pytorch') for drf in drf_list}
    generate_model_paths = {drf: (drf_paths[drf] + '/last_checkpoint.pytorch') for drf in drf_list}
    # print(generate_model_paths)
    # model_path = config['model_path']
    logger.info(f'Loading model from {generate_model_paths}...')
    load_checkpoint(generate_model_paths, generate_models)

    refine_model_paths = {drf: (drf_paths[drf]+'/refine_last_checkpoint.pytorch') for drf in drf_list}
    # refine_model_paths = {drf: (drf_paths[drf]+'/refine_best_checkpoint.pytorch') for drf in drf_list}
    # refine_model_path = config['refine_model_path']
    logger.info(f'Loading refine model from {refine_model_paths}...')
    load_checkpoint(refine_model_paths, refine_models)

    # refine_model_train_path = config['refine_model_train_path']
    # logger.info(f'Loading refine model from {refine_model_train_path}...')
    # utils.load_checkpoint(refine_model_train_path, refine_model_train)

    # Load device
    device = config['device']

    # Send models to device
    logger.info(f"Sending the models to '{device}'")
    for drf, model in generate_models.items():
        generate_models[drf] = model.to(device)
        refine_models[drf] = refine_models[drf].to(device)
    # refine_model_train = refine_model_train.to(device)

    # Create output path
    output_dir = config['loaders'].get('output_dir', None)
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f'Saving predictions to: {output_dir}')

    # Create predictor instance
    predictor = _get_predictor(refine_models, generate_models, output_dir, config)

    for test_loader in get_test_loaders(config):
        # for a,b in test_loader:
        #     print(a)
        #     print(b)
        # print('ssssssssssss')
        # print(test_loader.dataset)
        # run the model prediction on the test_loader and save the results in the output_dir
        predictor(test_loader)


if __name__ == '__main__':
    main_sequential()