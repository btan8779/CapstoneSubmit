import numpy as np
from model.Model import get_class
import torch
from torch.utils.data import DataLoader, RandomSampler
from Config_doc.logger import get_logger
from tqdm import tqdm
import _collections_abc


logger = get_logger('DataLoader')

def get_slice_builder(raws, labels, weight_maps, config):
    assert 'name' in config
    logger.info(f"Slice builder config: {config}")
    slice_builder_cls = get_loader_classes(config['name'])
    return slice_builder_cls(raws, labels, weight_maps, **config)

def get_loader_classes(class_name):
    modules = [
        'data.datasets'
    ]
    return get_class(class_name, modules)
            

def get_train_loaders(config):
    """
    Returns dictionary containing the training and validation loaders (torch.utils.data.DataLoader).

    :param config: a top level configuration object containing the 'loaders' key
    :return: dict {
        'train': <train_loader>
        'val': <val_loader>
    }
    """
    assert 'loaders' in config, 'Could not find data loaders configuration'
    loaders_config = config['loaders']
    train_dataset_path = loaders_config['train']['file_paths'][0]
    val_dataset_path = loaders_config['val']['file_paths'][0]
    # load data augmentation configuration

    #     # load slice builder config

    logger.info('Creating training and validation set loaders...')

    # get dataset class
    dataset_cls_str = loaders_config.get('dataset', None)
    if dataset_cls_str is None:
        dataset_cls_str = 'H5Dataset'
        logger.warning(f"Cannot find dataset class in the config. Using default '{dataset_cls_str}'.")
    dataset_class = get_loader_classes(dataset_cls_str)
    assert set(loaders_config['train']['file_paths']).isdisjoint(loaders_config['val']['file_paths']), \
        "Train and validation 'file_paths' overlap. One cannot use validation data for training!"
    train_datasets = dataset_class(file_path=train_dataset_path, phase='train')

    val_datasets = dataset_class(file_path=val_dataset_path, phase='val')
    
    num_workers = loaders_config.get('num_workers', 1)
    logger.info(f'Number of workers for train/val dataloader: {num_workers}')
    batch_size = loaders_config.get('batch_size', 1)
    if torch.cuda.device_count() > 1 and not config['device'].type == 'cpu':
        logger.info(
            f'{torch.cuda.device_count()} GPUs available. Using batch_size = {torch.cuda.device_count()} * {batch_size}')
        batch_size = batch_size * torch.cuda.device_count()
    logger.info(f'Batch size for train/val loader: {batch_size}')
   
    return {
        'train': DataLoader(train_datasets, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers ,pin_memory=True, drop_last=True),
        # don't shuffle during validation: useful when showing how predictions for a given batch get better over time
        'val': DataLoader(val_datasets, batch_size=batch_size, shuffle=False, num_workers=num_workers,pin_memory=True)
    }

def get_test_loaders(config):
    """
    Returns test DataLoader.

    :return: generator of DataLoader objects
    """

    assert 'loaders' in config, 'Could not find data loaders configuration'
    loaders_config = config['loaders']

    logger.info('Creating test set loaders...')

    # get dataset class
    dataset_cls_str = loaders_config.get('dataset', None)
    if dataset_cls_str is None:
        dataset_cls_str = 'StandardHDF5Dataset'
        logger.warning(f"Cannot find dataset class in the config. Using default '{dataset_cls_str}'.")
    dataset_class = get_loader_classes(dataset_cls_str)

    test_datasets = dataset_class.create_datasets(loaders_config, phase='test')

    num_workers = loaders_config.get('num_workers', 1)
    logger.info(f'Number of workers for the dataloader: {num_workers}')

    batch_size = loaders_config.get('batch_size', 1)
    if torch.cuda.device_count() > 1 and not config['device'].type == 'cpu':
        logger.info(
            f'{torch.cuda.device_count()} GPUs available. Using batch_size = {torch.cuda.device_count()} * {batch_size}')
        batch_size = batch_size * torch.cuda.device_count()

    logger.info(f'Batch size for dataloader: {batch_size}')

    # use generator in order to create data loaders lazily one by one
    for test_dataset in test_datasets:
        logger.info(f'Loading test set from: {test_dataset.file_path}...')
        if hasattr(test_dataset, 'prediction_collate'):
            collate_fn = test_dataset.prediction_collate
        else:
            collate_fn = default_prediction_collate

        yield DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers,
                         collate_fn=collate_fn)


def default_prediction_collate(batch):
    """
    Default collate_fn to form a mini-batch of Tensor(s) for HDF5 based datasets
    """



    error_msg = "batch must contain tensors or slice; found {}"
    if isinstance(batch[0], torch.Tensor):
        return torch.stack(batch, 0)
    elif isinstance(batch[0], tuple) and isinstance(batch[0][0], slice):
        return batch
    elif isinstance(batch[0], _collections_abc.Sequence):
        transposed = zip(*batch)
        return [default_prediction_collate(samples) for samples in transposed]

    raise TypeError((error_msg.format(type(batch[0]))))