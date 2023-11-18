import argparse
import torch
import yaml
from Config_doc.logger import get_logger

logger = get_logger('Config')

def load_config():
    # Create Argument Parser(创建解析器), the description here is to give a brief description about what the process does
    parser = argparse.ArgumentParser(description='2DModel')
    # Add argument information, you need to run python documentName.py --config yamlDocumentPath.yml in terminal to run the code
    parser.add_argument('--config', type=str, help='Path to the YAML config file', required=True)
    # args is the config yaml document
    args = parser.parse_args()
    # load the information in the config document
    config = yaml.safe_load(open(args.config, 'r'))
    # Get a device to train on, if no device information in config document set the device to None
    device_str = config.get('device', None)
    if device_str is not None:
        logger.info(f"Device specified in config: '{device_str}'")
        if device_str.startswith('cuda') and not torch.cuda.is_available():
            logger.warning('CUDA not available, using CPU')
            device_str = 'cpu'
    else:
        device_str = "cuda:0" if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using '{device_str}' device")

    device = torch.device(device_str)
    config['device'] = device
    # Return the config yaml document with the adjustment of whether the gpu is availible and modified it if gou is not avalible or lost the device information
    return config


def _load_config_yaml(config_file):
    return yaml.safe_load(open(config_file, 'r'))
