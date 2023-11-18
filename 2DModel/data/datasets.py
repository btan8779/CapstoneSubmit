from torch.utils.data import Dataset, DataLoader
import glob
import h5py
from Config_doc.logger import get_logger
from data.transforms import Transformer
from data.trans import transform_2d_image, test_transform_image
import numpy as np
import torch
from data.dataloader import default_prediction_collate, get_slice_builder
import os
from itertools import chain

logger = get_logger('Dataset')


class SliceBuilder:
    """
    Builds the position of the patches in a given raw/label/weight ndarray based on the the patch and stride shape
    """

    def __init__(self, raw_dataset, label_dataset, weight_dataset, patch_shape, stride_shape, **kwargs):
        """
        :param raw_dataset: ndarray of raw data
        :param label_dataset: ndarray of ground truth labels
        :param weight_dataset: ndarray of weights for the labels
        :param patch_shape: the shape of the patch DxHxW
        :param stride_shape: the shape of the stride DxHxW
        :param kwargs: additional metadata
        """

        patch_shape = tuple(patch_shape)
        stride_shape = tuple(stride_shape)
        skip_shape_check = kwargs.get('skip_shape_check', False)
        if not skip_shape_check:
            self._check_patch_shape(patch_shape)

        self._raw_slices = self._build_slices(raw_dataset, patch_shape, stride_shape)
        if label_dataset is None:
            self._label_slices = None
        else:
            # take the first element in the label_dataset to build slices
            self._label_slices = self._build_slices(label_dataset, patch_shape, stride_shape)
            assert len(self._raw_slices) == len(self._label_slices)
        if weight_dataset is None:
            self._weight_slices = None
        else:
            self._weight_slices = self._build_slices(weight_dataset, patch_shape, stride_shape)
            assert len(self.raw_slices) == len(self._weight_slices)

    @property
    def raw_slices(self):
        return self._raw_slices

    @property
    def label_slices(self):
        return self._label_slices

    @property
    def weight_slices(self):
        return self._weight_slices

    @staticmethod
    def _build_slices(dataset, patch_shape, stride_shape):
        """Iterates over a given n-dim dataset patch-by-patch with a given stride
        and builds an array of slice positions.

        Returns:
            list of slices, i.e.
            [(slice, slice, slice, slice), ...] if len(shape) == 4
            [(slice, slice, slice), ...] if len(shape) == 3
        """
        slices = []
        if dataset.ndim == 4:
            in_channels, i_z, i_y, i_x = dataset.shape
        else:
            i_z, i_y, i_x = dataset.shape

        k_z, k_y, k_x = patch_shape
        s_z, s_y, s_x = stride_shape
        z_steps = SliceBuilder._gen_indices(i_z, k_z, s_z)
        for z in z_steps:
            y_steps = SliceBuilder._gen_indices(i_y, k_y, s_y)
            for y in y_steps:
                x_steps = SliceBuilder._gen_indices(i_x, k_x, s_x)
                for x in x_steps:
                    slice_idx = (
                        slice(z, z + k_z),
                        slice(y, y + k_y),
                        slice(x, x + k_x)
                    )
                    if dataset.ndim == 4:
                        slice_idx = (slice(0, in_channels),) + slice_idx
                    slices.append(slice_idx)
        return slices

    @staticmethod
    def _gen_indices(i, k, s):
        assert i >= k, 'Sample size has to be bigger than the patch size'
        for j in range(0, i - k + 1, s):
            yield j
        if j + k < i:
            yield i - k

    @staticmethod
    def _check_patch_shape(patch_shape):
        assert len(patch_shape) == 3, 'patch_shape must be a 3D tuple'
        assert patch_shape[1] >= 64 and patch_shape[2] >= 64, 'Height and Width must be greater or equal 64'


def calculate_stats(images):
    """
    Calculates min, max, mean, std given a list of ndarrays
    """
    # flatten first since the images might not be the same size
    flat = np.concatenate(
        [img.ravel() for img in images]
    )
    return {'pmin': np.percentile(flat, 1), 'pmax': np.percentile(flat, 99.6), 'mean': np.mean(flat),
            'std': np.std(flat)}


class H5Dataset(Dataset):
    def __init__(self, file_path,
                 phase,
                 raw_internal_path='raw',
                 label_internal_path='label',
                 weight_internal_path=None,
                 global_normalization=None):
        """
        :param file_path: path to H5 file containing raw data as well as labels and per pixel weights (optional)
        :param phase: 'train' for training, 'val' for validation, 'test' for testing; data augmentation is performed
            only during the 'train' phase
        :para'/home/adrian/workspace/ilastik-datasets/VolkerDeconv/train'm slice_builder_config: configuration of the SliceBuilder
        :param transformer_config: data augmentation configuration
        :param mirror_padding (int or tuple): number of voxels padded to each axis
        :param raw_internal_path (str or list): H5 internal path to the raw dataset
        :param label_internal_path (str or list): H5 internal path to the label dataset
        :param weight_internal_path (str or list): H5 internal path to the per pixel weights
        """
        assert phase in ['train', 'val', 'test']
        self.phase = phase
        self.file_path = file_path
        self.raw_internal_path = raw_internal_path
        self.label_internal_path = label_internal_path
        self.weight_internal_path = weight_internal_path
        self.global_normalization = global_normalization
        self.image_list = glob.glob(self.file_path + "/*.h5")

    def __len__(self):
        logger.info(f'dataset length is{len(self.image_list)}')
        return len(self.image_list)

    def __getitem__(self, index):

        with h5py.File(self.image_list[index], 'r') as input_file:
            self.raw = input_file['raw'][:]
            self.label = input_file['label'][:]
            if self.phase == 'test':
                self.transform = test_transform_image()
                trans_data = self.transform(input_file)
                trans_raw = trans_data['raw']
                trans_label = trans_data['label']
                return trans_data, index

            else:
                self.transform = transform_2d_image()
                trans_data = self.transform(input_file)
                # print(input_file['raw'][:])
                # print(input_file['label'][:])
                # trans_data = self.transform(input_file)

                # raw_trans_data = self.transformer(torch.tensor(np.array(input_file['raw'])))
                # label_trans_data = self.transformer(torch.tensor(np.array(input_file['label'])))
                # logger.info('finish the transform')
                trans_raw = trans_data['raw'] #self.transform(self.raw)
                trans_label = trans_data['label'] #self.transform(self.label)
                # s
                # self.raw_transform = self.transformer.raw_transform()
                # self.label_transform = self.transformer.label_transform()
                # raw_trans = self.raw_transform(input_file['raw'])
                # label_trans = self.raw_transform(input_file['label'])

                return trans_raw, trans_label

    # @staticmethod
    # def create_h5_file(file_path):
    #     return h5py.File(file_path, 'r')


class ConfigDataset(Dataset):
    def __getitem__(self, index):
        raise NotImplementedError
    def __len__(self):
        raise NotImplementedError

    @classmethod
    def create_datasets(cls, dataset_config, phase):
        """
        Factory method for creating a list of datasets based on the provided config.

        Args:
            dataset_config (dict): dataset configuration
            phase (str): one of ['train', 'val', 'test']

        Returns:
            list of `Dataset` instances
        """
        raise NotImplementedError

    @classmethod
    def prediction_collate(cls, batch):
        """Default collate_fn. Override in child class for non-standard datasets."""
        return default_prediction_collate(batch)


class AbstractHDF5Dataset(ConfigDataset):
    """
    Implementation of torch.utils.data.Dataset backed by the HDF5 files, which iterates over the raw and label datasets
    patch by patch with a given stride.
    """

    def __init__(self, file_path,
                 phase,
                 slice_builder_config,
                 transformer_config,
                 mirror_padding=(1, 32, 32),
                 raw_internal_path='raw',
                 label_internal_path='label',
                 pre_prediction_path='predictions',
                 weight_internal_path=None,
                 global_normalization=True):
        """
        :param file_path: path to H5 file containing raw data as well as labels and per pixel weights (optional)
        :param phase: 'train' for training, 'val' for validation, 'test' for testing; data augmentation is performed
            only during the 'train' phase
        :para'/home/adrian/workspace/ilastik-datasets/VolkerDeconv/train'm slice_builder_config: configuration of the SliceBuilder
        :param transformer_config: data augmentation configuration
        :param mirror_padding (int or tuple): number of voxels padded to each axis
        :param raw_internal_path (str or list): H5 internal path to the raw dataset
        :param label_internal_path (str or list): H5 internal path to the label dataset
        :param weight_internal_path (str or list): H5 internal path to the per pixel weights
        """
        assert phase in ['train', 'val', 'test']
        if phase in ['train', 'val']:
            mirror_padding = None

        if mirror_padding is not None:
            if isinstance(mirror_padding, int):
                mirror_padding = (mirror_padding,) * 3
            else:
                assert len(mirror_padding) == 3, f"Invalid mirror_padding: {mirror_padding}"

        self.mirror_padding = mirror_padding
        self.phase = phase
        self.file_path = file_path

        input_file = self.create_h5_file(file_path)

        self.raw = self.fetch_and_check(input_file, raw_internal_path)

        if global_normalization:
            stats = calculate_stats(self.raw)
        else:
            stats = {'pmin': None, 'pmax': None, 'mean': None, 'std': None}

        self.transformer = test_transform_image()
        self.raw_transform = self.transformer

        if phase != 'test':
            self.prediction = None
            # create label/weight transform only in train/val phase
            self.label_transform = self.transformer  # .label_transform()
            self.label = self.fetch_and_check(input_file, label_internal_path)

            if weight_internal_path is not None:
                # look for the weight map in the raw file
                self.weight_map = self.fetch_and_check(input_file, weight_internal_path)
                self.weight_transform = self.transformer  # .weight_transform()
            else:
                self.weight_map = None

            self._check_volume_sizes(self.raw, self.label)
        else:
            # 'test' phase used only for predictions so ignore the label dataset
            self.label = None
            # self.weight_map = None
            if weight_internal_path is not None:
                # look for the weight map in the raw file
                self.weight_map = self.fetch_and_check(input_file, weight_internal_path)
                self.weight_transform = self.transformer  # .weight_transform()
            else:
                self.weight_map = None
            # self.weight_map = self.fetch_and_check(input_file, weight_internal_path)
            # self.weight_transform = self.transformer#.weight_transform()
            # preliminary prediction path
            # self.prediction = self.fetch_and_check(input_file, pre_prediction_path)
            # add mirror padding if needed
            if self.mirror_padding is not None:
                z, y, x = self.mirror_padding
                pad_width = ((z, z), (y, y), (x, x))
                if self.raw.ndim == 4:
                    channels = [np.pad(r, pad_width=pad_width, mode='reflect') for r in self.raw]
                    self.raw = np.stack(channels)
                    if self.weight_map != None:
                        w_channels = [np.pad(r, pad_width=pad_width, mode='reflect') for r in self.weight_map]
                        self.weight_map = np.stack(w_channels)
                    # pre_channels = [np.pad(r, pad_width=pad_width, mode='reflect') for r in self.prediction]
                    # self.prediction = np.stack(pre_channels)
                else:
                    self.raw = np.pad(self.raw, pad_width=pad_width, mode='reflect')
                    if self.weight_map != None:
                        self.weight_map = np.pad(self.weight_map, pad_width=pad_width, mode='reflect')
                    # self.prediction = np.pad(self.prediction, pad_width=pad_width, mode='reflect')

        # build slice indices for raw and label data sets

        slice_builder = get_slice_builder(self.raw, self.label, self.weight_map, slice_builder_config)
        self.raw_slices = slice_builder.raw_slices
        self.label_slices = slice_builder.label_slices

        self.weight_slices = slice_builder.weight_slices

        self.patch_count = len(self.raw_slices)
        logger.info(f'Number of raw patches: {self.patch_count}')

    @staticmethod
    def fetch_and_check(input_file, internal_path):
        ds = input_file[internal_path][:]
        if ds.ndim == 2:
            # expand dims if 2d
            ds = np.expand_dims(ds, axis=0)
        return ds

    def __getitem__(self, idx):
        if idx >= len(self):
            raise StopIteration

        # get the slice for a given index 'idx'
        raw_idx = self.raw_slices[idx]
        # get the raw data patch for a given slice
        raw_patch_transformed = self.raw_transform(self.raw[raw_idx])
        # get the slice for a given index 'idx'
        if self.weight_slices != None:
            weight_idx = self.weight_slices[idx]
            # get the raw data patch for a given slice
            weight_patch_transformed = self.weight_transform(self.weight_map[weight_idx])
        else:
            weight_patch_transformed = None

        if self.phase == 'test':
            # prediction_patch_transformed = self.raw_transform(self.prediction[raw_idx])
            # discard the channel dimension in the slices: predictor requires only the spatial dimensions of the volume
            if len(raw_idx) == 4:
                raw_idx = raw_idx[1:]
            # return raw_patch_transformed, prediction_patch_transformed, raw_idx
            return raw_patch_transformed, raw_idx  # weight_patch_transformed,
        else:
            # get the slice for a given index 'idx'
            label_idx = self.label_slices[idx]
            label_patch_transformed = self.label_transform(self.label[label_idx])
            if self.weight_map is not None:
                weight_idx = self.weight_slices[idx]
                weight_patch_transformed = self.weight_transform(self.weight_map[weight_idx])
                return raw_patch_transformed, label_patch_transformed, weight_patch_transformed
            # return the transformed raw and label patches
            return raw_patch_transformed, label_patch_transformed

    def __len__(self):
        return self.patch_count

    @staticmethod
    def create_h5_file(file_path):
        try:
            logger.info('copy data if data is already h5 file')
            input_file = h5py.File(file_path, 'r')
            data = input_file
            logger.info(f'get input file from {file_path}')
        except Exception:
            raise NotImplementedError
        return data

    @staticmethod
    def _check_volume_sizes(raw, label):
        def _volume_shape(volume):
            if volume.ndim == 3:
                return volume.shape
            return volume.shape[1:]

        assert raw.ndim in [3, 4], 'Raw dataset must be 3D (DxHxW) or 4D (CxDxHxW)'
        assert label.ndim in [3, 4], 'Label dataset must be 3D (DxHxW) or 4D (CxDxHxW)'

        assert _volume_shape(raw) == _volume_shape(label), 'Raw and labels have to be of the same size'

    @classmethod
    def create_datasets(cls, dataset_config, phase):
        phase_config = dataset_config[phase]

        # load data augmentation configuration
        transformer_config = phase_config['transformer']
        # load slice builder config
        slice_builder_config = phase_config['slice_builder']
        # load files to process
        file_paths = phase_config['file_paths']
        # file_paths may contain both files and directories; if the file_path is a directory all H5 files inside
        # are going to be included in the final file_paths
        file_paths = cls.traverse_h5_paths(file_paths)

        # load tracer information
        # with open(tracer_infor_path) as file:
        #     document = yaml.full_load(file)

        datasets = []
        drf_list = ['drf100','drf50','drf10','drf4','drf2','drf20']
        # for file_path in file_paths[0:200]:
        for file_path in file_paths:
            image_info = file_path.split('/')[-1]
            patient_info = image_info.split('.')[0]
            patient_num = patient_info.split('_')[1]
            patient_drf = patient_info.split('_')[0]
            # patient_number = int(patient_num)
            # print(patient_num)
            if patient_drf in drf_list:
                try:
                    logger.info(f'Loading {phase} set from: {file_path}...')
                    dataset = cls(file_path=file_path,
                                  phase=phase,
                                  slice_builder_config=slice_builder_config,
                                  transformer_config=transformer_config,
                                  mirror_padding=dataset_config.get('mirror_padding', None),
                                  raw_internal_path=dataset_config.get('raw_internal_path', 'raw'),
                                  label_internal_path=dataset_config.get('label_internal_path', 'label'),
                                  pre_prediction_path=dataset_config.get('pre_prediction_path', 'predictions'),
                                  weight_internal_path=dataset_config.get('weight_internal_path', None),
                                  global_normalization=dataset_config.get('global_normalization', None))
                    datasets.append(dataset)
                except Exception:
                    logger.error(f'Skipping {phase} set: {file_path}', exc_info=True)
        return datasets

    @staticmethod
    def traverse_h5_paths(file_paths):
        assert isinstance(file_paths, list)
        results = []
        for file_path in file_paths:
            if os.path.isdir(file_path):
                # if file path is a directory take all H5 files in that directory
                iters = [glob.glob(os.path.join(file_path, ext)) for ext in ['*.h5', '*.hdf', '*.hdf5', '*.hd5']]
                for fp in chain(*iters):
                    results.append(fp)
            else:
                results.append(file_path)
        return results


class StandardHDF5Dataset(AbstractHDF5Dataset):
    """
    Implementation of the HDF5 dataset which loads the data from all of the H5 files into the memory.
    Fast but might consume a lot of memory.
    """

    def __init__(self, file_path, phase, slice_builder_config, transformer_config, mirror_padding=(16, 32, 32),
                 raw_internal_path='raw', label_internal_path='label', weight_internal_path=None,
                 global_normalization=True):
        super().__init__(file_path=file_path,
                         phase=phase,
                         slice_builder_config=slice_builder_config,
                         transformer_config=transformer_config,
                         mirror_padding=mirror_padding,
                         raw_internal_path=raw_internal_path,
                         label_internal_path=label_internal_path,
                         weight_internal_path=weight_internal_path,
                         global_normalization=global_normalization)

    @staticmethod
    def create_h5_file(file_path):
        return h5py.File(file_path, 'r')


class LazyHDF5Dataset(AbstractHDF5Dataset):
    """Implementation of the HDF5 dataset which loads the data lazily. It's slower, but has a low memory footprint."""

    def __init__(self, file_path, phase, slice_builder_config, transformer_config, mirror_padding=(1, 32, 32),
                 raw_internal_path='raw', label_internal_path='label', pre_prediction_path='predictions',
                 weight_internal_path=None,
                 global_normalization=False):
        super().__init__(file_path=file_path,
                         phase=phase,
                         slice_builder_config=slice_builder_config,
                         transformer_config=transformer_config,
                         mirror_padding=mirror_padding,
                         raw_internal_path=raw_internal_path,
                         label_internal_path=label_internal_path,
                         pre_prediction_path=pre_prediction_path,
                         weight_internal_path=weight_internal_path,
                         global_normalization=global_normalization)

        logger.info("Using modified HDF5Dataset!")

    @staticmethod
    def create_h5_file(file_path):
        return LazyHDF5File(file_path)


class LazyHDF5File:
    """Implementation of the LazyHDF5File class for the LazyHDF5Dataset."""

    def __init__(self, path, internal_path=None):
        self.path = path
        self.internal_path = internal_path
        if self.internal_path:
            with h5py.File(self.path, "r") as f:
                self.ndim = f[self.internal_path].ndim
                self.shape = f[self.internal_path].shape

    def ravel(self):
        with h5py.File(self.path, "r") as f:
            data = f[self.internal_path][:].ravel()
        return data

    def __getitem__(self, arg):
        if isinstance(arg, str) and not self.internal_path:
            return LazyHDF5File(self.path, arg)

        if arg == Ellipsis:
            return LazyHDF5File(self.path, self.internal_path)

        with h5py.File(self.path, "r") as f:
            data = f[self.internal_path][arg]

        return data

