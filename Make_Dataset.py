import yaml
import os
import random
# from utils import SliceBuilder
import SimpleITK as sitk
import h5py
import shutil
import numpy as np
import tqdm

# data_nii_path = '/mnt/HDD3/btan8779/Data_nii'
#
# tracer_yaml_path = '/mnt/HDD3/btan8779/Challenge_Data/tracer_infor_total.yaml'
# total_yaml_path = '/mnt/HDD3/btan8779/Challenge_Data/dataset_infor.yaml'
# siemens_yaml_path = '/mnt/HDD3/btan8779/Challenge_Data/siemens_dataset_infor.yaml'
# explorer_yaml_path = '/mnt/HDD3/btan8779/Challenge_Data/explorer_dataset_infor.yaml'
# test_yaml_path = '/mnt/HDD3/btan8779/Challenge_Data/test_dataset_infor.yaml'
# test_mixed_dataset = '/mnt/HDD3/btan8779/test_dataset'
# test_test_dataset = '/mnt/HDD3/btan8779/test_dataset/test'
# # new_dataset_yaml_path = '/mnt/HDD3/btan8779/new_dataset/dataset_infor.yaml'
# new_dataset_path = '/mnt/HDD3/btan8779/Data_h5'
# dataset_2d_path = '/mnt/HDD3/btan8779/new_dataset_2d'
# new_dataset_denoise_path = '/mnt/HDD3/btan8779/new_dataset_denoise'
# full_test_path = '/mnt/HDD3/btan8779/new_dataset/test'
# cut_test_path = '/mnt/HDD3/btan8779/new_dataset/cut_test'
# siemens_patient_path = '/mnt/HDD3/btan8779/Challenge_Data/Subject'
# siemens_patient_list = os.listdir(siemens_patient_path)
# # print(siemens_patient_list)
# explorer_patient_path = '/mnt/HDD3/btan8779/Challenge_Data/Explorer'
# explorer_patient_list = os.listdir(explorer_patient_path)
# # print(explorer_patient_list)
# all_patient_list = os.listdir(data_nii_path)
# Siemens_dataset_info_path = "/mnt/HDD3/btan8779/Siemens_dataset_infor.yaml"
# explorer_dataset_info_path = "/mnt/HDD3/btan8779/Explorer_dataset_infor.yaml"

# test_full_size_path = '/mnt/HDD3/btan8779/new_dataset/test'
# test_cut_size_path = '/mnt/HDD3/btan8779/new_dataset/test'
# siemens_2d_path = '/mnt/HDD3/btan8779/CapstoneData/2d_h5_siemens'

class SliceBuilder:
    """
    Builds the position of the patches in a given raw/label/weight ndarray based on the the patch and stride shape
    """

    def __init__(self, raw_dataset, label_dataset, weight_dataset, patch_shape, stride_shape,type, **kwargs):
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

        self._raw_slices = self._build_slices(raw_dataset, patch_shape, stride_shape,type = type)
        if label_dataset is None:
            self._label_slices = None
        else:
            # take the first element in the label_dataset to build slices
            self._label_slices = self._build_slices(label_dataset, patch_shape, stride_shape,type=type)
            assert len(self._raw_slices) == len(self._label_slices)
        if weight_dataset is None:
            self._weight_slices = None
        else:
            self._weight_slices = self._build_slices(weight_dataset, patch_shape, stride_shape,type=type)
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
    def _build_slices(dataset, patch_shape, stride_shape,type):
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
        elif dataset.ndim == 3:
            i_z, i_y, i_x = dataset.shape
        else:
            i_y, i_x = dataset.shape

        if type =='3d':
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
        elif type =='2d':
            k_y, k_x = patch_shape
            s_y, s_x = stride_shape
            y_steps = SliceBuilder._gen_indices(i_y, k_y, s_y)
            for y in y_steps:
                x_steps = SliceBuilder._gen_indices(i_x, k_x, s_x)
                for x in x_steps:
                    slice_idx = (
                        slice(y, y + k_y),
                        slice(x, x + k_x)
                    )
                    # if dataset.ndim == 4:  # if there are channels
                    #     slice_idx = (slice(0, dataset.shape[0]),) + slice_idx
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
        # assert len(patch_shape) == 3, 'patch_shape must be a 3D tuple'
        # assert patch_shape[1] >= 64 and patch_shape[2] >= 64, 'Height and Width must be greater or equal 64'
        if len(patch_shape) == 3:
            assert patch_shape[1] >= 64 and patch_shape[2] >= 64, 'Height and Width must be greater or equal 64'
        elif len(patch_shape) == 2:
            assert patch_shape[0] >= 64 and patch_shape[1] >= 64, 'Height and Width must be greater or equal 64'
        else:
            raise ValueError('patch_shape must be a 2D or 3D tuple')



def denoise(img, weight=0.1, eps=1e-3, num_iter_max=200):
    """Perform total-variation denoising on a grayscale image.

    Parameters
    ----------
    img : array
        2-D input data to be de-noised.
    weight : float, optional
        Denoising weight. The greater `weight`, the more
        de-noising (at the expense of fidelity to `img`).
    eps : float, optional
        Relative difference of the value of the cost
        function that determines the stop criterion.
        The algorithm stops when:
            (E_(n-1) - E_n) < eps * E_0
    num_iter_max : int, optional
        Maximal number of iterations used for the
        optimization.

    Returns
    -------
    out : array
        De-noised array of floats.

    Notes
    -----
    Rudin, Osher and Fatemi algorithm.
    """
    u = np.zeros_like(img)
    px = np.zeros_like(img)
    py = np.zeros_like(img)

    nm = np.prod(img.shape[:2])
    tau = 0.125

    i = 0
    while i < num_iter_max:
        u_old = u
        # x and y components of u's gradient
        ux = np.roll(u, -1, axis=1) - u
        uy = np.roll(u, -1, axis=0) - u
        # update the dual variable
        px_new = px + (tau / weight) * ux
        py_new = py + (tau / weight) * uy

        norm_new = np.maximum(1, np.sqrt(px_new ** 2 + py_new ** 2))
        px = px_new / norm_new
        py = py_new / norm_new
        # calculate divergence
        rx = np.roll(px, 1, axis=1)
        ry = np.roll(py, 1, axis=0)
        div_p = (px - rx) + (py - ry)

        # update image
        u = img + weight * div_p
        # calculate error
        error = np.linalg.norm(u - u_old) / np.sqrt(nm)
        if i == 0:
            err_init = error
            err_prev = error
        else:
            # break if error small enough
            if np.abs(err_prev - error) < eps * err_init:
                break
            else:
                err_prev = error

        # don't forget to update iterator
        i += 1

    return u


class tracer:

    def __init__(self, yaml_path, patient_list):
        self.fdg = []
        self.dota = []
        self.ga = []
        self.wrong = []
        with open(yaml_path) as file:
            document = yaml.full_load(file)
        for patient in patient_list:
            # print(document)
            # for docuement in documents:
            for dataset_type, case_list in document.items():
                if patient == dataset_type:
                    if case_list == 'FDG':
                        self.fdg.append(dataset_type)
                    elif case_list == 'GA68':
                        self.ga.append(dataset_type)
                    elif case_list == 'DOTA':
                        self.dota.append(dataset_type)
                    else:
                        print(dataset_type, 'wrong!!!')
                        self.wrong.append(dataset_type)
        print('total FDG : ', len(self.fdg))
        print('total 68GA : ', len(self.ga))
        print('total DOTA : ', len(self.dota))
        print('wrong : ', self.wrong)

    def dota_list(self):
        return self.dota

    def fdg_list(self):
        return self.fdg

    def ga_list(self):
        return self.ga

    def wrong_list(self):
        return self.wrong


def divide_dataset(dataset_list, ratio_train, ratio_val):
    all_num =len(dataset_list)
    train_list = random.sample(dataset_list, round(ratio_train*all_num))
    rest_list = list(set(dataset_list) - set(train_list))
    val_list = random.sample(rest_list, round(ratio_val*all_num))
    test_list = list(set(rest_list) - set(val_list))

    return train_list, val_list, test_list

def split_dataset(tracer_infor_yaml_path, dataset_yaml_path,type):
    data_infor = {}
    fdg = []
    ga68 = []
    dota = []
    siemens = []
    with open(tracer_infor_yaml_path) as file:
        document = yaml.full_load(file)
        for patient, tracer_type in document.items():
            # print(patient, tracer_type)
            patient_num = int(patient[-3:])
            # print(patient_num)
            if patient_num < 320:
                if tracer_type == 'FDG':
                    fdg.append(patient)
                elif tracer_type == 'GA68':
                    ga68.append(patient)
                elif tracer_type == 'DOTA':
                    dota.append(patient)
                else:
                    print('wrong', patient)
            else:
                # print(siemens)
                siemens.append(patient)
    if type == 'siemens':
        # siemens = list(range(321,678))
        # print(siemens)
        siemens_train, siemens_val, siemens_test = divide_dataset(siemens, 0.8, 0.1)
        print('siemens train', len(siemens_train))
        print('siemens val', len(siemens_val))
        print('siemens test', len(siemens_test))
        train_list = siemens_train
        val_list = siemens_val
        test_list = siemens_test
        print('total train ', len(train_list))
        print('total val ', len(val_list))
        print('total test ', len(test_list))
    elif type == 'explorer':
        u_fdg_train, u_fdg_val, u_fdg_test = divide_dataset(fdg, 0.8, 0.1)
        print('u fdg train', len(u_fdg_train))
        print('u fdg val', len(u_fdg_val))
        print('u fdg test', len(u_fdg_test))
        u_ga_train, u_ga_val, u_ga_test = divide_dataset(ga68, 0.6, 0.2)
        print('u ga68 train', len(u_ga_train))
        print('u ga68 val', len(u_ga_val))
        print('u ga68 test', len(u_ga_test))
        u_dota_train, u_dota_val, u_dota_test = divide_dataset(dota, 0.7, 0.1)
        print('u dota train', len(u_dota_train))
        print('u dota val', len(u_dota_val))
        print('u dota test', len(u_dota_test))
        train_list =  u_fdg_train + u_ga_train + u_dota_train
        val_list =  u_fdg_val + u_ga_val + u_dota_val
        test_list =  u_fdg_test + u_ga_test + u_dota_test
        print('total train ', len(train_list))
        print('total val ', len(val_list))
        print('total test ', len(test_list))
    elif type == 'total':
        # siemens = list(range(321,678))
        # print(siemens)
        siemens_train, siemens_val, siemens_test = divide_dataset(siemens, 0.8, 0.1)
        print('siemens train', len(siemens_train))
        print('siemens val', len(siemens_val))
        print('siemens test', len(siemens_test))
        u_fdg_train, u_fdg_val, u_fdg_test = divide_dataset(fdg, 0.8, 0.1)
        print('u fdg train', len(u_fdg_train))
        print('u fdg val', len(u_fdg_val))
        print('u fdg test', len(u_fdg_test))
        u_ga_train, u_ga_val, u_ga_test = divide_dataset(ga68, 0.6, 0.2)
        print('u ga68 train', len(u_ga_train))
        print('u ga68 val', len(u_ga_val))
        print('u ga68 test', len(u_ga_test))
        u_dota_train, u_dota_val, u_dota_test = divide_dataset(dota, 0.7, 0.1)
        print('u dota train', len(u_dota_train))
        print('u dota val', len(u_dota_val))
        print('u dota test', len(u_dota_test))
        train_list = siemens_train + u_fdg_train + u_ga_train + u_dota_train
        val_list = siemens_val + u_fdg_val + u_ga_val + u_dota_val
        test_list = siemens_test + u_fdg_test + u_ga_test + u_dota_test
        print('total train ', len(train_list))
        print('total val ', len(val_list))
        print('total test ', len(test_list))
    data_infor['train'] = train_list
    data_infor['val'] = val_list
    data_infor['test'] = test_list
    print(data_infor)
    with open(dataset_yaml_path, 'w') as file:
        yaml.dump(data_infor, file, default_flow_style=False)


def make_array_into_patches_h5(new_dataset_path, case_num, drf_num, raw_array, label_array, patch_shape=[32, 256, 256], stride_shape=[20, 256, 256]):
    print('make_array_into_patches_h5 function')
    slice_generator = SliceBuilder(raw_dataset=raw_array,label_dataset=label_array, weight_dataset=None,patch_shape=patch_shape,stride_shape=stride_shape)
    raw_slices = slice_generator._build_slices(dataset=raw_array, patch_shape=patch_shape,stride_shape=stride_shape)
    print(len(raw_slices))
    for slice in raw_slices:
        index_num = raw_slices.index(slice)
        cropped_data= raw_array[slice]
        cropped_label= label_array[slice]
        # print(cropped_label.shape,cropped_data.shape)
        slice_data_path = os.path.join(new_dataset_path,case_num+'_'+'drf'+drf_num+'_'+str(index_num)+'.h5')
        # print(slice_data_path)
        save_h5py_file = h5py.File(slice_data_path, 'w')
        save_h5py_file.create_dataset('raw', data=cropped_data, compression='gzip', compression_opts=9)
        save_h5py_file.create_dataset('label', data=cropped_label, compression='gzip', compression_opts=9)
        save_h5py_file.close()


def build_new_drf_dataset(original_dataset_path, yaml_path, new_drf_path):
    i = 1
    with open(yaml_path) as file:
        document = yaml.full_load(file)
        # for docuement in documents:
        for dataset_type, case_list in document.items():
            #the line below should be removed in the end
            # if dataset_type == 'val':

                print(dataset_type, ":", case_list)
                new_dataset_path = os.path.join(new_drf_path, dataset_type)
                print(new_dataset_path)
                if not os.path.exists(new_dataset_path):
                    os.makedirs(new_dataset_path)
                else:
                    pass
                for case in case_list[0:2]:
                    case = str(case)
                    case_num = case[-3:]
                    # if case_num < 210:
                    print(dataset_type, 'patient', case_num)
                    for drf in drf_list:
                        print('drf number',drf)
                        low_dose_data = sitk.ReadImage(os.path.join(original_dataset_path,case,f'drf_{drf}.nii.gz'))
                        full_dose_data = sitk.ReadImage(os.path.join(original_dataset_path,case,'Full_dose.nii.gz'))
                        width, length = low_dose_data.GetSize()[0], low_dose_data.GetSize()[1]
                        print('original size',low_dose_data.GetSize())
                        cropped_low_dose_data = low_dose_data[width//2-128:width//2+128,length//2-128:length//2+128:]
                        cropped_full_dose_data = full_dose_data[width//2-128:width//2+128,length//2-128:length//2+128:]
                        print('after crop size', cropped_low_dose_data.GetSize())
                        low_dose_array = sitk.GetArrayFromImage(cropped_low_dose_data)
                        full_dose_array = sitk.GetArrayFromImage(cropped_full_dose_data)
                        print(low_dose_array.shape, full_dose_array.shape)
                        if dataset_type !='test':
                            # print(case)
                            make_array_into_patches_h5(new_dataset_path,case,drf,low_dose_array,full_dose_array)
                        else:
                            # print(case)
                            case_data_path = os.path.join(new_dataset_path,'drf'+drf+'_'+case+'.h5')
                            save_h5py_file = h5py.File(case_data_path, 'w')
                            save_h5py_file.create_dataset('raw', data=low_dose_array, compression='gzip', compression_opts=9)
                            save_h5py_file.create_dataset('label', data=full_dose_array, compression='gzip', compression_opts=9)
                            save_h5py_file.close()


                    print('finished number', i)
                    i += 1


def make_array_into_patches_with_single_h5(new_dataset_path, case_num,  drf_data_dict, patch_shape=[16, 256, 256], stride_shape=[10, 256, 256]):
    print('make_array_into_patches_h5 function')
    raw_array = drf_data_dict['raw']
    label_dict = drf_data_dict['label']
    slice_generator = SliceBuilder(raw_dataset=raw_array,label_dataset=None, weight_dataset=None,patch_shape=patch_shape,stride_shape=stride_shape,type = '3d')
    raw_slices = slice_generator._build_slices(dataset=raw_array, patch_shape=patch_shape,stride_shape=stride_shape,type = '3d')
    print(len(raw_slices))
    for idx, slice in enumerate(raw_slices):
        cropped_data= raw_array[slice]
        print(cropped_data.shape)
        slice_data_path = os.path.join(new_dataset_path,case_num+'_'+str(idx)+'.h5')
        print(slice_data_path)
        with h5py.File(slice_data_path, 'w') as save_h5py_file:
            save_h5py_file.create_dataset('raw', data=cropped_data, compression='gzip', compression_opts=9)
            # Save labels in the same h5 file
            label_group = save_h5py_file.create_group('label')
            for drf_key, drf_array in label_dict.items():
                cropped_label = drf_array[slice]
                label_group.create_dataset(drf_key, data=cropped_label, compression='gzip', compression_opts=9)
                print(cropped_label.shape)



def build_new_drf_dataset_with_single_h5(original_dataset_path, yaml_path, new_drf_path):
    i = 1
    with open(yaml_path) as file:
        document = yaml.full_load(file)
        # for docuement in documents:
        for dataset_type, case_list in document.items():
            #the line below should be removed in the end
            # if dataset_type == 'val':

                # print(dataset_type, ":", case_list)
                new_dataset_path = os.path.join(new_drf_path, dataset_type)
                # print(new_dataset_path)
                if not os.path.exists(new_dataset_path):
                    os.makedirs(new_dataset_path)
                else:
                    pass
                for case in case_list:
                    case = str(case)
                    case_num = case[-3:]
                    # if case_num < 210:
                    # print(dataset_type, 'patient', case_num)
                    drf_data_dict = {}
                    drf_data_dict['label'] = {}
                    for drf in drf_list:
                        # print('drf number',drf)
                        dose_data = sitk.ReadImage(os.path.join(original_dataset_path,case,f'{drf}.nii.gz'))
                        width, length = dose_data.GetSize()[0], dose_data.GetSize()[1]
                        # print('original size',dose_data.GetSize())
                        cropped_dose_data = dose_data[width//2-128:width//2+128,length//2-128:length//2+128:]
                        # print('after crop size', cropped_dose_data.GetSize())
                        dose_array = sitk.GetArrayFromImage(cropped_dose_data)
                        # print(dose_array.shape)
                        if drf == 'drf_100':
                            drf_data_dict['raw'] = dose_array
                        else:
                            drf_data_dict['label'][drf] = dose_array
                    if dataset_type !='test':
                        print(case)
                        make_array_into_patches_with_single_h5(new_dataset_path,case,drf_data_dict)
                    else:
                        print(case)
                        # case_data_path = os.path.join(new_dataset_path,case+'.h5')
                        #
                        # with h5py.File(case_data_path, 'w') as f:
                        #     # Create a group for label
                        #     label_group = f.create_group('label')
                        #
                        #     # Iterate over the keys and items inside the label dictionary
                        #     for key, value in drf_data_dict['label'].items():
                        #         label_group.create_dataset(key, data=value, compression='gzip', compression_opts=9)
                        #
                        #     # Save the raw data
                        #     f.create_dataset('raw', data=drf_data_dict['raw'], compression='gzip', compression_opts=9)

                    print('finished number', i)
                    i += 1




def make_array_into_patches_with_single_h5_2d(new_dataset_path, case_num,  drf_data_dict, patch_shape=[256, 256], stride_shape=[256, 256]):
    print('make_array_into_patches_h5 function')
    raw_array = drf_data_dict['raw']
    label_dict = drf_data_dict['label']
    slice_count = raw_array.shape[0]

    for slice_idx in range(slice_count):
        current_raw_slice = raw_array[slice_idx]
        print(current_raw_slice.shape)
        slice_generator = SliceBuilder(raw_dataset=current_raw_slice,label_dataset=None, weight_dataset=None, patch_shape=patch_shape, stride_shape=stride_shape,type = '2d')
        print(current_raw_slice.shape)
        raw_slices = slice_generator._build_slices(dataset=current_raw_slice, patch_shape=patch_shape, stride_shape=stride_shape,type = '2d')
        print(raw_slices)

        for idx, patch_slice in enumerate(raw_slices):
            cropped_raw = current_raw_slice[patch_slice]
            
            # Create the patch data path
            patch_data_path = os.path.join(new_dataset_path, f"{case_num}_slice_{slice_idx}_patch_{idx}.h5")
            
            # with h5py.File(patch_data_path, 'w') as save_h5py_file:
            #     save_h5py_file.create_dataset('raw', data=cropped_raw, compression='gzip', compression_opts=9)

            #     # Save labels in the same h5 file
            #     label_group = save_h5py_file.create_group('label')
            #     for drf_key, drf_array in label_dict.items():
            #         current_label_slice = drf_array[slice_idx]
            #         cropped_label = current_label_slice[patch_slice]
            #         label_group.create_dataset(drf_key, data=cropped_label, compression='gzip', compression_opts=9)



def build_new_drf_dataset_with_single_h5_2d(original_dataset_path, yaml_path, new_drf_path):
    i = 1
    with open(yaml_path) as file:
        document = yaml.full_load(file)
        # for docuement in documents:
        for dataset_type, case_list in document.items():
            #the line below should be removed in the end
            # if dataset_type == 'val':

                # print(dataset_type, ":", case_list)
                new_dataset_path = os.path.join(new_drf_path, dataset_type)
                # print(new_dataset_path)
                if not os.path.exists(new_dataset_path):
                    os.makedirs(new_dataset_path)
                else:
                    pass
                for case in case_list:
                    case = str(case)
                    case_num = case[-3:]
                    # if case_num < 210:
                    # print(dataset_type, 'patient', case_num)
                    drf_data_dict = {}
                    drf_data_dict['label'] = {}
                    for drf in drf_list:
                        # print('drf number',drf)
                        dose_data = sitk.ReadImage(os.path.join(original_dataset_path,case,f'{drf}.nii.gz'))
                        width, length = dose_data.GetSize()[0], dose_data.GetSize()[1]
                        # print('original size',dose_data.GetSize())
                        cropped_dose_data = dose_data[width//2-128:width//2+128,length//2-128:length//2+128:]
                        # print('after crop size', cropped_dose_data.GetSize())
                        dose_array = sitk.GetArrayFromImage(cropped_dose_data)
                        # print(dose_array.shape)
                        if drf == 'drf_100':
                            drf_data_dict['raw'] = dose_array
                        else:
                            drf_data_dict['label'][drf] = dose_array
                    if dataset_type !='test':
                        print(case)
                        make_array_into_patches_with_single_h5_2d(new_dataset_path,case,drf_data_dict)
                    else:
                        print(case)
                        # case_data_path = os.path.join(new_dataset_path,case+'.h5')

                        # with h5py.File(case_data_path, 'w') as f:
                        #     # Create a group for label
                        #     label_group = f.create_group('label')

                        #     # Iterate over the keys and items inside the label dictionary
                        #     for key, value in drf_data_dict['label'].items():
                        #         label_group.create_dataset(key, data=value, compression='gzip', compression_opts=9)

                        #     # Save the raw data
                        #     f.create_dataset('raw', data=drf_data_dict['raw'], compression='gzip', compression_opts=9)                      

                    print('finished number', i)
                    i += 1


drf_list = ['Full_dose','drf_2','drf_4', 'drf_10', 'drf_20', 'drf_50', 'drf_100']
data_nii_path = '/mnt/HDD3/btan8779/Data_nii/'
siemens_yaml_path = "/mnt/HDD3/btan8779/test_dataset/test_dataset_infor.yaml"
    # "/mnt/HDD3/btan8779/CapstoneData/subset_dataset_infor.yaml"
    # 'D:\\CapstoneData\\test_dataset\\test_dataset_infor.yaml'
#"/mnt/HDD3/btan8779/CapstoneData/subset_dataset_infor.yaml"
    # '/mnt/HDD3/btan8779/CapstoneData/siemens_dataset_infor.yaml'
new_dataset_path = "/mnt/HDD3/btan8779/test_dataset/test_3d_single_size_16/"
    # 'D:\\CapstoneData\\test_dataset\\test_drf_single_2d'
# "/mnt/HDD3/btan8779/CapstoneData/dataset_100/3D/3d_single/"
# data_path = "D:\\CapstoneData\\test_dataset\\test_drf_single\\train\\patient095_1.h5"
# with h5py.File(data_path, 'r') as f:
#     raw = f['raw']
#     label_dic = f['label']
#     print(raw.shape)
#     # print(label_dic.type)
#     drf_50 = label_dic['drf_50']
#     drf_20 = label_dic['drf_20']
#     drf_10 = label_dic['drf_10']
#     drf_4 = label_dic['drf_4']
#     drf_2 = label_dic['drf_2']
#     full = label_dic['Full_dose']
#     print(drf_50.shape,drf_20.shape,drf_10.shape,drf_4.shape,drf_2.shape,full.shape)
    # '/mnt/HDD3/btan8779/test_dataset/test_3d'
# test_dataset = '/mnt/HDD3/btan8779/test_dataset/Data_nii'
# test_h5 = '/mnt/HDD3/btan8779/test_dataset/test_3d'
# test_yaml = '/mnt/HDD3/btan8779/test_dataset/test_dataset_infor.yaml'
build_new_drf_dataset_with_single_h5(data_nii_path,siemens_yaml_path, new_dataset_path)
# make_3d_image_into_patches_h5_2d(new_dataset_path, case_num, drf_num, image_array, label_array)


# def make_array_into_patches_h5_2d(new_dataset_path, case_num, drf_num, raw_array, label_array, patch_shape=[256,256], stride_shape=[256,256]):
#     print('make_array_into_patches_h5 function')
#     slice_generator = SliceBuilder(raw_dataset=raw_array,label_dataset=label_array, weight_dataset=None,patch_shape=patch_shape,stride_shape=stride_shape)
#     raw_slices = slice_generator._build_slices(dataset=raw_array, patch_shape=patch_shape,stride_shape=stride_shape)
#     print(len(raw_slices))
#     for slice in raw_slices:
#         index_num = raw_slices.index(slice)
#         cropped_data= raw_array[slice]
#         cropped_label= label_array[slice]
#         print(cropped_label.shape,cropped_data.shape)
#         slice_data_path = os.path.join(new_dataset_path,case_num+'_'+'drf'+drf_num+'_'+str(index_num)+'.h5')
#         print(slice_data_path)
#         # save_h5py_file = h5py.File(slice_data_path, 'w')
#         # save_h5py_file.create_dataset('raw', data=cropped_data, compression='gzip', compression_opts=9)
#         # save_h5py_file.create_dataset('label', data=cropped_label, compression='gzip', compression_opts=9)
#         # save_h5py_file.close()


# import os
# import h5py

def make_3d_image_into_patches_h5_2d(new_dataset_path, case_num, drf_num, image_array, label_array, patch_size=256, stride=128):
    print('make_3d_image_into_patches_h5_2d function')
    num_slices = min(image_array.shape[0],label_array.shape[0])  # Number of 2D slices

    for index_num in range(num_slices):
        slice_data_path = os.path.join(new_dataset_path, f'{case_num}_drf{drf_num}_{index_num}.h5')
        # print(slice_data_path)
        if not os.path.exists(slice_data_path):

            slice_data = image_array[index_num]
            slice_label = label_array[index_num]

            height, width = slice_data.shape
            for h in range(0, height - patch_size + 1, stride):
                for w in range(0, width - patch_size + 1, stride):
                    patch_data = slice_data[h:h+patch_size, w:w+patch_size]
                    patch_label = slice_label[h:h+patch_size, w:w+patch_size]
                    # print(patch_data.shape,patch_label.shape)
                    if not os.path.exists(slice_data_path):
                        save_h5py_file = h5py.File(slice_data_path, 'w')
                        save_h5py_file.create_dataset('raw', data=patch_data, compression='gzip', compression_opts=9)
                        save_h5py_file.create_dataset('label', data=patch_label, compression='gzip', compression_opts=9)
                        save_h5py_file.close()



def build_new_drf_dataset_2d(original_dataset_path, yaml_path, new_drf_path):
    i = 1
    with open(yaml_path) as file:
        document = yaml.full_load(file)
        # for docuement in documents:
        for dataset_type, case_list in document.items():
            #the line below should be removed in the end
            if dataset_type != 'test':
                pass
            else:

                print(dataset_type, ":", case_list)
                new_dataset_path = os.path.join(new_drf_path, dataset_type)
                print(new_dataset_path)
                if not os.path.exists(new_dataset_path):
                    os.makedirs(new_dataset_path)
                else:
                    pass
                for case in case_list:
                    case = str(case)
                    case_num = case[-3:]
                    # if case_num < 210:
                    print(dataset_type, 'patient', case_num)
                    for drf in drf_list:
                            print('drf number', drf)
                        # if not os.path.exists(os.path.join(new_dataset_path, f'''{case_num}_drf{drf}_{'643'}.h5''')):
                            low_dose_data = sitk.ReadImage(os.path.join(original_dataset_path,case,f'drf_{drf}.nii.gz'))
                            full_dose_data = sitk.ReadImage(os.path.join(original_dataset_path,case,'Full_dose.nii.gz'))
                            width, length, height = low_dose_data.GetSize()[0], low_dose_data.GetSize()[1], low_dose_data.GetSize()[2]
                            print('original size',low_dose_data.GetSize())
                            cropped_low_dose_data = low_dose_data[width//2-128:width//2+128,length//2-128:length//2+128:]
                            cropped_full_dose_data = full_dose_data[width//2-128:width//2+128,length//2-128:length//2+128:]
                            print('after crop size', cropped_low_dose_data.GetSize())
                            low_dose_array = sitk.GetArrayFromImage(cropped_low_dose_data)
                            full_dose_array = sitk.GetArrayFromImage(cropped_full_dose_data)
                            print(low_dose_array.shape, full_dose_array.shape)
                            if dataset_type =='test':
                                make_3d_image_into_patches_h5_2d(new_dataset_path,case,drf,low_dose_array,full_dose_array)
                            else:
                                pass
                            # else:
                            #     case_data_path = os.path.join(new_dataset_path,'drf'+drf+'_'+case+'.h5')
                            #     save_h5py_file = h5py.File(case_data_path, 'w')
                            #     save_h5py_file.create_dataset('raw', data=low_dose_array, compression='gzip', compression_opts=9)
                            #     save_h5py_file.create_dataset('label', data=full_dose_array, compression='gzip', compression_opts=9)
                            #     save_h5py_file.close()



                    print('finished number', i)
                    i += 1


def make_3d_dataset_to_2d(drf_mix_dataset, new_dataset_path): # (in new dataset)
    phases = ['train', 'val', 'test']
    for phase in phases:
        print(phase)
        dataset_path =os.path.join(drf_mix_dataset,phase)
        all_files = os.listdir(dataset_path)
        new_phase_path = os.path.join(new_dataset_path,phase)
        print(new_phase_path)
        # if not os.path.exists(new_phase_path):
        #     os.makedirs(new_phase_path)
        # print(os.listdir(new_dataset_path))
        if phase == 'test':
            # print(dataset_path,new_phase_path)
            shutil.copytree(dataset_path,new_phase_path)
        else:
            if not os.path.exists(new_phase_path):
                os.makedirs(new_phase_path)
                # print(os.listdir(new_dataset_path))
            for file in all_files:
                # print(file)
                file_name = file.split('.')[0]
                # print(file_name)
                file_path = os.path.join(dataset_path, file)
                f1 = h5py.File(file_path,'r')
                raw_3d = np.array(f1['raw'])
                label_3d = np.array(f1['label'])
                # print(raw_3d.shape,label_3d.shape)
                for i in range(raw_3d.shape[0]):
                    raw_slice = raw_3d[i,...]
                    labe_slice = label_3d[i,...]
                    print(raw_slice.shape,labe_slice.shape)
                    case_data_path = os.path.join(new_phase_path,file_name+'_'+str(i)+'.h5')
                    print(case_data_path)
                    if os.path.exists(case_data_path):
                        pass
                    else:
                        save_h5py_file = h5py.File(case_data_path, 'w')
                        save_h5py_file.create_dataset('raw', data=raw_slice, compression='gzip', compression_opts=9)
                        save_h5py_file.create_dataset('label', data=labe_slice, compression='gzip', compression_opts=9)
                        save_h5py_file.close()


def make_denoised_dataset(old_dataset, new_dataset):
    drfs = ['train', 'val', 'test']
    for drf in drfs:
        drf_dataset = os.path.join(old_dataset, drf)
        new_drf_dataset = os.path.join(new_dataset, drf)
        if not os.path.exists(new_drf_dataset):
            os.makedirs(new_drf_dataset)
        else:
            pass
        all_files = os.listdir(drf_dataset)
        for file in all_files:
            # print(drf, file)
            file_path = os.path.join(drf_dataset, file)
            new_file_path = os.path.join(new_drf_dataset,file)
            # print(file_path)
            # print(new_file_path)
            f1 = h5py.File(file_path,'r')
            low_dose_array = np.array(f1['raw'])
            high_dose_array = np.array(f1['label'])
            f1.close()
            # print(new_file_path)
            # print(low_dose_array)
            # print(high_dose_array)
            # print(low_dose_array.shape,high_dose_array.shape)
            denoised_low_dose = np.zeros_like(low_dose_array)
            denoised_high_dose = np.zeros_like(high_dose_array)
            for i in range(low_dose_array.shape[0]):
                low_dose_slice = low_dose_array[i,...]
                high_dose_slice = high_dose_array[i,...]
                new_low_dose_slice = denoise(low_dose_slice,weight=10)
                new_high_dose_slice = denoise(high_dose_slice,weight=10)
                denoised_low_dose[i,...] = new_low_dose_slice
                denoised_high_dose[i,...] = new_high_dose_slice

            # print(denoised_low_dose)
            # print(denoised_high_dose)
                # print(new_low_dose_slice.shape)
                # print(new_high_dose_slice.shape)
            f2 = h5py.File(new_file_path,'w')
            f2.create_dataset('raw', data=denoised_low_dose, compression='gzip', compression_opts=9)
            f2.create_dataset('label', data=denoised_high_dose, compression='gzip', compression_opts=9)
            f2.close()

def make_3d_dataset_to_2d(drf_mix_dataset, new_dataset_path):
    phases = ['train', 'val', 'test']
    for phase in phases:
        print(phase)
        dataset_path =os.path.join(drf_mix_dataset,phase)
        all_files = os.listdir(dataset_path)
        new_phase_path = os.path.join(new_dataset_path,phase)
        if not os.path.exists(new_phase_path):
            os.makedirs(new_phase_path)
        if phase == 'test':
            shutil.copytree(dataset_path,new_phase_path)
        else:
            for file in all_files:
                print(file)
                file_name = file.split('.')[0]
                file_path = os.path.join(dataset_path, file)
                f1 = h5py.File(file_path,'r')
                raw_3d = np.array(f1['raw'])
                label_3d = np.array(f1['label'])
                for i in range(raw_3d.shape[0]):
                    raw_slice = raw_3d[i,...]
                    labe_slice = label_3d[i,...]
                    print(raw_slice.shape,labe_slice.shape)
                    case_data_path = os.path.join(new_phase_path,file_name+'_'+str(i)+'.h5')
                    if os.path.exists(case_data_path):
                        pass
                    else:
                        save_h5py_file = h5py.File(case_data_path, 'w')
                        save_h5py_file.create_dataset('raw', data=raw_slice, compression='gzip', compression_opts=9)
                        save_h5py_file.create_dataset('label', data=labe_slice, compression='gzip', compression_opts=9)
                        save_h5py_file.close()




# data_nii_path = '/mnt/HDD3/btan8779/Data_nii'
# siemens_yaml_path = '/mnt/HDD3/btan8779/CapstoneData/siemens_dataset_infor.yaml'
# new_dataset_path = '/mnt/HDD3/btan8779/test_dataset/test_3d'
# # test_dataset = '/mnt/HDD3/btan8779/test_dataset/Data_nii'
# # test_h5 = '/mnt/HDD3/btan8779/test_dataset/test_3d'
# # test_yaml = '/mnt/HDD3/btan8779/test_dataset/test_dataset_infor.yaml'
# build_new_drf_dataset(data_nii_path,siemens_yaml_path, new_dataset_path)
# build_new_drf_dataset(test_dataset,test_yaml,test_h5)
# build_new_drf_dataset_2d(data_nii_path,total_yaml_path, dataset_2d_path)
# make_3d_dataset_to_2d(data_nii_path,dataset_2d_path)
# make_denoised_dataset(new_dataset_path,new_dataset_denoise_path)
# build_new_drf_dataset_2d(data_nii_path,siemens_yaml_path,siemens_2d_path)