from locale import normalize
import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
import SimpleITK as sitk
import yaml


def MaxMinNormalizer(data):
    # print(data)
    data = np.clip(data, 0.0, 8.0)
    data_max = np.max(data)
    data_min = np.min(data)
    data_normalize = (data - data_min) / (data_max - data_min)
    return data_normalize

def visualiztaion(patient_num):
    drf_name = f'drf100_patient{patient_num}_predictions.h5'#'drf100_229_predictions.h5'"D:\CapstoneData\test_3d\ouput\drf2_patient438_predictions.h5"
    prediction_name = f'patient{patient_num}_predictions.h5'
    dataset_path = f'/mnt/HDD3/btan8779/CapstoneData/dataset_100/3D/3d_single/test/patient{patient_num}.h5'
        # '/mnt/HDD3/btan8779/CapstoneData/3d_subset/test/drf100_patient416.h5'
    # '/media/mingjian/NewVolume/DATA_Jane/low_to_high_PET/Dmix_dataset/test/drf100_229.h5'
    # output_AEGAN = '/media/mingjian/NewVolume/DATA_Jane/low_to_high_PET/Dmix_dataset/output/3DGAN_ARNet_uExplorer_ForDRF100'
    output_seperate = '/mnt/HDD3/btan8779/TEST_OUTPUT/10-18/3D/define_G_sequential_seperate/'
    output_totalback = '/mnt/HDD3/btan8779/TEST_OUTPUT/10-18/3D/define_G_sequential_total_back/'
    output = '/mnt/HDD3/btan8779/TEST_OUTPUT/10-18/3D/define_G/'

    # output_AEGAN_path = os.path.join(output_AEGAN, drf_name)
    output_seperate_path = os.path.join(output_seperate, prediction_name)
    output_totalback_path = os.path.join(output_totalback, prediction_name)
    output_path = os.path.join(output, drf_name)
    # 580 for brain 320 for liver, 350 for lung, 50
    # 550 for brain, 470 for liver,
    # slice_num = 580
    slice_num = 140
    datafile = h5py.File(dataset_path,'r')

    normalized_low = MaxMinNormalizer(np.array(datafile['raw']))
    normalized_high = MaxMinNormalizer(np.array(datafile['label']['Full_dose']))
    normalized_2 = MaxMinNormalizer(np.array(datafile['label']['drf_10']))
    low_slice = 1.0 - normalized_low[:,slice_num,...]
    high_slice = 1.0 - normalized_high[:,slice_num,...]
    slice_2 = 1.0 - normalized_2[:,slice_num,...]
    # low_dose_array = np.array(datafile['raw'])[:,slice_num,...]
    # high_dose_array = np.array(datafile['label'])[:,slice_num,...]
    datafile.close()
    output_file = h5py.File(output_path,'r')
    output_seperate_file = h5py.File(output_seperate_path,'r')
    output_totalback_file = h5py.File(output_totalback_path,'r')
    # output_UNet_file = h5py.File(output_UNet_path, 'r')
    normalized = MaxMinNormalizer(np.array(output_file['predictions']))
    normalized_seperate = MaxMinNormalizer(np.array(output_seperate_file['predictions']))
    normalized_totalback = MaxMinNormalizer(np.array(output_totalback_file['predictions']))
    # normalized_UNet = MaxMinNormalizer(np.array(output_UNet_file['predictions']))

    slice = 1.0 - normalized[0,...][:,slice_num,...]
    seperate_slice = 1.0 - normalized_seperate[0,...][:,slice_num,...]
    totalback_slice = 1.0 - normalized_totalback[0,...][:,slice_num,...]
    # UNet_slice = 1.0 - normalized_UNet[0,...][:,slice_num,...]
    # output_AEGAN_array = np.array(output_AEGAN_file['predictions'])[0,...][:,slice_num,...]
    # output_ARNet_array = np.array(output_ARNet_file['predictions'])[0,...][:,slice_num,...]
    output_file.close()
    output_seperate_file.close()
    output_totalback_file.close()
    # output_UNet_file.close()

    # high_dose_array = MaxMinNormalizer(high_dose_array)
    # normalized_high = 1.0 - high_dose_array
    # low_dose_array = MaxMinNormalizer(low_dose_array)
    # normalized_low = 1.0 - low_dose_array
    # output_AEGAN_array = MaxMinNormalizer(output_AEGAN_array)
    # normalized_AEGAN = 1.0 - output_AEGAN_array
    # output_ARNet_array = MaxMinNormalizer(output_ARNet_array)
    # normalized_ARNet = 1.0 - output_ARNet_array

    # Assuming you have some image data for low_slice, slice, separate_slice, totalback_slice, high_slice
    # Rotate and mirror the images
    # low_slice = np.flipud(np.rot90(low_slice, 2))
    # slice = np.flipud(np.rot90(slice, 2))
    # seperate_slice = np.flipud(np.rot90(seperate_slice, 2))
    # totalback_slice = np.flipud(np.rot90(totalback_slice, 2))
    # high_slice = np.flipud(np.rot90(high_slice, 2))
    low_slice = low_slice[::-1]
    slice = slice[::-1]
    seperate_slice = seperate_slice[::-1]
    totalback_slice = totalback_slice[::-1]
    high_slice = high_slice[::-1]

    # Create a figure with wider spacing
    plt.figure(figsize=(20, 5))

    # Plot each image with titles
    plt.subplot(1, 5, 1)
    plt.imshow(low_slice, cmap='jet_r')
    plt.title('Low')
    plt.subplot(1, 5, 2)
    plt.imshow(slice, cmap='jet_r')
    plt.title('Unet32')
    plt.subplot(1, 5, 3)
    plt.imshow(seperate_slice, cmap='jet_r')
    plt.title('Seperate Back')
    plt.subplot(1, 5, 4)
    plt.imshow(totalback_slice, cmap='jet_r')
    plt.title('Total Back')
    # plt.subplot(1, 6, 5)
    # plt.imshow(slice_2, cmap='jet_r')
    # plt.title('drf2')
    plt.subplot(1, 5, 5)
    plt.imshow(high_slice, cmap='jet_r')
    plt.title('High Slice')

    # Add a total title
    plt.suptitle(f'patient_{patient_num}', fontsize=16)

    # plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # plt.figure()
    # plt.subplot(1,5,1)
    # plt.imshow(low_slice,cmap='jet_r')
    # plt.subplot(1,5,2)
    # plt.imshow(slice,cmap='jet_r')
    # plt.subplot(1,5,3)
    # plt.imshow(seperate_slice,cmap='jet_r')
    # plt.subplot(1,5,4)
    # plt.imshow(totalback_slice,cmap='jet_r')
    # plt.subplot(1,5,5)
    # plt.imshow(high_slice,cmap='jet_r')
    # plt.show()





def visualiztaion_example(patient_num,drf_num):
    drf_name = f'{drf_num}_patient{patient_num}_predictions.h5'#'drf100_229_predictions.h5'"D:\CapstoneData\test_3d\ouput\drf2_patient438_predictions.h5"
    prediction_name = f'patient{patient_num}_predictions.h5'
    dataset_path = f'/mnt/HDD3/btan8779/CapstoneData/dataset_100/3D/3d_single/test/patient{patient_num}.h5'
        # '/mnt/HDD3/btan8779/CapstoneData/3d_subset/test/drf100_patient416.h5'
    # '/media/mingjian/NewVolume/DATA_Jane/low_to_high_PET/Dmix_dataset/test/drf100_229.h5'
    # output_AEGAN = '/media/mingjian/NewVolume/DATA_Jane/low_to_high_PET/Dmix_dataset/output/3DGAN_ARNet_uExplorer_ForDRF100'
    output_seperate = '/mnt/HDD3/btan8779/TEST_OUTPUT/final/2d_without_spectrum/'
    output_totalback = '/mnt/HDD3/btan8779/TEST_OUTPUT/final/2d_with_spectrum_new/'
    output = '/mnt/HDD3/btan8779/TEST_OUTPUT/10-18/3D/define_G/'
    output_step = '/mnt/HDD3/btan8779/TEST_OUTPUT/10-24/3D/sequential_seperate_100-20-4/'
    output_100 = '/mnt/HDD3/btan8779/TEST_OUTPUT/final/3D_100-full/'

    # output_AEGAN_path = os.path.join(output_AEGAN, drf_name)
    output_seperate_path = os.path.join(output_seperate, drf_name)
    output_totalback_path = os.path.join(output_totalback, drf_name)
    output_path = os.path.join(output, drf_name)
    # output_step_path = os.path.join(output_step, prediction_name)
    # output_100 = os.path.join(output_100, drf_name)
    # 580 for brain 320 for liver, 350 for lung, 50
    # 550 for brain, 470 for liver,
    # slice_num = 580
    slice_num = 140
    datafile = h5py.File(dataset_path,'r')
    # print(datafile['label'])
    if drf_num == 'drf100':
        normalized_low = MaxMinNormalizer(np.array(datafile['raw']))
    elif drf_num == 'drf50':
        normalized_low = MaxMinNormalizer(np.array(datafile['label']['drf_50']))
    elif drf_num == 'drf20':
        normalized_low = MaxMinNormalizer(np.array(datafile['label']['drf_20']))
    elif drf_num == 'drf10':
        normalized_low = MaxMinNormalizer(np.array(datafile['label']['drf_10']))
    elif drf_num == 'drf4':
        normalized_low = MaxMinNormalizer(np.array(datafile['label']['drf_4']))
    elif drf_num == 'drf2':
        normalized_low = MaxMinNormalizer(np.array(datafile['label']['drf_2']))
    # normalized_low = MaxMinNormalizer(np.array(datafile['raw']))
    normalized_high = MaxMinNormalizer(np.array(datafile['label']['Full_dose']))
    # normalized_50 = MaxMinNormalizer(np.array(datafile['label']['drf_50']))
    # normalized_20 = MaxMinNormalizer(np.array(datafile['label']['drf_20']))
    # normalized_10 = MaxMinNormalizer(np.array(datafile['label']['drf_10']))
    # normalized_4 = MaxMinNormalizer(np.array(datafile['label']['drf_4']))
    # normalized_2 = MaxMinNormalizer(np.array(datafile['label']['drf_2']))
    low_slice = 1.0 - normalized_low[:,slice_num,...]
    high_slice = 1.0 - normalized_high[:,slice_num,...]
    # slice_50 = 1.0 - normalized_50[:, slice_num, ...]
    # slice_20 = 1.0 - normalized_20[:, slice_num, ...]
    # slice_10 = 1.0 - normalized_10[:, slice_num, ...]
    # slice_4 = 1.0 - normalized_4[:, slice_num, ...]
    # slice_2 = 1.0 - normalized_2[:, slice_num, ...]
    # low_dose_array = np.array(datafile['raw'])[:,slice_num,...]
    # high_dose_array = np.array(datafile['label'])[:,slice_num,...]
    datafile.close()
    output_file = h5py.File(output_path,'r')
    output_seperate_file = h5py.File(output_seperate_path,'r')
    output_totalback_file = h5py.File(output_totalback_path,'r')
    # output_step_file = h5py.File(output_step_path, 'r')
    # output_100_file = h5py.File(output_100, 'r')
    # output_UNet_file = h5py.File(output_UNet_path, 'r')
    normalized = MaxMinNormalizer(np.array(output_file['predictions']))
    normalized_seperate = MaxMinNormalizer(np.array(output_seperate_file['predictions']))
    normalized_totalback = MaxMinNormalizer(np.array(output_totalback_file['predictions']))
    # normalized_step = MaxMinNormalizer(np.array(output_step_file['predictions']))
    # normalized_100 = MaxMinNormalizer(np.array(output_100_file['predictions']))

    slice = 1.0 - normalized[0,...][:,slice_num,...]
    seperate_slice = 1.0 - normalized_seperate[0,...][:,slice_num,...]
    totalback_slice = 1.0 - normalized_totalback[0,...][:,slice_num,...]
    # slice_step = 1.0 - normalized_step[0, ...][:, slice_num, ...]
    # slice_100= 1.0 - normalized_100[0,...][:,slice_num,...]
    # # output_AEGAN_array = np.array(output_AEGAN_file['predictions'])[0,...][:,slice_num,...]
    # output_ARNet_array = np.array(output_ARNet_file['predictions'])[0,...][:,slice_num,...]
    output_file.close()
    output_seperate_file.close()
    output_totalback_file.close()
    # output_step_file.close()
    # output_100_file.close()
    # output_UNet_file.close()

    # high_dose_array = MaxMinNormalizer(high_dose_array)
    # normalized_high = 1.0 - high_dose_array
    # low_dose_array = MaxMinNormalizer(low_dose_array)
    # normalized_low = 1.0 - low_dose_array
    # output_AEGAN_array = MaxMinNormalizer(output_AEGAN_array)
    # normalized_AEGAN = 1.0 - output_AEGAN_array
    # output_ARNet_array = MaxMinNormalizer(output_ARNet_array)
    # normalized_ARNet = 1.0 - output_ARNet_array

    # Assuming you have some image data for low_slice, slice, separate_slice, totalback_slice, high_slice
    # Rotate and mirror the images
    # low_slice = np.flipud(np.rot90(low_slice, 2))
    # slice = np.flipud(np.rot90(slice, 2))
    # seperate_slice = np.flipud(np.rot90(seperate_slice, 2))
    # totalback_slice = np.flipud(np.rot90(totalback_slice, 2))
    # high_slice = np.flipud(np.rot90(high_slice, 2))
    low_slice = low_slice[::-1]
    # slice_50 = slice_50[::-1]
    # slice_10 = slice_10[::-1]
    # slice_20 = slice_20[::-1]
    # slice_2 = slice_2[::-1]
    slice = slice[::-1]
    high_slice = high_slice[::-1]
    seperate_slice = seperate_slice[::-1]
    totalback_slice = totalback_slice[::-1]
    # slice_step = slice_step[::-1]
    # slice_100 = slice_100[::-1]

    # Create a figure with wider spacing
    plt.figure(figsize=(18, 7))
    # plt.figure(figsize=(15, 7))

    # Plot each image with titles
    plt.subplot(1, 4, 1)
    plt.imshow(low_slice, cmap='jet_r')
    plt.title('Low dose')
    plt.subplot(1, 4, 2)
    plt.imshow(seperate_slice, cmap='jet_r')
    plt.title('without spectral constraint')
    plt.subplot(1, 4, 3)
    plt.imshow(totalback_slice, cmap='jet_r')
    plt.title('with soectral constraint')
    # plt.subplot(1, 5, 4)
    # plt.imshow(slice, cmap='jet_r')
    # plt.title('Pixel-Net')
    # plt.subplot(1, 6, 5)
    # plt.imshow(slice, cmap='jet_r')
    # plt.title('Pixel-Net)')
    # plt.subplot(1, 7, 6)
    # plt.imshow(slice_2, cmap='jet_r')
    # plt.title('Low dose(DRF2)')
    plt.subplot(1, 4, 4)
    plt.imshow(high_slice, cmap='jet_r')
    plt.title('Full dose')

    # Add a total title
    plt.suptitle(f'Visualization of {drf_num}_patient{patient_num}', fontsize=16)

    # plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # plt.figure()
    # plt.subplot(1,5,1)
    # plt.imshow(low_slice,cmap='jet_r')
    # plt.subplot(1,5,2)
    # plt.imshow(slice,cmap='jet_r')
    # plt.subplot(1,5,3)
    # plt.imshow(seperate_slice,cmap='jet_r')
    # plt.subplot(1,5,4)
    # plt.imshow(totalback_slice,cmap='jet_r')
    # plt.subplot(1,5,5)
    # plt.imshow(high_slice,cmap='jet_r')
    # plt.show()


yaml_path = "/mnt/HDD3/btan8779/CapstoneData/subset_dataset_infor.yaml"

patient_num_list = []
with open(yaml_path) as file:
    document = yaml.full_load(file)
    for dataset_type, case_list in document.items():
        if dataset_type == 'test':
            for patient in case_list:
                patient_num = patient[-3:]
                patient_num_list.append(patient_num)
print(patient_num_list)
drf_list = ['drf100','drf50','drf20','drf10','drf4','drf2']
for patient_num in patient_num_list[0:5]:
    for drf in drf_list:
        print(drf,patient_num)
        visualiztaion_example(patient_num,drf)