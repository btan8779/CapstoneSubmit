from locale import normalize
import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
import SimpleITK as sitk

drf_name = 'drf2_patient400_predictions.h5'#'drf100_229_predictions.h5'"D:\CapstoneData\test_3d\ouput\drf2_patient438_predictions.h5"
dataset_path = '/mnt/HDD3/btan8779/CapstoneData/3d_subset/test/drf2_patient400.h5'
# '/media/mingjian/NewVolume/DATA_Jane/low_to_high_PET/Dmix_dataset/test/drf100_229.h5'
# output_AEGAN = '/media/mingjian/NewVolume/DATA_Jane/low_to_high_PET/Dmix_dataset/output/3DGAN_ARNet_uExplorer_ForDRF100'
# output_ARGAN = '/media/mingjian/NewVolume/DATA_Jane/low_to_high_PET/Dmix_dataset/output/3DUNet_ARNet_uExplorer'
# output_UNet = '/media/mingjian/NewVolume/DATA_Jane/low_to_high_PET/Dmix_dataset/output/UNet3D_uExplorer'
output = '/mnt/HDD3/btan8779/TEST_OUTPUT/10-10/3D/without_spectrum/'

def MaxMinNormalizer(data):
    data= np.clip(data,0.0,8.0)
    data_max = np.max(data)
    data_min = np.min(data)
    data_normalize = (data-data_min)/(data_max-data_min)
    return data_normalize

# output_AEGAN_path = os.path.join(output_AEGAN, drf_name)
# output_ARGAN_path = os.path.join(output_ARGAN, drf_name)
# output_UNet_path = os.path.join(output_UNet, drf_name)
output_path = os.path.join(output, drf_name)
# 580 for brain 320 for liver, 350 for lung, 50 
# 550 for brain, 470 for liver, 
# slice_num = 580
slice_num = 140
datafile = h5py.File(dataset_path,'r')
normalized_low = MaxMinNormalizer(np.array(datafile['raw']))
normalized_high = MaxMinNormalizer(np.array(datafile['label']))
low_slice = 1.0 - normalized_low[:,slice_num,...]
high_slice = 1.0 - normalized_high[:,slice_num,...]
# low_dose_array = np.array(datafile['raw'])[:,slice_num,...]
# high_dose_array = np.array(datafile['label'])[:,slice_num,...]
datafile.close()
output_file = h5py.File(output_path,'r')
# output_AEGAN_file = h5py.File(output_AEGAN_path,'r')
# output_ARNet_file = h5py.File(output_ARGAN_path,'r')
# output_UNet_file = h5py.File(output_UNet_path, 'r')
normalized = MaxMinNormalizer(np.array(output_file['predictions']))
# normalized_AEGAN = MaxMinNormalizer(np.array(output_AEGAN_file['predictions']))
# normalized_ARNet = MaxMinNormalizer(np.array(output_ARNet_file['predictions']))
# normalized_UNet = MaxMinNormalizer(np.array(output_UNet_file['predictions']))

slice = 1.0 - normalized[0,...][:,slice_num,...]
# AEGAN_slice = 1.0 - normalized_AEGAN[0,...][:,slice_num,...]
# ARNet_slice = 1.0 - normalized_ARNet[0,...][:,slice_num,...]
# UNet_slice = 1.0 - normalized_UNet[0,...][:,slice_num,...]
# output_AEGAN_array = np.array(output_AEGAN_file['predictions'])[0,...][:,slice_num,...]
# output_ARNet_array = np.array(output_ARNet_file['predictions'])[0,...][:,slice_num,...]
output_file.close()
# output_AEGAN_file.close()
# output_ARNet_file.close()
# output_UNet_file.close()

# high_dose_array = MaxMinNormalizer(high_dose_array)
# normalized_high = 1.0 - high_dose_array
# low_dose_array = MaxMinNormalizer(low_dose_array)
# normalized_low = 1.0 - low_dose_array
# output_AEGAN_array = MaxMinNormalizer(output_AEGAN_array)
# normalized_AEGAN = 1.0 - output_AEGAN_array
# output_ARNet_array = MaxMinNormalizer(output_ARNet_array)
# normalized_ARNet = 1.0 - output_ARNet_array

plt.figure()
plt.subplot(1,5,1)
plt.imshow(low_slice,cmap='jet_r')
plt.subplot(1,5,2)
plt.imshow(slice,cmap='jet_r')
# plt.subplot(1,5,3)
# plt.imshow(ARNet_slice,cmap='gray')
# plt.subplot(1,5,4)
# plt.imshow(AEGAN_slice,cmap='gray')
plt.subplot(1,5,3)
plt.imshow(high_slice,cmap='jet_r')
plt.show()


