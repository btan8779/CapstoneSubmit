from numpy import inf, nan
import os
import yaml
from sklearn.metrics import label_ranking_loss
import h5py
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, mean_squared_error, structural_similarity, normalized_root_mse
# from utils import MaxMinNormalizer
from statistics import mean

def MaxMinNormalizer(data):
    data_max = np.max(data)
    data_min = np.min(data)
    data_normalize = (data-data_min)/(data_max-data_min)
    return data_normalize


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

def compute_nrmse(real, pred):
    mse = np.mean(np.square(real - pred))
    nrmse = np.sqrt(mse) / (np.max(real)-np.min(real))
    return nrmse

def compute_mse(real, pred):
    mse = np.mean(np.square(real-pred))
    return mse


def compute_psnr(real, pred):
    PIXEL_MAX = np.max(real)
    psnr = 20 * np.log10(PIXEL_MAX / np.sqrt(np.mean(np.square(real - pred))))
    return psnr


def compute_ssim(real, pred):
    ssim = structural_similarity(real/ float(np.max(real)), pred/ float(np.max(pred)))
    return ssim

output_path = '/mnt/HDD3/btan8779/TEST_OUTPUT/10-5/3D/try' #'D:/CapstoneData/test_3d/ouput/'
    # '/mnt/HDD3/btan8779/TEST_OUTPUT/10-8/2D/all_drf/best_train_no_Specturm/'
raw_data_path = '/mnt/HDD3/btan8779/CapstoneData/3d_subset/test/'
# tracer_yaml_path = '/media/mingjian/NewVolume/DATA_Jane/low_to_high_PET/tracer_infor.yaml'
# abnormal_patients = ['8','76']
drf_list = ['drf100','drf50','drf10','drf20','drf4','drf2']
patients = os.listdir(output_path)
final_PSNR_pred, final_PSNR_raw = [], []
final_SSIM_pred, final_SSIM_raw = [], []
final_MSE_pred, final_MSE_raw = [], []
# with open(tracer_yaml_path) as file:
#     tracer_dict = yaml.full_load(file)
# print(tracer_dict)
for patient in patients:
    PSNR_pred, PSNR_raw = [],[]
    SSIM_pred, SSIM_raw = [], []
    MSE_pred, MSE_raw = [], []
    patient_title = patient.split('.')[0]
    print(patient_title)
    # print(patient_title)
    patient_drf = patient_title.split('_')[0]
    patient_num = patient_title.split('_')[1]
    # if patient_num in tracer_dict.keys():
    #     tracer = tracer_dict[patient_num]
    # else:
    #     tracer = None
    patient_name = patient_drf+'_'+patient_num
    # patient_number = int(patient_num)
    if patient_drf in drf_list:
        print(patient_num)
        patient_path = os.path.join(raw_data_path,patient_name+'.h5')
        prediction_path = os.path.join(output_path,patient)
        # pred_nii_save_path = os.path.join(output_path, patient_name + '_predictions.h5')
        pred_nii_save_path = os.path.join(output_path,patient_name+'_predictions.nii.gz')
        f1 = h5py.File(patient_path,'r') 
        f2 = h5py.File(prediction_path,'r') 
        raw = np.array(f1['raw'])
        label = np.array(f1['label'])
        prediction = np.array(f2['predictions'])[0,...]
        data_range = label.max() - label.min()
        print(raw.shape)
        print(prediction.shape)
        # denoised_array = np.zeros_like(raw)
        # for i in range(raw.shape[0]):
        #     low_dose_slice = raw[i,...]
        #     new_slice = denoise(low_dose_slice,weight=0.1)
        #     denoised_array[i,...] = new_slice
        # prediction = denoised_array
        #
        # denoised_data = sitk.GetImageFromArray(prediction)
        # wriiter = sitk.ImageFileWriter()
        # wriiter.SetFileName(pred_nii_save_path)
        # wriiter.Execute(denoised_data)
        # print(raw.shape, label.shape, prediction.shape)
        # f1.close()
        # f2.close()
        # pred_path = os.path.join(patient_path,patient+'.h5_raw.nii.gz')
        # label_path = os.path.join(patient_path,patient+'.h5_label.nii.gz')
        # pred_data, label_data = sitk.ReadImage(pred_path), sitk.ReadImage(label_path)
        # pred_array, label_array = sitk.GetArrayFromImage(pred_data), sitk.GetArrayFromImage(label_data)
        PSNR_pred = compute_psnr(label, prediction)
        NRMSE_pred = compute_nrmse(label,prediction)
        SSIM_pred = compute_ssim(label, prediction)
        PSNR_raw = compute_psnr(label, raw)
        NRMSE_raw = compute_nrmse(label, raw)
        SSIM_raw = compute_ssim(label, raw)

        # for i in range(raw.shape[0]):
        #     # print(i)
        #     raw_slice = raw[i,...]
        #     label_slice = label[i,...]
        #     prediction_slice = prediction[i,...]
        #     data_range = label_slice.max() - label_slice.min()
        #     # normalized_pred, normalized_raw,normalized_label = MaxMinNormalizer(prediction), MaxMinNormalizer(raw),MaxMinNormalizer(label)
        #     psnr_pred = peak_signal_noise_ratio(label_slice, prediction_slice, data_range=data_range)
        #     ssim_pred = structural_similarity(label_slice,prediction_slice)
        #     mse_pred = normalized_root_mse(label_slice, prediction_slice)
        #     psnr_raw = peak_signal_noise_ratio(label_slice, raw_slice, data_range=data_range)
        #     ssim_raw = structural_similarity(label_slice,raw_slice)
        #     mse_raw = normalized_root_mse(label_slice, raw_slice)
        #     if psnr_pred != -inf and psnr_pred != nan and psnr_raw!=-inf and psnr_raw != nan and psnr_raw>0:
        #         PSNR_raw.append(psnr_raw)
        #         SSIM_raw.append(ssim_raw)
        #         MSE_raw.append(mse_raw)
        #         PSNR_pred.append(psnr_pred)
        #         SSIM_pred.append(ssim_pred)
        #         MSE_pred.append(mse_pred)
        print('prediction PSNR is :', PSNR_pred)
        print('prediction SSIM is :', SSIM_pred)
        print('prediction MSE is :', NRMSE_pred)
        print('raw PSNR is :', PSNR_raw)
        print('raw SSIM is :', SSIM_raw)
        print('raw MSE is :', NRMSE_raw)
        final_PSNR_pred.append(PSNR_pred)
        final_SSIM_pred.append(SSIM_pred)
        final_MSE_pred.append(NRMSE_pred)
        final_PSNR_raw.append(PSNR_raw)
        final_SSIM_raw.append(SSIM_raw)
        final_MSE_raw.append(NRMSE_raw)
       
        # mean_psnr_pred = mean(PSNR_pred)
        # mean_ssim_pred = mean(SSIM_pred)
        # mean_mse_pred = mean(MSE_pred)
        # print("prediction mean PSNR is :", mean_psnr_pred)
        # print("prediction mean SSIM is :", mean_ssim_pred)
        # print("prediction mean MSE is :", mean_mse_pred)
        # mean_psnr_raw = mean(PSNR_raw)
        # mean_ssim_raw = mean(SSIM_raw)
        # mean_mse_raw = mean(MSE_raw)
        # print("raw mean PSNR is :", mean_psnr_raw)
        # print("raw mean SSIM is :", mean_ssim_raw)
        # print("raw mean MSE is :", mean_mse_raw)
        # final_PSNR_pred.append(mean_psnr_pred)
        # final_SSIM_pred.append(mean_ssim_pred)
        # final_MSE_pred.append(mean_mse_pred)
        # final_PSNR_raw.append(mean_psnr_raw)
        # final_SSIM_raw.append(mean_ssim_raw)
        # final_MSE_raw.append(mean_mse_raw)

# x = range(0, len(final_MSE_pred))
# y = final_MSE_pred
# plt.plot(x, y)
# plt.xlabel("Patients")
# plt.ylabel("NRMSE")
# plt.title("DRF 10 siemens")
# plt.show()
print('totoal patients : ', len(final_PSNR_pred))
psnr_pred_final = mean(final_PSNR_pred)
ssim_pred_final = mean(final_SSIM_pred)
mse_pred_final = mean(final_MSE_pred)
print("prediction final PSNR is :", psnr_pred_final)
print("prediction final SSIM is :", ssim_pred_final)
print("prediction final NRMSE is :", mse_pred_final)
psnr_raw_final = mean(final_PSNR_raw)
ssim_raw_final = mean(final_SSIM_raw)
mse_raw_final = mean(final_MSE_raw)
print("raw final PSNR is :", psnr_raw_final)
print("raw final SSIM is :", ssim_raw_final)
print("raw final NRMSE is :", mse_raw_final)