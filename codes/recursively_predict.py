import numpy as np
import torch
import os
from model import U_Net
from utility import non_land_mask


def crop(sic_array, sic_time, row_idx=0, col_idx=0, img_size=32, device='cpu'):
    """
    crop the sic data (448*304) to obtain a small area of interest
    :param device: the device on which to train our networks, defaults CPU
    :param sic_time: the time flag of the sic array
    :param sic_array: the ndarray that contains the SIC data, with the dimension of S*W*H
    :param row_idx: row index
    :param col_idx: col index
    :param img_size: the image size of our small area, defaulting 64*64
    :return: time, cropped_sic
    """
    time = sic_time
    sic = sic_array

    # Since the original image size of 448*304 is not a multiple of 32,
    # we have to reshape the data to 448*320 by filling the zeros
    filled_zeros = np.zeros((sic.shape[0], sic.shape[1], 320 - sic.shape[2]))
    filled_sic = np.append(sic, filled_zeros, axis=2)  # s*448*320

    ri, ci = row_idx * img_size, col_idx * img_size
    rj, cj = (row_idx + 1) * img_size, (col_idx + 1) * img_size
    if rj > filled_sic.shape[1] or cj > filled_sic.shape[2]:
        raise IndexError
    else:
        cropped_sic = filled_sic[:, ri:rj, ci:cj]

    # adding the channel dimension
    cropped_sic = cropped_sic[:, np.newaxis, :, :]
    # convert to tensor
    input_sic = torch.from_numpy(cropped_sic).to(dtype=torch.float32, device=device)

    return input_sic, time


def recur(sic_array, sic_time, model_dir='resulting_melting/None_model/', img_size=64, recur_flag=1):
    """
    perform the recursive prediction
    :param recur_flag: the recursive step
    :param sic_array: the ndarray that contains the SIC data, with the dimension of S*W*H
    :param sic_time: the input time
    :param img_size: the image size
    :param model_dir: the directory that contains all trained models
    :return: the prediction, the target, and the time
    """

    output_time = sic_time
    if recur_flag == 1:
        time_list = [str(time[0])[:7] for time in sic_time]
        output_time = np.array(time_list, dtype='datetime64') + 1

    if recur_flag == 2:
        output_time = sic_time + 1

    complete_prediction = np.zeros((len(sic_time), 448, 320))

    model_files = os.listdir(model_dir)  # a list that contains all trained model

    H, W, s = 0, 0, img_size
    load_model = U_Net()

    for i in range(len(model_files)):
        if model_files[i].endswith('.pt'):
            if len(model_files[i]) == 11:
                H = int(model_files[i][6])
                W = int(model_files[i][7])
            else:
                H = int(model_files[i][6:8])
                W = int(model_files[i][8])

            hi, wi = H * s, W * s
            hj, wj = (H + 1) * s, (W + 1) * s

            load_model.load_state_dict(torch.load(model_dir + model_files[i]))

            testX, _ = crop(sic_array=sic_array, sic_time=sic_time, row_idx=H, col_idx=W, img_size=s)

            prediction = load_model(testX)
            prediction = prediction.detach().numpy().squeeze()

            # complete_testY[:, hi:hj, wi: wj] = testY.detach().numpy().squeeze()
            complete_prediction[:, hi:hj, wi: wj] = prediction

    # Since the original image size of 448*304,  we have to reshape the data
    return complete_prediction[:, :, :304], output_time


def melting_recur():
    result = 'result_melting/none.npz'
    res = np.load(result, allow_pickle=True)
    target = res['target']

    non_land = non_land_mask()

    pred1 = res['model']  # ndarray 26*448*304
    pred1 = pred1 * non_land
    time1 = res['time']

    pred2, time2 = recur(sic_array=pred1, sic_time=time1, model_dir='result_melting/None_model/', recur_flag=1)
    pred2 = pred2 * non_land
    # 2018-05~2018-09 2019-05~2019-09 2020-05~2020-09 2021-05~2021-09
    pred_index2 = [2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24]
    new_time2 = time2[pred_index2]
    new_pred2 = pred2[pred_index2]
    target_index2 = [index + 1 for index in pred_index2]
    target2 = target[target_index2]

    pred3, time3 = recur(sic_array=pred2, sic_time=time2, model_dir='result_melting/None_model/', recur_flag=2)
    pred3 = pred3 * non_land
    # 2018-06~2018-09 2019-06~2019-09 2020-06~2020-09 2021-06~2021-09
    pred_index3 = [2, 3, 4, 5, 8, 9, 10, 11, 14, 15, 16, 17, 20, 21, 22, 23]
    new_time3 = time3[pred_index3]
    new_pred3 = pred3[pred_index3]
    target_index3 = [index + 2 for index in pred_index3]
    target3 = target[target_index3]

    diff2 = np.abs(target2 - new_pred2)
    diff_sum2 = np.sum(diff2, axis=(1, 2))
    total = np.sum(non_land)
    MAE2 = diff_sum2 / total

    diff3 = np.abs(target3 - new_pred3)
    diff_sum3 = np.sum(diff3, axis=(1, 2))
    MAE3 = diff_sum3 / total
    return new_time2, MAE2, new_time3, MAE3


def icing_recur():
    result = 'result_icing/none.npz'
    res = np.load(result, allow_pickle=True)
    target = res['target']

    non_land = non_land_mask()

    pred1 = res['model']  # ndarray 26*448*304
    pred1 = pred1 * non_land
    time1 = res['time']

    pred2, time2 = recur(sic_array=pred1, sic_time=time1, model_dir='result_icing/None_model/', recur_flag=1)
    pred2 = pred2 * non_land
    # 2018-01~03 2018-11~12 2019-01~03 2019-11~12 2020-01~03 2020-11~12 2021-01~03 2021-11~12
    pred_index2 = [1, 2, 3, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 23, 24]
    new_time2 = time2[pred_index2]
    new_pred2 = pred2[pred_index2]
    target_index2 = [index + 1 for index in pred_index2]
    target2 = target[target_index2]

    pred3, time3 = recur(sic_array=pred2, sic_time=time2, model_dir='result_melting/None_model/', recur_flag=2)
    pred3 = pred3 * non_land
    # 2018-01~03 2018-12 2019-01~03 2019-12 2020-01~03 2020-12 2021-01~03 2021-12
    pred_index3 = [0, 1, 2, 5, 6, 7, 8, 11, 12, 13, 14, 17, 18, 19, 20, 23]
    new_time3 = time3[pred_index3]
    new_pred3 = pred3[pred_index3]
    target_index3 = [index + 2 for index in pred_index3]
    target3 = target[target_index3]

    diff2 = np.abs(target2 - new_pred2)
    diff_sum2 = np.sum(diff2, axis=(1, 2))
    total = np.sum(non_land)
    MAE2 = diff_sum2 / total

    diff3 = np.abs(target3 - new_pred3)
    diff_sum3 = np.sum(diff3, axis=(1, 2))
    MAE3 = diff_sum3 / total
    return new_time2, MAE2, new_time3, MAE3


t2, m2, t3, m3 = icing_recur()
