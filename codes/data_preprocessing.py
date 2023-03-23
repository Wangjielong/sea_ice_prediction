"""
data preprocessing of sea ice concentration (sic)
"""
import functools
import netCDF4 as nC
import numpy as np
import torch
import random
import math
from torch.nn import functional as F
from torch.utils.data import Dataset
import os
from utility import non_land_mask

FILEPATH = 'data/SICcdr_seaice_conc_monthly.nc'
FILEPATH2 = 'data/monthly/'


def read_latest_sic(filepath=FILEPATH2):
    """
    read the latest sic files
    :param filepath: the directory that contains the latest files
    :return: time: ndarray
             sic: ndarray
    """
    sic_files = os.listdir(filepath)  # a list that contains filenames

    data_len = len(sic_files)
    sic = np.zeros((data_len, 448, 304))
    time = []

    var = 'cdr_seaice_conc_monthly'
    var2 = 'time'

    for idx, filename in enumerate(sic_files):
        nf = nC.Dataset(filepath + filename)

        temp_sic = nf[var][:]
        temp_sic = np.array(temp_sic)  # shape: 1*448*304
        # valid values lie between 0 and 1 ie., [0, 1]
        temp_sic[temp_sic > 1] = 0
        temp_sic[temp_sic < 0] = 0
        temp_sic = temp_sic.squeeze()
        sic[idx] = temp_sic

        temp_time = nf[var2][:]
        temp_time = nC.num2date(temp_time, units='days since 1601-01-01 00:00:00').data
        time.append(temp_time)

    time = np.array(time)

    return time, sic


def read_sic(filepath=FILEPATH):
    """
    read the sic file
    :param filepath: the filepath that contains the sic data
                    defaults ./data/SICcdr_seaice_conc_monthly.nc
    :return: time: ndarray
             sic: ndarray
    """
    nf = nC.Dataset(filepath)
    var = 'cdr_seaice_conc_monthly'
    sic = nf[var][:]
    sic = np.array(sic)  # shape: 511*448*304
    # valid values lie between 0 and 1 ie., [0, 1]
    sic[sic > 1] = 0
    sic[sic < 0] = 0

    var2 = 'time'
    var2_data = nf[var2][:]
    # shape: 511*1
    # 1978-11~2021-05
    time = nC.num2date(var2_data, units='days since 1601-01-01 00:00:00').data[:, np.newaxis]

    # add the latest data
    # 2021-06~2021-12
    time_latest, sic_latest = read_latest_sic(filepath=FILEPATH2)
    time = np.concatenate((time, time_latest), axis=0)
    sic = np.concatenate((sic, sic_latest), axis=0)

    non_land = non_land_mask()
    sic = sic * non_land

    return time, sic


def crop_sic(filepath=FILEPATH, row_idx=0, col_idx=0, img_size=32):
    """
    crop the sic data (448*304) to obtain a small area of interest
    :param filepath: the filepath that contains the sic data
    :param row_idx: row index
    :param col_idx: col index
    :param img_size: the image size of our small area, defaulting 32*32
    :return: time, cropped_sic
    """

    time, sic = read_sic(filepath)

    # Since the original image size of 448*304 is not a multiple of 32,
    # we have to reshape the data to 448*320 by filling the zeros
    filled_zeros = np.zeros((sic.shape[0], sic.shape[1], 320 - sic.shape[2]))
    filled_sic = np.append(sic, filled_zeros, axis=2)  # 518*448*320

    ri, ci = row_idx * img_size, col_idx * img_size
    rj, cj = (row_idx + 1) * img_size, (col_idx + 1) * img_size
    if rj > filled_sic.shape[1] or cj > filled_sic.shape[2]:
        raise IndexError
    else:
        cropped_sic = filled_sic[:, ri:rj, ci:cj]

    return time, cropped_sic


def data_split(filepath=FILEPATH, row_idx=0, col_idx=0, img_size=32, lag=1):
    """
    using the last month observation to forecast 1-ahead value
    split the data into inputs and targets
    :param filepath: the filepath that contains the sic data
    :param lag: the lagged observations used to perform the prediction, defaults 1
    :param row_idx: row index
    :param col_idx: col index
    :param img_size: the image size of our small area, defaulting 32*32
    :return: inputs_t, outputs_t, input_sic, target_sic
    """
    time, cropped_sic = crop_sic(filepath, row_idx, col_idx, img_size)
    input_t = time[:-lag, ...]
    target_t = time[lag:, ...]
    input_sic = cropped_sic[:-lag, ...]
    target_sic = cropped_sic[lag:, ...]

    # there is a change
    # input_sic_last_month = cropped_sic[11:-1, ...]
    # last_month = time[11:-1, ...]
    #
    # last_year_t = time[:-12, ...]
    # input_sic_last_year = cropped_sic[:-12, ...]
    #
    # target_sic_new = cropped_sic[12:, ...]
    # target_t_new = time[12:, ...]
    #
    # input_sic = input_sic_last_month * input_sic_last_year
    # target_sic = target_sic_new
    # target_t = target_t_new
    # input_t = last_month

    # we just consider the melting season, i.e., Apr-Sep
    # flag = np.full(input_t.size, True, dtype=bool)
    #
    # for ii in range(len(input_t)):
    #     input_month = input_t[ii][0].month
    #     target_month = target_t[ii][0].month
    #     if (3 <= input_month <= 8) and (target_month == (input_month + 1)):
    #         flag[ii] = False

    # reverse the flag to icing

    # input_t = input_t[flag]
    # target_t = target_t[flag]
    # input_sic = input_sic[flag]
    # target_sic = target_sic[flag]

    return input_t, target_t, input_sic, target_sic


def augment_tr(training_data, augmentation_type):
    """
    augment the training data
    :param augmentation_type: the dict used to provide specific augmentations
    :param training_data: training Tensor in N*C*H*W
    :return: augmented training data
    """
    transform_matrix = torch.tensor([
        [1, 0, 0],
        [0, 1, 0]
    ], dtype=torch.float32)  # *2*3

    # flip the image up-down
    if 'flipud' == augmentation_type:
        transform_matrix[1, 1] = -1

    # flip the image left-right
    if 'fliplr' == augmentation_type:
        transform_matrix[0, 0] = -1

    # rotate the image
    if 'rotate' == augmentation_type:
        angle_rad = random.random() * math.pi * 2
        s = math.sin(angle_rad)
        c = math.cos(angle_rad)
        transform_matrix = torch.tensor([
            [c, -s, 0],
            [s, c, 0]
        ], dtype=torch.float32)

    # using expand or broadcast
    theta = torch.zeros((len(training_data), 2, 3))  # N*2*3
    theta += transform_matrix

    grid = F.affine_grid(theta, training_data.size(), align_corners=False)
    device = training_data.device
    augmented_data = F.grid_sample(training_data, grid.to(device=device), align_corners=False)  # N*C*H*W

    return augmented_data


@functools.lru_cache(1)
def tr_val_test_gen(filepath=FILEPATH, row_idx=0, col_idx=0,
                    img_size=32, lag=1, shuffle=False, device='cpu', augmentation_type=None):
    """
    generate the sets of training(80%), validating(10%), and testing(10%)
    :param filepath: the filepath that contains the sic data
    :param row_idx: row index
    :param col_idx: col index
    :param img_size: the image size of our small area, defaulting 32*32
    :param lag: the lagged observations used to perform the prediction, defaults 1
    :param shuffle: whether randomly pick data as different sets, defaults False
    :param augmentation_type: Augment the training data
    :param device: the device on which to train our networks, defaults CPU
    :return:
    """
    input_t, target_t, input_sic, target_sic = data_split(filepath, row_idx, col_idx, img_size, lag)

    input_sic = input_sic[:, np.newaxis, :, :]  # adding the channel dimension
    target_sic = target_sic[:, np.newaxis, :, :]  # adding the channel dimension

    input_sic = torch.from_numpy(input_sic).to(dtype=torch.float32, device=device)
    target_sic = torch.from_numpy(target_sic).to(dtype=torch.float32, device=device)

    assert len(input_sic) == len(target_sic)
    data_len = len(input_sic)

    indices = range(data_len)
    if shuffle:
        # set a seed for reproducibility. Since you are generating random values, setting a seed
        # will ensure that the values generated are the same if the seed set is the same each time the code is run
        np.random.seed(1)
        indices = np.random.permutation(range(data_len))
    split_idx1 = int(len(indices) * 0.80)
    split_idx2 = int(len(indices) * 0.9)

    trX = input_sic[indices[:split_idx1]]
    trY = target_sic[indices[:split_idx1]]
    trY_T = target_t[indices[:split_idx1]]

    if augmentation_type:
        augmented_trX = augment_tr(trX, augmentation_type)
        augmented_trY = augment_tr(trY, augmentation_type)
        trX = torch.cat((trX, augmented_trX), dim=0)
        trY = torch.cat((trY, augmented_trY), dim=0)
        trY_T = np.concatenate((trY_T, trY_T), axis=0)

    valX = input_sic[indices[split_idx1:split_idx2]]
    valY = target_sic[indices[split_idx1:split_idx2]]
    valY_T = target_t[indices[split_idx1:split_idx2]]

    testX = input_sic[indices[split_idx2:]]
    testY = target_sic[indices[split_idx2:]]
    testY_T = target_t[indices[split_idx2:]]

    return trX, trY, trY_T, valX, valY, valY_T, testX, testY, testY_T


class SicDataset(Dataset):
    def __init__(self, filepath=FILEPATH, row_idx=0, col_idx=0,
                 img_size=32, lag=1, shuffle=False, device='cpu', augmentation_type=None, sets_str=None):
        """
        implementing our own datasets by subclassing the Pytorch Dataset
        :param filepath: the filepath that contains the sic data
        :param row_idx: row index
        :param col_idx: col index
        :param img_size: the image size of our small area, defaulting 32*32
        :param lag: the lagged observations used to perform the prediction, defaults 1
        :param shuffle: whether randomly pick data as different sets, defaults False
        :param augmentation_type: Augment the training data
        :param device: the device on which to train our networks, defaults CPU
        :param sets_str: training, validating or testing
        """
        (trX, trY, trY_T, valX, valY, valY_T,
         testX, testY, testY_T) = tr_val_test_gen(filepath, row_idx, col_idx, img_size,
                                                  lag, shuffle, device, augmentation_type)
        if sets_str == 'training':
            self.X = trX
            self.Y = trY
            self.Y_T = trY_T
        elif sets_str == 'validating':
            self.X = valX
            self.Y = valY
            self.Y_T = valY_T
        elif sets_str == 'testing':
            self.X = testX
            self.Y = testY
            self.Y_T = testY_T
        else:
            raise ValueError('Must be training, validating or testing')

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
