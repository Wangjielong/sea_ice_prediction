from data_preprocessing import SicDataset
import numpy as np
import os
from model import U_Net
import torch
import matplotlib.pyplot as plt
from utility import non_land_mask


def infer(model_dir='None_model/', img_size=64):
    """
    perform the inference after training our model
    :param img_size: the image size
    :param model_dir: the directory that contains all trained models
    :return: the prediction, the target, and the time
    """

    complete_prediction = np.zeros((SicDataset(sets_str='testing').X.shape[0], 448, 320))
    complete_testY = np.zeros((SicDataset(sets_str='testing').X.shape[0], 448, 320))

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

            test_dataset = SicDataset(row_idx=H, col_idx=W, sets_str='testing', img_size=s)
            testX = test_dataset.X
            testY = test_dataset.Y

            prediction = load_model(testX)
            prediction = prediction.detach().numpy().squeeze()

            complete_testY[:, hi:hj, wi: wj] = testY.detach().numpy().squeeze()
            complete_prediction[:, hi:hj, wi: wj] = prediction

    # Since the original image size of 448*304,  we have to reshape the data
    return complete_testY[:, :, :304], complete_prediction[:, :, :304], SicDataset(sets_str='testing').Y_T


non_land = non_land_mask()

target_test, model_test, testY_T = infer('result_whole/None_model/')

target_test = target_test * non_land
model_test = model_test * non_land

np.savez('result_whole/none.npz', target=target_test, model=model_test, time=testY_T)

# diff_test = np.abs(model_test - target_test)
# mae = np.mean(diff_test * non_land, axis=(1, 2))
#
# fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 3))
# idx = 0
# img1 = axs[0].imshow(model_test[idx], cmap='jet', vmin=-1, vmax=1)
# axs[0].set_title(f'pre:{testY_T[idx][0].year}-{testY_T[idx][0].month}')
#
# img2 = axs[1].imshow(target_test[idx], cmap='jet', vmin=-1, vmax=1)
# axs[1].set_title(f'target:{testY_T[idx][0].year}-{testY_T[idx][0].month}')
#
# img3 = axs[2].imshow(diff_test[idx], cmap='jet', vmin=0, vmax=1)
# axs[2].set_title(f'diff:{testY_T[idx][0].year}-{testY_T[idx][0].month}')
#
# plt.colorbar(mappable=img3)
# plt.show()
