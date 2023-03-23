import torch
from data_preprocessing import SicDataset
from torch.utils.data import DataLoader
from model import U_Net
import torch.optim as optim
from training import train
import os
import numpy as np
import matplotlib.pyplot as plt

row = 6
col = 2
weight_mode = None

period = 'result_whole/' + str(weight_mode)
model_path = str(period) + '_model/'
loss_path = str(period) + '_model_loss/'
flag_path = str(period) + '_tr_flag/'

if not os.path.exists(model_path):
    os.makedirs(model_path)
    os.makedirs(loss_path)
    os.makedirs(flag_path)

n_epochs = 50
patience = 7  # How long to wait after last time validation loss improved.
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tr_dataset = SicDataset(row_idx=row, col_idx=col, sets_str='training', device=device, shuffle=False, img_size=64,
                        )

if torch.all(tr_dataset.X == 0):
    print('cannot train the model')
    tr_flag = False
    np.save(flag_path + 'model_' + str(row) + str(col) + '_tr_flag.npy', tr_flag)
else:
    tr_flag = True
    np.save(flag_path + 'model_' + str(row) + str(col) + '_tr_flag.npy', tr_flag)

    val_dataset = SicDataset(row_idx=row, col_idx=col, sets_str='validating', device=device, shuffle=False, img_size=64)
    tr_loader = DataLoader(tr_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)

    model = U_Net().to(device)
    optimizer = optim.Adam(model.parameters())
    model_trained, tr_loss, val_loss = train(epochs=n_epochs, patience=patience, optimizer=optimizer, model=model,
                                             tr_loader=tr_loader, val_loader=val_loader, device=device,
                                             weight_period=weight_mode)

    model_trained.to(device='cpu')

    np.save(loss_path + 'model_' + str(row) + str(col) + '_tr_loss.npy', tr_loss)
    np.save(loss_path + 'model_' + str(row) + str(col) + '_val_loss.npy', val_loss)
    torch.save(model_trained.state_dict(), model_path + 'model_' + str(row) + str(col) + '.pt')

    # evaluate the model performance
    model_trained.eval()
    test_dataset = SicDataset(row_idx=row, col_idx=col, sets_str='testing', device='cpu', shuffle=False, img_size=64)
    model_test = model_trained(test_dataset.X)

    testY_T = test_dataset.Y_T
    model_test = model_test.squeeze().to(device='cpu').detach().numpy()
    target_test = test_dataset.Y.squeeze().to(device='cpu').detach().numpy()

    diff_test = np.abs(model_test - target_test)

    for idx in range(len(diff_test)):
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 3))
        img1 = axs[0].imshow(model_test[idx], cmap='jet', vmin=-1, vmax=1)
        axs[0].set_title(f'pre:{testY_T[idx][0].year}-{testY_T[idx][0].month}')

        img2 = axs[1].imshow(target_test[idx], cmap='jet', vmin=-1, vmax=1)
        axs[1].set_title(f'target:{testY_T[idx][0].year}-{testY_T[idx][0].month}')

        img3 = axs[2].imshow(diff_test[idx], cmap='jet', vmin=0, vmax=1)
        axs[2].set_title(f'diff:{testY_T[idx][0].year}-{testY_T[idx][0].month}')

        plt.colorbar(mappable=img3)

        fig.savefig(f'graph/{idx}.tif', format='tiff', bbox_inches='tight')
