from torch.utils.data import DataLoader
from training import train
import torch.optim as optim
import torch
from data_preprocessing import SicDataset
import numpy as np
from model import U_Net
import os

n_epochs = 100
patience = 7  # How long to wait after last time validation loss improved.
device = 'cuda' if torch.cuda.is_available() else 'cpu'

Hs = 14  # 448/32=14 total number of rows
Ws = 10  # 320/32=10 total number of columns
s = 32  # image size

# need to change
period = 'melting'
model_path = str(period) + '_model/'
loss_path = str(period) + '_model_loss/'
flag_path = str(period) + '_tr_flag/'

if not os.path.exists(model_path):
    os.makedirs(model_path)
    os.makedirs(loss_path)
    os.makedirs(flag_path)

model_tr_loss = {}
model_val_loss = {}
model_tr_flag = {}

row = 13  # need to change
for H in range(row, row + 1):
    for W in range(Ws):

        print('H=', H, ', W=', W, 'is starting')
        tr_flag = True

        tr_dataset = SicDataset(row_idx=H, col_idx=W, sets_str='training', device=device, shuffle=False, img_size=s)
        if torch.all(tr_dataset.X == 0):
            tr_flag = False
            model_tr_flag['model_' + str(H) + str(W) + '_tr_flag'] = tr_flag
            continue

        val_dataset = SicDataset(row_idx=H, col_idx=W, sets_str='validating', device=device, shuffle=False, img_size=s)
        tr_loader = DataLoader(tr_dataset, batch_size=12, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=12, shuffle=True)

        model = U_Net().to(device)
        optimizer = optim.Adam(model.parameters())
        model_end, tr_loss, val_loss = train(epochs=n_epochs, patience=patience, optimizer=optimizer, model=model,
                                             tr_loader=tr_loader, val_loader=val_loader, device=device,
                                             weight_period=period)

        model_end.to(device='cpu')

        name = 'model_' + str(H) + str(W)
        model_tr_loss['model_' + str(H) + str(W)] = tr_loss
        model_val_loss['model_' + str(H) + str(W)] = val_loss
        model_tr_flag['model_' + str(H) + str(W) + '_tr_flag'] = tr_flag

        np.save(loss_path + 'model_' + str(H) + str(W) + '_tr_loss.npy', model_tr_loss)
        np.save(loss_path + 'model_' + str(H) + str(W) + '_val_loss.npy', model_val_loss)
        np.save(flag_path + 'model_' + str(H) + str(W) + 'tr_flag.npy', model_tr_flag)
        torch.save(model_end.state_dict(), model_path + 'model_' + str(H) + str(W) + '.pt')

        del model
        del model_end
        torch.cuda.empty_cache()

print('training ends')
