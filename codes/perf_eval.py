import matplotlib.pyplot as plt
from model import U_Net
import torch
from data_preprocessing import SicDataset
from utility import non_land_mask

device = 'cpu'
model_trained = U_Net().to(device)

model_trained.load_state_dict(torch.load('result/None_model/model_01.pt', map_location=device))
model_trained.eval()

tr_dataset = SicDataset(row_idx=0, col_idx=1, sets_str='training', device=device, shuffle=False, img_size=64)
test_dataset = SicDataset(row_idx=0, col_idx=1, sets_str='testing', device=device, shuffle=False, img_size=64)

model_test = model_trained(test_dataset.X)
model_tr = model_trained(tr_dataset.X)
testY_T = test_dataset.Y_T
trY_T = tr_dataset.Y_T

model_tr = model_tr.squeeze().to(device='cpu').detach().numpy()
target_tr = tr_dataset.Y.squeeze().to(device='cpu').detach().numpy()
model_test = model_test.squeeze().to(device='cpu').detach().numpy()
target_test = test_dataset.Y.squeeze().to(device='cpu').detach().numpy()

diff_test = model_test - target_test
non_land = non_land_mask()
mae = diff_test * non_land()

fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 3))
idx = 40
img1 = axs[0].imshow(model_test[idx], cmap='jet', vmin=-1, vmax=1)
axs[0].set_title(f'pre:{testY_T[idx][0].year}-{testY_T[idx][0].month}')

img2 = axs[1].imshow(target_test[idx], cmap='jet', vmin=-1, vmax=1)
axs[1].set_title(f'target:{testY_T[idx][0].year}-{testY_T[idx][0].month}')

img3 = axs[2].imshow(diff_test[idx], cmap='jet', vmin=-1, vmax=1)
axs[2].set_title(f'diff:{testY_T[idx][0].year}-{testY_T[idx][0].month}')

plt.colorbar(mappable=img1)
plt.show()
