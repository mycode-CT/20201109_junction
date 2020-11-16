""" Script to train the backbone network using Type-1 criteria on the cluster. """
import os

import matplotlib
import monai
import numpy as np
import torch
from monai.utils import set_determinism
from torch import nn
from torch import optim

from nnUnet import nnUnet
from utils import get_dataloader

# ------------------------------------------------
# Sugestion:
# Set a random seed for reproducibility
# ------------------------------------------------
seed = 1
set_determinism(seed=seed)
np.random.seed(seed)
# ------------------------------------------------

monai.config.print_config()

# matplotlib.use('TkAgg')
matplotlib.use('Agg')

pwd = os.getcwd()
# pwd = '/nfs/home/ctangwiriyasakul/DGXwork/projects/202009_sep/week01/20200904_Junction_DGX'

# Creating output dir
suffix = '/DGX_save9_ep100_p40_s10'
save_fold = pwd + suffix
# save_fold = pwd + '/save' #'/results_para'
os.mkdir(save_fold)

# ------------------------------------------------
# Loading data
# ------------------------------------------------
# train_root = '/nfs/home/ctangwiriyasakul/DGXwork/datasets/Junction/Training'
train_root = '/home/chayanin/PycharmProjects/2020_v06_vrienv005_MONAI/Data/neurovascular/Junction/Training'
batch_size = 1
patch_size = 30  # 24
num_samples = 2  # 10

train_loader = get_dataloader(train_root, batch_size, patch_size, num_samples, val_root=None)

# ------------------------------------------------
# Creating model
# ------------------------------------------------
# Sugestion:
# It should be run only with GPU. CPU training is too long for this kind of network
# ------------------------------------------------
device = torch.device("cuda")
# ------------------------------------------------

backbone = nnUnet(
    input_channels=1,
    base_num_channels=14,
    num_pool=3,
    num_classes=1
)

backbone = backbone.to(device)
# ------------------------------------------------
# Sugestion:
# Consider to use multiple GPUS with DataParallel
# example
# print(f"Let's use {torch.cuda.device_count()} GPUs!")
# backbone = torch.nn.DataParallel(backbone).to(device)
# ------------------------------------------------

optimizer = optim.Adam(backbone.parameters(), lr=1e-4)

# ------------------------------------------------
# Sugestion:
# Consider adding an learning rate scheduler
# ------------------------------------------------

# ------------------------------------------------
# Training loop
# ------------------------------------------------
# ------------------------------------------------
# Sugestion:
# Set this parameter to be easy to configure
# When training in bigger datasets, you will want to keep it small
# ------------------------------------------------
eval_freq = 10
# ------------------------------------------------
num_epochs = 10  # 25
epoch_loss_values = list()
for epoch in range(num_epochs):
    print(epoch)
    backbone.train()

    for patch_s in train_loader:
        batch_images, batch_labels = patch_s['image'], patch_s['label']

        # ------------------------------------------------
        # Sugestion:
        # Maybe consider to cast the type during the data loading?
        # this transformation might be good https://docs.monai.io/en/latest/transforms.html#casttotype
        # for example, CastToTyped(keys=['image', 'label', 'bilabel'], dtype="float32")
        # ------------------------------------------------
        batch_images = batch_images.type(torch.FloatTensor)
        batch_labels = batch_labels.type(torch.FloatTensor)
        # ------------------------------------------------

        optimizer.zero_grad()

        # ------------------------------------------------
        # Sugestion:
        # If you compute the loss outside the forward function,
        # it will be easier to use DataParallel and use multigpus
        # ------------------------------------------------
        _, x = backbone(batch_images.to(device))
        loss = nn.L1Loss(torch.sigmoid(x), batch_labels.to(device))

        print('BB', loss)
        loss.backward()
        optimizer.step()

        # ------------------------------------------------
        # Sugestion:
        # Use loss.item() to get the value of the loss and avoid parse it to .cpu().numpy() later
        # ------------------------------------------------
        epoch_loss_values.append(loss.item())

    if (epoch + 1) % eval_freq == 0:
        name = pwd + suffix + '/backbone_%d.pt' % epoch
        torch.save(backbone.state_dict(), name)

epoch_loss_values = np.array(epoch_loss_values)
np.save(save_fold + '/epoch_loss_values.npy', epoch_loss_values)
