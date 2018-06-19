from autoencoder import Autoencoder
import torch
import torch.nn as nn

from train import train 
import data
import save

img_size = (28, 28)
learning_rate = 0.005
n_epochs = 1
       
train_set, train_loader = data.get_data(img_size=img_size)
autoencoder = Autoencoder(img_size=img_size)
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)
loss_func = nn.MSELoss()

timestamp = save.get_timestamp_for_save()

train(
    n_epochs, 
    train_loader, 
    autoencoder,
    optimizer,
    img_size, 
    loss_func,
    train_set,
    save_path=save.get_save_path(timestamp)
    )
