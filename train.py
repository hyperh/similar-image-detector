# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import time
import math
import matplotlib.pyplot as plt
import numpy as np

from autoencoder import Autoencoder
import data
import save

NUM_TEST_IMG_DEFAULT = 5
steps_until_log = 100

def get_time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def log(epoch, step, loss, start_time, autoencoder, train_set, axes, save_path, num_test_img=NUM_TEST_IMG_DEFAULT):
    time_since = get_time_since(start_time)
    print(time_since, 'Epoch: {}, Step: {}'.format(epoch, step), '| train loss: %.4f' % loss.data.numpy())
    
    img_size = train_set.__getitem__(0)[0][0].size()
    
    for i in range(num_test_img):
        img = train_set.__getitem__(i)[0].unsqueeze(0)
        _, decoded_data = autoencoder(img)
        axes[1][i].clear()
        axes[1][i].imshow(np.reshape(decoded_data.data.numpy(), img_size))
        axes[1][i].set_xticks(())
        axes[1][i].set_yticks(())
    plt.savefig(
        get_save_file_path_figure(
            save_path, 
            get_figure_suffix(epoch, step)
            )
        )
    
def init_plot(save_path, num_test_img=NUM_TEST_IMG_DEFAULT):
    rows = 2
    figure, axes = plt.subplots(rows, num_test_img, figsize=(num_test_img * 2, rows * 2))
    plt.ion() # continuously plot
    for i in range(num_test_img):
        axes[0][i].imshow(train_set.__getitem__(i)[0][0])
    figure.savefig(get_save_file_path_figure(save_path, get_figure_suffix(0, 0)))
    
    return figure, axes

def get_figure_suffix(epoch, step):
    return 'epoch-{}-step-{}'.format(epoch, step)

def get_save_file_path_figure(save_path, suffix):
    return '{}/figure-{}.png'.format(save_path, suffix)

def get_save_file_path_model(save_path):
    return '{}/trained_model.pt'.format(save_path)

def save_loss_plot(save_path, losses):
    figure = plt.figure()
    plt.plot(losses)
    figure.savefig('{}/losses.png'.format(save_path))

def save_model(save_path, autoencoder):
    save_file_path = get_save_file_path_model(save_path)
    torch.save(autoencoder, save_file_path)

def train(n_epochs, train_loader, autoencoder, img_size, loss_func, train_set, save_path):
    save.mkdir(save_path)

    figure, axes = init_plot(save_path)        
    start = time.time()

    losses = []
    
    for epoch in range(n_epochs):
        for step, (x, _) in enumerate(train_loader):
            encoded, decoded = autoencoder(x)
            
            loss = loss_func(decoded, x) # mean square error
            losses.append(loss)
            optimizer.zero_grad() # clear gradients for this training step
            loss.backward() # backpropagation, compute gradients
            optimizer.step() # apply gradients
            
            if step % steps_until_log == 0:
                log(epoch, step, loss, start, autoencoder, train_set, axes, save_path)
    
    save_loss_plot(save_path, losses)
    save_model(save_path, autoencoder)
    print('Saved to {}'.format(save_path))

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
    img_size, 
    loss_func,
    train_set,
    save_path=save.get_save_path(timestamp)
    )
