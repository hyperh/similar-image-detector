# -*- coding: utf-8 -*-
import torch
import time
import matplotlib.pyplot as plt
import numpy as np

import time_utils
import save

NUM_TEST_IMG_DEFAULT = 5
steps_until_log = 100

def log(epoch, step, loss, start_time, autoencoder, axes, save_path, data, num_img=NUM_TEST_IMG_DEFAULT):
    time_since = time_utils.get_time_since(start_time)
    print(time_since, 'Epoch: {}, Step: {}'.format(epoch, step), '| train loss: %.4f' % loss.data.numpy())
    
    # img_tensor = data.__getitem__(0)[0]
    # num_channels = img_tensor.size()[0]
    
    for i in range(num_img):
        img = data.__getitem__(i)[0].unsqueeze(0) # Add a batch of size 1
        _, decoded_data = autoencoder(img)
        # https://www.cs.virginia.edu/~vicente/recognition/notebooks/image_processing_lab.html
        # imshow needs a numpy array with the channel dimension
        # as the the last dimension so we have to transpose things.
        decoded_img = decoded_data.data[0].numpy().transpose(1, 2, 0)

        # Scale img to between [0, 1]
        decoded_img = (decoded_img - np.min(decoded_img))/np.ptp(decoded_img)

        axes[1][i].clear()
        axes[1][i].imshow(decoded_img)
        axes[1][i].set_xticks(())
        axes[1][i].set_yticks(())
    plt.savefig(
        get_save_file_path_figure(
            save_path, 
            get_figure_suffix(epoch, step)
            )
        )
    
def init_plot(save_path, data, num_img=NUM_TEST_IMG_DEFAULT):
    rows = 2
    figure, axes = plt.subplots(rows, num_img, figsize=(num_img * 2, rows * 2))
    plt.ion() # continuously plot
    for i in range(num_img):
        img_tensor = data.__getitem__(i)[0]
        axes[0][i].imshow(img_tensor.numpy().transpose(1, 2, 0))
    figure.savefig(get_save_file_path_figure(save_path, get_figure_suffix(0, 0)))
    
    return figure, axes

def get_figure_suffix(epoch, step):
    return 'epoch-{}-step-{}'.format(epoch, step)

def get_save_path_figure(save_path):
    return '{}/sample-images'.format(save_path);

def get_save_file_path_figure(save_path, suffix):
    return '{}/figure-{}.png'.format(get_save_path_figure(save_path), suffix)

def get_save_file_path_model(save_path):
    return '{}/trained_model.pt'.format(save_path)

def save_loss_plot(save_path, losses):
    figure = plt.figure()
    plt.plot(losses)
    figure.savefig('{}/losses.png'.format(save_path))

def save_model(save_path, autoencoder):
    save_file_path = get_save_file_path_model(save_path)
    torch.save(autoencoder, save_file_path)

def train(
    n_epochs,
    train_loader,
    autoencoder,
    optimizer,
    img_size,
    loss_func,
    train_set,
    save_path
    ):
    save.mkdir(save_path)
    save.mkdir(get_save_path_figure(save_path))

    figure, axes = init_plot(save_path, data=train_set)        
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
                log(epoch, step, loss, start, autoencoder, axes, save_path, data=train_set)
    
    save_loss_plot(save_path, losses)
    save_model(save_path, autoencoder)
    print('Saved to {}'.format(save_path))
