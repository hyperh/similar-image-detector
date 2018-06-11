# -*- coding: utf-8 -*-
import torch
from torchvision import transforms, datasets
import matplotlib.pyplot as plt

def get_data(root='Wallpapers', img_size=(64, 64)):
    data_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor()
    ])
    
    dataset = datasets.ImageFolder(
        root=root,
        transform=data_transform)
    
    dataset_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4, shuffle=True,
        num_workers=4)

    show_sample(dataset)
    
    return dataset, dataset_loader

def show_sample(dataset):
    index = 0
    print(dataset.imgs[index][0])
    single_channel = dataset.__getitem__(index)[0][1]
    
    print(type(single_channel))
    print(single_channel.size())
    plt.imshow(single_channel.numpy())
    plt.show()
