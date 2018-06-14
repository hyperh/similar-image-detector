# -*- coding: utf-8 -*-
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

def get_data(root='Wallpapers', img_size=(32, 32)):
    data_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    dataset = MNIST('./data', transform=data_transform, download=True)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    show_sample(dataset)

    return dataset, dataloader

def show_sample(dataset):
    index = 0
    img = dataset.__getitem__(index)[0][0]
    print(img.size())
    plt.imshow(img.numpy())
    plt.show()
