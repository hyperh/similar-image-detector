# -*- coding: utf-8 -*-
import torch
from torchvision import transforms, datasets
import matplotlib.pyplot as plt

def get_data(root='Wallpapers', img_size=(28, 28), batch_size=4, shuffle=True):
    data_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    dataset = datasets.ImageFolder(
        root=root,
        transform=data_transform)
    
    dataset_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0 # must be 0 or else dataloader fails during training, need to investigate
    )

    # show_sample(dataset)
    
    return dataset, dataset_loader

def show_sample(dataset):
    index = 0
    img = dataset.__getitem__(index)[0][0]
    print(img.size())
    plt.imshow(img.numpy())
    plt.show()
