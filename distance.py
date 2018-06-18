import torch
import matplotlib.pyplot as plt
import itertools
import operator
import numpy as np
import data

def get_all_img_paths(dataset):
	# Return a numpy array so we can use mask arrays
	return np.array([i[0] for i in dataset.imgs])

path = 'trained_model.pt'
autoencoder = torch.load(path)

img_size = (28, 28)
dataset, dataset_loader = data.get_data(
	img_size=img_size, batch_size=1, shuffle=False)

img_paths = get_all_img_paths(dataset)
num_imgs = len(img_paths)
print('{} images found'.format(num_imgs))

encodeds = []
for i, (x, y) in enumerate(dataset_loader):
	if i % 100 == 0 or i == num_imgs - 1:
		print('{}/{}: {:.0%} done Encoding...'.format(i, num_imgs, i/num_imgs))
	encoded = autoencoder(x)
	num_elem = list(itertools.accumulate(encoded[0].size(), operator.mul))[-1]
	flattened = encoded[0].view(-1, num_elem)
	encodeds.append(flattened)

encodeds_tensor = torch.cat(encodeds, 0)

pdist = torch.nn.PairwiseDistance(p=2)

for i, encoded in enumerate(encodeds):
	all_same_encoded = [encoded for e in encodeds]
	all_same_encoded_tensor = torch.cat(all_same_encoded, 0)

	distances = pdist(all_same_encoded_tensor, encodeds_tensor)	
	threshold = 1.00000e-02 * 1
	similar = [True if d < threshold else False for d in distances]

	if sum(similar) > 1:
		print('{} is similar to {}'.format(img_paths[i], img_paths[similar]))
