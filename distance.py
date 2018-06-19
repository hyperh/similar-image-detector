import csv
import torch
import itertools
import operator
import numpy as np

def get_all_img_paths(dataset):
	# Return a numpy array so we can use mask arrays
	return np.array([i[0] for i in dataset.imgs])

def get_encodeds(dataset, dataset_loader, autoencoder):
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
	return encodeds

def get_encodeds_tensor(encodeds):
	"""
	Parameters
	----------
	encodeds: list<torch.Tensor>

	Returns
	-------
	torch.Tensor A tensor of size [NUM_IMAGES, FLATTENED]
	"""

	encodeds_tensor = torch.cat(encodeds, 0)
	return encodeds_tensor

def calc_distances(encodeds, encodeds_tensor):
	"""
	Parameters
	----------
	encodeds: list<torch.Tensor>
		A list of length of NUM_IMAGES, each entry containing a tensor of size [1, FLATTENED]
	encodeds_tensor: torch.Tensor
		A tensor of size [NUM_IMAGES, FLATTENED]

	Returns
	-------
	list<torch.Tensor>
		A list of length NUM_IMAGES, each entry containing a tensor of size NUM_IMAGES
	"""
	pdist = torch.nn.PairwiseDistance(p=2)

	all_distances = []

	for i, encoded in enumerate(encodeds):
		all_same_encoded = [encoded for e in encodeds]
		all_same_encoded_tensor = torch.cat(all_same_encoded, 0)

		distances = pdist(all_same_encoded_tensor, encodeds_tensor)	
		all_distances.append(distances)

	return all_distances

def save(save_path, data):
	file_name = 'similar.csv'
	file_path = '{}/{}'.format(save_path, file_name)
	with open(file_path, 'w') as myfile:
	    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
	    wr.writerow(data)

def get_similar_images_boolean_mask(distances, threshold=1.00000e-01 * 1):
	return [True if d < threshold else False for d in distances]
	
def find_similar_images(save_path, img_paths, all_distances, threshold=1.00000e-01 * 1):
	"""
	Parameters
	----------
	img_paths: list<string>
		A list of length NUM_IMAGES
	all_distances: list<torch.Tensor>
		A list of length NUM_IMAGES, each entry containing a tensor of size NUM_IMAGES
	threshold: float

	Returns
	-------
	list<list<string>>
		A list of length NUM_IMAGES, each entry being an image and containing a list (of length 0 to NUM_IMAGES) of images to which they are similar to
	"""	
	all_similar = []
	for i, distances in enumerate(all_distances):
		similar = get_similar_images_boolean_mask(distances, threshold)
		all_similar.append(img_paths[similar])

	save(save_path, all_similar)
	return all_similar
