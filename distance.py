import csv
import torch
import itertools
import operator
import numpy as np
import json

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
	file_name = 'similar.json'
	file_path = '{}/{}'.format(save_path, file_name)

	json_serializable = [d.tolist() for d in data]

	with open(file_path, 'w') as myfile:
		json.dump(json_serializable, myfile)

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
	list<numpy.ndarray<string>>
		A list of length NUM_IMAGES, each entry being an image and containing an ndarray (of length 0 to NUM_IMAGES) of images to which they are similar to
	"""	
	all_similar = {}

	for i, distances in enumerate(all_distances):
		similar_mask = get_similar_images_boolean_mask(distances, threshold)
		similar_images = img_paths[similar_mask]

		# Trim list to only unique entries that have at least 1 similar image
		if len(similar_images) > 1:
			key = str(similar_images)
			all_similar[key] = similar_images

	all_similar_list = []
	for _, value in all_similar.items():
		all_similar_list.append(value)

	save(save_path, all_similar_list)
	return all_similar_list
