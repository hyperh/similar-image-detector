import time
import torch

import data
import distance
import results
import time_utils

start = time.time()
time_since = time_utils.get_time_since(start)

folder = 'saves/2018-06-18-23-05-40.687803'
model_path = '{}/trained_model.pt'.format(folder)
img_size = (28, 28)
autoencoder = torch.load(model_path)
thresholds = [0.01, 0.1, 0.25, 0.5, 1, 1.05, 1.1, 1.2, 1.5, 2]

dataset, dataset_loader = data.get_data(
	img_size=img_size, batch_size=1, shuffle=False)

img_paths = distance.get_all_img_paths(dataset)
encodeds = distance.get_encodeds(dataset, dataset_loader, autoencoder)
time_since = time_utils.get_time_since(start)
print('{}: Calculating distances...'.format(time_since))
all_distances = distance.calc_distances(encodeds)

num_similar = []
for threshold in thresholds:
	time_since = time_utils.get_time_since(start)
	print('{}: Finding similar images for threshold {}...'.format(time_since, threshold))
	similar_images = distance.find_similar_images(folder, img_paths, all_distances, threshold)
	result_file = 'results_threshold_{}.html'.format(threshold)
	results.save_results_html(result_file, folder, similar_images)
	num_similar.append(len(similar_images))

results.save_similar_images_plot(folder, num_similar, thresholds)

print('Done!')