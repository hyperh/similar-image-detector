import torch
import data
import distance

model_path = './saves/2018-06-18-21-29-41.802806/trained_model.pt'
img_size = (28, 28)
autoencoder = torch.load(model_path)

dataset, dataset_loader = data.get_data(
	img_size=img_size, batch_size=1, shuffle=False)

img_paths = distance.get_all_img_paths(dataset)
encodeds = distance.get_encodeds(dataset, dataset_loader, autoencoder)
encodeds_tensor = distance.get_encodeds_tensor(encodeds)
all_distances = distance.calc_distances(encodeds, encodeds_tensor)
similar_images = distance.find_similar_images(img_paths, all_distances, threshold=1.00000e-01 * 1)

for i in similar_images:
	print(len(i))