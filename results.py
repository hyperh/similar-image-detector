import json

def create_styles(styles):
	"""
	Parameters
	----------
	style: dict<string: list<string>>
	"""
	return {key: ';'.join(value) for key, value in styles.items()}

def Image(file):
	styles = create_styles({
		"wrapper": [
			'display: flex',
			'flex-direction: column',
			'margin: 0 10px'
		],
		"img": [
			'width: 200px'
		]
	})

	return """
	<a href="./{file}" style="{style_wrapper}">
		<img src="{file}" style="{style_img}">
		{file}
	</a>
	""".format(
		style_wrapper=styles['wrapper'],
		style_img=styles['img'],
		file=file)

def ImageGroup(name, children):
	styles = create_styles({
		"wrapper": [
			'display: flex',
			'flex-direction: row',
			'margin: 10px 0'
		],
	})

	return """
	<div style="{style_wrapper}">
		<h2>{name}</h2>
		{children}
	</div>
	""".format(
		name=name,
		style_wrapper=styles['wrapper'],
		children=children)

def get_image_group(name, imgs):
	images = ''.join([Image(img) for img in imgs])
	image_group = ImageGroup(name=name, children=images)
	return image_group

def get_results_html(data):
	image_groups = ''.join([get_image_group(i + 1, d) for i, d in enumerate(data)])

	html = """
	<html>
		<h1>Results</h1>
		<div>{results}</div>
	</html>
	""".format(results=image_groups)
	return html

def save(save_path, data):
	file_name = 'results.html'
	file_path = '{}/{}'.format(save_path, file_name)
	with open(file_path, 'w') as myfile:
		myfile.write(data)

def save_results_html(save_path, data):
	results = get_results_html(data)
	save(save_path, results)

path = 'saves/2018-06-18-23-05-40.687803/similar.json'
with open(path) as f:
    data = json.load(f)

save_results_html('.', data)