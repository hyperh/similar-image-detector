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
			'flex-direction: row'
		]
	})

	return """
<div style="{style_wrapper}">
	<img src="{file}">
	<a href="./{file}">{file}</a>
</div>
	""".format(
		style_wrapper=styles['wrapper'],
		file=file)

def get_images(imgs):
	links = ''.join([Image(img) for img in imgs])
	formatted = """
<div>{links}</div>
	""".format(links=links)
	return formatted

def get_img_groups(data):
	groups = ''.join([get_images(img_group) for img_group in data])
	formatted = """
<div>{groups}</div>
	""".format(groups=groups)
	return formatted

def get_results_html(data):
	results = get_img_groups(data)

	html = """
<html>
	<h1>Results</h1>
	<div>{results}</div>
</html>
	""".format(results=results)
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