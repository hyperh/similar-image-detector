import pathlib
import datetime

def mkdir(dir):
	pathlib.Path(dir).mkdir(parents=True, exist_ok=True)

def get_timestamp_for_save():
    return str(datetime.datetime.now()).replace(':', '-').replace(' ', '-')

def get_save_path(timestamp):
    return './saves/{}'.format(timestamp)