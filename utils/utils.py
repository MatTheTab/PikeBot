import json
import re 

def get_paths():
    '''
    Function that reads your config specified in config.json and returns a dictionary of paths 

    Specify your base_path to the folder where you keep PikeBot 
    '''
    with open('../config.json') as f:
        config = json.load(f)
    base_path=config['base_path']
    for key, value in config.items():
        config[key] = value.replace('{base_path}', base_path)
    return config