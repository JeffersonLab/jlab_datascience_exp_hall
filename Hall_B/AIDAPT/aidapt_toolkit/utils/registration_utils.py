from aidapt_toolkit.registration import make
import os
import yaml

def load_module(directory):
    with open(os.path.join(directory, 'make_arguments.yaml'), 'r') as f:
        make_arguments = yaml.safe_load(f)
    
    id = make_arguments.pop('id')

    module = make(id=make_arguments['id'], **make_arguments)
    module.load(directory)

    return module
