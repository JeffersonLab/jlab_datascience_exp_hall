import numpy as np
import logging
from jlab_datascience_toolkit.core.jdst_data_parser import JDSTDataParser
from Hall_B.AIDAPT.utils.config_utils import verify_config
import yaml
import inspect
from pathlib import Path
import os

aidapt_numpy_reader_log = logging.getLogger('AIDAPT Parser V0 Logger')


class AIDAPTNumpyReaderV0(JDSTDataParser):
    """Reads a list of .npy files and concatenates them along a given axis.

    Required intialization arguments: `config: dict`
    Optional intialization arguments: `name: str`

    Required configuration keys: `filepaths: str | list[str]`
    Optional configuration keys: `axis: int = 0`

    Attributes
    ----------
    module_name : str
        Name of the module
    config: dict
        Configuration information

    Methods
    -------
    get_info()
        Prints this docstring
    load(path)
        Loads this module from `path/self.module_name`
    save(path)
        Saves this module from `path/self.module_name`
    load_data(path)
        Loads all files listed in `config['filepaths']` and concatenates them 
        along the `config['axis']` axis
    save_data(path)
        Does nothing
    load_config(path)
        Does nothing
    save_config(path)
        Does nothing

    """

    def __init__(self, config: dict, name: str = "AIDAPT Numpy Reader V0"):
        self.module_name = name

        # Required config keys contains only keys that do not have reasonable
        #   defaults.
        self.required_config_keys = ['filepaths']
        verify_config(config, self.required_config_keys)
        self.config = config

        if isinstance(self.config['filepaths'], str):
            self.config['filepaths'] = [self.config['filepaths']]

        if not 'axis' in self.config:
            aidapt_numpy_reader_log.debug('Setting axis to default value: 0')
            self.config['axis'] = 0

    def get_info(self):
        """ Prints the docstring for the AIDAPTNumpyReaderV0 module"""
        print(inspect.getdoc(self))

    def load(self, path: str):
        """ Load the entire module state from `path/<self.module_name>/`

        Args:
            path (str): Path to folder containing a `self.module_name` folder.
        """
        base_path = Path(path)
        save_dir = base_path.joinpath(self.module_name)
        with open(save_dir.joinpath('config.yaml'), 'r') as f:
            loaded_config = yaml.safe_load(f)

        self.config.update(loaded_config)

    # Save the entire module state to a folder under <path> called
    # <self.module_name>
    def save(self, path: str):
        """Save the entire module state to a folder under `path` called 
        `self.module_name`

        Args:
            path (str): Location to save the module folder
        """
        base_path = Path(path)
        save_dir = base_path.joinpath(self.module_name)
        os.makedirs(save_dir)
        with open(save_dir.joinpath('config.yaml'), 'w') as f:
            yaml.safe_dump(self.config, f)

    def load_data(self):
        data_list = []
        for file in self.config['filepaths']:
            aidapt_numpy_reader_log.debug(f'Loading {file} ...')
            data_list.append(np.load(file))

        return np.concatenate(data_list, axis=self.config['axis'])

    # Unimplemented functions below
    def save_data(self, path: str):
        pass

    def load_config(self, path: str):
        pass

    def save_config(self, path: str):
        pass
