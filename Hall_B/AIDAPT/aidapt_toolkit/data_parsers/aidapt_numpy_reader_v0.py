from jlab_datascience_toolkit.core.jdst_data_parser import JDSTDataParser
from pathlib import Path
from omegaconf import OmegaConf
import numpy as np
import logging
import yaml
import inspect
import os

aidapt_numpy_reader_log = logging.getLogger('AIDAPT Parser V0 Logger')


class AIDAPTNumpyReaderV0(JDSTDataParser):
    """Reads a list of .npy files and concatenates them along a given axis.

    Optional intialization arguments: 
        `config: dict`
        `name: str`

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

    def __init__(self, config: dict = None, name: str = "AIDAPT Numpy Reader V0"):
        # It is important not to use default mutable arguments in python
        #   (lists/dictionaries), so we set config to None and update later
        self.module_name = name

        # Set default config
        self.config = dict(filepaths=[], axis=0)
        # Update configuration with new configuration
        if config is not None:
            self.config.update(config)

        # To handle strings and lists of strings, we convert the former here
        if isinstance(self.config['filepaths'], str):
            self.config['filepaths'] = [self.config['filepaths']]

    def get_info(self):
        """ Prints the docstring for the AIDAPTNumpyReaderV0 module"""
        print(inspect.getdoc(self))

    def load(self, path: str):
        """ Load the entire module state from `path/<self.module_name>/`

        Args:
            path (str): Path to folder containing a `self.module_name` folder.
        """
        base_path = Path(path)
        save_dir = base_path
        # save_dir = base_path.joinpath(self.module_name)
        with open(save_dir.joinpath('config.yaml'), 'r') as f:
            loaded_config = yaml.safe_load(f)

        self.config.update(loaded_config)

    def save(self, path: str):
        """Save the entire module state to a folder under `path` called 
        `self.module_name`

        Args:
            path (str): Location to save the module folder
        """
        base_path = Path(path)
        save_dir = base_path
        # save_dir = base_path.joinpath(self.module_name)
        os.makedirs(save_dir)
        with open(save_dir.joinpath('config.yaml'), 'w') as f:
            OmegaConf.save(self.config, f)

    def load_data(self) -> np.ndarray:
        """ Loads all files listed in `config['filepaths']` and concatenates 
        them along the `config['axis']` axis

        Returns:
            np.ndarray: A single array of concatenated data
        """
        data_list = []
        for file in self.config['filepaths']:
            aidapt_numpy_reader_log.debug(f'Loading {file} ...')
            data_list.append(np.load(file))

        # Check for empty data and return nothing if empty
        if not data_list:
            aidapt_numpy_reader_log.warn(
                'load_data() returning None. This is probably not what you '
                'wanted. Ensure that your configuration includes the key '
                '"filepaths"')
            return 
        
        return np.concatenate(data_list, axis=self.config['axis'])

    # Unimplemented functions below
    def save_data(self, path: str):
        aidapt_numpy_reader_log.warning(
            'save_data() is currently unimplemented.')
        pass

    def load_config(self, path: str):
        aidapt_numpy_reader_log.warn(
            'load_config() is currently unimplemented. '
            'Did you mean load()?')
        pass

    def save_config(self, path: str):
        aidapt_numpy_reader_log.warn(
            'save_config() is currently unimplemented.'
            ' Did you mean save()?')
        pass
