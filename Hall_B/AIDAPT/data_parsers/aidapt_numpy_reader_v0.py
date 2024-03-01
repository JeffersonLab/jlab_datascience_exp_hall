import os
import numpy as np
import logging
from jlab_datascience_toolkit.core.jdst_data_parser import JDSTDataParser
from Hall_B.AIDAPT.utils.math_utils import four_vector_dot, get_alpha
import yaml
from pathlib import Path
import inspect

aidapt_numpy_reader_log = logging.getLogger('AIDAPT Parser V0 Logger')


class AIDAPTNumpyReaderV0(JDSTDataParser):
    """Reads a list of .npy files and concatenates them along a given axis.

    Required intialization arguments: `config: dict`, `name: str`

    Required configuration keys: `filepaths: str` or `list[str]`
    Optional configuration keys: `axis: int = 0`

    Attributes
    ----------
    module_name : str
        Name of the module
    config: dict
        Configuration information

    Methods
    -------
    load_config(path)
        Loads and verifies a configuration dictionary from <path>
    save_config(path)
        Saves the current configuration dictionary to a yaml file located at <path>
    load(path)
        Currently does nothing
    save(path)
        Currently does nothing
    load_data(path)
        Loads all files listed in `config['filepaths']` and concatenates them along the `config['axis']` axis
    save_data(path)
        Currently does nothing

    """

    def __init__(self, config: dict, name: str = "AIDAPT Numpy Reader V0"):
        self.module_name = name
        self.required_config_keys = ['filepaths']
        self.config = config
        self._verify_config()

        if isinstance(self.config['filepaths'], str):
            self.config['filepaths'] = [self.config['filepaths']]

        if not 'axis' in self.config:
            aidapt_numpy_reader_log.debug('Setting axis to default value: 0')
            self.config['axis'] = 0

    def get_info(self):
        print(inspect.getdoc(self))

    def load_config(self, path):
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        self._verify_config(config)
        self.config = config

    def save_config(self, path):
        with open(path, 'w') as f:
            yaml.safe_dump(self.config, f)

    def load(self):
        aidapt_numpy_reader_log.debug('Nothing to load...')
        pass

    def save(self):
        aidapt_numpy_reader_log.debug('Nothing to save...')
        pass

    def load_data(self):
        data_list = []
        for file in self.config['filepaths']:
            aidapt_numpy_reader_log.debug(f'Loading {file} ...')
            data_list.append(np.load(file))

        return np.concatenate(data_list, axis=self.config['axis'])

    def save_data(self):
        pass

    def _verify_config(self):
        missing_keys = []
        for key in self.required_config_keys:
            if key not in self.config:
                missing_keys.append(key)

        if missing_keys:
            error_msg = f'Config is missing the following required keys: {", ".join(missing_keys)}'
            aidapt_numpy_reader_log.error(error_msg)
            raise KeyError(error_msg)
