import os
import numpy as np
from jlab_datascience_toolkit.core.jdst_data_prep import JDSTDataPrep
import inspect
import yaml


class NumpyStandardScaler(JDSTDataPrep):
    """ Performs standard normal scaling on numpy ndarrays.

    Assumes data is formatted as (samples, features, ...) and that 
    normalization is desired over all dimensions (1 scaler per feature 
    dimension)
    """

    # We fill config with a default of None since the module has no 
    #       configuration parameters
    def __init__(self, config=None, name='numpy_standard_scaler') -> None:
        self.mean = 0
        self.std = 1
        self.module_name = name

    def get_info(self):
        print(inspect.getdoc(self))

    # Nothing to load
    def load_config(self, path):
        pass

    # Nothing to save
    def save_config(self, path):
        pass

    def train(self, data: np.ndarray):
        x = data
        if data.ndim == 1:
            x = x[:, np.newaxis]

        self.mean = np.mean(x, axis=0)
        self.std = np.std(x, axis=0)

    def run(self, data: np.ndarray):
        x = data
        if data.ndim == 1:
            x = x[:, np.newaxis]

        return (x - self.mean) / self.std

    def reverse(self, data: np.ndarray):
        x = data
        if data.ndim == 1:
            x = x[:, np.newaxis]

        return x * self.std + self.mean

    def save_data(self):
        pass

    def save(self, path):
        os.makedirs(path)
        fullpath = os.path.join(path, f'standard_scaler_params.npz')
        np.savez(fullpath, mean=self.mean, std=self.std)

    def load(self, path):
        fullpath = os.path.join(path, f'standard_scaler_params.npz')
        module_params = np.load(fullpath)
        self.mean = module_params['mean']
        self.std = module_params['std']
