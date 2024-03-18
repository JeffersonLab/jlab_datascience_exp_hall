import os
import numpy as np
import logging
from jlab_datascience_toolkit.core.jdst_data_prep import JDSTDataPrep
from Hall_B.AIDAPT.utils.math_utils import four_vector_dot, get_alpha
from Hall_B.AIDAPT.utils.config_utils import verify_config
import yaml
import inspect

aidapt_lab2invariants_log = logging.getLogger('AIDAPT Lab2Invariants Logger')


class LabVariablesToInvariants(JDSTDataPrep):
    """Converts a numpy array of size (n,16) and converts it to invariants 
    used by AIDAPT models

    Required intialization arguments: `config: dict`, `name: str`

    Required configuration keys: None
    Optional configuration keys: `MP: float = 0.93827`

    Attributes
    ----------
    module_name : str
        Name of the module
    config: dict
        Configuration information

    Methods
    -------
    get_info()
        Returns this docstring
    load_config(path)
        Loads and verifies a configuration dictionary from <path>
    save_config(path)
        Saves the current configuration dictionary to a yaml file located at
        <path>
    load(path)
        Currently does nothing
    save(path)
        Currently does nothing
    save_data(path)
        Currently does nothing
    run(data)
        Converts lab data to invariants
    reverse(data)
        Currently does nothing, returns input data

    """

    def __init__(self, config: dict, name: str = 'AIDAPT Lab2Invariants'):
        self.module_name = name
        self.required_config_keys = []
        verify_config(config, self.required_config_keys)
        self.config = config

        # Constants used in lab_to_com and lab_to_inv
        if 'MP' not in self.config:
            self.config['MP'] = 0.93827

    def get_info(self):
        print(inspect.getdoc(self))

    def load_config(self, path):
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        verify_config(config, self.required_config_keys)
        self.config = config

    def save_config(self, path):
        with open(path, 'w') as f:
            yaml.safe_dump(self.config, f)

    def load(self):
        aidapt_lab2invariants_log.debug('Nothing to load...')
        pass

    def save(self):
        aidapt_lab2invariants_log.debug('Nothing to save...')
        pass

    def save_data(self):
        pass

    def run(self, data):
        # Remove unused fields in data
        output = np.delete(data, [0, 3], axis=1)
        # Convert to invariants
        output = self._lab_to_inv(output)
        return output

    def reverse(self, data):
        warn_string = f'{self.module_name} currently cannot be reversed. ' \
            'Returning original data.'
        aidapt_lab2invariants_log.warning(warn_string)
        return data

    def _lab_to_com(self, plab, s):
        output = plab.copy()
        MP = self.config['MP']
        c1 = (MP**2 + s)/(2*MP*np.sqrt(s))
        c2 = (MP**2 - s)/(2*MP*np.sqrt(s))
        output[:, 0] = c1*plab[:, 0] + c2*plab[:, 3]
        output[:, 3] = c2*plab[:, 0] + c1*plab[:, 3]
        return output

    def _lab_to_inv(self, data):
        """ Take lab data and transform to physical invariants

        Args:
            data (np.ndarray): Input data (either vertex or detector)

        Returns: 
            (np.ndarray): Data array containing:

                sppim: square of (recoil proton + pi-)
                spipm: square of (pi+ + pi-)
                tpip:  square of (photon - pi+)
                alpha: ???
                s:     square of (photon + target proton)
                
                phi:   azimuth of pi+ (value not used, so removed)
                MX:    Missing mass of the pi- (value not used, so removed)
        """

        N = data.shape[0]
        nu = data[:, 1]

        rec_idx = [11, 2, 5, 8]
        pi_plus_idx = [12, 3, 6, 9]
        pi_minus_idx = [13, 4, 7, 10]
        recoil_proton = data[:, rec_idx]
        pi_plus = data[:, pi_plus_idx]
        pi_minus = data[:, pi_minus_idx]

        beam = np.zeros((N, 4))
        beam[:, 0] = nu
        beam[:, 3] = nu
        target = np.zeros((N, 4))
        target[:, 0] = self.config['MP']

        s = four_vector_dot(beam+target, beam+target)
        sppim = four_vector_dot(recoil_proton+pi_minus, recoil_proton+pi_minus)
        spipm = four_vector_dot(pi_plus+pi_minus, pi_plus+pi_minus)
        tpip = four_vector_dot(pi_plus-beam, pi_plus-beam)

        # alpha = np.zeros(N)
        alpha = get_alpha(self._lab_to_com(pi_plus, s),
                          self._lab_to_com(pi_minus, s))

        return np.stack([sppim, spipm, tpip, alpha, s], axis=1)
