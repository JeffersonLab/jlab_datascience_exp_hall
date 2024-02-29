import os
import numpy as np
import logging
from jlab_datascience_toolkit.core.jdst_data_parser import JDSTDataParser
from Hall_B.AIDAPT.utils.math_utils import four_vector_dot, get_alpha
import yaml

aidapt_parser_v0_log = logging.getLogger('AIDAPT Parser V0 Logger')


class AIDAPTParserV0(JDSTDataParser):
    def __init__(self, config, name='aidapt_parser_v0') -> None:

        # config keys are always {module_name}.{key}
        self.required_cfg_keys = ['detector_filepath', 'vertex_filepath']
        self.module_name = name

        self.check_required_cfg(config)
        self.config = config

        # Constants used in lab_to_com and lab_to_inv
        self.MP = 0.93827
        self.MPI = 0.1395

    def check_required_cfg(self, cfg):
        missing_keys = []
        for key in self.required_cfg_keys:
            if key not in cfg:
                missing_keys.append(key)

        if missing_keys:
            raise KeyError(
                f'Config is missing the following required keys: {", ".join(missing_keys)}')

    def get_info(self):
        print(f'Module Name == {self.module_name}\n')
        print('Module loads data from directories and labels and images for use in a conditional GAN.')
        print('Proper configuration requires a directory for vertex and detector files.')
        print(
            'All ''ps_\{type\}_\{index\}.csv'' files in given directories are used.')

    def forward(self):
        return self.load_data(
            self.config['detector_filepath'],
            self.config['vertex_filepath']
        )

    # Nothing needs to be saved here since we load directories named in config file
    def save(self):
        pass

    # Nothing can be loaded here since forward will load and return data
    def load(self):
        pass

    def load_config(self, path):
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        self.check_required_cfg(config)
        self.config = config

    def save_config(self, path):
        with open(path, 'w') as f:
            yaml.safe_dump(self.config, f)

    # It's possible that it would be better not to allow this as
    #       it doesn't ensure the config matches the loaded data
    def load_data(self, detector_filepath, vertex_filepath):
        aidapt_parser_v0_log.debug(
            f'Loading detector files from {detector_filepath}')
        aidapt_parser_v0_log.debug(
            f'Loading vertex files from {vertex_filepath}')

        labels = self.load_npy_data(vertex_filepath, 'vertex')
        images = self.load_npy_data(detector_filepath, 'detector')

        labels = np.delete(labels, [0, 3], axis=1)
        images = np.delete(images, [0, 3], axis=1)

        labels = self._lab_to_inv(labels)
        images = self._lab_to_inv(images)

        return labels, images[:, :-1]

    def save_data(self, data, path):
        aidapt_parser_v0_log.warning(
            'aidapt_parser_v0.save_data() currently does nothing.')
        pass

    def load_npy_data(self, datapath, type='vertex'):
        # TODO: Update this to just read all ps_{type} files from path
        data_list = []
        for i in range(4):
            filepath = os.path.join(datapath, f'ps_{type}_{i}.npy')
            data_list.append(np.load(filepath))
            aidapt_parser_v0_log.debug(f'Loading {filepath}')
        return np.concatenate(data_list)

    def _lab_to_com(self, plab, s):
        output = plab.copy()
        c1 = (self.MP**2 + s)/(2*self.MP*np.sqrt(s))
        c2 = (self.MP**2 - s)/(2*self.MP*np.sqrt(s))
        output[:, 0] = c1*plab[:, 0] + c2*plab[:, 3]
        output[:, 3] = c2*plab[:, 0] + c1*plab[:, 3]
        return output

    def _lab_to_inv(self, data):
        N = data.shape[0]
        nu = data[:, 1]

        rec_idx = [11, 2, 5, 8]
        pip_idx = [12, 3, 6, 9]
        pim_idx = [13, 4, 7, 10]
        recoil_proton = data[:, rec_idx]
        pi_plus = data[:, pip_idx]
        pi_minus = data[:, pim_idx]

        beam = np.zeros((N, 4))
        beam[:, 0] = nu
        beam[:, 3] = nu
        target = np.zeros((N, 4))
        target[:, 0] = self.MP

        s = four_vector_dot(beam+target, beam+target)
        sppim = four_vector_dot(recoil_proton+pi_minus, recoil_proton+pi_minus)
        spipm = four_vector_dot(pi_plus+pi_minus, pi_plus+pi_minus)
        tpip = four_vector_dot(pi_plus-beam, pi_plus-beam)

        # alpha = np.zeros(N)
        alpha = get_alpha(self._lab_to_com(pi_plus, s),
                          self._lab_to_com(pi_minus, s))

        return np.stack([sppim, spipm, tpip, alpha, s], axis=1)
