from jlab_datascience_toolkit.core.jdst_data_prep import JDSTDataPrep
import pandas as pd
import numpy as np
import yaml
import logging
import os

class PunziDataPrep(JDSTDataPrep):

    '''
    Data preperation module for the punzi net in Hall D.
    '''

    # Initialize:
    #*********************************************
    def __init__(self,path_to_cfg,user_config={}):
        # Set the name specific to this module:
        self.module_name = "punzi_data_prep"

        # Load the configuration:
        self.config = self.load_config(path_to_cfg,user_config)
 
        # Save this config, if a path is provided:
        if 'store_cfg_loc' in self.config:
            self.save_config(self.config['store_cfg_loc'])

        # Register data frame related settings:
        self.feature_names = self.config.get('feature_names',None)
        self.target_names = self.config.get('target_names',None)

        # Make sure that feature and target names are actually set:
        assert self.feature_names is not None, logging.error(f">>> {self.module_name}: No feature names provided. Please provide a list of feature names <<<")
        assert self.target_names is not None, logging.error(f">>> {self.module_name}: No target names provided. Please provide a list of target names <<<")

        # Get the number of features:
        self.n_features = len(self.feature_names)

        # Physics info:

        # Weights for each bkg. decay mode:
        self.target_luminosity =  self.config.get('target_luminosity',1000)
        self.bkg_decay_branches =  self.config.get('bkg_decay_branches',None)
        self.luminosity_per_branch =  self.config.get('luminosity_per_branch',None)
        self.luminosity_factor =  self.config.get('luminosity_factor',1.0)

        # Include mass ranges:
        self.mass_range_params = self.config.get('mass_range_params',None)
        self.loc_mass_bin_width = self.config.get('loc_mass_bin_width',"")
        self.n_mass_sigma = self.config.get('n_mass_sigma',2.0)
        self.matrix_features = self.config.get('matrix_features',['sig_m_range','gen_mass','range_idx_low','range_idx_high'])

        # Sanity checks:
        assert self.bkg_decay_branches is not None, logging.error(f">>> {self.module_name}: No BKG. decay branches have been provided. Please provide a list of decay branches. <<<")
        assert self.luminosity_per_branch is not None, logging.error(f">>> {self.module_name}: No luminosities per branch have been provided. Please provide a list with a luminosity value per branch. <<<")
        self.n_decays = len(self.bkg_decay_branches)

        assert self.mass_range_params is not None, logging.error(f">>> {self.module_name}: No mass range parameters have been provided. Please provide a list with: [mass_start, mass_end,n_mass_points]. <<<")

    #*********************************************
        
    # Handle configurations:
    #*********************************************
    # Load the config:
    def load_config(self,path_to_cfg,user_config):
        with open(path_to_cfg, 'r') as file:
            cfg = yaml.safe_load(file)
        
        # Overwrite config with user settings, if provided
        try:
            if bool(user_config):
              #++++++++++++++++++++++++
              for key in user_config:
                cfg[key] = user_config[key]
              #++++++++++++++++++++++++
        except:
            logging.exception(">>> " + self.module_name +": Invalid user config. Please make sure that a dictionary is provided <<<") 

        return cfg
    
    #-----------------------------

    # Store the config:
    def save_config(self,path_to_config):
        with open(path_to_config, 'w') as file:
           yaml.dump(self.config, file)
    #*********************************************
           
    # Provide information about this module:
    #*********************************************
    def get_info(self):
        print("  ")
        print("***   Info: PunziDataPrep   ***")
        print("Input(s):")
        print("i) Full path to .yaml configuration file ") 
        print("ii) Optional: User configuration, i.e. a python dict with additonal / alternative settings")
        print("What this module does:")
        
        print("***   Info: PunziDataPrep   ***")
        print("  ")
    #*********************************************
        
    # Define the run method:
    #*********************************************
    # Calculate the weight for each decay branch:
    def calc_weight_per_decay_branch(self):
        weight_dict = {}

        #++++++++++++++++++++++++
        for i in range(self.n_decays):
             weight_dict[self.bkg_decay_branches[i]] = self.target_luminosity / (self.luminosity_per_branch[i] * self.luminosity_factor)
        #++++++++++++++++++++++++

        return weight_dict
    
    #-------------------------------

    # Add weights to the data frame:
    def add_weights_to_df(self,dataFrame):
        # Get the weights per decay:
        self.decay_weights = self.calc_weight_per_decay_branch()
        
        # Introduce a new target variable:
        dataFrame['weights'] = 1.0
        # Weight every non-signal according to the expected luminosity:
        is_signal = dataFrame['signal']
        dataFrame.loc[~is_signal, 'weights'] = dataFrame[~is_signal].category.map(self.decay_weights)

        # Now determine the expected fraction for the signal events:
        expected_signal_fraction = dataFrame[~is_signal]['weights'].sum() / dataFrame[is_signal]['weights'].sum()
        dataFrame.loc[is_signal, 'weights'] = expected_signal_fraction
        dataFrame['weights'] = dataFrame['weights'].astype('single')

    #-------------------------------

    # Load .csv file which contains the bin width for each mass value:
    def load_mass_bin_width_df(self):
        bin_width_df = pd.read_csv(self.loc_mass_bin_width,sep="\t")
        bin_width_df.columns = ['m', 'w', 'e']
        bin_width_df = bin_width_df.set_index('m')

        return bin_width_df
    
    #-------------------------------

    # Add mass ranges to the dataframe:
    def add_mass_ranges_to_df(self,dataFrame):
        # Get the width par mass first:
        self.width_per_mass = self.load_mass_bin_width_df()
        # Define the mass ranges, based on the configuration:
        self.mass_ranges = np.arange(self.mass_range_params[0],self.mass_range_params[1],self.mass_range_params[2])
        
        mass_in_gev = ((self.mass_ranges / 1000).reshape(-1, 1))
        mass_bin_width = self.width_per_mass.loc[self.mass_ranges].w.values

        low_mass_bound = (mass_in_gev**2 - self.n_mass_sigma * mass_bin_width.reshape(-1, 1))
        high_mass_bound = (mass_in_gev**2 + self.n_mass_sigma  * mass_bin_width.reshape(-1, 1))
        acceptance = (low_mass_bound < dataFrame['M2'].values) & (dataFrame['M2'].values < high_mass_bound)
        
        range_idx_low = np.ones(len(dataFrame), dtype=int) * acceptance.shape[0] + 100
        range_idx_high = np.ones(len(dataFrame), dtype=int) * -2
    
        # Make sure, that no index is 'out of bounce':
        #++++++++++++++++++++++++++++++++
        for i, row in enumerate(acceptance):
            range_idx_low[row & (range_idx_low > i)] = i
            range_idx_high[row & (range_idx_high < i)] = i
        #++++++++++++++++++++++++++++++++

        range_idx_low[range_idx_low == acceptance.shape[0] + 100] = -1

        dataFrame['sig_m_range'] = (dataFrame['gen_mass'].map({mass: 1 for mass in self.mass_ranges}) == 1).astype('int')
        dataFrame['range_idx_low'] = range_idx_low
        dataFrame['range_idx_high'] = range_idx_high
    #-------------------------------

    # Now put it all together:
    def run(self,dataFrame):
        # Add the weights:
        self.add_weights_to_df(dataFrame)
        # And add mass ranges:
        self.add_mass_ranges_to_df(dataFrame)

        return dataFrame
    #*********************************************
        
    # We can not reverse this operation:
    #*********************************************
    def reverse(self):
        pass    
    #*********************************************

    # Module checkpointing: Not implemented yet and maybe not 
    # necessary, ao we leave these functions blank for now
    #*********************************************
    def load(self):
        return 0

    #-----------------------------

    def save(self):
        return 0
    #*********************************************


    # No data saving for now
    #*********************************************
    def load_data(self):
        return 0

    #-----------------------------

    def save_data(self):
        return 0
    #*********************************************