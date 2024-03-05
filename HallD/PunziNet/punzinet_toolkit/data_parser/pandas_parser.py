from jlab_datascience_toolkit.core.jdst_data_parser import JDSTDataParser
import pandas as pd
import yaml
import logging
import os

class PandasParser(JDSTDataParser):

    '''
    Pandas data parser that reads in strings of file paths and returns a single pandas file. 
    The file format is specified by the user.
    '''

    # Initialize:
    #*********************************************
    def __init__(self,path_to_cfg,user_config={}):
        # Set the name specific to this module:
        self.module_name = "pandas_parser"

        # Load the configuration:
        self.config = self.load_config(path_to_cfg,user_config)
 
        # Save this config, if a path is provided:
        if 'store_cfg_loc' in self.config:
            self.save_config(self.config['store_cfg_loc'])

        # Run sanity check(s):
        # i) Make sure that the provide data path(s) are list objects:
        if isinstance(self.config['data_loc'],list) == False:
            logging.error(">>> " + self.module_name +": The data path(s) must be a list object, e.g. data_loc: [path1,path2,...] <<<")
    #*********************************************
            
    # Provide information about this module:
    #*********************************************
    def get_info(self):
        print("  ")
        print("***   Info: PandasParser   ***")
        print("Input(s):")
        print("i) Full path to .yaml configuration file ") 
        print("ii) Optional: User configuration, i.e. a python dict with additonal / alternative settings")
        print("What this module does:")
        print("i) Read in multiple pandas files that are specified in a list of strings")
        print("ii) Combine single files into one")
        print("Output(s):")
        print("i) Single pandas file")
        print("***   Info: PandasParser   ***")
        print("  ")
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
           
    # Load .npy file(s):
    #*********************************************
    # Load a single file:
    def load_single_file(self,path_to_file,file_format):
        try:
            # Load file, dependening on the file format: (please add more formats, if you need them)
            if file_format.lower() == "csv":
                return pd.read_csv(path_to_file)
            elif file_format.lower() == "json":
                return pd.read_json(path_to_file)
            elif file_format.lower() == "feather":
                return pd.read_feather(path_to_file)
            else:
                logging.error(f">>> " + self.module_name + ": File format {file_format} does either not exist or is not implemented <<<")
        except:
            logging.exception(">>> " + self.module_name + ": File with provided format does not exist <<<")

    
    #-----------------------------

    # Load multiple files which represent the final data:
    def load_data(self):
        try:
            collected_data = []
            pd_file_format = self.config['file_format']
            #+++++++++++++++++++++
            for path in self.config['data_loc']:
                collected_data.append(self.load_single_file(path,pd_file_format))
            #+++++++++++++++++++++

            return pd.concat(collected_data,axis=self.config['event_axis'])
        except:
            logging.exception(">>> " + self.module_name + ": Please check the provided data path which must be a list. <<<")
    #*********************************************
            
    # Save the data:
    #*********************************************
    def save_data(self,data):
        try:
           os.makedirs(self.config['data_store_loc'],exist_ok=True) 
           file_format = self.config['file_format']
           # Store data accroding to the provided format (add more if needed):
           if file_format.lower() == "csv":
                data.to_csv(self.config['data_store_loc'],index=self.config['use_pd_index'])
           elif file_format.lower() == "json":
                data.to_json(self.config['data_store_loc'],index=self.config['use_pd_index'])
           elif file_format.lower() == "feather":
                data.to_feather(self.config['data_store_loc'],index=self.config['use_pd_index'])
           else:
                logging.error(f">>> " + self.module_name + ": File format {file_format} does either not exist or is not implemented <<<")
           
        except:
           logging.exception(">>> " + self.module_name + ": Please provide a valid name for storing the data in .npy format. <<<")
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