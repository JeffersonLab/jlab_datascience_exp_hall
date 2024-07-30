from jlab_datascience_toolkit.core.jdst_model import JDSTModel
from punzinet_toolkit.utils.architectures.torch_dense_architecture import TorchDenseArchitecture
from punzinet_toolkit.utils.optimizers.torch_optimizers import TorchOptimizers
from punzinet_toolkit.utils.trainers.punzi_net_trainer import PunziNetTrainer
import torch
import yaml
import inspect
import logging

class PunziNet(torch.nn.Module,JDSTModel):
    '''
    Core class for a mlp neural network, using the puniz loss function and double training mechanism.
    '''

    # Initialize:
    #*********************************************
    def __init__(self,path_to_cfg,user_config={}):
        # Set the name specific to this module:
        self.module_name = "punzi_net"
    
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
        print(inspect.getdoc(self))
    #*********************************************


