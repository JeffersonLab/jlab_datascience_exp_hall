from jlab_datascience_toolkit.core.jdst_model import JDSTModel
from punzinet_toolkit.utils.architectures.torch_dense_architecture import TorchDenseArchitecture
from punzinet_toolkit.utils.optimizers.torch_optimizer import TorchOptimizer
from punzinet_toolkit.utils.trainers.punzinet_trainer import PunziNetTrainer
import torch
import yaml
import inspect
import logging
import os
import pandas as pd
from sklearn.model_selection import train_test_split

class PunziNet(torch.nn.Module,JDSTModel):
    '''
    Core class for a mlp neural network, using the punzi loss function and double training mechanism.
    '''

    # Initialize:
    #*********************************************
    def __init__(self,path_to_cfg,user_config={}):
        super().__init__()
        # Set the name specific to this module:
        self.module_name = "punzi_net"

        # Load the configuration:
        self.config = self.load_config(path_to_cfg,user_config)
 
        # Save this config, if a path is provided:
        if 'store_cfg_loc' in self.config:
            self.save_config(self.config['store_cfg_loc'])

        # Get the device (CPU, GPU) that we are running on:
        self.torch_device = self.config['torch_device']

        # Input data: feature names, import dataframe columns, etc.
        self.input_feature_names = self.config['input_feature_names']
        self.target_feature_names = self.config['target_feature_names']
        self.weight_names = self.config['weight_names']
        self.sigma_component_names = self.config['sigma_component_names']
        
        # Store the model somewher:
        self.model_store_loc = self.config['model_store_loc']
        os.makedirs(self.model_store_loc,exist_ok=True)
        # Decide if we soter the model weights or its scripted version:
        self.store_scripted_model = self.config['store_scripted_model']

        # Load an existing model, or its weights:
        model_load_loc = self.config['model_load_loc']
        # Check if the model or its weights have been stored:
        self.load_scripted_model = self.config['load_scripted_model']

        # Set up the model architecture:
        n_inputs = len(self.input_feature_names)
        n_outputs = len(self.target_feature_names)
        architecture = self.config['architecture']
        activations = self.config['activations']
        weight_initializers = self.config['weight_initializers']
        bias_initializers = self.config['bias_initializers']
        dropout_percents = self.config['dropout_percents']
        batchnorms = self.config['batchnorms']
        output_activation = self.config['output_activation']
        output_weight_initializer = self.config['output_weight_initializer']
        output_bias_initializer = self.config['output_bias_initializer']

        # Use the torch architecture class to quickly set up a model without doing any coding here:
        torch_mlp_architecture = TorchDenseArchitecture()
        mlp_architecture = torch_mlp_architecture.get_dense_architecture(
            n_inputs=n_inputs,
            n_outputs=n_outputs,
            architecture=architecture,
            activations=activations,
            weight_inits=weight_initializers,
            bias_inits=bias_initializers,
            dropouts=dropout_percents,
            batchnorms=batchnorms,
            output_activation=output_activation,
            output_weight_init=output_weight_initializer,
            output_bias_init=output_bias_initializer
        )
        self.model = None
        # Before turning this into a mlp, we check if there is already a complete model stored somewhere:
        if model_load_loc != "" and model_load_loc is not None and self.load_scripted_model == True:
            self.model = self.load(model_load_loc)
        else:
            self.model = torch.nn.Sequential(mlp_architecture).to(self.torch_device)

            if model_load_loc != "" and model_load_loc is not None:
                state_dict = self.load(model_load_loc)
                self.model.load_state_dict(state_dict)
        
        # Set up the optimizers:
        
        # Use the optimizer tool to quickly call an optimizer:
        torch_optimizer = TorchOptimizer()

        # BCE: 
        bce_optimizer_str = self.config['bce_optimizer']
        bce_learning_rate = self.config['bce_learning_rate']
        bce_lr_scheduler_mode = self.config['bce_lr_scheduler_mode']
        bce_lr_scheduler_factor = self.config['bce_lr_scheduler_factor']
        bce_lr_scheduler_patience = self.config['bce_lr_scheduler_patience']
        bce_lr_scheduler_threshold = self.config['bce_lr_scheduler_threshold']
        bce_lr_scheduler_threshold_mode = self.config['bce_lr_scheduler_threshold_mode']

        # Get an optimizer for the BCE training:
        self.bce_optimizer = torch_optimizer.get_optimizer(self.model,bce_optimizer_str,bce_learning_rate)
        # Set up the learning rate scheduler, as it was suggested in the punzi paper:
        self.bce_lr_scheduler = None
        if bce_lr_scheduler_mode is not None and bce_lr_scheduler_factor > 0.0:
           self.bce_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.bce_optimizer,bce_lr_scheduler_mode,bce_lr_scheduler_factor,bce_lr_scheduler_patience,threshold=bce_lr_scheduler_threshold,threshold_mode=bce_lr_scheduler_threshold_mode)

        # Punzi:
        punzi_optimizer_str = self.config['punzi_optimizer']
        punzi_learning_rate = self.config['punzi_learning_rate']
        punzi_lr_scheduler_mode = self.config['punzi_lr_scheduler_mode']
        punzi_lr_scheduler_factor = self.config['punzi_lr_scheduler_factor']
        punzi_lr_scheduler_patience = self.config['punzi_lr_scheduler_patience']
        punzi_lr_scheduler_threshold = self.config['punzi_lr_scheduler_threshold']
        punzi_lr_scheduler_threshold_mode = self.config['punzi_lr_scheduler_threshold_mode']

        # Get an optimizer for the Punzi training:
        self.punzi_optimizer = torch_optimizer.get_optimizer(self.model,punzi_optimizer_str,punzi_learning_rate)
        # Set up the learning rate scheduler, as it was suggested in the punzi paper:
        self.punzi_lr_scheduler = None
        if punzi_lr_scheduler_mode is not None and punzi_lr_scheduler_factor > 0.0:
           self.punzi_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.punzi_optimizer,punzi_lr_scheduler_mode,punzi_lr_scheduler_factor,punzi_lr_scheduler_patience,threshold=punzi_lr_scheduler_threshold,threshold_mode=punzi_lr_scheduler_threshold_mode)

        # Training:
        self.validation_split = self.config['validation_split']
        # We are unsing the punzi trainer to make our lives easier:
        punzi_loss_constant_a = self.config['punzi_loss_constant_a']
        punzi_loss_constant_b = self.config['punzi_loss_constant_b']
        punzi_loss_scale = self.config['punzi_loss_scale']
        n_mass_hypotheses = self.config['n_mass_hypotheses']
        n_gen_signal = self.config['n_gen_signal']
        target_luminosity = self.config['target_luminosity']

        self.punzi_trainer = PunziNetTrainer(
            bce_loss_function_string="binary_crossentropy",
            punzi_loss_a=punzi_loss_constant_a,
            punzi_loss_b=punzi_loss_constant_b,
            punzi_loss_scale=punzi_loss_scale,
            n_mass_hypotheses=n_mass_hypotheses,
            n_gen_signal=n_gen_signal,
            target_luminosity=target_luminosity,
            snapshot_folder=self.model_store_loc,
            store_scripted_model=self.store_scripted_model,
            torch_device=self.torch_device
        )

        # BCE:
        self.n_epochs_bce = self.config['n_epochs_bce']
        self.read_epochs_bce = self.config['read_epochs_bce']
        self.mon_epochs_bce = self.config['mon_epochs_bce']
        self.snapshot_epochs_bce = self.config['snapshot_epochs_bce']
        self.batch_size_bce = self.config['batch_size_bce']

        # Punzi:
        self.n_epochs_punzi = self.config['n_epochs_punzi']
        self.read_epochs_punzi = self.config['read_epochs_punzi']
        self.mon_epochs_punzi = self.config['mon_epochs_punzi']
        self.snapshot_epochs_punzi = self.config['snapshot_epochs_punzi']
        self.batch_size_punzi = self.config['batch_size_punzi']
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
    
    # Check the input data type:
    #*********************************************
    def check_input_data_type(self,data):
        if type(data) == pd.DataFrame:
            return True
        else:
            logging.error(f">>> {self.module_name}: The input data type {type(data)} is not a pandas dataframe <<<")
            return False
    #*********************************************
    
    # Define a forward / predcit function:
    #*********************************************
    def forward(self,x):
        return self.model(x)
    
    #---------------

    def predict(self,x):
        return self.forward(x)
    #*********************************************
   
    # Get the sequential model itself:
    #*********************************************
    def get_model(self):
        return self.model
    #*********************************************
    
    # Run the training: --> We just use the punzi trainer so everything becomes a one-liner...
    #*********************************************
    def train(self,dataframe):
        if self.check_input_data_type(dataframe) == True:
          # Collect all the columns that we need from the dataframe to train our model:
          x_in = dataframe[self.input_feature_names].values
          y_in = dataframe[self.target_feature_names].values
          w_in = dataframe[self.weight_names].values
          s_in = dataframe[self.sigma_component_names].values

          # Create a validation data set:
          x,x_test,y,y_test,w,w_test,s,s_test = train_test_split(x_in,y_in,w_in,s_in,test_size=self.validation_split)
          
          return self.punzi_trainer.run(
            model=self.model,
            bce_optimizer=self.bce_optimizer,
            punzi_optimizer=self.punzi_optimizer,
            bce_lr_scheduler=self.bce_lr_scheduler,
            punzi_lr_scheduler=self.punzi_lr_scheduler,
            x=x,
            y=y,
            w=w,
            s=s,
            x_test=x_test,
            y_test=y_test,
            w_test=w_test,
            s_test=s_test,
            n_epochs_bce=self.n_epochs_bce,
            batch_size_bce=self.batch_size_bce,
            mon_epochs_bce=self.mon_epochs_bce,
            read_epochs_bce=self.read_epochs_bce,
            snapshot_epochs_bce=self.snapshot_epochs_bce,
            n_epochs_punzi=self.n_epochs_punzi,
            batch_size_punzi=self.batch_size_punzi,
            mon_epochs_punzi=self.mon_epochs_punzi,
            read_epochs_punzi=self.read_epochs_punzi,
            snapshot_epochs_punzi=self.snapshot_epochs_punzi
          )
    #*********************************************
    
    # Load / save a model:
    #*********************************************
    def save(self,path):
        if self.store_scripted_model:
            scripted_model = torch.jit.script(self.model)
            scripted_model.save(path+".pt")
        else:
            torch.save(self.model.state_dict(),path+".pt")

    #-----------------------

    def load(self,path):
        if self.load_scripted_model:
            return torch.jit.load(path+".pt")
        else:
            return torch.load(path+".pt")
    #*********************************************




