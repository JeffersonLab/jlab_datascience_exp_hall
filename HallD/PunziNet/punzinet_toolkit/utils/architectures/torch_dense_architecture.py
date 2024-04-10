import torch
from torch import nn
import math
from collections import OrderedDict
import logging

class TorchDenseArchitecture(object):
    '''
    Helper class to efficiently set up a PyTorch MLP with dense layers
    '''

    # Initialize:
    #**************************
    def __init__(self):
       
        # Default layer parameters:
        # Dropout layer:
        self.dropout_inplace = False
        # Linear layer:
        self.linearL_bias = True
        self.linearL_device = None
        self.linearL_dtype = None

        # Batch normalization:
        self.batchnormL_eps = 1e-5
        self.batchnormL_momentum = 0.1
        self.batchnormL_affine = True,
        self.batchnormL_track_running_stats=True
        self.batchnormL_device = None
        self.batchnormL_dtype = None
        
        # Parameters for various activation functions: (these are the default values from pytorch)
        self.elu_alpha = 1.0
        self.leaky_relu_slope = 0.01
        self.selu_inplace = False
        # Please feel free to add more parameters for more activation functions...
    #**************************

    # Set the activation functions:
    #**************************
    def set_activation_function(self,act_func_str):
        if act_func_str.lower() == "relu":
            return nn.ReLU()
        
        if act_func_str.lower() == "leaky_relu":
            return nn.LeakyReLU(self.leaky_relu_slope)
        
        if act_func_str.lower() == "elu":
            return nn.ELU(self.elu_alpha)
        
        if act_func_str.lower() == "selu":
            return nn.SELU(self.selu_inplace)
        
        if act_func_str.lower() == "tanh":
            return nn.Tanh()
        
        if act_func_str.lower() == "sigmoid":
            return nn.Sigmoid()
        
        if act_func_str.lower() == "softmax":
            return nn.Softmax()
        
        # If no activation is provided or set to 'linear', then return -1:
        if act_func_str.lower() == "linear" or act_func_str == "" or act_func_str is None:
            return -1
        
        # Add more activations (see here: https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity)
    #**************************

    # Set the weight initialization:
    # This is quite important!!!
    #************************** 
    def initialize_linear_layer(self,layer,layer_activation,weight_init,bias_init):
        # Get the weights and bias first:
        w = None
        b = None

        if layer.weight is not None:
           w = layer.weight.data
        if layer.bias is not None:
           b = layer.bias.data

        # Handle weight initialization:
        if weight_init.lower() != "default" and w is not None: #--> Default here means the default pytorch implementation...
           if layer_activation.lower == 'linear' or layer_activation.lower() == 'tanh' or layer_activation.lower() == 'sigmoid' or layer_activation.lower() == 'softmax':
               if weight_init.lower() == 'normal':
                   torch.nn.init.xavier_normal_(w)
               if weight_init.lower() == 'uniform':
                   torch.nn.init.xavier_uniform_(w)

           if layer_activation.lower() == 'relu' or layer_activation.lower() == 'leaky_relu' or layer_activation.lower() == 'elu':
               a_slope = 0.0
               if layer_activation.lower() == 'leaky_relu':
                   a_slope = self.leaky_relu_slope

               if weight_init.lower() == 'normal':
                  torch.nn.init.kaiming_normal_(w,a=a_slope,nonlinearity=layer_activation.lower())
               if weight_init.lower() == 'uniform':
                  torch.nn.init.kaiming_uniform_(w,a=a_slope,nonlinearity=layer_activation.lower())
          
           # Add the lecun initialization for selu activation:
           # Following this definition: https://www.tensorflow.org/api_docs/python/tf/keras/initializers/LecunNormal 
           if layer_activation.lower() == 'selu':
              stddev = 1. / math.sqrt(w.size(1))
              torch.nn.init.normal_(w,mean=0.0,std=stddev)

        # Handle bias initialization: #--> Default here means the default pytorch implementation...
        if bias_init.lower() != "default" and b is not None:
            if bias_init.lower() == "normal":
                torch.nn.init.normal_(b)
            if bias_init.lower() == "uniform":
                torch.nn.init.uniform_(b)
            if bias_init.lower() == "ones":
                torch.nn.init.ones_(b)
            if bias_init.lower() == "zeros":
                torch.nn.init.zeros_(b) 

        # Add more initialization methods...
    #**************************

    # Set up the layers for a dense mlp:
    #**************************
    def get_dense_architecture(self,n_inputs,n_outputs,architecture,activations,weight_inits,bias_inits,dropouts,batchnorms,output_activation,output_weight_init,output_bias_init):
        # Get the number of layers:
        n_layers = len(architecture)
        # First, make sure that the dimensionality is correct:
        assert n_inputs > 0, logging.error(f">>> TorchDenseArchitecture: Number of inputs {n_inputs} has to be positive <<<")
        assert n_outputs > 0, logging.error(f">>> TorchDenseArchitecture: Number of outputs {n_outputs} has to be positive <<<")
        assert n_layers == len(activations), logging.error(f">>> TorchDenseArchitecture: Number of hidden layers {len(architecture)} does not match the number of activations {len(activations)} <<<")
        assert n_layers == len(dropouts), logging.error(f">>> TorchDenseArchitecture: Number of hidden layers {len(architecture)} does not match the number of dropout values {len(dropouts)} <<<")
        assert n_layers == len(batchnorms), logging.error(f">>> TorchDenseArchitecture: Number of hidden layers {len(architecture)} does not match the number of batchnorm values {len(batchnorms)} <<<")
        assert n_layers == len(weight_inits), logging.error(f">>> TorchDenseArchitecture: Number of hidden layers {len(architecture)} does not match the number of weight initializations {len(weight_inits)} <<<")
        assert n_layers == len(bias_inits), logging.error(f">>> TorchDenseArchitecture: Number of hidden layers {len(architecture)} does not match the number of bias initializations {len(bias_inits)} <<<")

        # Now we can set up the mlp:
        mlp_layers = OrderedDict()
        
        # Take care of the hidden units:
        n_prev_nodes = n_inputs
        #++++++++++++++++++++++++++
        for i in range(n_layers):
            layer_name = 'layer_' + str(i)
            act_name = 'activation_' + str(i)
            dropout_name = 'dropout_' + str(i)
            batchnorm_name = 'batchnorm_' + str(i)
            
            # Add some neurons
            mlp_layers[layer_name] = nn.Linear(
                in_features=n_prev_nodes,
                out_features=architecture[i],
                bias=self.linearL_bias,
                device=self.linearL_device,
                dtype=self.linearL_dtype
            )
            
            # Set the activation function:
            layer_activation = self.set_activation_function(activations[i])
            if layer_activation != -1:
                mlp_layers[act_name] = layer_activation
            
            # Now initialize the layer properly:
            self.initialize_linear_layer(mlp_layers[layer_name],activations[i],weight_inits[i],bias_inits[i])
            
            # Add a batch normalization (if requested):
            if batchnorms[i] == True:
                mlp_layers[batchnorm_name] = nn.BatchNorm1d(
                    num_features = architecture[i],
                    eps=self.batchnormL_eps,
                    momentum=self.batchnormL_momentum,
                    affine=self.batchnormL_affine,
                    track_running_stats=self.batchnormL_track_running_stats,
                    device=self.batchnormL_device,
                    dtype=self.batchnormL_dtype
                )
            
            
            # Include a dropout, if requested:
            if dropouts[i] > 0.0:
                mlp_layers[dropout_name] = nn.Dropout(p=dropouts[i],inplace=self.dropout_inplace)
            
            
            n_prev_nodes = architecture[i]
        #++++++++++++++++++++++++++
        
        # Add an output:
        mlp_layers['output_layer'] = nn.Linear(n_prev_nodes,n_outputs)
        output_act = self.set_activation_function(output_activation)
        if output_act != -1:
            mlp_layers['output_activation'] = output_act
       
        # Initialize the output:
        self.initialize_linear_layer(mlp_layers['output_layer'],output_activation,output_weight_init,output_bias_init)

        # And return it:
        return mlp_layers
    #**************************