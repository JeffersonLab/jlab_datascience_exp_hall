import torch
import logging

class TorchLossFunctions(object):

    # Initialize:
    #**************************
    def __init__(self,loss_fn_str):
        self.loss_fn_str = loss_fn_str
    #**************************

    # Get the loss function:
    #**************************
    def get_loss_function(self):
        if self.loss_fn_str.lower() == 'mse':
            return torch.nn.MSELoss()
        elif self.loss_fn_str.lower() == 'mae':
            return torch.nn.L1Loss()
        elif self.loss_fn_str.lower() == 'huber':
            return torch.nn.HuberLoss()
        elif self.loss_fn_str.lower() == 'categorical_crossentropy':
            return torch.nn.CrossEntropyLoss()
        elif self.loss_fn_str.lower() == 'binary_crossentropy':
            return torch.nn.BCELoss()
        else:
            logging.warning(">>> TorchLossFunctions: This loss function is (currently) not implemented. Going to use default: MSE <<<")
            return torch.nn.MSELoss()
        
        # Add more losses (see here: https://pytorch.org/docs/stable/nn.html#loss-functions)
    #**************************

    