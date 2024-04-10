import torch
import logging

class TorchOptimizer(object):
    '''
    Helper class to retreive a torch optimizer
    '''

    # Initialize:
    #**************************
    def __init__(self):
        # Parameters for various optimizers: (these are the default values from pytorch)
        # Adam: (for details please see: https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam)
        self.adam_betas = (0.9, 0.999)
        self.adam_eps =  1e-08
        self.adam_weight_decay = 0
        self.adam_amsgrad = False
        self.adam_foreach = None
        self.adam_maximize = False
        self.adam_capturable = False
        self.adam_differentiable = False
        self.adam_fused = None

        # SGD: (details can be found here: https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD)
        self.sgd_momentum = 0
        self.sgd_dampening = 0
        self.sgd_weight_decay = 0
        self.sgd_nesterov = False
        self.sgd_maximize = False
        self.sgd_foreach = None
        self.sgd_differentiable = False

        # RMSprop (details can be found here: https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html#torch.optim.RMSprop)
        self.rmsprop_alpha = 0.99
        self.rmsprop_eps = 1e-08
        self.rmsprop_weight_decay = 0
        self.rmsprop_momentum = 0
        self.rmsprop_centered = False
        self.rmsprop_maximize = False
        self.rmsprop_foreach = None
        self.rmsprop_differentiable = False

        # Please feel free to add more optimizers...
    #**************************

   
    # Set the optimizer:
    #**************************
    def get_optimizer(self,model,optimizer_name,learning_rate):
        if optimizer_name.lower() == 'adam':
            return torch.optim.Adam(
                params=model.parameters(),
                lr=learning_rate,
                betas=self.adam_betas,
                eps=self.adam_eps,
                weight_decay=self.adam_weight_decay,
                amsgrad=self.adam_amsgrad,
                foreach=self.adam_foreach,
                maximize=self.adam_maximize,
                capturable=self.adam_capturable,
                differentiable=self.adam_differentiable,
                fused=self.adam_fused
                )
        
        elif optimizer_name.lower() == 'sgd':
            return torch.optim.SGD(
                params=model.parameters(),
                lr=learning_rate,
                momentum=self.sgd_momentum,
                dampening=self.sgd_dampening,
                weight_decay=self.sgd_weight_decay,
                nesterov=self.sgd_nesterov,
                maximize=self.sgd_maximize,
                foreach=self.sgd_foreach,
                differentiable=self.sgd_differentiable
                )
        
        elif optimizer_name.lower() == 'rmsprop':
            return torch.optim.RMSprop(
                params=model.parameters(),
                lr=learning_rate,
                alpha=self.rmsprop_alpha,
                eps=self.rmsprop_eps,
                weight_deacy=self.rmsprop_weight_decay,
                momentum=self.rmsprop_momentum,
                centered=self.rmsprop_centered,
                maximize=self.rmsprop_maximize,
                foreach=self.rmsprop_foreach,
                differentiable=self.rmsprop_differentiable
            )
        
        else:
            
            logging.warning(">>> TorchOptimizer: This optimizer is (currently) not implemented. Going to use default: Adam <<<")
            return torch.optim.Adam(
                params=model.parameters(),
                lr=learning_rate,
                betas=self.adam_betas,
                eps=self.adam_eps,
                weight_deacy=self.adam_weight_decay,
                amsgrad=self.adam_amsgrad,
                foreach=self.adam_foreach,
                maximize=self.adam_maximize,
                capturable=self.adam_capturable,
                differentiable=self.adam_differentiable,
                fused=self.adam_fused
                )
    #**************************

   