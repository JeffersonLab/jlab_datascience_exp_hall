import torch
from torchmetrics import MeanMetric, Accuracy
from punzinet_toolkit.utils.loss_functions import TorchLossFunctions, PunziNetLoss
import logging

class PunziNetTrainer(object):
    '''
    Helper class to define the individual training steps of a punzi-net. The punzi net uses two training steps: (i) Training via a regualre binary cross-entropy (bce-training)
    and (ii) The punzi-net training (uses a custom loss function)
    '''
    
    # Initialize
    #********************
    def __init__(self,bce_loss_function_strig,punzi_loss_a,punzi_loss_b,punzi_loss_scale,n_mass_hypotheses,bce_device,punzi_device):
        # Get the bce loss function:
        self.bce_loss_function = TorchLossFunctions(bce_loss_function_strig)

        # Get the punzi loss function (this is a custom loss):
        self.punzi_loss = PunziNetLoss(n_mass_hypotheses,punzi_loss_a,punzi_loss_b,punzi_loss_scale,punzi_device)

        # Register the devices for each training (GPU vs. CPU)
        self.bce_device = bce_device
        self.punzi_device = punzi_device

        # Define performance trackers:

        # BCE:
        self.train_bce_loss_tracker = MeanMetric().to(self.bce_device)
        self.test_bce_loss_tracker = MeanMetric().to(self.bce_device)
        self.train_bce_acc_tracker = Accuracy(task='binary').to(self.bce_device)
        self.test_bce_acc_tracker = Accuracy(task='binary').to(self.bce_device)
        # Punzi:
        self.train_punzi_loss_tracker = MeanMetric().to(self.punzi_device)
        self.test_punzi_loss_tracker = MeanMetric().to(self.punzi_device)
        self.train_punzi_acc_tracker = Accuracy(task='binary').to(self.punzi_device)
        self.test_punzi_acc_tracker = Accuracy(task='binary').to(self.punzi_device)
    #********************

    # Sample batches from a given data tensor:
    #********************
    def get_data_batches(self,data_list,batch_dim):
        sample_size = data_list[0].size()[0]
        idx = None
        if batch_dim <= 0: # --> Use the entire data, but shuffle it:
          idx = torch.randint(low=0,high=sample_size,size=(sample_size,),device=data_list[0].device)
        else:
          idx = torch.randint(low=0,high=sample_size,size=(batch_dim,),device=data_list[0].device)  

        batched_data = []
        #++++++++++++++++
        for el in data_list:
            batched_data.append(el[idx].to(el.device))
        #++++++++++++++++

        return batched_data 
    #********************

    # BCE Training and Testing:
    #********************
    # Training:
    def bce_train_step(self,model,bce_optimizer,x,y,w):
       # Reset the optimizer:
       bce_optimizer.zero_grad(set_to_none=True)
       # Get the model predictions:
       model_predictions = torch.squeeze(model.predict(x))
       # Compute the weighted loss:
       weighted_loss = self.bce_loss_function(model_predictions,y)*w / torch.sum(w)
       # which leads to the loss:
       loss = torch.sum(weighted_loss)

       # Run backpropagation:
       loss.backward()
       # And weight update:
       bce_optimizer.step()
       
       # Register the predictions via the trackers:

       # Record the loss:
       self.train_bce_loss_tracker.update(loss)
       # Record the accuracy:
       self.train_bce_acc_tracker.update(model_predictions,y)
    
    #-----------------------

    # Testing:
    def bce_test_step(self,model,x,y,w):
       # Get the model predictions:
       model_predictions = torch.squeeze(model.predict(x))
       # Compute the weighted loss:
       weighted_loss = self.bce_loss_function(model_predictions,y)*w / torch.sum(w)
       # which leads to the loss:
       loss = torch.sum(weighted_loss)

       # Register the predictions via the trackers:

       # Record the loss:
       self.test_bce_loss_tracker.update(loss)
       # Record the accuracy:
       self.test_bce_acc_tracker.update(model_predictions,y)
    #********************


    # Punzi Training and Testing:
    #********************
    # Training:
    def punzi_train_step(self,model,punzi_optimizer,x,y,w,s,n_gen_signal,target_luminosity):
       # Reset optimizer:
       punzi_optimizer.zero_grad(set_to_none=True)
       # Get the model predictions:
       model_predictions = torch.squeeze(model.predict(x))
       # Compute punzi loss:
       punzi_loss = torch.sum(torch.mean(self.punzi_loss.compute(s,model_predictions,w,n_gen_signal,target_luminosity)))

       # Run backpropagaion:
       punzi_loss.backward()
       # Weight update:
       punzi_optimizer.step()

       # Register the predictions via the trackers:

       # Record the loss:
       self.train_punzi_loss_tracker.update(punzi_loss)
       # Record the accuracy:
       self.train_punzi_acc_tracker.update(model_predictions,y)
    
    #-----------------------

    # Testing:
    def punzi_test_step(self,model,x,y,w,s,n_gen_signal,target_luminosity):
       # Get the model predictions:
       model_predictions = torch.squeeze(model.predict(x))
       # Compute punzi loss:
       punzi_loss = torch.sum(torch.mean(self.punzi_loss.compute(s,model_predictions,w,n_gen_signal,target_luminosity)))

        # Register the predictions via the trackers:

       # Record the loss:
       self.test_punzi_loss_tracker.update(punzi_loss)
       # Record the accuracy:
       self.test_punzi_acc_tracker.update(model_predictions,y)
    #********************

    # Now put it all together:
    #********************
    def run(self,model,bce_optimizer,punzi_optimizer,bce_lr_scheduler,punzi_lr_scheduler,x,y,w,s,x_test,y_test,w_test,s_test,n_epochs_bce,batch_size_bce,mon_epochs_bce,read_epochs_bce,n_epochs_punzi,batch_size_punzi,mon_epochs_punzi,read_epochs_punzi,n_gen_signal,target_luminosity):
        # Collect tracker results:
        bce_training_loss = []
        bce_testing_loss = []
        bce_training_acc = []
        bce_testing_acc = []
        punzi_training_loss = []
        punzi_testing_loss = []
        punzi_training_acc = []
        punzi_testing_acc = []

        
    #********************
