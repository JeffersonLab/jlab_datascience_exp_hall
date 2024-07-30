from punzinet_toolkit.utils.loss_functions import TorchLossFunctions, PunziNetLoss
import torch
import numpy as np
from torchmetrics import MeanMetric, Accuracy
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

    # Sample batches from a given list of data:
    # This data is a simple numpy array / data frame object and is turned into a torch tensor when the batch is created
    # This way, we do not overload the GPU memory 
    #********************
    def get_data_batches(self,data_list,batch_dim,torch_device):
        sample_size = data_list[0].shape[0]
        idx = None
        if batch_dim <= 0: # --> Use the entire data, but shuffle it:
          idx = np.random.choice(sample_size,sample_size)
        else:
          idx = np.random.choice(sample_size,batch_dim)

        batched_data = []
        #++++++++++++++++
        for dat in data_list:
            batched_data.append(torch.as_tensor(dat[idx],device=torch.float32,device=torch_device))
        #++++++++++++++++

        return batched_data 
    #********************

    # BCE Training and Testing:
    #********************
    # Training:
    def bce_train_step(self,model,bce_optimizer,bce_training_data):
       x,y,w = bce_training_data
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
    def bce_test_step(self,model,bce_test_data):
       x,y,w = bce_test_data
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
    def punzi_train_step(self,model,punzi_optimizer,punzi_training_data,n_gen_signal,target_luminosity):
       x,y,w,s = punzi_training_data
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
    def punzi_test_step(self,model,punzi_test_data,n_gen_signal,target_luminosity):
       x,y,w,s = punzi_test_data
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

        has_test_data = False
        if x_test is not None and y_test is not None and w_test is not None and s_test is not None:
           has_test_data = True
           logging.info(">>> Punzi Trainer: Test data available <<<")

        #-----------------------------------------------------------------------

        # Run the BCE training first, according to the original paper:
        if n_epochs_bce > 0:
           #++++++++++++++++++++++++++++++
           for bce_epoch in range(1,1+n_epochs_bce):
              # Draw a random batch from data:
              bce_training_data = self.get_data_batches([x,y,w],batch_size_bce,self.bce_device)
              # Update model parameters:
              self.bce_train_step(model,bce_optimizer,bce_training_data)
              # Update the learning rate scheduler if existent:
              if bce_lr_scheduler is not None:
                 bce_lr_scheduler.step()

              # Test, if test data is available:
              if has_test_data:
                 bce_test_data = self.get_data_batches([x_test,y_test,w_test],batch_size_bce,self.bce_device)
                 self.bce_test_step(model,bce_test_data)

              # Read out losses and accuracy:
              if bce_epoch % read_epochs_bce == 0:
                 bce_training_loss.append(self.train_bce_loss_tracker.compute().detach().cpu().item())
                 bce_training_acc.append(self.train_bce_acc_tracker.compute().detach().cpu().item())
                 self.train_bce_loss_tracker.reset()
                 self.train_bce_acc_tracker.reset()

                 if has_test_data:
                    bce_testing_loss.append(self.test_bce_loss_tracker.compute().detach().cpu().item())
                    bce_testing_acc.append(self.test_bce_acc_tracker.compute().detach().cpu().item())
                    self.test_bce_loss_tracker.reset()
                    self.test_bce_acc_tracker.reset()

              # Print out loss values on screen:
              if bce_epoch % mon_epochs_bce == 0:
                 print(" ")
                 print(f"BCE Epoch: {bce_epoch} / {n_epochs_bce}")
                 if len(bce_training_loss) > 0 and len(bce_training_acc) > 0:
                    print(f"BCE Training Loss: {round(bce_training_loss[-1],4)}")
                    print(f"BCE Training Acc.: {round(bce_training_acc[-1],4)}")

                 if len(bce_testing_loss) > 0 and len(bce_testing_acc) > 0:
                    print(f"BCE Testing Loss: {round(bce_testing_loss[-1],4)}")
                    print(f"BCE Testing Acc.: {round(bce_testing_acc[-1],4)}")
           #++++++++++++++++++++++++++++++
           print(" ")
        else:
           logging.warning(">>> Punzi Trainer: BCE training is inactive <<<")
        
        #-----------------------------------------------------------------------

        # Now run the training using the custom punzi loss:
        if n_epochs_punzi > 0:
           #++++++++++++++++++++++++++++++
           for punzi_epoch in range(1,1+n_epochs_punzi):
              # Get training data:
              punzi_training_data = self.get_data_batches([x,y,w,s],batch_size_punzi,self.punzi_device)
              # Update model parameters:
              self.punzi_train_step(model,punzi_optimizer,punzi_training_data,n_gen_signal,target_luminosity)
              # Update punzi learning rate scheduler, if existent:
              if punzi_lr_scheduler is not None:
                 punzi_lr_scheduler.step()

              if has_test_data:
                 punzi_testing_data = self.get_data_batches([x_test,y_test,w_test,s_test],batch_size_punzi,self.punzi_device)
                 # Run a test step:
                 self.punzi_test_step(model,punzi_testing_data,n_gen_signal,target_luminosity)

              # Record losses and accuracies:
              if punzi_epoch % read_epochs_punzi == 0:
                 punzi_training_loss.append(self.train_punzi_loss_tracker.compute().detach().cpu().item())
                 punzi_training_acc.append(self.train_punzi_acc_tracker.compute().detach().cpu().item())
                 self.train_punzi_loss_tracker.reset()
                 self.train_punzi_acc_tracker.reset()

                 if has_test_data:
                    punzi_testing_loss.append(self.test_punzi_loss_tracker.compute().detach().cpu().item())
                    punzi_testing_acc.append(self.test_punzi_acc_tracker.compute().detach().cpu().item())
                    self.test_punzi_loss_tracker.reset()
                    self.test_punzi_acc_tracker.reset()
              
              # Print out loss values on screen:
              if punzi_epoch % mon_epochs_punzi == 0:
                 print(" ")
                 print(f"Punzi Epoch: {punzi_epoch} / {n_epochs_punzi}")
                 if len(punzi_training_loss) > 0 and len(punzi_training_acc) > 0:
                    print(f"Punzi Training Loss: {round(punzi_training_loss[-1],4)}")
                    print(f"Punzi Training Acc.: {round(punzi_training_acc[-1],4)}")

                 if len(punzi_testing_loss) > 0 and len(punzi_testing_acc) > 0:
                    print(f"Punzi Testing Loss: {round(punzi_testing_loss[-1],4)}")
                    print(f"Punzi Testing Acc.: {round(punzi_testing_acc[-1],4)}")
           #++++++++++++++++++++++++++++++
           print(" ")
        else:
           logging.warning(">>> Punzi Trainer: Punzi training is inactive <<<")
        
        #-----------------------------------------------------------------------

        # Return the training history:
        return {
           'bce_training_loss':bce_training_loss,
           'bce_training_accuracy':bce_training_acc,
           'bce_testing_loss':bce_testing_loss,
           'bce_testing_accuracy':bce_testing_acc,
           'punzi_training_loss':punzi_training_loss,
           'punzi_training_accuracy':punzi_training_acc,
           'punzi_testing_loss':punzi_testing_loss,
           'punzi_testing_accuracy':punzi_testing_acc
        }
    #********************
