from punzinet_toolkit.utils.loss_functions.torch_loss_functions import TorchLossFunctions
from punzinet_toolkit.utils.loss_functions.punzinet_loss import PunziNetLoss
import torch
import torch
import numpy as np
from torchmetrics import MeanMetric, Accuracy
import logging
import os

class PunziNetTrainer(object):
    '''
    Helper class to define the individual training steps of a punzi-net. The punzi net uses two training steps: (i) Training via a regualre binary cross-entropy (bce-training)
    and (ii) The punzi-net training (uses a custom loss function)
    '''
    
    # Initialize
    #********************
    def __init__(self,bce_loss_function_string,punzi_loss_a,punzi_loss_b,punzi_loss_scale,n_mass_hypotheses,n_gen_signal,target_luminosity,snapshot_folder,store_scripted_model,torch_device):
        # Get the bce loss function:
        torch_losses = TorchLossFunctions(bce_loss_function_string)
        self.bce_loss_function = torch_losses.get_loss_function()

        # Get the punzi loss function (this is a custom loss):
        self.punzi_loss = PunziNetLoss(n_mass_hypotheses,punzi_loss_a,punzi_loss_b,punzi_loss_scale,torch_device)
        self.n_gen_signal = n_gen_signal
        self.target_luminosity = target_luminosity

        # Register the devices for each training (GPU vs. CPU)
        self.torch_device = torch_device

        # Allow checkpoinint, i.e. store the model every i-th epoch.
        # Thus we need to decide if we want to store the entire model or just its weights:
        self.store_scripted_model = store_scripted_model
        self.snapshot_folder = snapshot_folder
        os.makedirs(self.snapshot_folder,exist_ok=True)

        # Define performance trackers:

        # BCE:
        self.train_bce_loss_tracker = MeanMetric().to(self.torch_device)
        self.test_bce_loss_tracker = MeanMetric().to(self.torch_device)
        self.train_bce_acc_tracker = Accuracy(task='binary').to(self.torch_device)
        self.test_bce_acc_tracker = Accuracy(task='binary').to(self.torch_device)
        # Punzi:
        self.train_punzi_loss_tracker = MeanMetric().to(self.torch_device)
        self.test_punzi_loss_tracker = MeanMetric().to(self.torch_device)
        self.train_punzi_acc_tracker = Accuracy(task='binary').to(self.torch_device)
        self.test_punzi_acc_tracker = Accuracy(task='binary').to(self.torch_device)
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
            batched_data.append(torch.squeeze(torch.as_tensor(dat[idx],dtype=torch.float32,device=torch_device)))
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
       model_predictions = torch.squeeze(model(x))
       # Compute the weighted loss:
       weighted_loss = self.bce_loss_function(model_predictions,y)*w / torch.sum(w)
       # which leads to the loss:
       loss = torch.sum(weighted_loss)

       # Run backpropagation:
       loss.backward()
       # And weight update:
       #bce_optimizer.step()
       
       # Register the predictions via the trackers:

       # Record the loss:
       self.train_bce_loss_tracker.update(loss)
       # Record the accuracy:
       self.train_bce_acc_tracker.update(model_predictions,y)

       return loss
    
    #-----------------------

    # Testing:
    def bce_test_step(self,model,bce_test_data):
       x,y,w = bce_test_data
       # Get the model predictions:
       model_predictions = torch.squeeze(model(x))
       # Compute the weighted loss:
       print(model_predictions)
       weighted_loss = self.bce_loss_function(model_predictions,y)*w / torch.sum(w)
       # which leads to the loss:
       loss = torch.sum(weighted_loss)

       # Register the predictions via the trackers:

       # Record the loss:
       self.test_bce_loss_tracker.update(loss)
       # Record the accuracy:
       self.test_bce_acc_tracker.update(model_predictions,y)

       return loss
    #********************


    # Punzi Training and Testing:
    #********************
    # Training:
    def punzi_train_step(self,model,punzi_optimizer,punzi_training_data):
       x,y,w,s = punzi_training_data
       # Reset optimizer:
       punzi_optimizer.zero_grad(set_to_none=True)
       # Get the model predictions:
       model_predictions = torch.squeeze(model(x))
       # Compute punzi loss:
       punzi_loss = torch.sum(torch.mean(self.punzi_loss.compute(s,model_predictions,w,self.n_gen_signal,self.target_luminosity)))

       # Run backpropagaion:
       punzi_loss.backward()
       # Weight update:
       #punzi_optimizer.step()

       # Register the predictions via the trackers:

       # Record the loss:
       self.train_punzi_loss_tracker.update(punzi_loss)
       # Record the accuracy:
       self.train_punzi_acc_tracker.update(model_predictions,y)

       return punzi_loss
    
    #-----------------------

    # Testing:
    def punzi_test_step(self,model,punzi_test_data):
       x,y,w,s = punzi_test_data
       # Get the model predictions:
       model_predictions = torch.squeeze(model(x))
       # Compute punzi loss:
       punzi_loss = torch.sum(torch.mean(self.punzi_loss.compute(s,model_predictions,w,self.n_gen_signal,self.target_luminosity)))

        # Register the predictions via the trackers:

       # Record the loss:
       self.test_punzi_loss_tracker.update(punzi_loss)
       # Record the accuracy:
       self.test_punzi_acc_tracker.update(model_predictions,y)

       return punzi_loss
    #********************
    
    # Store the model at a given epoch, i.e. take a snapshot:
    #********************
    def take_model_snapshot(self,model,path,current_epoch,n_epochs):
        epoch_str = str(current_epoch) + 'epochs'
        epoch_str = epoch_str.zfill(6 + len(str(n_epochs)))
        
        # Store the entire model using torch script, if provided by the user:
        if self.store_scripted_model:
           scripted_model = torch.jit.script(model)
           scripted_model.save(path+"_"+epoch_str+".pt")
        else:
           torch.save(model.state_dict(),path+"_weights_"+epoch_str+".pt")
    #********************

    # Now put it all together:
    #********************
    def run(self,model,bce_optimizer,punzi_optimizer,bce_lr_scheduler,punzi_lr_scheduler,x,y,w,s,x_test,y_test,w_test,s_test,n_epochs_bce,batch_size_bce,mon_epochs_bce,read_epochs_bce,snapshot_epochs_bce,n_epochs_punzi,batch_size_punzi,mon_epochs_punzi,read_epochs_punzi,snapshot_epochs_punzi):
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
              bce_training_data = self.get_data_batches([x,y,w],batch_size_bce,self.torch_device)
              # Update model parameters:
              bce_train_loss = self.bce_train_step(model,bce_optimizer,bce_training_data)
              
              # Test, if test data is available:
              if has_test_data:
                 bce_test_data = self.get_data_batches([x_test,y_test,w_test],batch_size_bce,self.torch_device)
                 bce_test_loss = self.bce_test_step(model,bce_test_data)

              # Update the learning rate scheduler if existent:
              if bce_lr_scheduler is not None:
                 if has_test_data:
                    bce_lr_scheduler.step(bce_test_loss)
                 else:
                    bce_lr_scheduler.step(bce_train_loss)

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

              # Take a snapshot of the model during training:
              if bce_epoch % snapshot_epochs_bce == 0:
                 self.take_model_snapshot(model,self.snapshot_folder+"/bce_model_snapshot",bce_epoch,n_epochs_bce)
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
              punzi_training_data = self.get_data_batches([x,y,w,s],batch_size_punzi,self.torch_device)
              # Update model parameters:
              train_loss = self.punzi_train_step(model,punzi_optimizer,punzi_training_data)
              

              if has_test_data:
                 punzi_testing_data = self.get_data_batches([x_test,y_test,w_test,s_test],batch_size_punzi,self.torch_device)
                 # Run a test step:
                 test_loss = self.punzi_test_step(model,punzi_testing_data)

              # Update punzi learning rate scheduler, if existent:
              if punzi_lr_scheduler is not None:
                 if has_test_data:
                    punzi_lr_scheduler.step(test_loss)
                 else:
                    punzi_lr_scheduler.step(train_loss)

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

              # Take a snapshot of the model during training:
              if punzi_epoch % snapshot_epochs_punzi == 0:
                 self.take_model_snapshot(model,self.snapshot_folder+"/punzi_model_snapshot",punzi_epoch,n_epochs_punzi)
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
