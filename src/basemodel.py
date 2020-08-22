
import os
import copy
import time as t
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
from torchsummary import summary
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from mngrdevice import DeviceMngr
from mngrdata import DataMngr
from mngrplot import PlotMngr
from settings import Settings, HyperParamsDistrib


class MultiClassBaseModel(nn.Module):
    """
    Base model for image multi-classification task
    """

    def __init__(self, setting):
        super(MultiClassBaseModel, self).__init__()
        self.model_name = self.__class__.__name__
        self.version = int(t.time())
        self.setting = setting
        self.model_path = self.create_checkpoint_path()

        # Available after training
        self.cost_function = nn.CrossEntropyLoss(reduction='sum')
        self.optimizer = None
        self.lr_scheduler = None
        self.epoch_results = None

        # Available after evaluation
        self.class_names = None
        self.confusion_matrix = None
        self.classification_report = None
        return

    def init_optimizer(self):
        """
        Initialize optimizer and learning-rate scheduler 
        Need to be called after initialization of network, final number of parameters in network need to be known
        """
        # Set optimizer
        self.optimizer = optim.Adam(
            params=self.parameters(),
            lr=self.setting.learning_rate, 
            weight_decay=self.setting.weight_decay)

        # Set learning rate schedule
        # This scheduler reads a metric quantity and if no improvement is seen for a patience number of epochs, the learning rate is reduced by a factor
        # Defines whether the metric quantity is increasing or decreasing during training
        mode = 'min' if self.setting.loss_optim else 'max'           
        self.lr_scheduler = ReduceLROnPlateau(
            optimizer=self.optimizer,
            mode=mode,                     
            factor=self.setting.lr_factor,
            patience=self.setting.lr_patience,
            verbose=True)
        return


    @torch.no_grad()
    def score(self, outputs, targets):
        """
        Score function used as metric for controling training process with training and validation set
        This method calculate the score based on the batch target and predicted values
        Count total number of correct predictions
        """
        # Calculate 
        _, predictions = torch.max(outputs, dim=1)          # returns (values, indices) for maximum values over given dimension, dim is the dimension to reduce
        total = len(predictions)                            # number instances in batch
        correct = (predictions == targets).sum().item()     # count total correct predictions

        # Debugging
        if self.setting.debug:
            print('[SCORE] \tTargets:{} \tOutputs:{}, \tPredictions:{}'.format(
                targets.requires_grad, outputs.requires_grad, predictions.requires_grad))

        return correct


    def gradient_clipping(self):
        """
        Clip gradients with given methods in settings attribute

        About Gradient clipping
            Gradient clipping norm
                The norm is computed over all gradients together, as if they were concatenated into a single vector. Gradients are modified in-place. [0]
                The average value of gradient norms is a good initial trial [1]
                Each gradient is multiplied with given max_norm / total norm, which returns this method  
            Gradient clipping value
                Clips gradient of an iterable of parameters at specified value. Gradients are modified in-place. [3]
                It's generally tuned as a hyperparameter with clipping range [-1 ... +1] [2]
            Source: 
                [0] https://pytorch.org/docs/master/generated/torch.nn.utils.clip_grad_norm_.html
                [1] https://towardsdatascience.com/what-is-gradient-clipping-b8e815cdfb48
                [2] https://www.reddit.com/r/MachineLearning/comments/3n8g28/gradient_clipping_what_are_good_values_to_clip_at/
                [3] https://pytorch.org/docs/master/generated/torch.nn.utils.clip_grad_value_.html
        """
        # Gradient clipping norm
        if self.setting.grad_clip_norm:

            if self.setting.debug:
                for p in self.parameters():
                    print('[BEFORE NORM CLIPING] Gradient:\n{}'.format(p.grad))
                    break

            total_norm = clip_grad_norm_(self.parameters(), max_norm=self.setting.gc_max_norm)

            if self.setting.debug:
                for p in self.parameters():
                    print('[AFTER NORM CLIPING] with total_norm={:.6f}\nGradient:\n{}'.format(total_norm, p.grad))
                    break

        # Gradient clipping value
        if self.setting.grad_clip_value:

            if self.setting.debug:
                for p in self.parameters():
                    print('[BEFORE VALUE CLIPING] Gradient:\n{}'.format(p.grad))
                    break

            clip_grad_value_(self.parameters(), clip_value=self.setting.gc_value)

            if self.setting.debug:
                for p in self.parameters():
                    print('[BEFORE VALUE CLIPING] Gradient:\n{}'.format(p.grad))
                    break
        return

    def forward_step(self, batch):
        """
        Forward propagation for one batch with calculation of loss and score
        """
        # Send batch data to device (GPU) and unpack it
        inputs, targets = self.setting.device.move(batch)

        # Calculate predictions
        outputs = self(inputs)

        # Calculate loss
        loss = self.cost_function(outputs, targets)

        # Calculate score
        score = self.score(outputs, targets)

        # Debugging
        if self.setting.debug:
            print('[FORWARD STEP] \tInputs:{}, \tTargets:{} \tOutputs:{}, \tLoss: {}'.format(
                inputs.requires_grad, targets.requires_grad, outputs.requires_grad, loss.requires_grad))

        return loss, score

    @torch.enable_grad()
    def training_step(self, batch):
        """
        Traning step as forward propagation for one batch with calculation of loss and score
        """
        # Forward propagation
        loss, score = self.forward_step(batch)

        # Debugging
        if self.setting.debug:
            print('[TRAIN STEP] \tLoss: {}'.format(loss.requires_grad))

        return loss, score

    @torch.no_grad()
    def validation_step(self, batch):
        """
        Validation step as forward propagation for one batch with calculation of loss and score
        """
        # Forward propagation
        loss, score = self.forward_step(batch)

        # Debugging
        if self.setting.debug:
            print('[VALID STEP] \tLoss: {}'.format(loss.requires_grad))

        return loss, score

    @torch.enable_grad()
    def train_model(self, dataloader):
        """
        Train model with given dataset
        """
        print('=== Training Phase ===')
        self.train()

        epoch_loss = 0
        epoch_score = 0

        for batch in tqdm(dataloader):
            # Clear the existing gradients though, otherwise gradients will be accumulated to follow gradients
            self.optimizer.zero_grad()

            # Forward propagation with loss and score calculation
            loss, score = self.training_step(batch)

            # Backpropagate the error, calculate the gradients
            loss.backward()

            # Gradient clipping
            self.gradient_clipping()

            # Update parameters using learning rate and calculated gradients
            self.optimizer.step()

            # Add accumulated losses and scores from this batch
            epoch_loss += loss.item()
            epoch_score += score

            # Debugging
            if self.setting.debug:
                print('[TRAIN MODEL] \tLoss: {}'.format(loss.requires_grad))

            # Perform only one batch in sanity-check mode
            if self.setting.sanity_check:
                break

        # Calculate average from accumulated losses and scores on entire dataset
        epoch_loss /= len(dataloader.dataset)
        epoch_score /= len(dataloader.dataset)

        return epoch_loss, epoch_score

    @torch.no_grad()
    def valid_model(self, dataloader):
        """
        Validate model with given dataset
        """
        print('=== Validation Phase ===')
        # This will change behaviour of layers such as dropout and batch-normalization
        self.eval()    

        epoch_loss = 0
        epoch_score = 0

        for batch in tqdm(dataloader):
            # Forward propagation with loss and score calculation
            loss, score = self.validation_step(batch)

            # Add accumulated losses and scores from this batch
            epoch_loss += loss.item()
            epoch_score += score

            # Debugging
            if self.setting.debug:
                print('[VALID MODEL] \tLoss: {}'.format(loss.requires_grad))

            # Perform only one batch in sanity-check mode
            if self.setting.sanity_check:
                break

        # Calculate average from accumulated losses and scores on entire dataset
        epoch_loss /= len(dataloader.dataset)
        epoch_score /= len(dataloader.dataset)

        return epoch_loss, epoch_score


    @torch.no_grad()
    def end_epoch(self, train_loss, train_score, valid_loss, valid_score, learning_rate, epoch):
        """
        Print results at end of epoch
        """
        self.epoch_results['train_loss'].append(train_loss)
        self.epoch_results['valid_loss'].append(valid_loss)

        self.epoch_results['train_score'].append(train_score)
        self.epoch_results['valid_score'].append(valid_score)

        self.epoch_results['learning_rate'].append(learning_rate)
        self.epoch_results['train_epochs'] = epoch

        # Print results
        print()
        print('EPOCH {}/{}'.format(epoch, self.setting.epochs))
        print('Train Loss: \t{:.6f} \tValid Loss: \t{:.6f}'.format(train_loss, valid_loss))
        print('Train Accuracy: {:.3f}% \tValid Accuracy: {:.3f}%'.format(train_score*100, valid_score*100))
        print('Learning rate:\t{}'.format(learning_rate))
        print()
        return

    def fit(self, trainset, validset, resuming=False):
        """
        Fit the given training dataset into the model with controling of training process with validation dataset by calculating losses and scores per epoch
        After this method, the instance from which this method is called hold reference to best-achieved model on the training process
        Returns model with best achieved metric on validation set
        """
        # Initialize
        best_valid_score = -1
        best_valid_loss = float('inf')
        best_params = copy.deepcopy(self.state_dict()) 
        epochs_no_improve = 0

        self.epoch_results = {'train_loss':[], 'train_score':[], 'valid_loss':[], 'valid_score':[], 'learning_rate':[], 'train_epochs':0}

        if not resuming:
            self.init_optimizer()

        # Training process
        print('\n=== START TRAINING ===\n')

        for epoch in range(self.setting.epochs):

            # Before epoch - initialization
            curr_learning_rate = self.get_learning_rate()

            # Training phase
            train_loss, train_score = self.train_model(trainset)

            # Validation phase
            valid_loss, valid_score = self.valid_model(validset)

            # After epoch - show performance on traning and validation set 
            self.end_epoch(train_loss, train_score, valid_loss, valid_score, curr_learning_rate, epoch + 1)

            # Save with best score
            if self.setting.loss_optim:
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    best_params = copy.deepcopy(self.state_dict())
                    self.save_checkpoint()
                    print('Best validation loss is achieved and copied current parameters')
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
            else:
                if valid_score > best_valid_score:
                    best_valid_score = valid_score
                    best_params = copy.deepcopy(self.state_dict())
                    self.save_checkpoint()
                    print('Best validation score is achieved and copied current parameters')
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

            # Update learning rate if there is no improvement
            if self.setting.loss_optim:
                self.lr_scheduler.step(valid_loss)
            else:
                self.lr_scheduler.step(valid_score)
            if curr_learning_rate != self.get_learning_rate():
                self.load_state_dict(best_params)
                print('No improvement after {} consecutive epochs, the learning rate is changed and training is continued with best parameters'.format(self.setting.lr_patience))

            # Early stopping
            if self.setting.early_stop and self.setting.es_patience + 1 == epochs_no_improve:
                print('Early stopped after trained {} epochs, no improvement in last {} consecutive epochs'.format(epoch + 1, epochs_no_improve))
                break

            print('\n')

        # Update epoch results after best model epoch
        best_checkpoint = self.load_checkpoint(path=self.model_path)                               # load best model environment
        self.epoch_results['train_epochs'] = best_checkpoint['epoch_results']['train_epochs']      # set number of trained epochs for best model
        best_checkpoint['epoch_results'] = self.epoch_results                                      # remeber epochs after best model
        self.update_checkpoint(best_checkpoint, path=self.model_path)

        print('\n=== TRAINING IS FINISHED ===\n')

        return self
    
    
    def inference_time(self, elapsed_times):
        """
        Print results of inference time
        """
        # TODO fix measurement
        elapsed_times = np.array(elapsed_times)
        device_type = self.setting.device.device.type
        batch_size = self.setting.batch_size

        total_time = np.sum(elapsed_times) * 1000
        mean_time = np.mean(elapsed_times / batch_size) * 1000
        std_time = np.std(elapsed_times / batch_size) * 1000

        print('\n')
        print('Inference time')
        print('\twith using device {} and with batch size {}'.format(device_type, batch_size))
        print('\tEntire dataset: \t{:.2f}s'.format(total_time))
        print('\tSingle instance: \t{:.2f}s ± {:.2f}s'.format(mean_time, std_time))
        return

    def eval_score(self, y_targets, y_preds):
        """
        Returns score of a model for a given target and predicted values. 
        This score will be used in the hyper-parameter tuning process to determine the model with better hyper-parameters.
        """
        accuracy = accuracy_score(y_targets, y_preds)
        print('', 'Accuracy: {:.2f}%'.format(accuracy*100), sep='\n')
        return accuracy

    @torch.no_grad()
    def evaluate(self, dataloader):
        """
        Evaluate model on given dataset

        About confusion matrix
            Source: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
        """
        # This will change behaviour of layers such as dropout and batch-normalization
        self.eval() 

        elapsed_times = []
        y_preds = np.array([], dtype=int)
        y_targets = np.array([], dtype=int)
        self.class_names = dataloader.dataset.classes
              
        # Evaluate on given dataset
        print('\n=== START EVALUATION ===\n')

        for batch in tqdm(dataloader):

            # Send batch data to device (GPU) and unpack it
            inputs, targets = self.setting.device.move(batch)

            # Calculate predictions
            start_time = t.time()   # TODO Fix measurement

            outputs = self(inputs)
            probs, predictions = torch.max(outputs, dim=1)

            elapsed_times.append(t.time() - start_time)

            # Move to CPU and convert to numpy array
            predictions = self.setting.device.move_to(predictions, 'cpu').numpy()
            y_preds = np.concatenate([y_preds, predictions], axis=0)

            targets = self.setting.device.move_to(targets, 'cpu').numpy()
            y_targets = np.concatenate([y_targets, targets], axis=0)

            # Debugging
            if self.setting.debug:
                print('Shape of predictions:', predictions.shape)
                print('Predictions:', predictions, sep='\n')
                print('Shape of y_preds:', y_preds.shape)
                print('Y_preds:', y_preds, sep='\n')

                print('Shape of targets:', targets.shape)
                print('Targets:', targets, sep='\n')
                print('Shape of y_targets:', y_targets.shape)
                print('Y_targets:', y_targets, sep='\n')
  
            # Perform only one batch in sanity-check mode
            if self.setting.sanity_check:
                break

        # Classification report
        self.classification_report = classification_report(y_targets, y_preds, target_names=self.class_names)
        print('', 'Classification report', self.classification_report, sep='\n')

        # Confusion matrix
        self.confusion_matrix = pd.DataFrame(
            data=confusion_matrix(y_targets, y_preds), 
            index=self.class_names, 
            columns=self.class_names,
            dtype=int)
        print('', 'Confusion matrix', '(rows = Actual, columns = Predicted)', '', self.confusion_matrix, sep='\n')

        # Inference time
        self.inference_time(elapsed_times)

        # Single score
        score = self.eval_score(y_targets, y_preds)

        print('\n=== EVALUATION IS FINISHED ===\n')

        return score


    def get_conv_outshape(self, H_in, W_in, layer):
        """
        Calculate height and width of output image after Conv2D or Pool2D operation
        Source: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
                https://pytorch.org/docs/master/generated/torch.nn.MaxPool2d.html
        """
        # Kernel size
        if type(layer.kernel_size) is int:
            kernel_size_H = layer.kernel_size
            kernel_size_W = layer.kernel_size
        else:
            kernel_size_H, kernel_size_W = layer.kernel_size

        # Stride
        if type(layer.stride) is int:
            stride_H = layer.stride
            stride_W = layer.stride
        else:
            stride_H, stride_W = layer.stride

        # Padding
        if type(layer.padding) is int:
            padding_H = layer.padding
            padding_W = layer.padding
        else:
            padding_H, padding_W = layer.padding

        # Dilation
        if type(layer.dilation) is int:
            dilation_H = layer.dilation
            dilation_W = layer.dilation
        else:
            dilation_H, dilation_W = layer.dilation

        # Calculate output height and weight
        H_out = int(np.floor(1 + (H_in + 2*padding_H - dilation_H * (kernel_size_H-1) - 1) / stride_H))
        W_out = int(np.floor(1 + (W_in + 2*padding_W - dilation_W * (kernel_size_W-1) - 1) / stride_W))
        
        # Print new dimensions
        if self.setting.debug:
            print('After {}: H={}, W={}'.format(layer._get_name(), H_out, W_out))

        return H_out, W_out

    def num_flat_features(self, x):
        """
        Returns number of neurons/parameters from non-flat layer
        """
        dimensions = x.size()[1:]  # get all dimensions except the batch dimension
        num_features = 1
        for dim in dimensions:
            num_features *= dim
        return num_features

    def get_learning_rate(self):
        """
        Returns current learning rate from optimizer
        """
        for param_group in self.optimizer.param_groups:
            return param_group['lr']


    def create_checkpoint_path(self, suffix=None, version=None):
        """
        Creates a checkpoint unique name and returns its path
        """
        # Create suffix
        if suffix is None:
            suffix = 'best_loss' if self.setting.loss_optim else 'best_score'

        # Crea version
        if version is None:
            version = self.version

        # Create name and path
        checkpoint_name = '{}-{}-{}.tar'.format(self.model_name, version, suffix)
        checkpoint_path = os.path.join(DataMngr.OUTPUT_DIR, checkpoint_name)

        return checkpoint_path

    def update_checkpoint(self, checkpoint, suffix=None, path=None):
        """
        Update given checkpoint
        """
        # Create name and path
        if path is None:
            path = self.create_checkpoint_path(suffix=suffix)

        # Update checkpoint
        torch.save(checkpoint, path)
        return

    def save_checkpoint(self, suffix=None, version=None, path=None):
        """
        Save current state to be used for either inference or resuming training
        Source: https://pytorch.org/tutorials/beginner/saving_loading_models.html
        """
        # Create name and path
        if path is None:
            path = self.create_checkpoint_path(suffix=suffix, version=version)

        # Collect current states
        checkpoint = {
            'epoch_results': self.epoch_results,                # information about each epoch
            'hparams': self.setting.get_hparams(),              # model hyper-parameters
            'model': self.state_dict(),                         # model parameters
            'optimizer': self.optimizer.state_dict(),           # optimizer parameters
            'lr_scheduler': self.lr_scheduler.state_dict()}     # lr_schduler parameters

        # Save checkpoint
        torch.save(checkpoint, path)
        return

    def get_last_checkpoint(self, suffix=None):
        """
        Returns last saved checkpoint
        """
        # Create suffix
        if suffix is None:
            suffix = 'best_loss' if self.setting.loss_optim else 'best_score'

        # Find last checkpoint by version
        last_version = -1
        for file_name in os.listdir(DataMngr.OUTPUT_DIR):
            if file_name.startswith(self.model_name) and suffix in file_name:
                curr_version = int(file_name.split('-')[1])
                if last_version < curr_version:
                    last_version = curr_version

        # Set path for last checkpoint
        last_checkpoint = '{}-{}-{}.tar'.format(self.model_name, last_version, suffix)
        path = os.path.join(DataMngr.OUTPUT_DIR, last_checkpoint)

        # Load checkpoint
        checkpoint = torch.load(path)

        return checkpoint

    def load_checkpoint(self, suffix=None, path=None):
        """
        Load saved checkpoint to be used for either inference or resuming training
        To continue training method fit() must be called with flag resuming=True
        Source: https://pytorch.org/tutorials/beginner/saving_loading_models.html
        """
        # Get checkpoint
        if path is None:
            checkpoint = self.get_last_checkpoint(suffix)
        else:
            checkpoint = torch.load(path)

        # Initalization
        self.init_optimizer()

        # Load parameters for each component
        self.setting.load_values(checkpoint['hparams'])
        self.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        self.setting.device.move(self)

        # Contains information about each epoch
        return checkpoint

    def print_summary(self, additional=True):
        """
        Print summary of model including layers, shapes and number of parameters
        """
        model_summary = summary(self, 
            input_size=self.setting.input_size, 
            batch_size=self.setting.batch_size,
            device=self.setting.device.device.type)
        print(model_summary)

        if additional:
            print(self)
        return



class ConvNet(MultiClassBaseModel):
    """
    Convolutional neural network
    """

    def __init__(self, setting):
        super().__init__(setting)
        self.model_name = self.__class__.__name__
        ch, h, w = self.setting.input_size

        # Features
        self.conv1 = nn.Conv2d(
            in_channels=ch, 
            out_channels=32, 
            kernel_size=3)
        h, w = self.get_conv_outshape(h, w, self.conv1)

        self.maxpool1 = nn.MaxPool2d(
            kernel_size=(2, 2),
            stride=(2, 2))
        h, w = self.get_conv_outshape(h, w, self.maxpool1)

        self.conv2 = nn.Conv2d(
            in_channels=32, 
            out_channels=64, 
            kernel_size=5,
            stride=2, 
            padding=1)
        h, w = self.get_conv_outshape(h, w, self.conv2)

        self.maxpool2 = nn.MaxPool2d(
            kernel_size=2,
            stride=2)
        h, w = self.get_conv_outshape(h, w, self.maxpool2)

        # Classifier
        self.num_flatten = h * w * 64
        self.fc1 = nn.Linear(self.num_flatten, 2048)
        self.fc2 = nn.Linear(2048, self.setting.num_classes)

        # Initialize weights
        self.init_weights()
        return

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)

        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)
        
        x = x.view(-1, self.num_flatten)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.setting.dropout_rate, training=self.training)

        x = self.fc2(x)
        return x

    def init_weights(self):
        """
        Weight initialization
        The general rule for setting the weights in a neural network is to set them to be close to zero without being too small
        """
        for m in self.modules():

            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        return

if __name__ == "__main__":

    # Create settings
    setting = Settings(
        input_size=(3, 32, 32),
        num_classes=10,
        batch_size=512,
        batch_norm=False,
        epochs=10,
        learning_rate=0.001,
        lr_factor=0.1,
        lr_patience=5,
        weight_decay=1E-5,
        dropout_rate=0.5,
        loss_optim=True,
        data_augment=False,
        early_stop=True,
        es_patience=10,
        grad_clip_norm=False,
        gc_max_norm=100,
        grad_clip_value=False,
        gc_value=100,
        distrib=None,
        sanity_check=False,
        debug=False)

    # Load data
    data = DataMngr(setting)
    trainset = data.load_train()
    validset = data.load_valid()

    # Create net
    convnet = ConvNet(setting)
    setting.device.move(convnet)
    convnet.print_summary()

    # Train model
    #convnet.fit(trainset, validset)

    # Plot training performance
    plot = PlotMngr()
    #plot.performance(convnet.epoch_results)

    # Load best model
    states = convnet.load_checkpoint()

    # Plot training performance
    plot.performance(states['epoch_results'])
    
    # Evaluate model on traning set
    convnet.evaluate(trainset)
    plot.confusion_matrix(convnet.confusion_matrix)

    # Evaluate model on validation set
    convnet.evaluate(validset)
    plot.confusion_matrix(convnet.confusion_matrix)
    
    # TODO
    # Final test
    #convnet.evaluate(testset)
    #plot.confusion_matrix(convnet.confusion_matrix)
    