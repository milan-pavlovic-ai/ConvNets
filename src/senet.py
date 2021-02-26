
import random
import numpy as np
import scipy.stats as stats

import torch
import torch.nn as nn
import torch.nn.functional as F

from mngrdata import DataMngr
from mngrplot import PlotMngr
from mngrtune import Tuner
from settings import Settings, HyperParamsDistrib
from basemodel import MultiClassBaseModel, Conv2dBlock


class SENet(MultiClassBaseModel):
    """
    SENet - Squeeze-and-Excitation Network
        Modifications
            - Added dropout layer after global average pooling

    Source: Squeeze-and-Excitation Networks
            https://arxiv.org/pdf/1709.01507.pdf

            Aggregated Residual Transformations for Deep Neural Networks
            https://arxiv.org/pdf/1611.05431.pdf
    """

    # Configuration for each residual block
    #   in format (num_filters, num_res_blocks, stride_for_first_conv2d_in_res_block)
    config = {
        '26': [(128, 2, 1), (256, 2, 2), (512, 2, 2), (1024, 2, 2)],
        '50': [(128, 3, 1), (256, 4, 2), (512, 6, 2), (1024, 3, 2)],
        '101': [(128, 3, 1), (256, 4, 2), (512, 23, 2), (1024, 3, 2)],
        '152': [(128, 3, 1), (256, 8, 2), (512, 36, 2), (1024, 3, 2)]
    }

    def __init__(self, setting):
        super().__init__(setting)

        # Features
        self.features = self.make_feature_layers()

        # Classifier
        self.classifier = self.make_classifier_layers()

        # Initialize parameters of layers
        if self.setting.init_params:
            self.init_params()

        return

    def make_feature_layers(self):
        """
        Create feature layers
        """
        # Configuration
        config = SENet.config[str(self.setting.kind)]
        block_type = SEBottleneck
        expansion = 2
        reduction = 16

        # Layers
        layers = []

        # Convolution and Max-Pool
        layers += [self.conv2d_block(num_filters=64, kernel_size=7, stride=2, padding=3)]
        layers += [self.maxpool2d(kernel_size=3, stride=2, padding=1)]

        # SE blocks
        for cfg_block in config:
            num_filters, num_repeat, stride = cfg_block
            
            # First se-block reduce height and weight with stride
            layers += [self.se_block(block_type, num_filters, expansion, reduction, stride)]

            # Add rest se-blocks
            for _ in range(1, num_repeat):
                layers += [self.se_block(block_type, num_filters, expansion, reduction)]
                
        # Adaptive Average-Pool
        layers += [self.adapt_avgpool2d(output_size=1)]

        return nn.Sequential(*layers)

    def se_block(self, block_type, num_filters, expansion, reduction, stride=1, cardinality=32):
        """
        Create residual block
        """
        # Dimensions synchronize
        if stride != 1 or self.in_channels != num_filters * expansion:
            input_channels = self.in_channels
            dim_synch = self.conv2d_block(
                self.in_channels, 
                num_filters * expansion, 
                set_output=True, 
                activation=False, 
                kernel_size=1, 
                stride=stride)
            self.in_channels = input_channels
        else:
            dim_synch = None
        
        # Create a block
        layers = block_type(self, 
            num_filters=num_filters, 
            expansion=expansion, 
            reduction=reduction,
            dim_synch=dim_synch, 
            stride=stride,
            groups=cardinality)
        
        return layers

    def make_classifier_layers(self):
        """
        Create classifier layers
        """
        layers = nn.Sequential(
            nn.Dropout(p=self.setting.dropout_rate),
            nn.Linear(self.num_flat_features(), self.setting.num_classes)
        )
        return layers

    def forward(self, x):
        """
        Forward propagation
        """
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Residual block
    """

    def __init__(self, network, num_filters, expansion, reduction, dim_synch=None, **kwargs):
        """
        Initialize layers
        """
        super().__init__()
        self.dim_synch = dim_synch
        self.se_block = nn.Sequential(
            Conv2dBlock(network, num_filters=num_filters, kernel_size=3, padding=1, **kwargs),
            Conv2dBlock(network, num_filters=num_filters * expansion, activation=False, kernel_size=3, padding=1),
            SEUnit(network, reduction)
        )
        return

    def forward(self, x):
        """
        Forward propagation
        """
        # Save previous state
        identity = x
        
        # Calculate block
        output = self.se_block(x)

        # Dimension synchronization
        if self.dim_synch is not None:
            identity = self.dim_synch(identity) 

        # Add shortcut / skip-connection
        output += identity

        # Activation
        output = F.relu(output)

        return output

class SEBottleneck(nn.Module):
    """
    Squeeze-and-Excitation Residual bottleneck
    """

    def __init__(self, network, num_filters, expansion, reduction, dim_synch=None, **kwargs):
        """
        Initialize layers
        """
        super().__init__()
        self.dim_synch = dim_synch
        self.se_bottleneck = nn.Sequential(
            Conv2dBlock(network, num_filters=num_filters, kernel_size=1),
            Conv2dBlock(network, num_filters=num_filters, kernel_size=3, padding=1, **kwargs),
            Conv2dBlock(network, num_filters=num_filters * expansion, activation=False, kernel_size=1),
            SEUnit(network, reduction)
        )
        return

    def forward(self, x):
        """
        Forward propagation
        """
        # Save previous state
        identity = x
        
        # Calculate bottleneck
        output = self.se_bottleneck(x)

        # Dimension synchronization
        if self.dim_synch is not None:
            identity = self.dim_synch(identity) 

        # Add shortcut / skip-connection
        output += identity

        # Activation
        output = F.relu(output)

        return output

class SEUnit(nn.Module):
    """
    Squeeze-and-Excitation unit
        After this unit the size of tensor is the same as before
    """

    def __init__(self, network, reduction, **kwargs):
        """
        Initialize layers
        """
        super().__init__()
        num_features = network.in_channels
        num_features_reduct = num_features // reduction

        self.squeeze = network.adapt_avgpool2d(set_output=False, output_size=1)

        self.excitation = nn.Sequential(
            nn.Linear(num_features, num_features_reduct, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(num_features_reduct, num_features, bias=False),
            nn.Sigmoid()
        )
        return

    def forward(self, x):
        """
        Forward propagation
        """
        batch_size, num_channels, _, _ = x.size()
        
        # Squeeze
        output = self.squeeze(x).view(batch_size, num_channels)
        
        # Exicitation
        output = self.excitation(output).view(batch_size, num_channels, 1, 1)
        
        # Scale
        output = x * output.expand_as(x)

        return output


def process_eval(model, trainset, validset, testset, tuning=False, results=None):
    """
    Process evaluation process
    """
    # Trained epochs
    train_epochs = model.epoch_results['train_epochs']
    print('Trained epochs: {}'.format(train_epochs))

    # Training time
    train_time = float(model.epoch_results['train_time']) / 60
    print('Training time: {:.2f} min'.format(train_time))

    # Plot training performance of best model
    plot = PlotMngr()
    plot.performance(model.epoch_results)

    # Evaluate model on traning set
    model.evaluate(trainset)
    plot.confusion_matrix(model.confusion_matrix)

    # Evaluate model on validation set
    model.evaluate(validset)
    plot.confusion_matrix(model.confusion_matrix)

    # Final evaluation on test set
    scores, times, fps = model.test(testset)
    plot.confusion_matrix(model.confusion_matrix)

    # Plot tuning process
    if tuning:
        if 'tuning_results' in results:
            results = results['tuning_results']
        plot.hyperparameters(results, model.setting.get_hparams_names())
    return

def process_fit():
    """
    Process the procedure of fitting model
    """
    # Create settings
    setting = Settings(
        kind=50,
        input_size=(3, 32, 32),
        num_classes=10,
        # Batch
        batch_size=256,
        batch_norm=True,
        # Epoch
        epochs=3,
        # Learning rate
        learning_rate=0.01,
        lr_factor=0.1,
        lr_patience=10,
        # Regularization
        weight_decay=1E-5,
        dropout_rate=0.5,
        # Metric
        loss_optim=False,
        # Data
        data_augment=False,
        # Early stopping
        early_stop=True,
        es_patience=15,
        # Gradient clipping
        grad_clip_norm=False,
        gc_max_norm=1,
        grad_clip_value=False,
        gc_value=1,
        # Initialization
        init_params=True,
        # Distributions
        distrib=None,
        # Environment
        num_workers=16,
        mixed_precision=True,
        test_sample_size=90,
        seed=21,
        sanity_check=False,
        debug=False)

    # Load data
    data = DataMngr(setting)
    trainset = data.load_train()
    validset = data.load_valid()

    # Create net
    model = SENet(setting)
    setting.device.move(model)
    model.print_summary(additional=False)

    # Train model
    model.fit(trainset, validset)

    # Evaluate model
    testset = data.load_test()
    process_eval(model, trainset, validset, testset, tuning=False, results=None)
    
    return

def process_tune():
    """
    Process the procedure of tuning model
    """
    # Hyper-parameters search space
    distrib = HyperParamsDistrib(
        # Batch
        batch_size      = [256],
        batch_norm      = [True],
        # Epoch
        epochs          = [3],
        # Learning rate
        learning_rate   = list(np.logspace(np.log10(0.01), np.log10(0.1), base=10, num=1000)),
        lr_factor       = list(np.logspace(np.log10(0.05), np.log10(0.5), base=10, num=1000)),
        lr_patience     = [10],
        # Regularization
        weight_decay    = list(np.logspace(np.log10(0.001), np.log10(0.001), base=10, num=1000)),
        dropout_rate    = stats.uniform(0.5, 0),
        # Metric
        loss_optim      = [False],
        # Data
        data_augment    = [True],
        data_norm       = [True],
        # Early stopping
        early_stop      = [True],
        es_patience     = [15],
        # Gradient clipping
        grad_clip_norm  = [False],
        gc_max_norm     = [1],
        grad_clip_value = [False],
        gc_value        = [10],
        # Initialization
        init_params     = [True]
    )

    # Create settings
    setting = Settings(
        kind=50,
        input_size=(3, 32, 32),
        num_classes=10,
        distrib=distrib,
        num_workers=16,
        mixed_precision=True,
        test_sample_size=90,
        seed=21,
        sanity_check=False,
        debug=False)

    # Create tuner
    tuner = Tuner(SENet, setting)

    # Search for best model in tuning process
    model, results = tuner.process(num_iter=1)

    # Load data for evaluation
    data = DataMngr(model.setting)
    trainset = data.load_train()
    validset = data.load_valid()
    testset = data.load_test()

    # Evaluate model
    process_eval(model, trainset, validset, testset, tuning=True, results=results)
    
    return

def process_load(path, resume=False, testing=False):
    """
    Process loading and resume training
    """
    # Create settings
    setting = Settings(
        kind=50,
        input_size=(3, 32, 32),
        num_classes=10,
        batch_size=256,
        num_workers=16,
        mixed_precision=True,
        test_sample_size=90,
        seed=21,
        sanity_check=False,
        debug=False)

    # Load checkpoint
    model = SENet(setting)
    model.setting.device.move(model)
    states = model.load_checkpoint(path=path)
    model.setting.show()

    # Load data
    data = DataMngr(model.setting)
    trainset = data.load_train()
    validset = data.load_valid()

    # Resume training
    if resume:
        model.setting.epochs = 100
        model.setting.show()
        model.fit(trainset, validset, resume=resume)

    # Evaluate model
    testset = data.load_test()
    if testing:
        scores, times, fps = model.test(testset)
        return model.model_name, scores
    process_eval(model, trainset, validset, testset, tuning=True, results=states)

    return


if __name__ == "__main__":
    
    #process_fit()

    process_tune()

    #process_load(path='data/output/SEResNet26-1600789031-tuned.tar', resume=False)