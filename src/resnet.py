
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
from basemodel import MultiClassBaseModel


class ResNet(MultiClassBaseModel):
    """
    ResNet - Residual Network

    Source: Deep Residual Learning for Image Recognition
            https://arxiv.org/pdf/1512.03385.pdf
    """

    # Configuration for each residual block
    #   in format (num_filters, num_res_blocks, stride_for_first_conv2d_in_res_block)
    config = {
        '18': ('basic', [(64, 2, 1), (128, 2, 2), (256, 2, 2), (512, 2, 2)]), 
        '34': ('basic', [(64, 3, 1), (128, 4, 2), (256, 6, 2), (512, 3, 2)]),
        '50': ('bottleneck', [(64, 3, 1), (128, 4, 2), (256, 6, 2), (512, 3, 2)]),
        '101': ('bottleneck', [(64, 3, 1), (128, 4, 2), (256, 23, 2), (512, 3, 2)]),
        '152': ('bottleneck', [(64, 3, 1), (128, 8, 2), (256, 36, 2), (512, 3, 2)])
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
        block_type_str, config = ResNet.config[str(self.setting.kind)]
        block_type = ResBottleneck if block_type_str == 'bottleneck' else ResBlock
        expansion = 4 if block_type_str == 'bottleneck' else 1

        # Layers
        layers = []

        # Convolution and Max-Pool
        layers += [self.conv2d_block(num_filters=64, kernel_size=7, stride=2, padding=3)]
        layers += [self.maxpool2d(kernel_size=3, stride=2, padding=1)]

        # Residual blocks
        for cfg_block in config:
            num_filters, num_repeat, stride = cfg_block
            
            # First residual block reduce height and weight with stride
            layers += [self.residual_block(block_type, num_filters, expansion, stride)]

            # Add rest residual blocks
            for _ in range(1, num_repeat):
                layers += [self.residual_block(block_type, num_filters, expansion)]
                
        # Adaptive Average-Pool
        layers += [self.adapt_avgpool2d(output_size=1)]

        return nn.Sequential(*layers)

    def residual_block(self, block_type, num_filters, expansion, stride=1):
        """
        Create residual block
        """
        # Dimensions synchronize
        if stride != 1 or self.in_channels != num_filters * expansion:
            dim_synch = self.conv2d_block(self.in_channels, num_filters * expansion, set_output=False, activation=False, kernel_size=1, stride=stride)
        else:
            dim_synch = None
        
        # Create a block
        layers = block_type(self, num_filters=num_filters, expansion=expansion, dim_synch=dim_synch, stride=stride)
        
        return layers

    def make_classifier_layers(self):
        """
        Create classifier layers
        """
        layers = nn.Sequential(
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

class ResBlock(nn.Module):
    """
    Residual block
    """

    def __init__(self, network, num_filters, expansion, dim_synch=None, **kwargs):
        """
        Initialize layers
        """
        super().__init__()
        self.dim_synch = dim_synch
        self.res_block = nn.Sequential(
            network.conv2d_block(num_filters=num_filters, kernel_size=3, padding=1, **kwargs),
            network.conv2d_block(num_filters=num_filters, activation=False, kernel_size=3, padding=1)
        )
        return

    def forward(self, x):
        """
        Forward propagation
        """
        # Save previous state
        identity = x
        
        # Calculate block
        output = self.res_block(x)

        # Dimension synchronization
        if self.dim_synch is not None:
            identity = self.dim_synch(identity) 

        # Add shortcut / skip-connection
        output += identity

        # Activation
        output = F.relu(output)

        return output

class ResBottleneck(nn.Module):
    """
    Residual bottleneck
    """

    def __init__(self, network, num_filters, expansion, dim_synch=None, **kwargs):
        """
        Initialize layers
        """
        super().__init__()
        self.dim_synch = dim_synch
        self.res_bottleneck = nn.Sequential(
            network.conv2d_block(num_filters=num_filters, kernel_size=1, **kwargs),
            network.conv2d_block(num_filters=num_filters, kernel_size=3, padding=1),
            network.conv2d_block(num_filters=num_filters * expansion, activation=False, kernel_size=1)
        )
        return

    def forward(self, x):
        """
        Forward propagation
        """
        # Save previous state
        identity = x
        
        # Calculate bottleneck
        output = self.res_bottleneck(x)

        # Dimension synchronization
        if self.dim_synch is not None:
            identity = self.dim_synch(identity) 

        # Add shortcut / skip-connection
        output += identity

        # Activation
        output = F.relu(output)

        return output


def process_eval(model, trainset, validset, testset, tuning=False, results=None):
    """
    Process evaluation process
    """
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
        kind=101,
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
        sanity_check=True,
        debug=True)

    # Load data
    data = DataMngr(setting)
    trainset = data.load_train()
    validset = data.load_valid()

    # Create net
    model = ResNet(setting)
    setting.device.move(model)
    model.print_summary()

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
        learning_rate   = list(np.logspace(np.log10(0.0001), np.log10(0.01), base=10, num=1000)),
        lr_factor       = list(np.logspace(np.log10(0.01), np.log10(1), base=10, num=1000)),
        lr_patience     = [10],
        # Regularization
        weight_decay    = list(np.logspace(np.log10(0.0009), np.log10(0.9), base=10, num=1000)),
        dropout_rate    = stats.uniform(0.3, 0.65),
        # Metric
        loss_optim      = [False],
        # Data
        data_augment    = [False],
        # Early stopping
        early_stop      = [True],
        es_patience     = [12],
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
        kind=101,
        input_size=(3, 32, 32),
        num_classes=10,
        distrib=distrib,
        num_workers=16,
        mixed_precision=True,
        test_sample_size=90,
        seed=21,
        sanity_check=False,
        debug=False)

    # Load data for evaluation
    data = DataMngr(setting)
    trainset = data.load_train()
    validset = data.load_valid()

    # Create tuner
    tuner = Tuner(ResNet, setting)

    # Search for best model in tuning process
    model, results = tuner.process(num_iter=3)

    # Evaluate model
    testset = data.load_test()
    process_eval(model, trainset, validset, testset, tuning=True, results=results)
    
    return

def process_load(resume_training=False):
    """
    Process loading and resume training
    """
    # Create settings
    setting = Settings(
        kind=101,
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
    model = ResNet(setting)
    model.setting.device.move(model)
    states = model.load_checkpoint(path='data/output/ResNet101-1600028289-tuned.tar', strict=False)
    model.setting.show()

    # Load data
    data = DataMngr(model.setting)
    trainset = data.load_train()
    validset = data.load_valid()

    # Resume training
    if resume_training:
        model.setting.epochs = 2
        model.setting.show()
        model.fit(trainset, validset, resume=True)

    # Evaluate model
    testset = data.load_test()
    process_eval(model, trainset, validset, testset, tuning=True, results=states)

    return


if __name__ == "__main__":
    
    #process_fit()

    #process_tune()

    process_load(resume_training=False)