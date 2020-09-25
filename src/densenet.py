
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


class DenseNet(MultiClassBaseModel):
    """
    DenseNet - Densely Connected Convolutional Networks 
            Effective implementation with bottleneck layers and compression
            Modifications
                Move dropout layer before convolution layer according to https://arxiv.org/pdf/1904.03392.pdf
                Added dropout layer after global average pooling

    Source: Densely Connected Convolutional Networks
            https://arxiv.org/pdf/1608.06993.pdf
            Configuration of DenseNet 161 model
            https://github.com/liuzhuang13/DenseNet
    """

    # Configuration 
    #   in format (growth_rate, list_of_dense_block_size, num_init_features) 
    config = {
        '121': (32, [6, 12, 24, 16], 64),
        '169': (32, [6, 12, 32, 32], 64),
        '201': (32, [6, 12, 48, 32], 64),
        '264': (32, [6, 12, 64, 48], 64),
        '161': (48, [6, 12, 36, 24], 96) 
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
        config = DenseNet.config[str(self.setting.kind)]
        growth_rate, dense_blocks_list, num_init_features = config

        # Layers
        layers = []

        # Construct initial features
        layers += [self.conv2d_block(num_filters=num_init_features, kernel_size=7, stride=2, padding=3)]
        layers += [self.maxpool2d(kernel_size=3, stride=2, padding=1)]
                
        # Add dense blocks
        for i, dense_block_size in enumerate(dense_blocks_list):
            layers += [DenseBlock(self, dense_block_size, growth_rate, bottleneck_factor=4)]
            
            # Add transition block if it is not last dense block
            if i != len(dense_blocks_list) - 1:
                layers += [TransitionBlock(self, compression_factor=2)]

        # Construct final features
        layers += [nn.BatchNorm2d(num_features=self.in_channels)]
        layers += [nn.ReLU(inplace=True)]
        layers += [self.adapt_avgpool2d(output_size=1)]

        return nn.Sequential(*layers)

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

class DenseLayer(nn.Module):
    """
    Dense layer
    """

    def __init__(self, network, growth_rate, bottleneck_factor):
        """
        Initalize layers
        """
        super().__init__()

        self.bottleneck = nn.Sequential(
            nn.BatchNorm2d(network.in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=network.setting.dropout_rate),
            network.conv2d(num_filters=bottleneck_factor * growth_rate, kernel_size=1)
        )

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(network.in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=network.setting.dropout_rate),
            network.conv2d(num_filters=growth_rate, kernel_size=3, padding=1)
        )
        return

    def forward(self, x):
        """
        Forward propagation
        """
        # Bottleneck layer
        output = self.bottleneck(x)
        output = self.conv_block(output)

        # Stack input and output, this will be input for next layer
        return torch.cat([x, output], dim=1)

class DenseBlock(nn.Module):
    """
    Dense block
    """

    def __init__(self, network, dense_block_size, growth_rate, bottleneck_factor):
        """
        Initialize layers
        """
        super().__init__()
        layers = []
        num_input_filters = network.in_channels

        # Create a dense block from dense layers
        for i in range(dense_block_size):
            layers += [DenseLayer(network, growth_rate, bottleneck_factor)]
            network.in_channels += num_input_filters + i * growth_rate
            if network.setting.debug:
                print('[{}] Input shape: ({} x {} x {})'.format(i+1, network.in_channels, network.height, network.width))

        self.dense_block = nn.Sequential(*layers)
        return

    def forward(self, x):
        """
        Forward propagation
        """
        return self.dense_block(x)

class TransitionBlock(nn.Sequential):
    """
    Transition block
    """

    def __init__(self, network, compression_factor):
        """
        Initialize layers
        """
        super().__init__()

        self.add_module('batch_norm', nn.BatchNorm2d(network.in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))

        num_filters = network.in_channels // compression_factor
        self.add_module('conv', network.conv2d(num_filters=num_filters, kernel_size=1))
        
        self.add_module(('avg_pool'), network.avgpool2d(kernel_size=2, stride=2))
        return


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
        kind='121',
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
        es_patience=12,
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
    model = DenseNet(setting)
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
        epochs          = [150],
        # Learning rate
        learning_rate   = list(np.logspace(np.log10(0.001), np.log10(0.1), base=10, num=1000)),
        lr_factor       = list(np.logspace(np.log10(0.01), np.log10(0.5), base=10, num=1000)),
        lr_patience     = [10],
        # Regularization
        weight_decay    = list(np.logspace(np.log10(0.00005), np.log10(0.005), base=10, num=1000)),
        dropout_rate    = stats.uniform(0.1, 0.4),
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
        kind='121',
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
    tuner = Tuner(DenseNet, setting)

    # Search for best model in tuning process
    model, results = tuner.process(num_iter=3)

    # Load data for evaluation
    data = DataMngr(setting)
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
        kind='121',
        input_size=(3, 32, 32),
        num_classes=10,
        num_workers=16,
        mixed_precision=True,
        test_sample_size=90,
        seed=21,
        sanity_check=False,
        debug=False)

    # Load checkpoint
    model = DenseNet(setting)
    model.setting.device.move(model)
    states = model.load_checkpoint(path=path)
    model.setting.show()

    # Load data
    data = DataMngr(model.setting)
    trainset = data.load_train()
    validset = data.load_valid()

    # Resume training
    if resume:
        model.setting.epochs = 2
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

    #process_tune()

    process_load(path='data/output/DenseNet121-1600733395-tuned.tar', resume=False)