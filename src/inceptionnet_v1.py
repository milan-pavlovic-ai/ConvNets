
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


class InceptionNetV1(MultiClassBaseModel):
    """
    Inception v1 - also known as GoogLeNet
        Modifications
            - Added Batch Normalization after each Convolutional layer
            - Added Padding with size of 1 for every Max-Pooling layer, to be able to handle small images
            - Removed Auxiliary classifiers connected to intermediate layer, they're not really useful and also mess up implementation

    Source: Going deeper with convolutions
            https://arxiv.org/pdf/1409.4842.pdf
    """

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
        Create features layers
        """
        layers = []

        # [1] Convolution
        layers += [self.conv2d_block(num_filters=64, kernel_size=7, stride=2, padding=3)]
        layers += [self.maxpool2d(kernel_size=3, stride=2, padding=1)]
        
        # [2] Convolution
        layers += [self.conv2d_block(num_filters=64, kernel_size=1)]
        layers += [self.conv2d_block(num_filters=192, kernel_size=3, padding=1)]
        layers += [self.maxpool2d(kernel_size=3, stride=2, padding=1)]

        # [3] Inception
        layers += [InceptionBlock(self, 64, 96, 128, 16, 32, 32)]
        layers += [InceptionBlock(self, 128, 128, 192, 32, 96, 64)]
        layers += [self.maxpool2d(kernel_size=3, stride=2, padding=1)]

        # [4] Inception
        layers += [InceptionBlock(self, 192, 96, 208, 16, 48, 64)]
        layers += [InceptionBlock(self, 160, 112, 224, 24, 64, 64)]
        layers += [InceptionBlock(self, 128, 128, 256, 24, 64, 64)]
        layers += [InceptionBlock(self, 112, 144, 288, 32, 64, 64)]
        layers += [InceptionBlock(self, 256, 160, 320, 32, 128, 128)]
        layers += [self.maxpool2d(kernel_size=3, stride=2, padding=1)]

        # [5] Inception
        layers += [InceptionBlock(self, 256, 160, 320, 32, 128, 128)]
        layers += [InceptionBlock(self, 384, 192, 384, 48, 128, 128)]
        layers += [self.adapt_avgpool2d(output_size=1)]

        return nn.Sequential(*layers)

    def make_classifier_layers(self):
        """
        Create classifier layers
        """
        # Create a classifier
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

class InceptionBlock(nn.Module):
    """
    Inception block
    """

    def __init__(self, network, ch_conv1x1, ch_conv3x3_red, ch_conv3x3, ch_conv5x5_red, ch_conv5x5, pool_proj):
        """
        Initialize layers
        """
        super().__init__()

        self.branch1 = network.conv2d_block(network.in_channels, ch_conv1x1, set_output=False, kernel_size=1)

        self.branch2 = nn.Sequential(
            network.conv2d_block(network.in_channels, ch_conv3x3_red, set_output=False, kernel_size=1),
            network.conv2d_block(ch_conv3x3_red, ch_conv3x3, set_output=False, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            network.conv2d_block(network.in_channels, ch_conv5x5_red, set_output=False, kernel_size=1),
            network.conv2d_block(ch_conv5x5_red, ch_conv5x5, set_output=False, kernel_size=5, padding=2)
        )

        self.branch4 = nn.Sequential(
            network.maxpool2d(set_output=True, kernel_size=3, stride=1, padding=1),
            network.conv2d_block(num_filters=pool_proj, set_output=True, kernel_size=1)
        )

        network.in_channels = ch_conv1x1 + ch_conv3x3 + ch_conv5x5 + pool_proj
        
        return

    def forward(self, x):
        """
        Forward propagation
        """
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]

        return torch.cat(outputs, dim=1)


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
        kind='',
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
    model = InceptionNetV1(setting)
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
        learning_rate   = list(np.logspace(np.log10(0.0075), np.log10(0.0075), base=10, num=1000)),
        lr_factor       = list(np.logspace(np.log10(0.05), np.log10(0.5), base=10, num=1000)),
        lr_patience     = [10],
        # Regularization
        weight_decay    = list(np.logspace(np.log10(0.00005), np.log10(0.00005), base=10, num=1000)),
        dropout_rate    = stats.uniform(0, 0.5),
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
        kind='',
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
    tuner = Tuner(InceptionNetV1, setting)

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

def process_load(path, resume=False):
    """
    Process loading and resume training
    """
    # Create settings
    setting = Settings(
        kind='',
        input_size=(3, 32, 32),
        num_classes=10,
        num_workers=16,
        mixed_precision=True,
        test_sample_size=90,
        seed=21,
        sanity_check=False,
        debug=False)

    # Load checkpoint
    model = InceptionNetV1(setting)
    model.setting.device.move(model)
    states = model.load_checkpoint(path=path)
    model.setting.show()

    # Load data
    data = DataMngr(model.setting)
    trainset = data.load_train()
    validset = data.load_valid()

    # Resume training
    if resume:
        model.setting.epochs = 10
        model.setting.learning_rate = 0.0005
        model.update_learning_rate()
        model.setting.show()
        model.fit(trainset, validset, resume=resume)

    # Evaluate model
    testset = data.load_test()
    process_eval(model, trainset, validset, testset, tuning=True, results=states)

    return


if __name__ == "__main__":
    
    #process_fit()

    #process_tune()

    process_load(path='data/output/InceptionNetV1-1600908899-tuned.tar', resume=False)