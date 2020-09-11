
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


class VGGNet(MultiClassBaseModel):
    """
    VGGNet - Visual Geometry Group Network

    Source: Very deep convolutional networks for large-scale image recognition
            https://arxiv.org/pdf/1409.1556v6.pdf
    """

    config = {
        '11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        '13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        '16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        '19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
    }

    def __init__(self, setting):
        """
        Initialization of model
        """
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
        Create features layers from configuration by using convolution and max-pooling operation
        """
        layers = []

        # Construct layers from configuration
        for element in VGGNet.config[str(self.setting.kind)]:

            if element == 'M':
                # Max-pooling layer
                layer = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
                self.save_conv_outshape(layer)
                layers += [layer]
            else:
                # Convolutional layer
                layer = nn.Conv2d(self.in_channels, element, kernel_size=(3, 3), stride=1, padding=1)
                self.save_conv_outshape(layer)
                layers += [layer]

                # Batch normalization layer
                if self.setting.batch_norm:
                    layers += [nn.BatchNorm2d(num_features=self.in_channels)]

                # Activation layer
                layers += [nn.ReLU()]
                
        return nn.Sequential(*layers)

    def make_classifier_layers(self):
        """
        Create classifier layers by using linear operation with dropout regularization method
        """
        layers = nn.Sequential(
            nn.Linear(self.num_flat_features(), 4096),
            nn.ReLU(),
            nn.Dropout(p=self.setting.dropout_rate),

            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=self.setting.dropout_rate),

            nn.Linear(4096, self.setting.num_classes)
        )
        
        return layers

    def init_params(self):
        """
        Parameters initialization
        Source: https://pytorch.org/docs/stable/nn.init.html
        """
        for m in self.modules():
            # Convolutional layer, weights use normal He initialization, biases are 0
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
            # Batch Normalization layer, weights are 1, biases are 0 
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            # Fully Connected layer, weights have normal distribution with mean 0 and std 0.01, biases are 0
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        return

    def forward(self, x):
        """
        Forward propagation
        """
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x


def process_fit():
    """
    Process the procedure of fitting model
    """
    # Create settings
    setting = Settings(
        kind=19,
        input_size=(3, 32, 32),
        num_classes=10,
        # Batch
        batch_size=512,
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
    model = VGGNet(setting)
    setting.device.move(model)
    model.print_summary()

    # Train model
    model.fit(trainset, validset)

    # Plot training performance
    plot = PlotMngr()
    plot.performance(model.epoch_results)

    # Evaluate model on traning set
    model.evaluate(trainset)
    plot.confusion_matrix(model.confusion_matrix)

    # Evaluate model on validation set
    model.evaluate(validset)
    plot.confusion_matrix(model.confusion_matrix)

    # Final evaluation on test set
    testset = data.load_test()
    scores, times, fps = model.test(testset)
    plot.confusion_matrix(model.confusion_matrix)
    return

def process_tune():
    """
    Process the procedure of tuning model
    """
    # Hyper-parameters search space
    distrib = HyperParamsDistrib(
        # Batch
        batch_size      = [32],
        batch_norm      = [True],
        # Epoch
        epochs          = [50],
        # Learning rate
        learning_rate   = list(np.logspace(np.log10(0.0001), np.log10(0.001), base=10, num=1000)),
        lr_factor       = list(np.logspace(np.log10(0.01), np.log10(1), base=10, num=1000)),
        lr_patience     = list(np.arange(10, 12)),
        # Regularization
        weight_decay    = list(np.logspace(np.log10(0.01), np.log10(0.9), base=10, num=1000)),
        dropout_rate    = stats.uniform(0.85, 0.15),
        # Metric
        loss_optim      = [False],
        # Data
        data_augment    = [False],
        # Early stopping
        early_stop      = [True],
        es_patience     = list(np.arange(12, 15)),
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
        kind=16,
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

    # Load checkpoint
    load_checkpoint = True
    if load_checkpoint:
        model = VGGNet(setting)
        setting.device.move(model)
        states = model.load_checkpoint(path='data/output/VGGNet16-1598556840-tuned.tar')
        model.setting.show()

        plot = PlotMngr()
        plot.performance(states['epoch_results'])

        model.evaluate(trainset)
        plot.confusion_matrix(model.confusion_matrix)

        model.evaluate(validset)
        plot.confusion_matrix(model.confusion_matrix)

        plot.hyperparameters(states['tuning_results'], setting.get_hparams_names())

    # Create tuner
    tuner = Tuner(VGGNet, setting)

    # Search for best model in tuning process
    model, tuning_results = tuner.process(num_iter=3)

    # Plot training performance of best model
    plot = PlotMngr()
    plot.performance(model.epoch_results)

    # Plot tuning process
    plot.hyperparameters(tuning_results, setting.get_hparams_names())

    # Evaluate model on traning set
    model.evaluate(trainset)
    plot.confusion_matrix(model.confusion_matrix)

    # Evaluate model on validation set
    model.evaluate(validset)
    plot.confusion_matrix(model.confusion_matrix)

    # Final evaluation on test set
    testset = data.load_test()
    scores, times, fps = model.test(testset)
    plot.confusion_matrix(model.confusion_matrix)
    return


if __name__ == "__main__":
    
    #process_fit()

    process_tune()

