
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


class MobileNetV1(MultiClassBaseModel):
    """
    MobileNet - Mobile network
        Modifications
            - Added dropout layer after global average pooling

    Source: MobileNets: Efficient Convolutional Neural Networks for Mobile VisionApplications
            https://arxiv.org/pdf/1704.04861.pdf
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
        Create feature layers
        """
        layers = nn.Sequential(
            Conv2dBlock(self, num_filters=32, kernel_size=3, stride=2, padding=1),

            Conv2dBlockDW(self, num_filters=64, kernel_size=3, padding=1),

            Conv2dBlockDW(self, num_filters=128, kernel_size=3, stride=2, padding=1),
            Conv2dBlockDW(self, num_filters=128, kernel_size=3, padding=1),

            Conv2dBlockDW(self, num_filters=256, kernel_size=3, stride=2, padding=1),
            Conv2dBlockDW(self, num_filters=256, kernel_size=3, padding=1),

            Conv2dBlockDW(self, num_filters=512, kernel_size=3, stride=2, padding=1),
            Conv2dBlockDW(self, num_filters=512, kernel_size=3, padding=1),
            Conv2dBlockDW(self, num_filters=512, kernel_size=3, padding=1),
            Conv2dBlockDW(self, num_filters=512, kernel_size=3, padding=1),
            Conv2dBlockDW(self, num_filters=512, kernel_size=3, padding=1),
            Conv2dBlockDW(self, num_filters=512, kernel_size=3, padding=1),

            Conv2dBlockDW(self, num_filters=1024, kernel_size=3, stride=2, padding=1),
            Conv2dBlockDW(self, num_filters=1024, kernel_size=3, padding=1),

            self.adapt_avgpool2d(output_size=1)
        )
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

class Conv2dBlockDW(nn.Sequential):
    """
    Convolution depth-wise block
    """

    def __init__(self, network, in_channels=None, num_filters=None, set_output=True, activation=True, **kwargs):
        """
        Initialize layers
        """
        super().__init__()

        # Convolution Depth-Wise layer
        self.add_module('conv_dw', network.conv2d_depthwise(in_channels, in_channels, set_output, **kwargs))

        # Batch normalization layer
        if network.setting.batch_norm:
            if in_channels is None:
                in_channels = network.in_channels
            self.add_module('bn_dw', nn.BatchNorm2d(num_features=in_channels))

        # Activation layer
        if activation:
            self.add_module('relu_dw', nn.ReLU(inplace=True))

        # Convolution Point-Wise layer
        self.add_module('conv_pw', network.conv2d(in_channels, num_filters, set_output, kernel_size=1))

        # Batch normalization layer
        if network.setting.batch_norm:
            self.add_module('bn_pw', nn.BatchNorm2d(num_features=num_filters))

        # Activation layer
        if activation:
            self.add_module('relu_pw', nn.ReLU(inplace=True))

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
    model = MobileNetV1(setting)
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
        learning_rate   = list(np.logspace(np.log10(0.09), np.log10(0.09), base=10, num=1000)),
        lr_factor       = list(np.logspace(np.log10(0.01), np.log10(0.5), base=10, num=1000)),
        lr_patience     = [10],
        # Regularization
        weight_decay    = list(np.logspace(np.log10(0.00001), np.log10(0.0001), base=10, num=1000)),
        dropout_rate    = stats.uniform(0, 0.1),
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
    tuner = Tuner(MobileNetV1, setting)

    # Search for best model in tuning process
    model, results = tuner.process(num_iter=3)

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
    model = MobileNetV1(setting)
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

    process_load(path='data/output/MobileNetV1-1600756521-tuned.tar', resume=False)