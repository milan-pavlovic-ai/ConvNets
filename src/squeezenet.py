
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


class SqueezeNet(MultiClassBaseModel):
    """
    SqueezeNet - Squeeze Network
        Modifications
            - Added Batch Normalization after each Convolutional layer

    Source: SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size
            https://arxiv.org/pdf/1602.07360.pdf
            SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters than SqueezeNet 1.0, without sacrificing accuracy.
            https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1
    """

    # Configuration
    config = {
        '1.0': [
            ('conv', 96, 7, 2),
            ('maxpool', 3, 2),
            ('fire', 16, 64, 64),
            ('fire', 16, 64, 64),
            ('fire', 32, 128, 128),
            ('maxpool', 3, 2),
            ('fire', 32, 128, 128),
            ('fire', 48, 192, 192),
            ('fire', 48, 192, 192),
            ('fire', 64, 256, 256),
            ('maxpool', 3, 2),
            ('fire', 64, 256, 256),
        ],
        '1.1': [
            ('conv', 64, 3, 2),
            ('maxpool', 3, 2),
            ('fire', 16, 64, 64),
            ('fire', 16, 64, 64),
            ('maxpool', 3, 2),
            ('fire', 32, 128, 128),
            ('fire', 32, 128, 128),
            ('maxpool', 3, 2),
            ('fire', 48, 192, 192),
            ('fire', 48, 192, 192),
            ('fire', 64, 256, 256),
            ('fire', 64, 256, 256),
        ]
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
        config = SqueezeNet.config[str(self.setting.kind)]

        # Layers
        layers = []

        # Construct feature layers
        for cfg in config:
            operation_type = cfg[0]

            if operation_type == 'fire':
                _, squeeze, expand_1x1, expand_3x3 = cfg
                layers += [Fire(self, squeeze, expand_1x1, expand_3x3)]

            elif operation_type == 'maxpool':
                _, kernel_size, stride = cfg
                layers += [self.maxpool2d(kernel_size=kernel_size, stride=stride)]

            elif operation_type == 'conv':
                _, num_filters, kernel_size, stride = cfg
                layers += [self.conv2d_block(num_filters=num_filters, kernel_size=kernel_size, stride=stride)]
            else:
                exit(1)
                
        return nn.Sequential(*layers)

    def make_classifier_layers(self):
        """
        Create classifier layers
        """
        layers = nn.Sequential(
            nn.Dropout(p=self.setting.dropout_rate),
            self.conv2d_block(num_filters=self.setting.num_classes, kernel_size=1),
            self.adapt_avgpool2d(output_size=1)
        )
        return layers

    def forward(self, x):
        """
        Forward propagation
        """
        x = self.features(x)
        x = self.classifier(x)
        return torch.flatten(x, start_dim=1)

class Fire(nn.Module):
    """
    Fire module
    """

    def __init__(self, network, squeeze_num, expand_1x1_num, expand_3x3_num):
        """
        Initialize layers
        """
        super().__init__()
        self.squeeze = network.conv2d_block(num_filters=squeeze_num, kernel_size=1)
        self.expand_1x1 = network.conv2d_block(network.in_channels, expand_1x1_num, set_output=False, kernel_size=1)
        self.expand_3x3 = network.conv2d_block(network.in_channels, expand_3x3_num, set_output=True, kernel_size=3, padding=1)
        network.in_channels = expand_1x1_num + expand_3x3_num
        return

    def forward(self, x):
        """
        Forward propagation
        """
        # Squeeze layer
        x = self.squeeze(x)

        # Expand layer
        branch1 = self.expand_1x1(x)
        branch2 = self.expand_3x3(x)
        output = [branch1, branch2]

        return torch.cat(output, dim=1)


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
        kind='1.1',
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
    model = SqueezeNet(setting)
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
        learning_rate   = list(np.logspace(np.log10(0.1), np.log10(0.1), base=10, num=1000)),
        lr_factor       = list(np.logspace(np.log10(0.05), np.log10(0.5), base=10, num=1000)),
        lr_patience     = [10],
        # Regularization
        weight_decay    = list(np.logspace(np.log10(0.00005), np.log10(0.0001), base=10, num=1000)),
        dropout_rate    = stats.uniform(0.1, 0.3),
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
        kind='1.1',
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
    tuner = Tuner(SqueezeNet, setting)

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
        kind='1.1',
        input_size=(3, 32, 32),
        num_classes=10,
        num_workers=16,
        mixed_precision=True,
        test_sample_size=90,
        seed=21,
        sanity_check=False,
        debug=False)

    # Load checkpoint
    model = SqueezeNet(setting)
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

    process_load(path='data/output/SqueezeNet1.1-1600730105-tuned.tar', resume=False)