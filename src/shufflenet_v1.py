
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


class ShuffleNetV1(MultiClassBaseModel):
    """
    ShuffleNet - Shuffle network 
        Implementation of 1.0x scaler factor with ability to choose group factor
        Modifications
            - Added dropout layer after global average pooling

    Source: ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices
            https://arxiv.org/pdf/1707.01083.pdf
    """

    # Configuration
    #   in format (stride, repeat, out_channels) for each block
    config = {
        'g1': [(2, 1, 144), (1, 3, 144), (2, 1, 288), (1, 7, 288), (2, 1, 576), (1, 3, 576)],
        'g2': [(2, 1, 200), (1, 3, 200), (2, 1, 400), (1, 7, 400), (2, 1, 800), (1, 3, 800)],
        'g3': [(2, 1, 240), (1, 3, 240), (2, 1, 480), (1, 7, 480), (2, 1, 960), (1, 3, 960)],
        'g4': [(2, 1, 272), (1, 3, 272), (2, 1, 544), (1, 7, 544), (2, 1, 1088), (1, 3, 1088)],
        'g8': [(2, 1, 384), (1, 3, 384), (2, 1, 768), (1, 7, 768), (2, 1, 1536), (1, 3, 1536)],
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
        config = ShuffleNetV1.config[str(self.setting.kind)]
        groups = int(self.setting.kind[1:])

        # Construct layers
        layers = []
        layers += [self.conv2d_block(num_filters=24, kernel_size=3, stride=2, padding=1)]
        layers += [self.maxpool2d(kernel_size=3, stride=2, padding=1)]

        # Create shuffle block
        for i, cfg_block in enumerate(config):
            stride, repeat, num_output_channels = cfg_block

            # Create shuffle unit
            for j in range(repeat):
                downsample = (stride == 2)
                first_conv = (i == 0 and j == 0)
                layers += [ShuffleUnit(self, num_output_channels, groups, stride, downsample, first_conv)]

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

class ShuffleUnit(nn.Module):
    """
    Shuffle Unit
    """

    def __init__(self, network, num_output_channels, groups, stride, downsample, first_conv):
        """
        Initialize layers
        """
        super().__init__()

        self.groups = groups
        self.groups_first_conv1x1 = 1 if first_conv else groups
        self.downsample = downsample
        bottleneck_channels = num_output_channels // 4
        num_channels_identity = network.in_channels
        if self.downsample:
            num_output_channels -= num_channels_identity

        # Group convolution for channel compression before conv3x3
        self.conv1x1_group_compress = Conv2dBlock(
            network, 
            num_filters=bottleneck_channels, 
            kernel_size=1, 
            groups=self.groups_first_conv1x1)

        # Convolution 3x3 depth-wise
        self.conv3x3_depthwise = Conv2dBlock(
            network, 
            num_filters=bottleneck_channels, 
            activation=False, 
            kernel_size=3, 
            stride=stride,
            padding=1, 
            groups=bottleneck_channels)

        # Group convolution for channel expansion
        self.conv1x1_group_expand = Conv2dBlock(
            network,
            num_filters=num_output_channels,
            activation=False,
            kernel_size=1,
            groups=self.groups)

        # Update number of channels
        if self.downsample:
            network.in_channels += num_channels_identity

        # Average pooling and Activation
        self.avg_pool = network.avgpool2d(set_output=False, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)
        return

    def forward(self, x):
        """
        Forward propagation
        """
        identity = x

        if self.downsample:
            identity = self.avg_pool(identity)

        output = self.conv1x1_group_compress(x)
        output = self.channel_shuffle(output)
        output = self.conv3x3_depthwise(output)
        output = self.conv1x1_group_expand(output)

        if self.downsample:
            output = torch.cat([identity, output], dim=1)
        else:
            output += identity

        return self.relu(output)

    def channel_shuffle(self, x):
        """
        Channel shuffle operation
            Usage of contiguous function: https://github.com/pytorch/pytorch/issues/764
        """
        batch_size, num_channels, height, width = x.data.size()
        group_channels = num_channels // self.groups

        # Reshape with group channels
        x = x.view(batch_size, self.groups, group_channels, height, width)

        # Random with transpose
        x = torch.transpose(x, 1, 2).contiguous()

        # Flatten
        x = x.view(batch_size, -1, height, width)

        return x


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
        kind='g3',
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
    model = ShuffleNetV1(setting)
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
        weight_decay    = list(np.logspace(np.log10(0.000005), np.log10(0.0001), base=10, num=1000)),
        dropout_rate    = stats.uniform(0, 0.3),
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
        kind='g4',
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
    tuner = Tuner(ShuffleNetV1, setting)

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
        kind='g4',
        input_size=(3, 32, 32),
        num_classes=10,
        num_workers=16,
        mixed_precision=True,
        test_sample_size=90,
        seed=21,
        sanity_check=False,
        debug=False)

    # Load checkpoint
    model = ShuffleNetV1(setting)
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

    process_load(path='data/output/ShuffleNetV1g4-1600946548-tuned.tar', resume=False)