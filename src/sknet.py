
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


class SKNet(MultiClassBaseModel):
    """
    SKNet - Selective Kernel Networks
        Modifications
            - Added dropout layer after global average pooling

    Source: Selective Kernel Networks
            https://arxiv.org/pdf/1903.06586.pdf

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
        config = SKNet.config[str(self.setting.kind)]
        block_type = SKBottleneck
        expansion = 2

        # Layers
        layers = []

        # Convolution and Max-Pool
        layers += [self.conv2d_block(num_filters=64, kernel_size=7, stride=2, padding=3)]
        layers += [self.maxpool2d(kernel_size=3, stride=2, padding=1)]

        # SK blocks
        for cfg_block in config:
            num_filters, num_repeat, stride = cfg_block
            
            # First sk-block reduce height and weight with stride
            layers += [self.sk_block(block_type, num_filters, expansion, stride)]

            # Add rest sk-blocks
            for _ in range(1, num_repeat):
                layers += [self.sk_block(block_type, num_filters, expansion)]

        # Adaptive Average-Pool
        layers += [self.adapt_avgpool2d(output_size=1)]

        return nn.Sequential(*layers)

    def sk_block(self, block_type, num_filters, expansion, stride=1, cardinality=32):
        """
        Create sk-block
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

class SKBottleneck(nn.Module):
    """
    Selective Kernel Bottleneck
    """

    def __init__(self, network, num_filters, expansion, dim_synch=None, **kwargs):
        """
        Initialize layers
        """
        super().__init__()
        self.dim_synch = dim_synch
        self.sk_bottleneck = nn.Sequential(
            Conv2dBlock(network, num_filters=num_filters, kernel_size=1),
            SKConv(network, **kwargs),
            Conv2dBlock(network, num_filters=num_filters * expansion, activation=False, kernel_size=1)
        )
        return

    def forward(self, x):
        """
        Forward propagation
        """
        # Save previous state
        identity = x
        
        # Calculate bottleneck
        output = self.sk_bottleneck(x)

        # Dimension synchronization
        if self.dim_synch is not None:
            identity = self.dim_synch(identity) 

        # Add shortcut / skip-connection
        output += identity

        # Activation
        output = F.relu(output)

        return output

class SKConv(nn.Module):
    """
    Selective Kernel convolution
        After this operation the size of tensor is the same as before
    """

    def __init__(self, network, num_paths=2, groups=32, reduction=16, min_descriptor=32, **kwargs):
        """
        Initialize layers
        """
        super().__init__()
        self.num_paths = num_paths
        self.num_features = network.in_channels
        self.kernels = nn.ModuleList([])
        self.attentions = nn.ModuleList([])
        descriptor_size = max(network.in_channels // reduction, min_descriptor)

        for i in range(num_paths):
            # Splits, list of different kernel operations
            self.kernels += [Conv2dBlock(
                network, 
                num_filters=network.in_channels,
                set_output=False,
                kernel_size=3,
                padding=1+i,
                dilation=1+i,
                groups=groups,
                **kwargs
            )]
            # Attentions, one attention for each kernel operation
            self.attentions += [nn.Sequential(
                nn.Conv2d(descriptor_size, network.in_channels, kernel_size=1)
            )]

        # Squeeze
        self.global_avgpool = network.adapt_avgpool2d(set_output=False, output_size=1)

        # Feature descriptor
        self.descriptor = Conv2dBlock(network, num_filters=descriptor_size, set_output=False, kernel_size=1)

        # Soft attention across channels is used to adap-tively select different spatial scales of information
        self.softmax = nn.Softmax(dim=1)

        return

    def forward(self, x):
        """
        Forward propagation
        """
        # Split: Create a feature map for each kernel
        feature_maps = [group_conv(x) for group_conv in self.kernels]

        # Fuse: Concatenate and sum feature maps element-wise
        feature_maps = torch.cat(feature_maps, dim=1)
        batch_size, num_channels, height, width = feature_maps.size()
        feature_maps = feature_maps.view(batch_size, self.num_paths, self.num_features, height, width)
        
        features = torch.sum(feature_maps, dim=1)
        features = self.global_avgpool(features)
        features = self.descriptor(features)

        # Select: Create soft attention vectors for selection
        attentions = [attention(features) for attention in self.attentions]
        attentions = torch.cat(attentions, dim=1)
        attentions = attentions.view(batch_size, self.num_paths, self.num_features, 1, 1)
        soft_attentions = self.softmax(attentions)

        output = torch.sum(feature_maps * soft_attentions, dim=1)

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
    model = SKNet(setting)
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
    tuner = Tuner(SKNet, setting)

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
    model = SKNet(setting)
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

    #process_load(path='data/802.11/output/ident/SKNet1D50-1604082963-best_score.tar', resume=True, visual=True)