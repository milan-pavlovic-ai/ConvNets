
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
        Modifications
            - Added Batch Normalization after each Convolutional layer
            - Added Global Average Pooling layer before classifier,
                a] it reduce number of total parameters
                b] the number of parameters does not depend on input size anymore
            - Added dropout layer after global average pooling
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
                layers += [self.maxpool2d(kernel_size=2, stride=2)]
            else:
                # Convolutional layer
                layers += [self.conv2d_block(num_filters=element, kernel_size=3, padding=1)]

        # Global average pooling
        layers += [self.adapt_avgpool2d(output_size=1)]

        return nn.Sequential(*layers)

    def make_classifier_layers(self):
        """
        Create classifier layers by using linear operation with dropout regularization method
        """
        layers = nn.Sequential(
            nn.Dropout(p=self.setting.dropout_rate),

            nn.Linear(self.num_flat_features(), 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.setting.dropout_rate),

            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.setting.dropout_rate),

            nn.Linear(4096, self.setting.num_classes)
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
        kind=16,
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
    model = VGGNet(setting)
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
        learning_rate   = list(np.logspace(np.log10(0.017), np.log10(0.019), base=10, num=1000)),
        lr_factor       = list(np.logspace(np.log10(0.05), np.log10(0.5), base=10, num=1000)),
        lr_patience     = [10],
        # Regularization
        weight_decay    = list(np.logspace(np.log10(0.00005), np.log10(0.00007), base=10, num=1000)),
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
        kind=11,
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
    tuner = Tuner(VGGNet, setting)

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
        kind=11,
        input_size=(3, 32, 32),
        num_classes=10,
        num_workers=16,
        mixed_precision=True,
        test_sample_size=90,
        seed=21,
        sanity_check=False,
        debug=False)

    # Load checkpoint
    model = VGGNet(setting)
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

    process_load(path='data/output/VGGNet11-1600959117-tuned.tar', resume=False)

