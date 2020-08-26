
import random
import numpy as np
import scipy.stats as stats

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from mngrdevice import DeviceMngr


class HyperParams:
    """
    Hyper-parameters for model tuning
    """

    def __init__(self):
        super().__init__()
        self.batch_size = None
        self.batch_norm = None
        self.epochs = None
        self.learning_rate = None
        self.lr_factor = None
        self.lr_patience = None
        self.weight_decay = None
        self.dropout_rate = None
        self.loss_optim = None
        self.data_augment = None
        self.early_stop = None
        self.es_patience = None
        self.grad_clip_norm = None
        self.gc_max_norm = None
        self.grad_clip_value = None
        self.gc_value = None
        self.init_params = None
        return

    def show(self):
        """
        Print members of current class
        """
        print(self.__class__.__name__)
        for item in self.__dict__.items():
            print(item)
        print()
        return

    def load_values(self, dictionary):
        """
        Set values to the class members from given dictionary
        """
        for key in dictionary.keys():
            setattr(self, key, dictionary[key])
        return

    def to_dict(self):
        """
        Get members and pack into dictionary
        """
        return self.__dict__


class HyperParamsDistrib(HyperParams):
    """
    Hyper-parameters distributions for model tuning

    About Gradient clipping
        Gradient clipping norm
            The average value of gradient norms is a good initial trial [1]
        Gradient clipping value
            It's generally tuned as a hyperparameter with clipping range [-1 ... +1] [2]
        Source: 
            [1] https://towardsdatascience.com/what-is-gradient-clipping-b8e815cdfb48
            [2] https://www.reddit.com/r/MachineLearning/comments/3n8g28/gradient_clipping_what_are_good_values_to_clip_at/
    """

    # Default distributions

    # Batch
    DEF_BATCH_SIZE = [int(2**i) for i in np.arange(1, 10)]
    DEF_BATCH_NORM = [False, True]

    # Epoch
    DEF_EPOCHS = list(np.arange(10, 55, 5))

    # Learning rate
    DEF_LEARNING_RATE = list(np.logspace(np.log10(0.001), np.log10(0.5), base=10, num=1000))
    DEF_LR_FACTOR = list(np.logspace(np.log10(0.01), np.log10(1), base=10, num=1000))
    DEF_LR_PATIENCE = list(np.arange(1, 10))

    # Regularization
    DEF_WEIGHT_DECAY = list(np.logspace(np.log10(1E-6), np.log10(0.5), base=10, num=1000))
    DEF_DROPOUT_RATE = stats.uniform(0, 0.9)

    # Metric
    DEF_LOSS_OPTIM = [False, True]

    # Data
    DEF_DATA_AUGMENT = [False, True]

    # Early stopping
    DEF_EARLY_STOP = [False, True]
    DEF_ES_PATIENCE = list(np.arange(10, 20))            # Should be always greater than learning rate patience

    # Gradient clipping
    DEF_GRAD_CLIP_NORM = [False, True]
    DEF_GC_MAX_NORM = stats.uniform(0.01, 10)
    DEF_GRAD_CLIP_VALUE = [False, True]
    DEF_GC_VALUE = stats.uniform(0.01, 10)

    # Initialization
    DEF_INIT_PARAMS = [False, True]

    def __init__(self,
        batch_size=None,
        batch_norm=None,
        epochs=None,
        learning_rate=None,
        lr_factor=None,
        lr_patience=None,
        weight_decay=None,
        dropout_rate=None,
        loss_optim = None,
        data_augment=None,
        early_stop=None,
        es_patience = None,
        grad_clip_norm = None,
        gc_max_norm = None,
        grad_clip_value = None,
        gc_value = None,
        init_params = None):

        super().__init__()

        # Distributions
        self.batch_size = batch_size
        self.batch_norm = batch_norm
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate
        self.loss_optim = loss_optim
        self.data_augment = data_augment
        self.early_stop = early_stop
        self.es_patience = es_patience
        self.grad_clip_norm = grad_clip_norm
        self.gc_max_norm = gc_max_norm
        self.grad_clip_value = grad_clip_value
        self.gc_value = gc_value
        self.init_params = init_params

        # Set default values for None
        for attrib, value in self.__dict__.items():
            if value is None:
                def_attrib = 'DEF_' + attrib.upper()
                def_value = getattr(self, def_attrib)
                setattr(self, attrib, def_value)
        return


class Settings(HyperParams):
    """
    Setting hyper-parameters and environment for model
    """

    # Default hyper-parameters
    
    # Batch
    DEF_BATCH_SIZE = 64
    DEF_BATCH_NORM = True
    
    # Epoch
    DEF_EPOCHS = 10
    
    # Learning rate
    DEF_LEARNING_RATE = 0.01
    DEF_LR_FACTOR = 0.1
    DEF_LR_PATIENCE = 10
    
    # Regularization
    DEF_WEIGHT_DECAY = 1E-4
    DEF_DROPOUT_RATE = 0.5
    
    # Metric
    DEF_LOSS_OPTIM = True

    # Data
    DEF_DATA_AUGMENT = False

    # Early stopping
    DEF_EARLY_STOP = False
    DEF_ES_PATIENCE = 15

    # Gradient clipping
    DEF_GRAD_CLIP_NORM = False
    DEF_GC_MAX_NORM = 1
    DEF_GRAD_CLIP_VALUE = False
    DEF_GC_VALUE = 1

    # Default hyper-parameters distributions
    DEF_DISTRIB = HyperParamsDistrib()

    # Default environment
    DEF_SANITY_CHECK = False
    DEF_DEBUG = False
    DEF_DEVICE = DeviceMngr()
    DEF_NUM_WORKERS = 15
    DEF_MIXED_PRECISION = False

    # Initialization
    DEF_INIT_PARAMS = True

    def __init__(self,
        kind,
        input_size,
        num_classes,
        batch_size=None,
        batch_norm=None,
        epochs=None,
        learning_rate=None,
        lr_factor=None,
        lr_patience=None,
        weight_decay=None,
        dropout_rate=None,
        loss_optim=None,
        data_augment=None,
        early_stop=None,
        es_patience = None,
        grad_clip_norm = None,
        gc_max_norm = None,
        grad_clip_value = None,
        gc_value = None,
        init_params = None,
        distrib=None,
        sanity_check=None,
        debug=None,
        device=None,
        num_workers=None,
        mixed_precision=None):

        super().__init__()

        # Custom
        self.kind = kind
        self.input_size = input_size
        self.num_classes = num_classes

        # Hyper-parameters
        self.batch_size = batch_size
        self.batch_norm = batch_norm
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate
        self.loss_optim = loss_optim
        self.data_augment = data_augment
        self.early_stop = early_stop
        self.es_patience = es_patience
        self.grad_clip_norm = grad_clip_norm
        self.gc_max_norm = gc_max_norm
        self.grad_clip_value = grad_clip_value
        self.gc_value = gc_value
        self.init_params = init_params

        # Hyper-parameters distributions
        self.distrib = distrib

        # Environment
        self.sanity_check = sanity_check
        self.debug = debug
        self.device = device
        self.num_workers = num_workers
        self.mixed_precision = mixed_precision

        # Set default values for None
        for attrib, value in self.__dict__.items():
            if value is None:
                def_attrib = 'DEF_' + attrib.upper()
                def_value = getattr(self, def_attrib)
                setattr(self, attrib, def_value)
        return

    def get_hparams(self):
        """
        Get hyper-parameters and pack into dictionary
        """
        # Get superclass fields
        dictionary = self.__class__.__bases__[0]().__dict__

        # Set values from subclass
        for field in dictionary.keys():
            dictionary[field] = getattr(self, field)

        return dictionary

    def get_hparams_names(self):
        """
        Returns hyper-parameters names
        """
        return list(self.get_hparams().keys())


if __name__ == "__main__":
    
    distrib = HyperParamsDistrib() 
    distrib.show()

    setting1 = Settings(
        kind=0,
        input_size=(3, 32, 32),
        num_classes=10,
        batch_size=1024,
        lr_factor=0.12)
    setting1.show()
    hparams = setting1.get_hparams()
    print(hparams)

    setting2 = Settings(
        kind=0,
        input_size=(3, 64, 64),
        num_classes=10,
        # Batch
        batch_size=None,
        batch_norm=None,
        # Epoch
        epochs=None,
        # Learning rate
        learning_rate=None,
        lr_factor=None,
        lr_patience=None,
        # Regularization
        weight_decay=None,
        dropout_rate=None,
        # Metric
        loss_optim = None,
        # Data
        data_augment=None,
        # Early stopping
        early_stop=None,
        es_patience = None,
        # Gradient clipping
        grad_clip_norm = None,
        gc_max_norm = None,
        grad_clip_value = None,
        gc_value = None,
        # Initialization
        init_params=None,
        # Distribution
        distrib=None,
        # Environment
        sanity_check=None,
        debug=None,
        device=None,
        num_workers=None,
        mixed_precision=None)
    setting2.show()
    hparams = setting2.get_hparams()
    print(hparams)

    setting2.load_values({
        'batch_size':8192, 
        'batch_norm':False, 
        'loss_optim':False, 
        'es_patience':17, 
        'grad_clip_norm':True,
        'gc_max_norm': 0.33,
        'grad_clip_value':False,
        'gc_value':0.25,
        'init_params':False,
        'mixed_precision':True,
        'num_workers':2})
    setting2.show()

