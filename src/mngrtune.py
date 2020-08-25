
import os
import copy
import time as t
import numpy as np
import scipy.stats as stats

from sklearn.model_selection import ParameterSampler
from basemodel import MultiClassBaseModel, ConvNet
from mngrdata import DataMngr
from mngrplot import PlotMngr
from settings import Settings, HyperParamsDistrib


class Tuner:
    """
    Manager for tuning model hyperparameters
    """

    def __init__(self, model_class, setting):
        super().__init__()
        self.model_class = model_class
        self.setting = setting
        
        # Best model information
        self.suffix = 'tuned'
        self.version = int(t.time())
        model_name = self.model_class.__name__ + str(self.setting.kind)
        self.best_model_name = '{}-{}-{}.tar'.format(model_name, self.version, self.suffix)
        self.best_model_path = os.path.join(DataMngr.OUTPUT_DIR, self.best_model_name)
        
        # Available after tuning
        self.results = None

    def process(self, num_iter=10):
        """
        Tuning hyper-parameters of model with random search method for given hyper-parameters distributions in setting attribute
        Random search method use holdout procedure for model evaluation

        Returns model with best metric achieved on samples from hyper-parameters distributions and tuning results
        Saved tuning checkpoint contains all information about best model and tuning history over all models
        History of the training process for each model in the tuning process is saved as an individual checkpoint

        About parameter sampler
            Source: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ParameterSampler.html
        """

        # Initialization
        best_score = -1
        data = DataMngr(self.setting)
        best_model_index = 0
        self.results = {'hparams':[], 'scores':[], 'best_model_index':0}

        # Load, split and transform data once if batch size is fixed
        fixed_batch_size = len(self.setting.distrib.batch_size) == 1
        if fixed_batch_size:
            trainset = data.load_train()
            validset = data.load_valid()

        # Create hyperparameters samples from given distributions
        hparams_samples = ParameterSampler(param_distributions=self.setting.distrib.to_dict(), n_iter=num_iter)
        
        # Tuning process
        print('\n=== START TUNING ===\n')

        for i, sample in enumerate(hparams_samples):

            # Adjust model setting with given sample
            self.setting.load_values(sample)
            print('SETTING {}/{}\n'.format(i+1, num_iter))
            self.setting.show()

            # Load, transform data and split into batches 
            if not fixed_batch_size:
                print('Load, transform data and split into batches...')
                data = DataMngr(self.setting)
                trainset = data.load_train()
                validset = data.load_valid()

            # Create model with sample setting
            model = self.model_class(self.setting)
            self.setting.device.move(model)

            # Fit the model with given training data and use validation data for controling training process
            model.fit(trainset, validset)

            # Evaluate model with validation data
            score = model.evaluate(validset)

            # Save the model with best hparams
            if best_score < score:
                best_score = score
                best_model_index = i
                model.save_checkpoint(path=self.best_model_path)      # Save as best tuned model

            # Remember history
            self.results['scores'].append(score)
            self.results['hparams'].append(self.setting.get_hparams())

        print('\n=== TUNING IS FINISHED ===\n')

        # Update results
        self.results['best_model_index'] = best_model_index

        # Create model
        model = self.model_class(self.setting)
        self.setting.device.move(model)

        # Load best parameters and other information into created model
        best_checkpoint = model.load_checkpoint(path=self.best_model_path)
        model.epoch_results = best_checkpoint['epoch_results']

        # Save history of tuning process
        best_checkpoint['tuning_results'] = self.results                           
        model.update_checkpoint(best_checkpoint, path=self.best_model_path)
        
        return model, self.results

    def process_cv(self, cv=5):
        """
        Tuning hyper-parameters of model with random search method for given hyper-parameters distributions in setting attribute
        Random search method use cross-validation procedure for model evaluation
        """
        raise NotImplementedError
        return


if __name__ == "__main__":

    # Hyper-parameters search space
    distrib = HyperParamsDistrib(
        # Batch
        batch_size      = [32, 64, 128, 256, 512, 1024],
        batch_norm      = [False],
        # Epoch
        epochs          = [10],
        # Learning rate
        learning_rate   = list(np.logspace(np.log10(0.001), np.log10(0.1), base=10, num=1000)),
        lr_factor       = list(np.logspace(np.log10(0.01), np.log10(1), base=10, num=1000)),
        lr_patience     = list(np.arange(1, 7)),
        # Regularization
        weight_decay    = list(np.logspace(np.log10(1E-6), np.log10(0.5), base=10, num=1000)),
        dropout_rate    = stats.uniform(0, 0.9),
        # Metric
        loss_optim      = [False, True],
        # Data
        data_augment    = [False],
        # Early stopping
        early_stop      = [False, True],
        es_patience     = list(np.arange(7, 12)),
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
        kind=0,
        input_size=(3, 32, 32),
        num_classes=10,
        distrib=distrib,
        sanity_check=False,
        debug=False)

    '''
    # Load checkpoint
    model = ConvNet(setting)
    setting.device.move(model)
    states = model.load_checkpoint(path='data/output/ConvNet-1598124935-tuned.tar')
    plot = PlotMngr()
    plot.performance(states['epoch_results'])
    plot.hyperparameters(states['tuning_results'], setting.get_hparams_names())
    '''

    # Create tuner
    tuner = Tuner(ConvNet, setting)

    # Search for best model in tuning process
    model, tuning_results = tuner.process(num_iter=3)

    # Plot training performance of best model
    plot = PlotMngr()
    plot.performance(model.epoch_results)

    # Plot tuning process
    plot.hyperparameters(tuning_results, setting.get_hparams_names())

    # Load data for evaluation
    data = DataMngr(setting)
    trainset = data.load_train()
    validset = data.load_valid()

    # Evaluate model on traning set
    model.evaluate(trainset)
    plot.confusion_matrix(model.confusion_matrix)

    # Evaluate model on validation set
    model.evaluate(validset)
    plot.confusion_matrix(model.confusion_matrix)


