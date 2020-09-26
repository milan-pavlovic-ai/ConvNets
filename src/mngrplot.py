
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from settings import Settings
from mngrutility import UtilityMngr


class PlotMngr:
    """
    Class for model performance visualization

    About colors
        Source: https://matplotlib.org/3.1.0/gallery/color/named_colors.html
    """

    def __init__(self):
        super().__init__()
        self.epoch_results = None       # information about each epoch
        self.num_epochs = None          # total number of epochs
        self.epochs = None              # list of epochs
        self.num_train_epochs = None    # trained number of epochs for best model
        self.alpha = 0.75               # transparency factor
        self.max_charts_plot = 8        # maximum number of charts per one plot, must be divisible with maximum charts per row
        self.max_charts_row = 2         # maximum number of charts per one row
        self.charts_colors = ['blue', 'red', 'magenta', 'purple', 'lime', 'orange', 'lightskyblue', 'teal']
        sns.set(style='darkgrid')


    def losses(self, axes):
        """
        Plot validation and traning loss for each epoch
        """
        # Set figure
        axes.set_title('Model Loss', fontsize=14)
        axes.set_ylabel('Loss')
        axes.set_xlabel('Epoch')

        # Plot traning curve
        train_losses = self.epoch_results['train_loss']
        axes.plot(self.epochs, train_losses, label='Traning', color='royalblue', alpha=self.alpha)

        # Plot validation curve
        valid_losses = self.epoch_results['valid_loss']
        axes.plot(self.epochs, valid_losses, label='Validation', color='forestgreen', alpha=self.alpha)

        # Plot number of trained epochs for best model
        x = [self.num_train_epochs, self.num_train_epochs]
        y = np.sort([train_losses[self.num_train_epochs - 1], valid_losses[self.num_train_epochs - 1]])
        axes.plot(x, y, marker='x', color='purple', lw=0, label='Best model', alpha=self.alpha)

        # Annotation for best model on training set
        text_value = 'Train\n{:.5f}'.format(train_losses[self.num_train_epochs - 1])
        axes.annotate(text_value,
            xy=(self.num_train_epochs, train_losses[self.num_train_epochs - 1]), 
            xytext=(-40, -30), 
            arrowprops=dict(arrowstyle='->', connectionstyle='angle3,angleA=10,angleB=-90', color='black', alpha=self.alpha),
            ha='center', va='center', 
            fontsize=9, 
            textcoords='offset points')
        
        # Annotation for best model on validation set
        text_value = 'Valid\n{:.5f}'.format(valid_losses[self.num_train_epochs - 1])
        axes.annotate(text_value,
            xy=(self.num_train_epochs, valid_losses[self.num_train_epochs - 1]), 
            xytext=(40, 30), 
            arrowprops=dict(arrowstyle='->', connectionstyle='angle3,angleA=10,angleB=-90', color='black', alpha=self.alpha),
            ha='center', va='center', 
            fontsize=9, 
            textcoords='offset points')

        # Add legend
        axes.legend(loc='best')

        return axes

    def accuracies(self, axes):
        """
        Plot validation and traning accuracy for each epoch
        """
        # Set figure
        axes.set_title('Model Accuracy', fontsize=14)
        axes.set_ylabel('Accuracy')
        axes.set_xlabel('Epoch')

        # Plot traning curve
        train_accuracy = np.array(self.epoch_results['train_score']) * 100
        axes.plot(self.epochs, train_accuracy, label='Traning', color='royalblue', alpha=self.alpha)

        # Plot validation curve
        valid_accuracy = np.array(self.epoch_results['valid_score']) * 100
        axes.plot(self.epochs, valid_accuracy, label='Validation', color='forestgreen', alpha=self.alpha)

        # Plot number of trained epoch for best model
        x = [self.num_train_epochs, self.num_train_epochs]
        y = np.sort([train_accuracy[self.num_train_epochs - 1], valid_accuracy[self.num_train_epochs - 1]])
        axes.plot(x, y, marker='x', color='purple', lw=0, label='Best model', alpha=self.alpha)

        # Annotation for best model on training set
        text_value = 'Train\n{:.2f}%'.format(train_accuracy[self.num_train_epochs - 1])
        axes.annotate(text_value,
            xy=(self.num_train_epochs, train_accuracy[self.num_train_epochs - 1]), 
            xytext=(-40, 30), 
            arrowprops=dict(arrowstyle='->', connectionstyle='angle3,angleA=10,angleB=-90', color='black', alpha=self.alpha),
            ha='center', va='center', 
            fontsize=9, 
            textcoords='offset points')
        
        # Annotation for best model on validation set
        text_value = 'Valid\n{:.2f}%'.format(valid_accuracy[self.num_train_epochs - 1])
        axes.annotate(text_value,
            xy=(self.num_train_epochs, valid_accuracy[self.num_train_epochs - 1]), 
            xytext=(40, -30), 
            arrowprops=dict(arrowstyle='->', connectionstyle='angle3,angleA=10,angleB=-90', color='black', alpha=self.alpha),
            ha='center', va='center', 
            fontsize=9, 
            textcoords='offset points')

        # Add legend
        axes.legend(loc='best')

        return axes

    def learning_rates(self, axes):
        """
        Plot learning rate for each epoch
        """
        # Set figure
        axes.set_title('Model Learning rate', fontsize=14)
        axes.set_ylabel('Learning rate')
        axes.set_xlabel('Epoch')

        # Plot learning rate curve
        learning_rates = self.epoch_results['learning_rate']
        axes.plot(self.epochs, learning_rates, label='Learning rate', color='royalblue', alpha=self.alpha)

        # Plot number of trained epoch for best model
        x = [self.num_train_epochs]
        y = [learning_rates[self.num_train_epochs - 1]]
        axes.plot(x, y, marker='x', color='purple', lw=0, label='Best model')

        # Annotation best model 
        text_value = 'Value\n{:.6f}'.format(learning_rates[self.num_train_epochs - 1]).rstrip('0')
        axes.annotate(text_value,
            xy=(self.num_train_epochs, learning_rates[self.num_train_epochs - 1]), 
            xytext=(-40, -30), 
            arrowprops=dict(arrowstyle='->', connectionstyle='angle3,angleA=10,angleB=-90', color='black', alpha=self.alpha),
            ha='center', va='center', 
            fontsize=9, 
            textcoords='offset points')

        # Add legend
        axes.legend(loc='best')

        return axes

    def performance(self, epoch_results):
        """
        Plot model performance including: loss, accuracy and learning rate
        """
        # Initialization
        self.epoch_results = epoch_results
        self.num_epochs = len(epoch_results['train_loss'])
        self.epochs = np.arange(1, self.num_epochs + 1, 1)
        self.num_train_epochs = epoch_results['train_epochs']

        # Plotting
        sns.set(style='darkgrid')
        fig, axes = plt.subplots(1, 3)

        self.losses(axes=axes[0])
        self.accuracies(axes=axes[1])
        self.learning_rates(axes=axes[2])

        # Show plot
        mng = plt.get_current_fig_manager()
        mng.full_screen_toggle()
        plt.show()
        plt.close()
        return


    def confusion_matrix(self, conf_matrix):
        """
        Plot given confusion matrix in dataframe format as heatmap
        """
        # Plot confusion matrix as heatmap
        plt.figure()
        axes = sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        axes.tick_params(axis='both', which='both', length=0)
        axes.set_xticklabels(axes.get_xticklabels(), rotation=-360, horizontalalignment='center')
        axes.set_title('Confusion matrix', fontsize=18, pad=20)
        axes.set_ylabel('Actual', fontsize=14)
        axes.set_xlabel('Predicted', fontsize=14)
        axes.xaxis.labelpad = 20

        # Show plot
        mng = plt.get_current_fig_manager()
        mng.full_screen_toggle()
        plt.show()
        plt.close()
        return


    def single_hyperparam(self, axes, hparam, hparam_values, scores, color, i_best_model):
        """
        Plot hyperparameter value vs score value for each evaluated model
        """
        # Set figure
        axes.set_title(hparam, fontsize=14)
        axes.set_ylabel('Score')
        #axes.set_xlabel(hparam)

        # Plotting
        axes.scatter(hparam_values, scores, marker='o', color=color, alpha=self.alpha)

        # Mark best model
        axes.scatter(hparam_values[i_best_model], scores[i_best_model], marker='D', color='green', alpha=self.alpha)

        # Annotation for best model 
        best_model_value = hparam_values[i_best_model]
        if type(best_model_value) is float or type(best_model_value) is np.float64:
            text_value = 'Value\n{:.6f}'.format(best_model_value).rstrip('0')
        else:
            text_value = 'Value\n{}'.format(str(best_model_value))
            
        axes.annotate(text_value,
            xy=(best_model_value, scores[i_best_model]), 
            xytext=(-40, -30), 
            arrowprops=dict(arrowstyle='->', connectionstyle='angle3,angleA=10,angleB=-90', color='black', alpha=self.alpha),
            ha='center', va='center', 
            fontsize=9, 
            textcoords='offset points')

        return axes

    def hyperparameters(self, results, hparams):
        """
        Plot hyperparameter vs score chart for each given hyperparameter
        """
        # Initialization
        scores = results['scores']
        settings = results['hparams']
        i_best_model = results['best_model_index']

        plots = UtilityMngr.split(hparams, part_size=self.max_charts_plot)

        # Plotting maximum number of charts per plot
        for plot in plots:

            # Set maximum number of charts per row
            num_columns = int(self.max_charts_plot / self.max_charts_row)
            fig, axes = plt.subplots(self.max_charts_row, num_columns)
            fig.subplots_adjust(hspace=0.35, wspace=0.25)

            # For each hyperameter collect values and draw a chart
            for i in range(self.max_charts_row):
                for j in range(num_columns):

                    # Check last row 
                    chart_index = i * num_columns + j
                    if chart_index >= len(plot):
                        break

                    # Collect value from each model setting
                    hparam = plot[chart_index]
                    hparam_values = []
                    for setting in settings:
                        hparam_values.append(setting[hparam])

                    # Calculate current axes
                    if self.max_charts_row == 1 and num_columns == 1:
                        curr_axes = axes
                    elif self.max_charts_row == 1 or num_columns == 1:
                        curr_axes = axes[chart_index]
                    else:
                        curr_axes = axes[i][j]

                    # Plot hyperparameter vs score chart
                    self.single_hyperparam(
                        axes=curr_axes,
                        hparam=hparam, 
                        hparam_values=hparam_values, 
                        scores=scores,
                        color=self.charts_colors[chart_index],
                        i_best_model=i_best_model)

            # Show i-th plot
            mng = plt.get_current_fig_manager()
            mng.full_screen_toggle()
            plt.show()
            plt.close()
        return


    def models(self, scores_by_models):
        """
        Plot descriptive statistics and distribution of scores per model
        """
        # Box plots + Violins
        fig, axes = plt.subplots()
        axes.boxplot(scores_by_models.values())
        axes.violinplot(scores_by_models.values(), showmeans=False, showextrema=False)
        axes.set_title('Models', fontsize=18, pad=20)
        axes.set_ylabel('Accuracy', fontsize=14)
        axes.set_xticklabels(scores_by_models.keys())

        # Show plot
        mng = plt.get_current_fig_manager()
        mng.full_screen_toggle()
        plt.show()
        plt.close()
        return

    def metrics_analysis(self):
        """
        Analysis of metrics
        """
        # Data
        results = pd.read_excel('data/results.xlsx', header=0)
        results.columns = ['Configurations', 'Complexity', 'Speed', 'Throughput', 'Training Time', 'Memory Usage', 'Accuracy']
        results = results.set_index('Configurations')
        print(results)
        
        # Correlations
        self.correlations(results)

        # Accuracy vs Complexity
        self.scatter_plot(
            data=results,
            x_label='Complexity',
            y_label='Accuracy',
            models=results.index,
            title='Accuracy vs Complexity',
            thresholds=[],
        )
        # Accuracy vs Throughput
        self.scatter_plot(
            data=results,
            x_label='Throughput',
            y_label='Accuracy',
            models=results.index,
            title='Accuracy vs Throughput',
            thresholds=[15, 30],
        )
        # Accuracy vs Training Time
        self.scatter_plot(
            data=results,
            x_label='Training Time',
            y_label='Accuracy',
            models=results.index,
            title='Accuracy vs Training Time',
            thresholds=[60],
        )
        # Accuracy vs Memory Usage
        self.scatter_plot(
            data=results,
            x_label='Memory Usage',
            y_label='Accuracy',
            models=results.index,
            title='Accuracy vs Memory Usage',
            thresholds=[512, 1024],
        )

        # Accuracy density
        self.scatter_plot(
            data=results,
            x_label='Accuracy density',
            y_label='Accuracy',
            models=results.index,
            title='Accuracy density',
            thresholds=[],
            x_data=results['Accuracy'] / results['Complexity']
        )

        return

    def correlations(self, data):
        """
        Calculate correlation matrix
        """
        # Calculate correlation
        corr_matrix = data.corr(method='pearson')
        print(corr_matrix)

        # Show as heatmap
        plt.figure()
        axes = sns.heatmap(corr_matrix, annot=True, cmap='Reds')
        axes.tick_params(axis='both', which='both', length=0)
        axes.set_xticklabels(axes.get_xticklabels(), rotation=-360, horizontalalignment='center')
        axes.set_title('Pearson correlation matrix', fontsize=18, pad=20)
        axes.xaxis.labelpad = 20
    
        plt.show()
        plt.close()
        return corr_matrix

    def scatter_plot(self, data, x_label, y_label, models, title, thresholds=None, x_data=None, y_data=None):
        """
        Scatter plot
        """
        # Set axes data
        if x_data is None:
            x_data = data[x_label]
        if y_data is None:
            y_data = data[y_label]

        # Scatter plot
        axes = sns.scatterplot(
            x=x_data, 
            y=y_data, 
            hue=models,  
            size=data['Complexity'],
            sizes=(100, 1000)
        )

        # Tresholds
        if thresholds is not None or len(thresholds) > 0:
            max_val = y_data.max() + 1
            for threshold in thresholds:
                sns.lineplot(
                    x=[threshold, threshold],
                    y=[0, max_val],
                    color='red', 
                    lw=2, 
                    axes=axes)
        
        # Settings
        axes.set_title(title, fontsize=18, pad=20)
        axes.set_xlabel(x_label, fontsize=12)
        axes.set_ylabel(y_label, fontsize=12)

        # Show plot
        #mng = plt.get_current_fig_manager()
        #mng.full_screen_toggle()
        plt.show()
        plt.close()
        return


if __name__ == "__main__":

    # Initialization
    plot = PlotMngr()

    # Performance
    epoch_results = {
        'train_loss': np.linspace(0.5, 0.001, 10),
        'valid_loss': np.linspace(0.7, 0.01, 10),  
        'train_score': np.linspace(0.71, 0.96, 10), 
        'valid_score': np.linspace(0.65, 0.88, 10),
        'learning_rate': np.linspace(0.01, 0.00001, 10),
        'train_epochs': 8
    }
    plot.performance(epoch_results)

    # Confusion matrix
    conf_matrix = pd.DataFrame([[1, 2, 2], [3, 4, 0], [3, 5, 1]])
    plot.confusion_matrix(conf_matrix)

    # Hyperparameters
    setting1 = Settings(kind=0, input_size=(3, 32, 32), num_classes=10, batch_size=32, learning_rate=0.001, lr_factor=0.1, lr_patience=20, early_stop=True, es_patience=25)
    setting2 = Settings(kind=0, input_size=(3, 32, 32), num_classes=10, batch_size=128, learning_rate=0.01, lr_factor=0.3, lr_patience=19, early_stop=True, es_patience=27)
    setting3 = Settings(kind=0, input_size=(3, 32, 32), num_classes=10, batch_size=64, learning_rate=0.01, lr_factor=0.5, lr_patience=25, early_stop=False, es_patience=0)
    results = {
        'scores': [56, 71, 65],
        'hparams': [setting1.get_hparams(), setting2.get_hparams(), setting3.get_hparams()],
        'best_model_index': 1
    }
    hparams = ['batch_size', 'learning_rate', 'lr_factor', 'lr_patience', 'early_stop', 'es_patience', 'gc_value']
    plot.hyperparameters(results, hparams=hparams)

    # Models statistics
    models_scores = {'vggnet': [0.75, 0.86], 'resnet': [0.74, 0.88], 'densenet': [0.68, 0.71]}
    #models_scores = UtilityMngr.get_models_scores()
    print(models_scores)
    #plot.models(models_scores)

    # Metrics analysis
    plot.metrics_analysis()
