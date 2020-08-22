
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm 
import seaborn as sns
import torch

from torchvision import datasets, transforms
from settings import Settings


class DataMngr:
    """
    Class for loading and preprocessing images from CINIC-10 dataset

    About data
        CINIC-10 is an augmented extension of CIFAR-10. 
        It contains the images from CIFAR-10 (60 000 images, 32x32 RGB pixels) and a selection of ImageNet database images (210,000 images downsampled to 32x32). 
        It contains 10 category same as CIFAR-10, except its larger and more difficult to train
        It's split into 3 equal subsets - train, validation and test - each of which contain 90 000 images.
        Paper source: https://arxiv.org/pdf/1810.03505v1.pdf
        Data source: https://github.com/BayesWatch/cinic-10

    About pin_memory
        Source: https://discuss.pytorch.org/t/pin-memory-vs-sending-direct-to-gpu-from-dataset/33891

    About normalization
        The idea is to normalize data per channels, because according to this paper [1] it avoids vanishing gradients
        For clarity is good to check batch normalization paper
        Source: [1] https://openreview.net/pdf?id=rJxBv4r22V    
                [2] https://stackoverflow.com/questions/45799926/why-batch-normalization-over-channels-only-in-cnn
    
    About reproducibility
        Using the same software and hardware and not using the specifc kind of methods in pytorch the deterministic behavior is guaranteed [1, 3]
        Also random transformations for data agumentation should not be used for determististic behavior [2]
        Source: [1] https://pytorch.org/docs/stable/notes/randomness.html
                [2] https://discuss.pytorch.org/t/how-to-get-deterministic-behavior/18177/4
                [3] https://discuss.pytorch.org/t/clarification-for-reproducibility-and-determinism/53928
    """

    ROOT_DIR = os.getcwd()
    OUTPUT_DIR = os.path.join(ROOT_DIR, 'data/output')
    CINIC_DIR = os.path.join(ROOT_DIR, 'data/CINIC-10')
    TRAIN_DIR = os.path.join(CINIC_DIR , 'train')
    VALID_DIR = os.path.join(CINIC_DIR , 'valid')
    TEST_DIR = os.path.join(CINIC_DIR , 'test')

    def __init__(self, setting):
        super().__init__()
        self.setting = setting
        self.batch_size = self.setting.batch_size
        self.data_augment = self.setting.data_augment
        #self.data_norm = self.setting.data_norm

        # Normalization parameters calculated from train dataset for each channel
        self.cinic_mean = [0.47889522, 0.47227842, 0.43047404]
        self.cinic_std = [0.24205776, 0.23828046, 0.25874835]

    def inv_normalized(self, tensor):
        """
        Return original tensor from normalized tensor with given mean and standard deviation
        Source: https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/3
        """
        for t, m, s in zip(tensor, self.cinic_mean, self.cinic_std):
            # Normalization for each channel = (t - m) / s
            t.mul_(s).add_(m)
        return tensor

    def info(self, data, title, visualize=False, show_normalized=False):
        """
        Show information about given dataset
        """
        # Print basic information
        print('\n', data.dataset, '\n')
        classes = data.dataset.classes
        print('Num of classes = {}\n{}\n'.format(len(classes), classes))

        # Initialize dictionary for counting
        totals = 0
        counts = {}
        for i in range(len(classes)):
            counts[i] = 0

        # Iterate over dataset batches
        for i, batch in enumerate(data):
            X, y = batch
            if i == 0:
                print('Batch', i+1)
                print('\t', 'X has', len(X), 'samples')
                print('\t', 'X has shape', X.shape)

            # Iterate over batch samples
            if visualize:
                for j, sample in enumerate(X):
                    if i == 0:
                        y_class = data.dataset.classes[y[j]]
                        print('\t\t', 'Sample {} is {} [{}]'.format(j+1, y_class, y[j]))
                        if not show_normalized:
                            sample = self.inv_normalized(sample)      # otherwise image will be clipped into range [0..1] to be able to display
                        image = sample.numpy().transpose((1, 2, 0))   # move channels to the end
                        plt.imshow(image)
                        plt.show()
                        plt.close()

            # Iterate over batch labels
            for j, label in enumerate(y):
                counts[int(label)] += 1
                totals += 1

        # Plot classes distribution
        sns.set(style='darkgrid')
        plt.figure()
        plt.title('Distribution of classes in {} set'.format(title), fontsize=14)
        plt.ylabel('Number of samples')
        plt.xlabel('Classes')
        plt.xticks(rotation=35)
        #plt.ylim(0, max(counts.values()))
        
        cmap = cm.get_cmap('tab10')
        colors = cmap(range(len(classes)))
        bars = plt.bar(x=counts.keys(), height=counts.values(), tick_label=classes, color=colors, alpha=0.7)

        for i, bar in enumerate(bars):
            y_pos = bar.get_height() * 0.95
            x_pos = bar.get_x() + bar.get_width() / 2
            val = counts[i] / totals * 100
            text = '{}\n{:.2f}%'.format(bar.get_height(), val)
            plt.text(x_pos, y_pos, text, color='black', ha='center', va='center', fontsize=10)

        plt.show()
        plt.close()
        return

    def load_train(self):
        """
        Load images for training
        """
        # Transforms on image
        transformers = transforms.Compose([
            #transforms.RandomHorizontalFlip(p=0.5), # Horizontally flip the given image randomly with a given probability
            transforms.ToTensor(),                   # Convert to tensor and automatically normalize to [0..1] range
            transforms.Normalize(mean=self.cinic_mean, std=self.cinic_std)])      # Normalize with given parameters calculated on training set

        # Load data
        trainset = torch.utils.data.DataLoader(
            datasets.ImageFolder(DataMngr.TRAIN_DIR, transformers), 
            batch_size=self.batch_size,                 # how many samples per batch to load
            shuffle=True,                               # set to True to have the data reshuffled at every epoch
            pin_memory=True,                            # if True, it will enables faster data transfer from CPU to GPU by using page-locked memory
            num_workers=15)                             # how many subprocesses to use for data loading

        return trainset

    def load_valid(self):
        """
        Load images for validation
            It is thus preferred to use the non-random versions of the transformations for the validation/test to get consistent predictions.
            For the validation/test dataset, we don't need any augmentation
        Source: https://discuss.pytorch.org/t/data-transformation-for-training-and-validation-data/32507
        """
        # Transforms on image
        transformers = transforms.Compose([
            transforms.ToTensor(),                  # Convert to tensor and automatically normalize to [0..1] range   
            transforms.Normalize(mean=self.cinic_mean, std=self.cinic_std)])      # Normalize with given parameters calculated on training set

        # Load data
        validset = torch.utils.data.DataLoader(
            datasets.ImageFolder(DataMngr.VALID_DIR, transformers), 
            batch_size=self.batch_size,                 # how many samples per batch to load
            shuffle=True,                               # set to True to have the data reshuffled at every epoch, because it's used to check training performance while traning model
            pin_memory=True,                            # if True, it will enables faster data transfer from CPU to GPU by using page-locked memory
            num_workers=15)                             # how many subprocesses to use for data loading

        return validset

    def load_test(self, sample_size=10):
        """
        Load images for testing
            It is thus preferred to use the non-random versions of the transformations for the validation/test to get consistent predictions.
            For the validation/test dataset, we don't need any augmentation
        Source: https://discuss.pytorch.org/t/data-transformation-for-training-and-validation-data/32507
        """
        # Transforms on image
        transformers = transforms.Compose([
            transforms.ToTensor(),                  # Convert to tensor and automatically normalize to [0..1] range
            transforms.Normalize(mean=self.cinic_mean, std=self.cinic_std)])      # Normalize with given parameters calculated on training set

        images = datasets.ImageFolder(DataMngr.TEST_DIR, transformers)
        num_images = len(images)
        batch_size = int(num_images / sample_size)

        # Load data
        testset = torch.utils.data.DataLoader(
            dataset=images, 
            batch_size=batch_size,                      # one sample per batch for testing, mostly because of inference time measuring
            shuffle=True,                               # data will be split into n segments for creating sample of metric for statistical testing
            pin_memory=True,                            # if True, it will enables faster data transfer from CPU to GPU by using page-locked memory
            num_workers=15)                             # how many subprocesses to use for data loading

        return testset

    @staticmethod
    def set_reproducible_mode(seed=0, deterministic=False):
        """
        Set reproducible behaviour with fixed seed for all devices (both CPU and CUDA) for pytorch library,
            and for all others used libraries such as numpy, random, etc. [1]
        
        Using the same software and hardware and not using the specifc kind of methods in pytorch the deterministic behavior is guaranteed [1, 4]
        If the absolute error is approx. 1e-6 it might be due to the usage of float values [3]
        Also random transformations for data agumentation should not be used for determististic behavior [3]
        You need to call this method every time before call initialization of data-loader and order of samples in batches is guaranteed !

        About determinism
            In deterministic algorithm, for a given particular input, the computer will always produce the same output going through the same states, 
            but in case of non-deterministic algorithm for the same input, the compiler may produce different output in different runs [2]

            Deterministic operation may have a negative single-run performance impact, depending on the composition of your model.
            Processing speed (e.g. the number of batches trained per second) may be lower than when the model functions non-deterministically. [1]

        Source: [1] https://pytorch.org/docs/stable/notes/randomness.html
                [2] https://www.tutorialspoint.com/difference-between-deterministic-and-non-deterministic-algorithms
                [3] https://discuss.pytorch.org/t/how-to-get-deterministic-behavior/18177/4
                [4] https://discuss.pytorch.org/t/clarification-for-reproducibility-and-determinism/53928
        """
        # Used libraries
        np.random.seed(seed)
        random.seed(seed)

        # Torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        return



if __name__ == "__main__":

    # Create setting
    setting = Settings(
        input_size=(3, 32, 32),
        num_classes=10,
        batch_size=16)

    data = DataMngr(setting)

    # Training set
    trainset = data.load_train()
    data.info(trainset, 'training', visualize=True, show_normalized=True)

    # Validation set
    validset = data.load_valid()
    data.info(validset, 'validation', visualize=True, show_normalized=False)

    # Test set
    DataMngr.set_reproducible_mode(seed=21)
    testset = data.load_test(sample_size=9000)

    print('=============')
    DataMngr.set_reproducible_mode(seed=21)
    #data.info(testset, 'test', visualize=True)
    for i, batch in enumerate(testset):
        X, y = batch
        if i < 10:
            print(y)

    print('=============')
    DataMngr.set_reproducible_mode(seed=21)
    #data.info(testset, 'test', visualize=True)
    for i, batch in enumerate(testset):
        X, y = batch
        if i < 10:
            print(y)