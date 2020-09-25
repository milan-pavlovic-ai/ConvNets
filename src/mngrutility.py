
import torch
import random
import numpy as np
import scipy.stats as stats
from functools import cmp_to_key

class UtilityMngr:
    """
    Static class for Utilities
    """

    @staticmethod
    def split(array, part_size):
        """
        Split array into equal parts with given split size except possible last part
        """
        chunks = []
        for i in np.arange(0, len(array)):
            start_ind = i * part_size
            array_part = array[start_ind : start_ind + part_size]
            if len(array_part) > 0:
                chunks.append(array_part)
        return chunks

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
   
    @staticmethod
    def get_models_scores():
        """
        Returns list of models with their names and scores
        """
        # Models
        models = {}

        # VGGNet
        import vggnet
        name, scores = vggnet.process_load('data/output/VGGNet11-1600959117-tuned.tar', testing=True)
        models[name] = scores

        # InceptionNet
        import inceptionnet_v1
        name, scores = inceptionnet_v1.process_load('data/output/InceptionNetV1-1600908899-tuned.tar', testing=True)
        models[name] = scores
        
        # ResNet
        import resnet
        name, scores = resnet.process_load('data/output/ResNet26-1600717670-tuned.tar', testing=True)
        models[name] = scores

        # SqueezeNet
        import squeezenet
        name, scores = squeezenet.process_load('data/output/SqueezeNet1.1-1600730105-tuned.tar', testing=True)
        models[name] = scores
        
        # DenseNet
        import densenet
        name, scores = densenet.process_load('data/output/DenseNet121-1600733395-tuned.tar', testing=True)
        models[name] = scores

        # MobileNet
        import mobilenet_v1
        name, scores = mobilenet_v1.process_load('data/output/MobileNetV1-1600756521-tuned.tar', testing=True)
        models[name] = scores

        # ShuffleNet
        import shufflenet_v1
        name, scores = shufflenet_v1.process_load('data/output/ShuffleNetV1g4-1600946548-tuned.tar', testing=True)
        models[name] = scores
        
        # SEResNet
        import se_resnet
        name, scores = se_resnet.process_load('data/output/SEResNet26-1600789031-tuned.tar', testing=True)
        models[name] = scores

        # SKResNet
        import sk_resnet
        name, scores = sk_resnet.process_load('data/output/SKResNet26-1600653648-best_score.tar', testing=True)
        models[name] = scores

        return models


if __name__ == "__main__":
    
    # Split
    array = np.arange(1, 15, 1)
    parts = UtilityMngr.split(array, part_size=5)
    print('Original array:', array, sep='\n')
    print('Parts:', parts, sep='\n')

    # Reproducibility
    rand_arr_1 = np.random.rand(1, 10)
    rand_arr_2 = np.random.rand(1, 10)
    print('Random np-array 1:', rand_arr_1, sep='\n')
    print('Random np-array 2:', rand_arr_2, sep='\n')

    rand_tensor_1 = torch.rand(1, 10)
    rand_tensor_2 = torch.rand(1, 10)
    print('Random tensor 1:', rand_tensor_1, sep='\n')
    print('Random tensor 2:', rand_tensor_2, sep='\n')

    UtilityMngr.set_reproducible_mode(seed=21)
    rand_arr_1 = np.random.rand(1, 10)
    UtilityMngr.set_reproducible_mode(seed=21)
    rand_arr_2 = np.random.rand(1, 10)
    print('[R] Random np-array 1:', rand_arr_1, sep='\n')
    print('[R] Random np-array 2:', rand_arr_2, sep='\n')

    UtilityMngr.set_reproducible_mode(seed=21)
    rand_tensor_1 = torch.rand(1, 10)
    UtilityMngr.set_reproducible_mode(seed=21)
    rand_tensor_2 = torch.rand(1, 10)
    print('[R] Random tensor 1:', rand_tensor_1, sep='\n')
    print('[R] Random tensor 2:', rand_tensor_2, sep='\n')