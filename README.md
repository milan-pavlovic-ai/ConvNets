# Convolutional Neural Networks
Architectures of convolutional neural networks for image classification in PyTorch.

This repository implements various architectures of Convolutional Neural Networks (CNNs) for image classification tasks using PyTorch.
Each model showcases unique advancements in CNN design, addressing key challenges in accuracy, computational efficiency, and model size.
The repository also provides clear instructions on how to train, evaluate, and experiment with each architecture, making it easier to understand and apply these models in real-world image classification tasks.

## Architectures

* ### VGGNet

    Very Deep Convolutional Networks for Large-Scale Image Recognition, 2014
    
    https://arxiv.org/pdf/1409.1556v6.pdf

    VGGNet introduced a simple yet effective architecture that stacks very deep networks of 3x3 convolutions.
    This deep structure allows for more complex feature extraction, and VGGNet achieved significant success in large-scale image classification tasks, though with a large number of parameters.
        

* ### InceptionNet-v1 (GoogLeNet)

    Going Deeper with Convolutions, 2014

    https://arxiv.org/pdf/1409.4842v1.pdf

    InceptionNet (or GoogLeNet) is known for its innovative approach to network architecture, combining convolutional layers with different filter sizes (1x1, 3x3, 5x5) in parallel.
    This allows the model to learn multi-scale features efficiently and helps in reducing computational cost through dimensionality reduction layers (1x1 convolutions).

* ### ResNet

    Deep Residual Learning for Image Recognition, 2015

    https://arxiv.org/pdf/1512.03385v1.pdf

    ResNet introduced the concept of residual learning by using skip connections that bypass one or more layers.
    This simple but powerful idea helps train very deep networks by mitigating the vanishing gradient problem, enabling networks with hundreds or even thousands of layers to be trained effectively.

* ### ResNeXt
    Aggregated Residual Transformations for Deep Neural Networks, 2016
    
    https://arxiv.org/pdf/1611.05431.pdf

    Aggregated residual transformations for deeper networks with better performance.
    ResNeXt builds on the idea of residual connections but introduces a cardinality dimension—multiple paths with shared parameters within the same block.
    This leads to a more efficient network with enhanced performance, allowing for greater depth and complexity without a significant increase in computation.

* ### SqueezeNet-v1

    SqueezeNet: AlexNet-Level Accuracy with 50x Fewer Parameters and 0.5Mb Model Size, 2016

    https://arxiv.org/pdf/1602.07360v4.pdf

    SqueezeNet significantly reduces the number of parameters compared to architectures like AlexNet while maintaining similar accuracy.
    The key innovation is the "fire module," which reduces the number of input channels using a 1x1 convolution before expanding them with a 3x3 convolution.
    This drastically decreases the model size, making it suitable for deployment in resource-constrained environments.

* ### DenseNet

    Densely Connected Convolutional Networks, 2017

    https://arxiv.org/pdf/1608.06993v5.pdf

    DenseNet connects each layer to every other layer in a feedforward manner.
    Each layer receives input from all preceding layers, which leads to efficient feature reuse, stronger gradient flow, and reduced parameter counts.
    DenseNet is known for its superior performance on tasks like object recognition and segmentation.

* ### MobileNet-v1

    MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications, 2017

    https://arxiv.org/pdf/1704.04861v1.pdf

    MobileNet is designed to be highly efficient, with a focus on mobile and embedded vision applications.
    It uses depthwise separable convolutions to drastically reduce the computational cost and number of parameters, achieving a good balance between performance and efficiency.

* ### ShuffleNet-v1

    ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices, 2017

    https://arxiv.org/pdf/1707.01083v2.pdf

    ShuffleNet also targets mobile devices and focuses on optimizing computational efficiency.
    It uses group convolutions combined with channel shuffle operations to enable high performance at low computational cost, achieving fast inference while maintaining competitive accuracy.

* ### SE-ResNet and SENet

    Squeeze-and-Excitation Networks, 2017

    https://arxiv.org/pdf/1709.01507v4.pdf

    SENet introduces the concept of "squeeze-and-excitation" blocks, which explicitly model inter-channel dependencies to recalibrate feature responses adaptively.
    This mechanism improves the representational power of the network, leading to better performance in tasks like image classification and object detection.

* ### SK-ResNet and SKNet

    Selective Kernel Networks, 2019

    https://arxiv.org/pdf/1903.06586v2.pdf

    SKNet further extends the idea of adaptive kernels with its selective kernel mechanism, which dynamically chooses the best kernel size based on the input.
    This adaptive approach helps improve performance on diverse datasets by allowing the network to choose the most relevant features for each task.


TO BE CONTINUED ...
