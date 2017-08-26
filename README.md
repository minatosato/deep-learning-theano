
# Deep Learning Implementation with Theano

## About
This repository is simple implementation of some deep learning algorithms.

### MNIST Examples
#### Usage
e.g.
```
cd example
python mlp.py
```

#### Files
- Multi-Layer Perceptron (`example/mlp.py`)
- Denoising Autoencoder (`example/da.py`)
- Sparse Autoencoder (`example/sa.py`)
- Convolutional Neural Network (`example/cnn.py`)


### Layers
in `dnn/layer.py`

- Fully-Connected Layer
- 2-Dimensional Convolutional Layer
- 2-Dimensional Max Pooling Layer
- Batch Normalization Layer


### Optimizers
in `dnn/optimizers.py`

- Stochastic Gradient Descent (SGD)
- Momentum Stochastic Gradient Descent (Momentum SGD)
- AdaGrad
- RMSprop
- AdaDelta
- Adam

## Requirements
- Python3
- [NumPy](http://www.numpy.org)
- [Theano](http://deeplearning.net/software/theano/) == 0.9.0
- [Scikit-Learn](http://scikit-learn.org/stable/)
- [tqdm](https://pypi.python.org/pypi/tqdm)
- [Pandas](http://pandas.pydata.org) (for visualization)
- [Matplotlib](http://matplotlib.org) (for visualization)

![](https://dl.dropboxusercontent.com/u/38631959/deep-learning-theano.png)
