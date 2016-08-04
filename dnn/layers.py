#!/usr/local/bin python
#! -*- coding: utf-8 -*-

import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal.pool import pool_2d
import numpy as np
from utils import *


class FullyConnectedLayer(object):
	def __init__(self, rng, input=None, n_input=784, n_output=10, activation=None, W=None, b=None):

		self.input = input

		if W is None:
			W_values = np.asarray(
				rng.uniform(low=-np.sqrt(6.0/(n_input+n_output)),
							high=np.sqrt(6.0/(n_input+n_output)),
							size=(n_input, n_output)),
				dtype=theano.config.floatX)
			if activation == sigmoid:
				W_values *= 4.0
			W = theano.shared(value=W_values, name='W', borrow=True)

		if b is None:
			b_values = np.zeros((n_output,), dtype=theano.config.floatX)
			b = theano.shared(value=b_values, name='b', borrow=True)

		self.W = W
		self.b = b

		linear_output = T.dot(input, self.W) + self.b

		if activation is None:
			self.output = linear_output
		else:
			self.output = activation(linear_output)

		self.params = [self.W, self.b]


class Convolutional2DLayer(object):
	def __init__(self, rng, input, filter_shape=None, image_shape=None):
		self.input = input
		self.rng = rng

		"""filter shape = n_feature_map, in channel, width, height"""
		"""channel * width * height"""
		fan_in = np.prod(filter_shape[1:]) # 1*2*3
		"""feature map """
		fan_out = filter_shape[0] * np.prod(filter_shape[2:]) # 0*2*3
		
		W_bound = np.sqrt(6.0 / (fan_in + fan_out))
		self.W = theano.shared(
			np.asarray(
				self.rng.uniform(
					low = -W_bound,
					high = W_bound,
					size = filter_shape
				),
				dtype = theano.config.floatX
			),
			borrow = True
		)

		b_values = np.zeros((filter_shape[0],),	dtype=theano.config.floatX)
		self.b = theano.shared(b_values, borrow=True)

		conv_out = conv.conv2d(
			input=self.input,
			filters=self.W,
			filter_shape=filter_shape,
			image_shape=image_shape
		)

		self.output = conv_out + self.b.dimshuffle('x', 0, 'x', 'x')
		self.params = [self.W, self.b]

class MaxPooling2DLayer(object):
	def __init__(self, input, poolsize=(2,2)):
		pooled_out = pool_2d(
			input=input,
			ds=poolsize,
			ignore_border=True,
			mode="max"
		)
		self.output = pooled_out


class BatchNormalizationLayer(object):
	def __init__(self, input, shape=None):
		self.shape = shape
		if len(shape) == 2: # for fully connnected
			gamma = theano.shared(value=np.ones(shape[1], dtype=theano.config.floatX), name="gamma", borrow=True)
			beta = theano.shared(value=np.zeros(shape[1], dtype=theano.config.floatX), name="beta", borrow=True)
			mean = input.mean((0,), keepdims=True)
			var = input.var((0,), keepdims=True)
		elif len(shape) == 4: # for cnn
			gamma = theano.shared(value=np.ones(shape[1:], dtype=theano.config.floatX), name="gamma", borrow=True)
			beta = theano.shared(value=np.zeros(shape[1:], dtype=theano.config.floatX), name="beta", borrow=True)
			mean = input.mean((0,2,3), keepdims=True)
			var = input.var((0,2,3), keepdims=True)
			mean = self.change_shape(mean)
			var = self.change_shape(var)

		self.params = [gamma, beta]
		self.output = gamma * (input - mean) / T.sqrt(var + 1e-5) + beta
	
	def change_shape(self, vec):
		ret = T.repeat(vec, self.shape[2]*self.shape[3])
		ret = ret.reshape(self.shape[1:])
		return ret





