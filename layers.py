#!/usr/local/bin python
#! -*- coding: utf-8 -*-

import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample
import numpy as np
from utils import *


class FullyConnectedLayer(object):
	def __init__(
		self,
		rng,
		input=None, 
		n_input=784,
		n_output=10,
		activation=None,
		W=None,
		b=None
	):

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
