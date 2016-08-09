#!/usr/local/bin python
#! -*- coding: utf-8 -*-

import sys,os
sys.path.append(os.pardir)

import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from dnn.layers import FullyConnectedLayer
from dnn.optimizers import *
from dnn.utils import *
from tqdm import tqdm


plt.style.use('ggplot')

"""Sparse Autoencoder"""
class SA(object):
	def __init__(self, rng,	input=None,	n_visible=784, 
		n_hidden=784, sparse_reg=1e-3, optimizer=Adam, W=None, b=None):
		self.rng = rng

		"""symbol definition"""
		self.index = T.lscalar()
		self.s_level = T.fscalar()
		if input == None:
			self.x = T.matrix('x')
		else:
			self.x = input

		"""network structure definition"""
		"""encoder"""
		self.h = FullyConnectedLayer(
			self.rng,
			input=self.x,
			n_input=n_visible,
			n_output=n_hidden,
			activation=sigmoid,
			W=W,
			b=b
		)
		"""decoder"""
		self.y = FullyConnectedLayer(
			self.rng,
			input=self.h.output,
			n_input=n_hidden,
			n_output=n_visible,
			activation=sigmoid
		)

		"""loss accuracy error"""
		self.metric = Metric(self.y.output, self.x)
		sparsity_penalty = self.sparsity_penalty(self.h.output, sparsity_level=self.s_level, sparse_reg=sparse_reg, n_units=n_hidden)
		self.loss = self.metric.mean_squared_error() + sparsity_penalty

		"""parameters (i.e., weights and biases) for whole networks"""
		self.params = self.h.params + self.y.params

		"""optimizer for learning parameters"""
		self.optimizer = optimizer(params=self.params)
		self.updates = self.optimizer.updates(self.loss)

	def kl_divergence(self, p, p_hat):
		return p * T.log(p/p_hat) + (1-p)*T.log((1-p)/(1-p_hat))

	def sparsity_penalty(self, h, sparsity_level=None, sparse_reg=1e-3, n_units=-1):
		sparsity_level = T.extra_ops.repeat(sparsity_level, n_units)
		avg_act = h.mean(axis=0)
		kl_div = self.kl_divergence(sparsity_level, avg_act)
		sparsity_penalty = sparse_reg * kl_div.sum()
		return sparsity_penalty

	def fit(self, x_train, x_valid, batchsize=128, n_epoch=10):
		self.batchsize = batchsize
		self.n_epoch = n_epoch

		"""data pre-processing"""
		self.x_train = shared_data(x_train, None)
		self.x_valid = shared_data(x_valid, None)
		self.n_train_batches = self.x_train.get_value(borrow=True).shape[0] / self.batchsize
		self.n_valid_batches = self.x_valid.get_value(borrow=True).shape[0] / self.batchsize

		self.train_model = theano.function(
			inputs = [self.index],
			outputs = self.loss,
			updates = self.updates,
			givens = {
				self.x: self.x_train[self.index*batchsize: (self.index+1)*batchsize],
				self.s_level: np.cast['float32'](1e-2),
			}
		)

		self.valid_model = theano.function(
			inputs = [self.index],
			outputs = self.loss,
			givens = {
				self.x: self.x_valid[self.index*batchsize: (self.index+1)*batchsize],
				self.s_level: np.cast['float32'](1e-2),
			}
		)


		epoch = 0
		loss = []
		val_loss = []

		while epoch < self.n_epoch:
			epoch += 1

			loss.append(0.0)
			for batch_index in tqdm(xrange(self.n_train_batches)):
				batch_loss = self.train_model(batch_index)
				loss[-1] += batch_loss
			loss[-1] /= self.n_train_batches
			print 'epoch: {}, train mean loss={}'.format(epoch, loss[-1])

			val_loss.append(0.0)
			for batch_index in xrange(self.n_valid_batches):
				batch_loss = self.valid_model(batch_index)
				val_loss[-1] += batch_loss
			val_loss[-1] /= self.n_valid_batches
			print 'epoch: {}, valid mean loss={}'.format(epoch, val_loss[-1])

		hist = {}
		hist["loss"] = loss
		hist["val_loss"] = val_loss
		return hist

	def get_encoder_params(self):
		return self.h.params


if __name__ == '__main__':
	random_state = 1234

	print 'fetch MNIST dataset'
	mnist = fetch_mldata('MNIST original')
	mnist.data   = mnist.data.astype(np.float32)
	mnist.data  /= 255
	mnist.target = mnist.target.astype(np.int32)

	x_train, x_valid,\
	y_train, y_valid \
	= train_test_split(mnist.data, mnist.target, random_state=random_state, test_size=0.1)
	print "... done"

	n_visible = x_train.shape[1]
	n_hidden = 784
	rng = np.random.RandomState(random_state)
	sa = SA(rng, n_visible=n_visible, n_hidden=n_hidden, sparse_reg=1e-3, optimizer=Adam)
	hist = sa.fit(x_train, x_valid, n_epoch=30)

	df = pd.DataFrame(hist)
	df.index += 1
	df.index.name = "epoch"

	df[["loss", "val_loss"]].plot(linewidth=2, alpha=0.6)
	plt.ylabel("loss function value")
	plt.title("sparse autoencoder example")
	plt.show()

	from PIL import Image
	image = Image.fromarray(tile_raster_images(
		X=sa.h.W.get_value(borrow=True).T,
	    img_shape=(28, 28), tile_shape=(15, 15),
	    tile_spacing=(1, 1)))
	image.show()
	image.save('sa_result.png')


