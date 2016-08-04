#!/usr/local/bin python
#! -*- coding: utf-8 -*-

import sys,os
sys.path.append(os.pardir)

import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
from tqdm import tqdm
import pandas as pd

# original
from dnn.layers import FullyConnectedLayer
from dnn.utils import *
from dnn.optimizers import *

plt.style.use('ggplot')

class MLP(object):
	def __init__(self, rng, n_input=784, n_hidden=[500, 500, 500], n_output=10, optimizer=AdaDelta):

		self.rng = rng
		self.batchsize = 100

		self.n_input = n_input
		self.n_hidden = n_hidden
		self.n_output = n_output
		self.n_layer = len(n_hidden)

		self.L1_reg = 0.0
		self.L2_reg = 0.001

		"""symbol definition"""
		self.index = T.lscalar()
		self.x = T.matrix('x')
		self.y = T.ivector('y')
		self.train = T.iscalar('train')

		"""network structure definition"""
		self.layers = []
		self.params = []
		for i in xrange(self.n_layer+1):
			"""for first hidden layer"""
			if i == 0:
				layer_n_input = self.n_input
				layer_n_output = self.n_hidden[0]
				layer_input = dropout(self.rng, self.x, self.train, p=0.1)
				activation=relu
			elif i != self.n_layer:
				layer_n_input = self.n_hidden[i-1]
				layer_n_output = self.n_hidden[i]
				layer_input = dropout(self.rng, self.layers[-1].output, self.train)
				activation=relu
			else:
				"""for output layer"""
				layer_n_input = self.n_hidden[-1]
				layer_n_output = self.n_output
				layer_input = self.layers[-1].output
				activation=None	


			layer = FullyConnectedLayer(
				self.rng,
				input=layer_input,
				n_input=layer_n_input,
				n_output=layer_n_output,
				activation=activation
			)
			self.layers.append(layer)
			self.params.extend(layer.params)

		"""regularization"""
		# self.L1 = abs(self.h1.W).sum() + abs(self.pred_y.W).sum()
		# self.L2 = abs(self.h1.W**2).sum() + abs(self.pred_y.W**2).sum()

		"""loss accuracy error"""
		self.metric = Metric(self.layers[-1].output, self.y)
		self.loss = self.metric.negative_log_likelihood()# + L1_reg*self.L1 + L2_reg*self.L2
		self.accuracy = self.metric.accuracy()
		self.errors = self.metric.errors()

		"""parameters (i.e., weights and biases) for whole networks"""
		# self.params

		"""optimizer for learning parameters"""
		self.optimizer = optimizer(params=self.params)
		
		"""definition for optimizing update"""
		self.updates = self.optimizer.updates(self.loss)


	def fit(self, x_train, y_train, x_valid, y_valid, batchsize=128, n_epoch=10):
		self.batchsize = batchsize
		self.n_epoch = n_epoch

		"""data pre-processing"""
		self.x_train, self.y_train = shared_data(x_train, y_train)
		self.x_valid, self.y_valid = shared_data(x_valid, y_valid)
		self.n_train_batches = self.x_train.get_value(borrow=True).shape[0] / self.batchsize
		self.n_valid_batches = self.x_valid.get_value(borrow=True).shape[0] / self.batchsize

		print "# of train mini-batches: " + str(self.n_train_batches)

		self.train_model = theano.function(
			inputs = [self.index],
			outputs = [self.loss, self.accuracy],
			updates = self.updates,
			givens = {
				self.x: self.x_train[self.index*self.batchsize: (self.index+1)*self.batchsize],
				self.y: self.y_train[self.index*self.batchsize: (self.index+1)*self.batchsize],
				self.train: np.cast['int32'](1)
			},
			mode = 'FAST_RUN'
		)

		self.valid_model = theano.function(
			inputs = [self.index],
			outputs = [self.loss, self.accuracy],
			givens = {
				self.x: self.x_valid[self.index*self.batchsize: (self.index+1)*self.batchsize],
				self.y: self.y_valid[self.index*self.batchsize: (self.index+1)*self.batchsize],
				self.train: np.cast['int32'](0)
			},
			mode = 'FAST_RUN'
		)


		epoch = 0
		acc, loss = [], []
		val_acc, val_loss = [], []

		while epoch < self.n_epoch:
			epoch += 1

			acc.append(0.0)
			loss.append(0.0)
			for batch_index in tqdm(xrange(self.n_train_batches)):
				batch_loss, batch_accuracy = self.train_model(batch_index)
				acc[-1]  += batch_accuracy
				loss[-1] += batch_loss
			acc[-1]  /= self.n_train_batches
			loss[-1] /= self.n_train_batches
			print 'epoch: {}, train mean loss={}, train accuracy={}'.format(epoch, loss[-1], acc[-1])

			val_acc.append(0.0)
			val_loss.append(0.0)
			for batch_index in xrange(self.n_valid_batches):
				batch_loss, batch_accuracy = self.valid_model(batch_index)
				val_acc[-1]  += batch_accuracy
				val_loss[-1] += batch_loss
			val_acc[-1]  /= self.n_valid_batches
			val_loss[-1] /= self.n_valid_batches
			print 'epoch: {}, valid mean loss={}, valid accuracy={}'.format(epoch, val_loss[-1], val_acc[-1])

		hist = {}
		hist["acc"] = acc
		hist["loss"] = loss
		hist["val_acc"] = val_acc
		hist["val_loss"] = val_loss
		return hist



if __name__ == '__main__':
	random_state = 1234

	print 'fetch MNIST dataset'
	mnist = fetch_mldata('MNIST original')
	mnist.data   = mnist.data.astype(np.float32)
	mnist.data  /= 255
	mnist.target = mnist.target.astype(np.int32)

	x_train, x_valid,\
	y_train, y_valid \
	= train_test_split(mnist.data, mnist.target, random_state=random_state)

	random_state = 1234
	n_input = x_train.shape[1]
	n_hidden = [784, 784]
	n_output = 10
	rng = np.random.RandomState(random_state)

	mlp = MLP(rng, n_input=n_input, n_hidden=n_hidden, n_output=n_output, optimizer=Adam)
	hist = mlp.fit(x_train, y_train, x_valid, y_valid, batchsize=128, n_epoch=10)

	df = pd.DataFrame(hist)
	df.index += 1
	df.index.name = "epoch"

	fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True)
	ax = df[["acc", "val_acc"]].plot(linewidth=2, alpha=0.6, ax=axes.flatten()[0])
	ax.set_ylabel("classification accuracy")
	ax = df[["loss", "val_loss"]].plot(linewidth=2, alpha=0.6, ax=axes.flatten()[1])
	ax.set_ylabel("loss function value")
	plt.title("mlp example")
	plt.show()







