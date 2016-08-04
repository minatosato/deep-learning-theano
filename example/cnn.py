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

from dnn.layers import FullyConnectedLayer
from dnn.layers import Convolutional2DLayer
from dnn.layers import MaxPooling2DLayer
from dnn.layers import BatchNormalizationLayer
from dnn.utils import *
from dnn.optimizers import *

from tqdm import tqdm
import pandas as pd

plt.style.use('ggplot')

class CNN(object):
	def __init__(self, rng, n_output=10, batchsize=128):
		self.rng = rng
		self.batchsize = batchsize

		self.index = T.lscalar()
		self.x = T.tensor4('x')
		self.y = T.ivector('y')
		self.train = T.iscalar('train')

		# layer0_input = self.x.reshape((self.batchsize, 1, 28, 28))
		layer0_input = self.x

		layer0 = Convolutional2DLayer(
			self.rng,
			layer0_input,
			filter_shape=(20, 1, 5, 5),
			image_shape=(self.batchsize, 1, 28, 28)
		)

		layer1 = BatchNormalizationLayer(layer0.output, shape=(self.batchsize, 20, 24, 24))

		layer2_input = layer1.output.reshape((self.batchsize, 20, 24, 24))
		layer2_input = relu(layer2_input)
		layer2_input = MaxPooling2DLayer(layer2_input, poolsize=(2, 2)).output

		layer2 = Convolutional2DLayer(
			self.rng,
			layer2_input,
			filter_shape=(50, 20, 5, 5),
			image_shape=(self.batchsize, 20, 12, 12)
		)

		layer3 = BatchNormalizationLayer(layer2.output, shape=(self.batchsize, 50, 8, 8))

		layer4_input = relu(layer3.output)
		layer4_input = MaxPooling2DLayer(layer4_input, poolsize=(2, 2)).output
		layer4_input = layer4_input.reshape((self.batchsize, 50*4*4))

		layer4 = FullyConnectedLayer(
			self.rng,
			dropout(self.rng, layer4_input, self.train),
			n_input=50*4*4,
			n_output=500
		)

		layer5_input = layer4.output

		layer5 = BatchNormalizationLayer(layer5_input, shape=(self.batchsize, 500))
		layer6_input = relu(layer5.output)
		layer6 = FullyConnectedLayer(
			self.rng,
			layer6_input,
			n_input=500,
			n_output=n_output
		)

		self.metric = Metric(layer6.output, self.y)
		self.loss = self.metric.negative_log_likelihood()
		self.accuracy = self.metric.accuracy()
		params = []
		params.extend(layer6.params)
		params.extend(layer5.params)
		params.extend(layer4.params)
		params.extend(layer3.params)
		params.extend(layer2.params)
		params.extend(layer1.params)
		params.extend(layer0.params)
		self.updates = AdaDelta(params=params).updates(self.loss)


	def fit(self, x_train, y_train, x_valid, y_valid, n_epoch=10):
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
	mnist.data   = mnist.data.astype(np.float32).reshape((len(mnist.data), 1, 28, 28))
	mnist.data  /= 255
	mnist.target = mnist.target.astype(np.int32)

	x_train, x_valid,\
	y_train, y_valid \
	= train_test_split(mnist.data, mnist.target, random_state=random_state)

	n_output = 10
	rng = np.random.RandomState(random_state)

	cnn = CNN(rng, n_output=n_output, batchsize=128)
	hist = cnn.fit(x_train, y_train, x_valid, y_valid, n_epoch=10)

	df = pd.DataFrame(hist)
	df.index += 1
	df.index.name = "epoch"

	fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True)
	ax = df[["acc", "val_acc"]].plot(linewidth=2, alpha=0.6, ax=axes.flatten()[0])
	ax.set_ylabel("classification accuracy")
	ax = df[["loss", "val_loss"]].plot(linewidth=2, alpha=0.6, ax=axes.flatten()[1])
	ax.set_ylabel("loss function value")
	plt.title("cnn example")
	plt.show()







