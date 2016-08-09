#!/usr/local/bin python
#! -*- coding: utf-8 -*-

import theano
import theano.tensor as T
import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split

def relu(x):
	return T.maximum(0, x)
def sigmoid(x):
	return T.nnet.sigmoid(x)
def tanh(x):
	return T.tanh(x)

def dropout(rng, x, train, p=0.5):
	masked_x = None
	if p > 0.0 and p < 1.0:
		seed = rng.randint(2 ** 30)
		srng = theano.tensor.shared_randomstreams.RandomStreams(seed)
		mask = srng.binomial(
			n=1,
			p=1.0-p,
			size=x.shape,
			dtype=theano.config.floatX
		)
		masked_x = x * mask
	else:
		masked_x = x
	return T.switch(T.neq(train, 0), masked_x, x*(1.0-p))

def get_corrupted_input(rng, x, train, corruption_level=0.3):
	masked_x = None
	if corruption_level > 0.0 and corruption_level < 1.0:
		seed = rng.randint(2 ** 30)
		srng = theano.tensor.shared_randomstreams.RandomStreams(seed)
		mask = srng.binomial(
			n=1,
			p=1.0-corruption_level,
			size=x.shape,
			dtype=theano.config.floatX
		)
		masked_x = x * mask
	return T.switch(T.neq(train, 0), masked_x, x)


class Metric(object):
	def __init__(self, x, y):
		self.x = x
		self.y = y

	def negative_log_likelihood(self):
		self.prob_of_y_given_x = T.nnet.softmax(self.x)
		return -T.mean(T.log(self.prob_of_y_given_x)[T.arange(self.y.shape[0]), self.y])

	def cross_entropy(self):
		self.prob_of_y_given_x = T.nnet.softmax(self.x)
		return T.mean(T.nnet.categorical_crossentropy(self.prob_of_y_given_x, self.y))

	def mean_squared_error(self):
		return T.mean((self.x - self.y) ** 2)

	def errors(self):
		if self.y.ndim != self.y_pred.ndim:
			raise TypeError('y should have the same shape as self.y_pred',
							('y', self.y.type, 'y_pred', self.y_pred.type))

		if self.y.dtype.startswith('int'):
			self.prob_of_y_given_x = T.nnet.softmax(self.x)
			self.y_pred = T.argmax(self.prob_of_y_given_x, axis=1)
			return T.mean(T.neq(self.y_pred, self.y))
		else:
			return NotImplementedError()

	def accuracy(self):
		if self.y.dtype.startswith('int'):
			self.prob_of_y_given_x = T.nnet.softmax(self.x)
			self.y_pred = T.argmax(self.prob_of_y_given_x, axis=1)
			return T.mean(T.eq(self.y_pred, self.y))
		else:
			return NotImplementedError()

def shared_data(x,y):
	shared_x = theano.shared(np.asarray(x, dtype=theano.config.floatX), borrow=True)
	if y is None:
		return shared_x

	shared_y = theano.shared(np.asarray(y, dtype=theano.config.floatX), borrow=True)

	return shared_x, T.cast(shared_y, 'int32')


def build_shared_zeros(shape, name):
    """ Builds a theano shared variable filled with a zeros numpy array """
    return theano.shared(
    	value=np.zeros(shape, dtype=theano.config.floatX), 
    	name=name, 
    	borrow=True
    )

def scale_to_unit_interval(ndar, eps=1e-8):
	ndar = ndar.copy()
	ndar -= ndar.min()
	ndar *= 1.0 / (ndar.max() + eps)
	return ndar

def tile_raster_images(
		X, img_shape, tile_shape,
		tile_spacing = (0, 0),
		scale_rows_to_unit_interval = True,
		output_pixel_vals=True
	):
	assert len(img_shape) == 2
	assert len(tile_shape) == 2
	assert len(tile_spacing) == 2
	out_shape = [(ishp + tsp) * tshp - tsp for ishp, tshp, tsp in zip(img_shape, tile_shape, tile_spacing)]
	if isinstance(X, tuple):
		assert len(X) == 4
		if output_pixel_vals:
			out_array = np.zeros((out_shape[0], out_shape[1], 4), dtype='uint8')
		else:
			out_array = np.zeros((out_shape[0], out_shape[1], 4), dtype=X.dtype)

		if output_pixel_vals:
			channel_defaults = [0, 0, 0, 255]
		else:
			channel_defaults = [0., 0., 0., 1.]

		for i in range(4):
			if X[i] is None:
				out_array[:, :, i] = np.zeros(out_shape, dtype='uint8' if output_pixel_vals else out_array.dtype ) + channel_defaults[i]
			else:
				out_array[:, :, i] = tile_raster_images(X[i], img_shape, tile_shape, tile_spacing, scale_rows_to_unit_interval, output_pixel_vals)
		return out_array
	else:
		H, W = img_shape
		Hs, Ws = tile_spacing
		out_array = np.zeros(out_shape, dtype='uint8' if output_pixel_vals else X.dtype)
		for tile_row in range(tile_shape[0]):
			for tile_col in range(tile_shape[1]):
				if tile_row * tile_shape[1] + tile_col < X.shape[0]:
					if scale_rows_to_unit_interval:
						this_img = scale_to_unit_interval(X[tile_row * tile_shape[1] + tile_col].reshape(img_shape))
					else:
						this_img = X[tile_row * tile_shape[1] + tile_col].reshape(img_shape)
					out_array[tile_row * (H+Hs): tile_row * (H + Hs) + H,tile_col * (W+Ws): tile_col * (W + Ws) + W] = this_img * (255 if output_pixel_vals else 1)
		return out_array















