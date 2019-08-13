import numpy as np
import os
import tensorflow as tf
import keras.backend as K
import pickle

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

from keras.models import Model
from keras.layers import Input, Lambda
from keras.optimizers import RMSprop
from test_getinput import test_getinput
from model_def import new_model


def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)



base_network = new_model((512,None,1))
base_network.load_weights('weights.h5', by_name=True)
input_a = Input(shape=(512,None,1))
input_b = Input(shape=(512,None,1))

processed_a = base_network(input_a)
processed_b = base_network(input_b)
distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])
model = Model([input_a, input_b], distance)


inp1 = test_getinput('rec.wav')
inp2 = test_getinput('x6uYqmx31kE_0000001.wav')

length1 = inp1.shape[1]
length2 = inp2.shape[1]

x = inp1.reshape(-1, 512, length1, 1)
y = inp2.reshape(-1, 512, length2, 1)

score = model.predict([x,y])

print(score)
	





























