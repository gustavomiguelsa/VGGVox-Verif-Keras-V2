
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Dense, ZeroPadding2D, Activation, BatchNormalization, Lambda, GlobalAveragePooling2D
from keras.models import Model, Sequential
import tensorflow as tf
import keras.backend as K

def l2_norm(x):
	x1 = x ** 2
	x2 = K.sum(x1, axis=-1)
	x3 = K.sqrt(x2 + 0.0000000001)
	return x/x3


def new_model(input1_shape):

	length = [None, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
	dim = [-1, 2, 5, 8, 11, 14, 17, 20, 23, 27, 30] 

	idx1 = length.index(input1_shape[1])
	n1 = dim[idx1]
	

	input_b1 = Input(shape=input1_shape, name='input1')
	pad_input_b1 = ZeroPadding2D(padding=(1,1),name='pad_1_b1')(input_b1)
	x1_b1 = Conv2D(96, (7, 7), strides=(2, 2), padding='valid', use_bias=True, name='conv1_b1')(pad_input_b1)
	x2_b1 = BatchNormalization(name='bn1_b1', epsilon=0.00001)(x1_b1)
	x3_b1 = Activation('relu', name='relu1_b1')(x2_b1)
	x4_b1 = MaxPooling2D(pool_size=(3, 3), strides=(2,2), padding='valid', name='pool1_b1')(x3_b1)
	pad_x4_b1 = ZeroPadding2D(padding=(1,1),name='pad_2_b1')(x4_b1)
	x5_b1 = Conv2D(256, (5, 5), strides=(2, 2), padding='valid', use_bias=True, name='conv2_b1')(pad_x4_b1)
	x6_b1 = BatchNormalization(name='bn2_b1', epsilon=0.00001)(x5_b1)
	x7_b1 = Activation('relu', name='relu2_b1')(x6_b1)
	x8_b1 = MaxPooling2D(pool_size=(3, 3), strides=(2,2), padding='valid', name='pool2_b1')(x7_b1)
	pad_x8_b1 = ZeroPadding2D(padding=(1,1),name='pad_3_b1')(x8_b1)
	x9_b1 = Conv2D(384, (3, 3), strides=(1, 1), padding='valid', use_bias=True, name='conv3_b1')(pad_x8_b1)
	x10_b1 = BatchNormalization(name='bn3_b1', epsilon=0.00001)(x9_b1)
	x11_b1 = Activation('relu', name='relu3_b1')(x10_b1)
	pad_x11_b1 = ZeroPadding2D(padding=(1,1),name='pad_4_b1')(x11_b1)
	x12_b1 = Conv2D(256, (3, 3), strides=(1, 1), padding='valid', use_bias=True, name='conv4_b1')(pad_x11_b1)
	x13_b1 = BatchNormalization(name='bn4_b1', epsilon=0.00001)(x12_b1)
	x14_b1 = Activation('relu', name='relu4_b1')(x13_b1)
	pad_x14_b1 = ZeroPadding2D(padding=(1,1),name='pad_5_b1')(x14_b1)
	x15_b1 = Conv2D(256, (3, 3), strides=(1, 1), padding='valid', use_bias=True, name='conv5_b1')(pad_x14_b1)
	x16_b1 = BatchNormalization(name='bn5_b1', epsilon=0.00001)(x15_b1)
	x17_b1 = Activation('relu', name='relu5_b1')(x16_b1)
	x18_b1 = MaxPooling2D(pool_size=(5, 3), strides=(3,2), padding='valid', name='pool5_b1')(x17_b1)
	x19_b1 = Conv2D(4096, (9, 1), strides=(1, 1), padding='valid', use_bias=True, name='fc6_b1')(x18_b1)
	x20_b1 = BatchNormalization(name='bn6_b1', epsilon=0.00001)(x19_b1)
	x21_b1 = Activation('relu', name='relu6_b1')(x20_b1)
	#x22_b1 = AveragePooling2D(pool_size=(1, 8), strides=(1, 1), padding='valid', name='pool6_b1')(x21_b1)
	x22_b1 = GlobalAveragePooling2D(name='pool6_1')(x21_b1)
	x23_b1 = Dense(1024, use_bias=True, name='fc7_b1')(x22_b1)
	x24_b1 = BatchNormalization(name='bn7_b1', epsilon=0.00001)(x23_b1)
	x25_b1 = Activation('relu', name='relu7_b1')(x24_b1)
	x0_s1 = Lambda(lambda x: l2_norm(x))(x25_b1)
	x1_s1 = Dense(1024, use_bias=True, name='fc8_s1')(x0_s1)
	
 

	model = Model(inputs=input_b1, outputs=x1_s1)
	return model


