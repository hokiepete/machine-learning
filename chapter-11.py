# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 13:46:36 2019

@author: pnola
"""

import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
he_init = tf.contrib.layers.variance_scaling_initializer()

#hidden1 = tf.fully_connected(x,n_hidden1, weights_initializer=he_init,scope='h1')

#hidden1 = tf.fully_connected(x,n_hidden1, activation_fn = tf.nn.elu)

def leaky_relu(z,name=None):
    return tf.maximum(0.01 * z, z, name=name)


#hidden1 = tf.fully_connected(x,n_hidden1, activation_fn = leaky_relu)

from tensorflow.contrib.layers import batch_norm

n_inputs = 28*28
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

x = tf.placeholder(tf.float32,shape=(None,n_inputs),name='x')
is_training = tf.placeholder(tf.bool,shape=(),name='is_training')
bn_params = {'is_training':is_training,'decay':0.99,'update_collections':None}
"""
hidden1 = fully_connected(x,n_hidden1,scope='hidden1',normalizer_fn=batch_norm,normalizer_params=bn_params)
hidden2 = fully_connected(hidden1,n_hidden2,scope='hidden2',normalizer_fn=batch_norm,normalizer_params=bn_params)
logits = fully_connected(hidden2,n_outputs,activation_fn=None,scope='outputs',normalizer_fn=batch_norm,normalizer_params=bn_params)
"""

with tf.contrib.framework.arg_scope(
        [fully_connected],
        normalizer_fn=batch_norm,
        normalizer_params=bn_params):
    hidden1 = fully_connected(x,n_hidden1,scope='hidden1')
    hidden2 = fully_connected(hidden1,n_hidden2,scope='hidden2')
    logits = fully_connected(hidden2,n_outputs,scope='outputs',activation_fn=None)




