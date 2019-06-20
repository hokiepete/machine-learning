import numpy as np
from sklearn.datasets import load_sample_images
import tensorflow as tf
dataset = np.array(load_sample_images().images,dtype=np.float32)
batch_size, height, width, channels = dataset.shape

"""
filters_test = np.zeros(shape=(7,7,channels,2),dtype=np.float32)
filters_test[:,3,:,0] = 1
filters_test[3,:,:,1] = 1

x = tf.placeholder(tf.float32,shape=(None,height,width,channels))
convolution = tf.nn.conv2d(x,filters_test,strides=[1,2,2,1],padding="SAME")

with tf.Session() as sess:
    output = sess.run(convolution,feed_dict={x:dataset})

import matplotlib.pyplot as plt
plt.imshow(output[0,:,:,1])

#"""
"""
x = tf.placeholder(tf.float32,shape=(None,height,width,channels))
lay_pool = tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID")
#lay_pool = tf.nn.avg_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID")

with tf.Session() as sess:
    output = sess.run(lay_pool,feed_dict={x:dataset})
    
import matplotlib.pyplot as plt
plt.close('all')
plt.imshow(output[0].astype(np.uint8))
#"""

filters_test = np.zeros(shape=(7,7,channels,2),dtype=np.float32)
filters_test[:,3,:,0] = 1
filters_test[3,:,:,1] = 1

x = tf.placeholder(tf.float32,shape=(None,height,width,channels))
convolution = tf.nn.conv2d(x,filters_test,strides=[1,2,2,1],padding="SAME")
lay_pool = tf.nn.max_pool(convolution,ksize=[1,4,4,1],strides=[1,1,1,1],padding="VALID")
#lay_pool = tf.nn.avg_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID")

filters_test = np.zeros(shape=(3,3,2,2),dtype=np.float32)
filters_test[:,1,:,0] = 1
filters_test[1,:,:,1] = 1


convolution2 = tf.nn.conv2d(lay_pool,filters_test,strides=[1,2,2,1],padding="SAME")
lay_pool2 = tf.nn.max_pool(convolution2,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID")
#lay_pool = tf.nn.avg_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID")


with tf.Session() as sess:
    output = sess.run(lay_pool2,feed_dict={x:dataset})

import matplotlib.pyplot as plt
plt.close('all')
plt.subplot(211)
plt.imshow(output[0,:,:,0])
plt.subplot(212)
plt.imshow(output[0,:,:,1])

