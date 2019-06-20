import numpy as np
from sklearn.datasets import load_sample_images
import tensorflow as tf
dataset = np.array(load_sample_images().images,dtype=np.float32)
batch_size, height, width, channels = dataset.shape

filters_test = np.zeros(shape=(7,7,channels,2),dtype=np.float32)
filters_test[:,3,:,0] = 1
filters_test[3,:,:,1] = 1

x = tf.placeholder(tf.float32,shape=(None,height,width,channels))
convolution = tf.nn.conv2d(x,filters_test,strides=[1,2,2,1],padding="SAME")

with tf.Session() as sess:
    output = sess.run(convolution,feed_dict={x:dataset})

import matplotlib.pyplot as plt
plt.imshow(output[0,:,:,1])