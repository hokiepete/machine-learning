"""
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
iris = load_iris()
x = iris.data[:,(2,3)] #petal length and width
y = (iris.target == 0).astype(np.int)
per_clf = Perceptron(random_state=42)
per_clf.fit(x,y)

print(per_clf.predict([[2,0.5]]))
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

import tensorflow as tf
feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(x_train)
dnn_clf = tf.contrib.learn.DNNClassifier(hidden_units=[300,100],n_classes=10,feature_columns=feature_columns)

dnn_clf.fit(x=x_train,y=y_train, batch_size=50,steps=400)

print(dnn_clf.evaluate(x_test,y_test))
"""

import tensorflow as tf
import numpy as np
"""
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
mnist = fetch_mldata('MNIST original')
x, y = mnist['data'], mnist['target']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
"""
n_inputs = 28*28
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

x = tf.placeholder(tf.float32, shape=(None, n_inputs), name='x')
y = tf.placeholder(tf.int64, shape=(None), name='y')

def neuron_layer(x,n_neurons,name,activation=None):
    with tf.name_scope(name):
        n_inputs = int(x.get_shape()[1])
        stddev = 2 / np.sqrt(n_inputs)
        init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
        w = tf.Variable(init, name='weights')
        b = tf.Variable(tf.zeros([n_neurons]), name='biases')
        z = tf.matmul(x,w) + b
        if activation=='relu':
            return tf.nn.relu(z)
        else:
            return z
        
with tf.name_scope('dnn'):
    hidden1 = neuron_layer(x,n_hidden1,'hidden1',activation='relu')
    hidden2 = neuron_layer(hidden1,n_hidden2,'hidden2',activation='relu')
    logits = neuron_layer(hidden2,n_outputs,'outputs')
"""    
from tensorflow.contrib.layers import fully_connected
with tf.name_scope('dnn'):
    hidden1 = fully_connected(x,n_hidden1,scope='hidden1')
    hidden2 = fully_connected(hidden1,n_hidden2,scope='hidden2')
    logits = fully_connected(hidden2,n_outputs,scope='outputs',activation_fn=None)
"""
    
with tf.name_scope('loss'):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=logits)
    loss = tf.reduce_mean(xentropy,name='loss')
    
learning_rate = 0.01
with tf.name_scope('train'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)
    
with tf.name_scope('eval'):
    correct = tf.nn.in_top_k(logits,y,1)
    accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))
    
init = tf.global_variables_initializer()
saver = tf.train.Saver()

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./tmp/data/')

n_epochs = 40
batch_size = 50

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            x_batch,y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op,feed_dict={x:x_batch,y:y_batch})
        acc_train = accuracy.eval(feed_dict={x:x_batch,y:y_batch})
        acc_test = accuracy.eval(feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print(epoch,"Train accuracy: ", acc_train, ", Test accuracy: ",acc_test)
    save_path = saver.save(sess,"./my_model_final.ckpt")


















