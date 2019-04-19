import tensorflow as tf

x=tf.Variable(3,name='x')
y=tf.Variable(4,name='y')
f = x*x*y+y+2

with tf.Session() as sess:
    x.initializer.run()
    y.initializer.run()
    results = f.eval()
    
init = tf.global_variables_initializer()

with tf.Session() as sess:
    init.run()
    results = f.eval()
    
sess = tf.InteractiveSession()
init.run()
results = f.eval()
sess.close()

x1 = tf.Variable(1)
print(x1.graph is tf.get_default_graph())

graph = tf.Graph()
with graph.as_default():
    x2 = tf.Variable(2)

print(x2.graph is graph)
print(x2.graph is tf.get_default_graph)

w = tf.constant(3)
x = w+2
y = x+5
z = x+3

with tf.Session() as sess:
    print(y.eval())
    print(z.eval())

with tf.Session() as sess:
    yval, zval = sess.run([y,z])
    print(yval)
    print(zval)

"""
import numpy as np
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
m,n = housing.data.shape
housing_data_plus_bias= np.c_[np.ones((m,1)),housing.data] 
x=tf.constant(housing_data_plus_bias,dtype=tf.float32,name='x')
y=tf.constant(housing.target.reshape(-1,1),dtype=tf.float32,name='y')
xt=tf.transpose(x)
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(xt,x)),xt),y)
with tf.Session() as sess:
    theta_value = theta.eval()
#"""
"""
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

n_epochs = 1000
learning_rate = 0.01

housing = fetch_california_housing()
scal = StandardScaler()
scal.fit(housing.data)
housing.data = scal.transform(housing.data)
m,n = housing.data.shape
housing_data_plus_bias= np.c_[np.ones((m,1)),housing.data] 
x=tf.constant(housing_data_plus_bias,dtype=tf.float32,name='x')
y=tf.constant(housing.target.reshape(-1,1),dtype=tf.float32,name='y')
theta = tf.Variable(tf.random_uniform([n+1,1],-1.0,1.0),name='theta')
ypred = tf.matmul(x,theta,name='predictions')
error = ypred-y
mse = tf.reduce_mean(tf.square(error),name='mse')
#gradients = 2/m*tf.matmul(tf.transpose(x),error)
#gradients = tf.gradients(mse,[theta])[0]
#training_op = tf.assign(theta,theta-learning_rate*gradients)
saver = tf.train.Saver()
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.9)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range(n_epochs):
        if epoch % 100 ==0:
            print('Epoch',epoch,'MSE=',mse.eval())
        sess.run(training_op)
    best_theta = theta.eval()
    save_path = saver.save(sess,"tmp/model.ckpt")
#"""
"""
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

n_epochs = 1000
learning_rate = 0.01

housing = fetch_california_housing()
scal = StandardScaler()
scal.fit(housing.data)
housing.data = scal.transform(housing.data)
m,n = housing.data.shape
housing_data_plus_bias= np.c_[np.ones((m,1)),housing.data] 
x=tf.placeholder(tf.float32,shape=(None,n+1),name='x') # constant(housing_data_plus_bias,dtype=tf.float32,name='x')
y=tf.placeholder(tf.float32,shape=(None,1),name='y') # constant(housing.target.reshape(-1,1),dtype=tf.float32,name='y')
theta = tf.Variable(tf.random_uniform([n+1,1],-1.0,1.0),name='theta')
ypred = tf.matmul(x,theta,name='predictions')
error = ypred-y
mse = tf.reduce_mean(tf.square(error),name='mse')
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.9)
training_op = optimizer.minimize(mse)

batch_size = 100
n_batches = int(np.ceil(m/batch_size))

def fetch_batch(epoch,batch_index,batch_size):
    xbatch = housing_data_plus_bias[batch_index:(batch_index+batch_size),:]
    ybatch = housing.target.reshape(-1,1)[batch_index:(batch_index+batch_size),:]
    return xbatch,ybatch

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            xbatch,ybatch = fetch_batch(epoch,batch_index,batch_size)
            sess.run(training_op,feed_dict={x:xbatch,y:ybatch})
        if epoch % 100 ==0:
            print('Epoch',epoch,'MSE=',mse.eval())
    best_theta = theta.eval()
"""















