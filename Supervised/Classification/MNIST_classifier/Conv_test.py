import numpy as np
import tensorflow as tf

X_raw = np.random.randn(20,100,100,1)

X = tf.placeholder(tf.float32, shape=(None,100,100,1))
F = tf.Variable(np.random.randn(4,4,1,4), dtype=tf.float32)
W = tf.Variable(np.random.randn(100,100,10), dtype=tf.float32)
conv = tf.nn.conv2d(X, F, strides=[1,2,2,1], padding='SAME')


# Initialize GPU options
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    a = sess.run(conv, feed_dict={X:X_raw})
    print(a.shape)
    print(np.sum(a))