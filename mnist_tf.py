from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import tensorflow as tf

import numpy as np
import os

from azureml.core.run import Run
from PlotUtils import plot_digits, plot_confusion_matrix

data_path = '/tmp'
os.makedirs('./outputs', exist_ok=True)
    
if "AZUREML_NATIVE_SHARE_DIRECTORY" in os.environ:
    print('use shared folder.')
    data_path = os.environ['AZUREML_NATIVE_SHARE_DIRECTORY']
else:
    print('shared volume not enabled.')

print('data path:', data_path)
        
run = Run.get_submitted_run()

print('fetching MNIST data...')
mnist = fetch_mldata('MNIST original', data_home=data_path)

# use the full set with 70,000 records
X_mnist, y_mnist = mnist['data'], mnist['target']

print('plotting some digits...')
example_images = np.r_[X_mnist[:12000:600], X_mnist[13000:30600:600], X_mnist[30600:60000:590]]
plot_digits(instances=example_images, images_per_row=10, save_file_name='./outputs/digits.png')

# use a random subset of n records to reduce training time.
#n = 5000
#shuffle_index = np.random.permutation(70000)[:n]
#X, y = mnist['data'][shuffle_index], mnist['target'][shuffle_index]

print('X: ', X_mnist.shape)
print('y: ', y_mnist.shape)
print('labels: ', np.unique(y_mnist))

X_train, X_test, y_train, y_test = train_test_split(X_mnist, y_mnist, test_size = 0.3, random_state = 42)

print("training a tensorflow model...")

n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
y = tf.placeholder(tf.int64, shape=(None), name='y')

with tf.name_scope('dnn'):
    hidden1 = tf.layers.dense(X, n_hidden1, name='hidden1', activation=tf.nn.relu)
    hidden2 = tf.layers.dense(hidden1, n_hidden2, name='hidden2', activation=tf.nn.relu)
    logits = tf.layers.dense(hidden2, n_outputs, name='outputs')

with tf.name_scope('loss'):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name='loss')

learning_rate = 0.01
with tf.name_scope('train'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
    training_op = optimizer.minimize(loss)

with tf.name_scope('eval'):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()

n_epochs = 5
batch_size = 50
train_size = X_train.shape[0]
n_batches = train_size // batch_size
n_batches = 10

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/")

print(X_train.shape, y_train.shape)
print(type(X_train), type(y_train))
print('n_batches:', n_batches)
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        batch_count = 0
        perm = np.arange(train_size)
        np.random.shuffle(perm)
        X_train, y_train = X_train[perm], y_train[perm]
        b_start = 0
        b_end = b_start + batch_size
        for _ in range(n_batches):
            X_batch, y_batch = X_train[b_start:b_end], y_train[b_start:b_end]
            b_start = b_start + batch_size
            b_end = min(b_start + batch_size, train_size)
            #X_batch, y_batch = mnist.train.next_batch(batch_size)
            #print(b_start, b_end)
            print(X_batch, y_batch)
            print(X_batch.shape, y_batch.shape)
            print(type(X_batch), type(y_batch))
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_train, y: y_train})
        acc_val = accuracy.eval(feed_dict={X: X_test, y: y_test})
        print(epoch, 'Train accuracy:', acc_train, 'Val accuracy;', acc_val)
        y_hat = np.argmax(logits.eval(feed_dict={X: X_test}), axis=1)

acc = np.average(np.int32(y_hat == y_test))
run.log('accuracy', acc)

print('Overall accuracy:', acc)

conf_mx = confusion_matrix(y_test, y_hat)
print('Confusion matrix:')
print(conf_mx)

row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums
np.fill_diagonal(norm_conf_mx, 0)

print('plotting a normalized confusionn matrix...')
plot_confusion_matrix(norm_conf_mx, save_file_name='./outputs/mx.png')


