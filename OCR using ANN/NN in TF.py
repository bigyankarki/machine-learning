# importing dependencies
import tensorflow as tf
import numpy as np
from datetime import datetime


# ----------------------------- CONSTRUCTION PHASE ------------------

# Defining parameters.
n_inputs = 28*28 # MNIST
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10 # 10 Classes of prediction

# defining placeholder variables.
x = tf.placeholder(tf.float32, shape=(None, n_inputs), name='x')
y = tf.placeholder(tf.int32, shape=(None), name='y')

# defining deep neural network.
with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(x, n_hidden1, name="hidden1", activation=tf.nn.relu)
    hidden2 = tf.layers.dense(hidden1, n_hidden2, name="hidden2", activation=tf.nn.relu)
    logits = tf.layers.dense(hidden2, n_outputs, name="outputs")


# defining loss function.
with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

# defining the neural network optimizer: the graident descent.
learning_rate = 0.01
with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

# measuring classifiers acuracy.
with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# -------------------VISUALIZATION --------------

# Making dir for different log files.
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)

# writing summary to TensorBoard.
# eval_summary = tf.summary.scalar("Cross Entropy", eval)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())


# initializing all variables.
init = tf.global_variables_initializer()
saver = tf.train.Saver()


# --------------------------- NOW EXECUTION PHASE --------------------------------
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/")

n_epochs = 20
batch_size = 50

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            x_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={x: x_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict={x: x_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)

    save_path = saver.save(sess, "./checkpoints/my_model_final.ckpt")

file_writer.close()
