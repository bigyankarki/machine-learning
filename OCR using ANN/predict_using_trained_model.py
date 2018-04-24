# importing dependencies
import tensorflow as tf
import numpy as np
from recognize import resize_image


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

# initializing all variables.
init = tf.global_variables_initializer()
saver = tf.train.Saver()

# ------------------------EXECUTION PHASE--------------------------


def return_prediction():
    with tf.Session() as sess:
        saver.restore(sess, './checkpoints/my_model_final.ckpt')
        x_new_scaled = [resize_image()]
        z = logits.eval(feed_dict={x: x_new_scaled})
        y_pred = np.argmax(z, axis=1)

    print("Digit predicted is", y_pred[0])
    return y_pred

    # with tf.Session() as sess:
    #     saver.restore(sess, "./checkpoints/my_model_final.ckpt") # or better, use save_path
    #     x_new_scaled = mnist.test.images[:20]
    #     Z = logits.eval(feed_dict={x: x_new_scaled})
    #     y_pred = np.argmax(Z, axis=1)
    #
    # print("Predicted classes:", y_pred)
    # print("Actual classes:   ", mnist.test.labels[:20])


if __name__ == "__main__":
    return_prediction()
