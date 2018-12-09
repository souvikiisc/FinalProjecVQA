import tensorflow as tf
from tensorflow.python.layers.core import Dense

def gated_tanh(feat, W, W_prime, scope_name):

    with tf.name_scope(scope_name) as scope:

        y_tilda = W(feat)
        g = tf.nn.sigmoid(W_prime(feat))
        y = tf.multiply(g, y_tilda)

        return y

def simple_relu(feat, out_dims, scope_name):
        with tf.name_scope(scope_name) as scope:
                W = Dense(out_dims, use_bias=True, name=scope_name)
                y = tf.nn.relu(W(feat))
                return y

def relu_layernorm(feat, out_dims, scope_name):
        with tf.name_scope(scope_name) as scope:
                W = Dense(out_dims, use_bias=True, name=scope_name)
                y = tf.contrib.layers.layer_norm(W(feat))
                y = tf.nn.leaky_relu(y)
                return y