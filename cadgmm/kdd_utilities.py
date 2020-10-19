"""
KDD  architecture.

Autoencoder and estimation network.
eature extraction.


Taken from Section 4.3 of Zong, Bo et al. “Deep Autoencoding Gaussian
Mixture Model for Unsupervised Anomaly Detection.” (2018).
"""

import tensorflow as tf
from utils.layers import *

init_kernel = tf.contrib.layers.xavier_initializer()

params = {
    'is_image': False,
    'learning_rate': 0.0001,
    'batch_size': 1024,
    'latent_dim': 1,
    'K': 4,
    'KNN': 15,
    'n_epochs': 300,
    "l1": 0.1,
    "l2": 0.005,
    "l3": 10
}


def encoder(x_inp, adj, n_samples, is_training=False, reuse=False):
    """ Encoder architecture in tensorflow

    Maps the data into the latent space

    Note:
        Provides histogram and distribution tensorflow summaries

    Args:
        x_inp (tensor): input data for the encoder.
        reuse (bool): sharing variables or not

    Returns:
        (tensor): last activation layer of the encoder

    """

    with tf.variable_scope('encoder', reuse=reuse):
        name_net = 'layer_1'
        net_mlp = x_inp
        with tf.variable_scope(name_net):
            net_mlp = tf.layers.dense(net_mlp,
                                      units=64,
                                      activation=tf.nn.tanh,
                                      kernel_initializer=init_kernel,
                                      name='fc')
        name_net = 'layer_2'
        with tf.variable_scope(name_net):
            net_mlp = tf.layers.dense(net_mlp,
                                      units=32,
                                      activation=tf.nn.tanh,
                                      kernel_initializer=init_kernel,
                                      name='fc')

        ### GCN branch
        net_gcn = tf.expand_dims(x_inp, 1)

        attns = []
        k = 2
        for _ in range(k):
            attns.append(NodeAttention_SP(bias_mat=adj, nb_nodes=n_samples,
                                          out_sz=32 // k, act=tf.nn.tanh)(net_gcn))

        net_gcn = tf.concat(attns, axis=-1)[0]

        ### Fusion module
        net = tf.add(net_mlp, net_gcn)
        ### Last layer
        name_net = 'layer_3'
        with tf.variable_scope(name_net):
            net = tf.layers.dense(net,
                                  units=8,
                                  kernel_initializer=init_kernel,
                                  name='fc')
    return net


def decoder(z_inp, n_features, is_training=False, getter=None, reuse=False):
    """ Decoder architecture in tensorflow

    Generates data from the latent space

    Note:
        Provides histogram and distribution tensorflow summaries

    Args:
        z_inp (tensor): variable in the latent space
        reuse (bool): sharing variables or not

    Returns:
        (tensor): last activation layer of the generator

    """
    with tf.variable_scope('decoder', reuse=reuse):
        name_net = 'layer_1'
        with tf.variable_scope(name_net):
            net = tf.layers.dense(z_inp,
                                  units=32,
                                  activation=tf.nn.tanh,
                                  kernel_initializer=init_kernel,
                                  name='fc')

        name_net = 'layer_2'
        with tf.variable_scope(name_net):
            net = tf.layers.dense(net,
                                  units=64,
                                  activation=tf.nn.tanh,
                                  kernel_initializer=init_kernel,
                                  name='fc')

        name_net = 'layer_3'
        # there actually are 121 features in kdd
        with tf.variable_scope(name_net):
            net = tf.layers.dense(net,
                                  units=n_features,
                                  kernel_initializer=init_kernel,
                                  name='fc')

    return net


def feature_extractor(x, x_r):
    """
    Computes the reconstruction features for the autoencoder

    Args:
        - x : [N, 121] input data
        - x_r : same shape - reconstructed thanks to the autoencoder

    Returns:
        - f : chosen features
              here relative Euclidean distance and cosine similarity
    """
    dist = tf.norm(x - x_r, keepdims=True, axis=1) / tf.norm(x, keepdims=True, axis=1)
    cosine_dist = tf.losses.cosine_distance(tf.nn.l2_normalize(x, 1), tf.nn.l2_normalize(x_r, 1), axis=1,
                                            reduction=tf.losses.Reduction.NONE)
    tf.summary.scalar("dist", tf.reduce_mean(dist), ["loss"])
    tf.summary.scalar("cosine", tf.reduce_mean(1 - cosine_dist), ["loss"])
    # tf.summary.scalar("cosine_dist", tf.reduce_mean(cosine_dist), ["loss"])
    return tf.concat([dist, 1 - cosine_dist], axis=-1)


def estimator(z_inp, K, is_training=False, getter=None, reuse=False):
    """ Estimation network architecture in tensorflow

    Computes the probability of x represented by z to be in the training data

    Note:
        Provides histogram and distribution tensorflow summaries

    Args:
        z_inp (tensor): variable in the latent space + reconstruction features
        reuse (bool): sharing variables or not

    Returns:
        logits (tensor): last activation layer of the estimation network (shape 1)

    """
    init_kernel = tf.contrib.layers.xavier_initializer()
    with tf.variable_scope('estimator', reuse=reuse, custom_getter=getter):
        name_layer = 'layer_1'
        with tf.variable_scope(name_layer):
            net = tf.layers.dense(z_inp,
                                  units=20,
                                  activation=tf.nn.tanh,
                                  kernel_initializer=init_kernel,
                                  name='fc')
            net = tf.layers.dropout(net, rate=0.5, name='dropout', training=is_training)

        name_layer = 'layer_2'
        with tf.variable_scope(name_layer):
            net = tf.layers.dense(net,
                                  units=8,
                                  activation=tf.nn.tanh,
                                  kernel_initializer=init_kernel,
                                  name='fc')
            net = tf.layers.dropout(net, rate=0.5, name='dropout', training=is_training)

        name_layer = 'layer_3'
        with tf.variable_scope(name_layer):
            net = tf.layers.dense(net,
                                  units=K,
                                  kernel_initializer=init_kernel,
                                  name='fc')
            logits = tf.nn.softmax(net)

    return logits
