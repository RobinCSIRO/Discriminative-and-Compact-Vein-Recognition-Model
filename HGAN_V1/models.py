from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial

import tensorflow as tf
import tensorflow.contrib.slim as slim


conv = partial(slim.conv2d, activation_fn=None)
deconv = partial(slim.conv2d_transpose, activation_fn=None)
relu = tf.nn.relu
lrelu = partial(tf.nn.leaky_relu, alpha=0.2)
batch_norm = partial(slim.batch_norm, scale=True, decay=0.9, epsilon=1e-5, updates_collections=None)


def discriminator(img1, img2, scope, dim=64, train=True):
    bn = partial(batch_norm, is_training=train)
    conv_bn_lrelu = partial(conv, normalizer_fn=bn, activation_fn=lrelu, biases_initializer=None)

    with tf.variable_scope(scope + '_discriminator', reuse=tf.AUTO_REUSE):
        img = tf.concat([img1, img2], axis=3) 
        net = lrelu(conv(img, dim, 4, 2))
        net = conv_bn_lrelu(net, dim * 2, 4, 2)
        net = conv_bn_lrelu(net, dim * 4, 4, 2)
        net = conv_bn_lrelu(net, dim * 8, 4, 1)
        net = conv(net, 1, 4, 1)

        return net


def generator(img, scope, dim=64, train=True):
    bn = partial(batch_norm, is_training=train)
    conv_bn_relu = partial(conv, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)
    deconv_bn_relu = partial(deconv, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)

    def _residule_block(x, dim):
        y = conv_bn_relu(x, dim, 3, 1)
        y = bn(conv(y, dim, 3, 1))
        return y + x

    with tf.variable_scope(scope + '_generator', reuse=tf.AUTO_REUSE):
        net = conv_bn_relu(img, dim, 7, 1)
        net = conv_bn_relu(net, dim * 2, 3, 2)
        net = conv_bn_relu(net, dim * 4, 3, 2)

        for i in range(9):
            net = _residule_block(net, dim * 4)

        net = deconv_bn_relu(net, dim * 2, 3, 2)
        net = deconv_bn_relu(net, dim, 3, 2)
        net = conv(net, 3, 7, 1)
        net = tf.nn.tanh(net)

        return net

def REG_GEN(img, scope, dim=64, train=True):
    #bn = partial(batch_norm, is_training=train)
    #conv_bn_relu = partial(conv, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)
    #deconv_bn_relu = partial(deconv, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)

    conv_relu = partial(conv, normalizer_fn=None, activation_fn=relu, biases_initializer=None)
    deconv_relu = partial(deconv, normalizer_fn=None, activation_fn=relu, biases_initializer=None)

    def _residule_block(x, dim):
        y = conv_relu(x, dim, 3, 1)
        #y = bn(conv(y, dim, 3, 1))
        return y + x

    with tf.variable_scope(scope + '_generator', reuse=tf.AUTO_REUSE):
        net1 = conv_relu(img, dim, 7, 1)
        net2 = conv_relu(net1, dim * 2, 3, 2)
        net3 = conv_relu(net2, dim * 4, 3, 2)
        net_res = net3

        for i in range(9):
            net_res = _residule_block(net_res, dim * 4)

        con1 = tf.concat([net_res, net3], axis=3) 
        shape1 = con1.get_shape().as_list()
        CH1 = shape1[3]
        con1 = conv(con1, CH1, 1, 1)
        net4 = deconv_relu(con1, dim * 2, 3, 2)
        net4 = conv(net4, dim * 2, 1, 1)
        con2 = tf.concat([net4, net2], axis=3)
        shape2 = con2.get_shape().as_list()
        CH2 = shape2[3]
        con2 = conv(con2, CH2, 1, 1) 
        net5 = deconv_relu(net4, dim, 3, 2)
        net6 = conv(net5, 3, 7, 1)
        net7 = tf.nn.tanh(net6)

        return net7
