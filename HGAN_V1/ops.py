import numpy as np
import tensorflow as tf

# Xavier : tf_contrib.layers.xavier_initializer()
# He : tf_contrib.layers.variance_scaling_initializer()
# Normal : tf.random_normal_initializer(mean=0.0, stddev=0.02)
# l2_decay : tf_contrib.layers.l2_regularizer(0.0001)

weight_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)
weight_regularizer = None

##################################################################################
# Basic Operations
##################################################################################
def lrelu(x, leak=0.2, name="lrelu", alt_relu_impl=False):

    with tf.variable_scope(name):
        if alt_relu_impl:
            f1 = 0.5 * (1 + leak)
            f2 = 0.5 * (1 - leak)
            # lrelu = 1/2 * (1 + leak) * x + 1/2 * (1 - leak) * |x|
            return f1 * x + f2 * abs(x)
        else:
            return tf.maximum(x, leak*x)

def relu(x, name="relu"):
    with tf.variable_scope(name):
        return tf.nn.relu(x, name)

def instance_norm(x, name="instance_norm"):

    with tf.variable_scope(name):
        epsilon = 1e-5
        mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
        scale = tf.get_variable('scale',[x.get_shape()[-1]], 
        initializer=tf.truncated_normal_initializer(mean=1.0, stddev=0.02))
        offset = tf.get_variable('offset',[x.get_shape()[-1]],initializer=tf.constant_initializer(0.0))
        out = scale*tf.div(x-mean, tf.sqrt(var+epsilon)) + offset

        return out

def batch_norm(x, scope='batch_norm', do_norm=True):
    return tf.layers.batch_normalization(x,momentum=0.9, epsilon=1e-3,center=True, scale=True,training=do_norm, name=scope)


def general_conv2d(inputconv, o_d=64, f_h=7, f_w=7, s_h=1, s_w=1, padding="VALID",name="conv2d"):
    with tf.variable_scope(name):
        
        conv = tf.layers.conv2d(inputconv, o_d, [f_h,f_w], [s_h,s_w], padding, activation=None)

        return conv



def general_deconv2d(inputconv, outshape, o_d=64, f_h=7, f_w=7, s_h=1, s_w=1, padding="VALID", name="deconv2d"):
    with tf.variable_scope(name):

        conv = tf.layers.conv2d_transpose(inputconv, o_d, [f_h, f_w], [s_h, s_w], padding, activation=None)
        
        return conv

def Drop_out(x, rate, training) :
    return tf.layers.dropout(inputs=x, rate=rate, training=training)

def Global_Average_Pooling(x, stride=1) :
    width = np.shape(x)[1]
    height = np.shape(x)[2]
    pool_size = [width, height]
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride) 

def Average_pooling(x, pool_size=[2,2], stride=2, padding='SAME'):
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

def Sigmoid(x) :
    return tf.nn.sigmoid(x)

def flatten(x) :
    return tf.layers.flatten(x)

def Concatenation(layers) :
    return tf.concat(layers, axis=3)

def Fully_connected(x, units, scope='fully_connected') :
    with tf.variable_scope(scope) :
        return tf.layers.dense(inputs=x, use_bias=False, units=units)

##################################################################################
# DENSE-block
##################################################################################
def bottleneck_layer(x, dim, dropout_rate, name="BN_DENSE_LAYER", is_training=True):  #dim should be self.ngf
    with tf.variable_scope(name):
        #x = batch_norm(x, scope=name+'batch_norm1', do_norm=is_training)
        x = relu(x, 'relu1')
        x = general_conv2d(x, dim*2, 1, 1, 1, 1, "SAME","c1")
        x = Drop_out(x, rate=dropout_rate, training=is_training)

        #x = batch_norm(x, scope=name+'batch_norm2', do_norm=is_training)
        x = relu(x, 'relu2')
        x = general_conv2d(x, dim, 3, 3, 1, 1, "SAME","c2")
        x = Drop_out(x, rate=dropout_rate, training=is_training)
            
        return x


def dense_block(input_x, nb_layers, filters, dropout_rate, name="DENSE_BLOCK", is_training=True):  #filters should be self.ngf
    with tf.variable_scope(name):
        layers_concat = list()
        layers_concat.append(input_x)

        x = bottleneck_layer(input_x, filters, dropout_rate, name=name+'_bottleN_' + str(0), is_training=is_training)

        layers_concat.append(x)

        for i in range(nb_layers - 1):
            x = Concatenation(layers_concat)
            x = bottleneck_layer(x, filters, dropout_rate, name=name+'_bottleN_' + str(i + 1), is_training=is_training)
            layers_concat.append(x)

        x = Concatenation(layers_concat)

        return x


def down_transition_layer(x, filters, dropout_rate, name="DTRANSI_LAYER", is_training=True):
    with tf.variable_scope(name):
        #x = batch_norm(x, scope=name+'batch_norm1', do_norm=is_training)
        x = relu(x, 'relu')
        shape = x.get_shape().as_list()
        in_channel = shape[3]
        filter_num = int(in_channel*0.6)
        x = general_conv2d(x, filter_num, 1, 1, 1, 1, "SAME","conv")
        x = Drop_out(x, rate=dropout_rate, training=is_training)
        x = Average_pooling(x, pool_size=[2,2], stride=2)

        return x

def up_transition_layer(x, filters, up_size, dropout_rate, name="UTRANSI_LAYER", is_training=True):
    with tf.variable_scope(name):
        #x = batch_norm(x, scope=name+'batch_norm1', do_norm=is_training)
        x = relu(x, 'relu')
        shape = x.get_shape().as_list()
        in_channel = shape[3]
        filter_num = int(in_channel*0.6)
        x = general_deconv2d(x, up_size, filter_num, 3, 3, 2, 2, "SAME","dconv")
        x = Drop_out(x, rate=dropout_rate, training=is_training)

        return x
##################################################################################
# DENSE-block
##################################################################################


##################################################################################
# Residual-block
##################################################################################
def build_DENSE_block(inputres, nb_layers, dim, dropout_rate, name="DENSE", is_training=True):
    
    with tf.variable_scope(name):

        out_res = dense_block(inputres, nb_layers, dim, dropout_rate, name=name+"dense_block", is_training=is_training)
  
        return out_res


def build_resnet_block_BN(inputres, dim, name="BN_Res", is_training=True):
    
    with tf.variable_scope(name):

        out_res = tf.pad(inputres, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        out_res = general_conv2d(out_res, dim, 3, 3, 1, 1, "VALID","c1")
        out_res = batch_norm(out_res, scope=name+'batch_norm1', do_norm=is_training)
        out_res = relu(out_res, 'relu1')
        out_res = tf.pad(out_res, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        out_res = general_conv2d(out_res, dim, 3, 3, 1, 1, "VALID","c2")
        out_res = batch_norm(out_res, scope=name+'batch_norm2', do_norm=is_training)
        
        return relu(out_res + inputres,'relu2')

def build_resnet_block(inputres, dim, name="resnet", is_training=True):
    
    with tf.variable_scope(name):

        mid_dim = int(dim*0.5)
        out_res1 = general_conv2d(inputres, mid_dim, 1, 1, 1, 1, "SAME", "c1") 
        out_res1 = tf.nn.relu(out_res1, 'relu1')
        out_res2 = general_conv2d(out_res1, mid_dim, 3, 3, 1, 1, "SAME", "c2") 
        out_res2 = tf.nn.relu(out_res2, 'relu2')
        out_res3 = general_conv2d(out_res2, dim, 1, 1, 1, 1, "SAME", "c3") 
       
        return tf.nn.relu(out_res3 + inputres)

##################################################################################
# Feature Fusion block
##################################################################################
def CON_FUSION(input_x, name):
    with tf.name_scope(name) :

        shape = input_x.get_shape().as_list()
        channel = shape[3]
        fm1 = relu(input_x, name=name+'relu') 
        fm1 = general_conv2d(fm1, channel, 3, 3, 1, 1, "SAME", name=name+"conv")

        fm2 = SE_Block(fm1, channel, 2, name=name+"SE")
        
        out = input_x + fm2

        return out
##################################################################################
# Feature Fusion block
##################################################################################

##################################################################################
# SE-block
##################################################################################
def SE_Block(input_x, out_dim, ratio, name):
    with tf.name_scope(name) :
        squeeze = Global_Average_Pooling(input_x)

        excitation = Fully_connected(squeeze, units=int(out_dim / ratio), scope=name+'_fc1')
        excitation = relu(excitation)
        excitation = Fully_connected(excitation, units=out_dim, scope=name+'_fc2')
        excitation = Sigmoid(excitation)

        excitation = tf.reshape(excitation, [-1,1,1,out_dim])
        re_scale = input_x * excitation

        return re_scale
##################################################################################
# SE-block
##################################################################################

##################################################################################
# Reconstruction Loss Function
##################################################################################
def L1_loss(x, y):
    loss = tf.reduce_mean(tf.abs(x - y))

    return loss

def L2_loss(x, y):
    loss = tf.reduce_mean(tf.nn.l2_loss(x - y))

    return loss

##################################################################################
# Loss function
##################################################################################

def discriminator_loss(loss_func, real, fake):
    real_loss = 0
    fake_loss = 0

    if loss_func.__contains__('wgan') :
        real_loss = -tf.reduce_mean(tf.nn.sigmoid(real))
        fake_loss = tf.reduce_mean(tf.nn.sigmoid(fake))

    if loss_func == 'lsgan' :
        real_loss = tf.reduce_mean(tf.squared_difference(real, 1.0))*0.5
        fake_loss = tf.reduce_mean(tf.square(fake))*0.5

    if loss_func == 'gan' or loss_func == 'dragan' :
        real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real), logits=real))
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake), logits=fake))

    if loss_func == 'hinge' :
        real_loss = tf.reduce_mean(relu(1.0 - real))
        fake_loss = tf.reduce_mean(relu(1.0 + fake))

    loss = real_loss + fake_loss

    return loss

def generator_loss(loss_func, fake):
    fake_loss = 0

    if loss_func.__contains__('wgan') :
        fake_loss = -tf.reduce_mean(tf.nn.sigmoid(fake))

    if loss_func == 'lsgan' :
        fake_loss = tf.reduce_mean(tf.squared_difference(fake, 1.0))

    if loss_func == 'gan' or loss_func == 'dragan' :
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake), logits=fake))

    if loss_func == 'hinge' :
        fake_loss = -tf.reduce_mean(tf.nn.sigmoid(fake))

    loss = fake_loss

    return loss
