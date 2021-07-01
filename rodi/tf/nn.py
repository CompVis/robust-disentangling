import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import add_arg_scope, arg_scope
import math


def model_arg_scope(**kwargs):
    """Create new counter and apply arg scope to all arg scoped nn
    operations."""
    counters = {}
    return arg_scope(
            [conv2d, residual_block, dense, activate],
            counters = counters, **kwargs)


def make_model(name, template, **kwargs):
    """Create model with fixed kwargs."""
    run = lambda *args, **kw: template(*args, **dict((k, v) for kws in (kw, kwargs) for k, v in kws.items()))
    return tf.make_template(name, run, unique_name_ = name)


def int_shape(x):
    return x.shape.as_list()


def get_name(layer_name, counters):
    ''' utlity for keeping track of layer names '''
    if not layer_name in counters:
        counters[layer_name] = 0
    name = layer_name + '_' + str(counters[layer_name])
    counters[layer_name] += 1
    return name


@add_arg_scope
def conv2d(x, num_filters, filter_size=[3, 3], stride=[1, 1], pad='SAME', init_scale=1., counters={}, init=False, **kwargs):
    ''' convolutional layer '''
    num_filters = int(num_filters)
    strides = [1] + stride + [1]
    name = get_name('conv2d', counters)
    initdist = "uniform"
    with tf.variable_scope(name):
        in_channels = int(x.get_shape()[-1])
        fan_in = in_channels*filter_size[0]*filter_size[1]
        stdv = math.sqrt(1.0 / fan_in)
        if initdist == "uniform":
            V_initializer = tf.random_uniform_initializer(minval = -stdv, maxval = stdv)
            b_initializer = tf.random_uniform_initializer(minval = -stdv, maxval = stdv)
        elif initdist == "normal":
            V_initializer = tf.random_normal_initializer(stddev = stdv)
            b_initializer = tf.random_normal_initializer(stddev = stdv)
        else:
            raise ValueError(initdist)
        V = tf.get_variable("V", filter_size + [in_channels, num_filters],
                initializer = V_initializer,
                dtype = tf.float32)
        b = tf.get_variable("b", [num_filters],
                initializer = b_initializer,
                dtype = tf.float32)
        if init:
            tmp = tf.nn.conv2d(x, V, [1] + stride + [1], pad) + tf.reshape(b, [1,1,1,num_filters])
            mean, var = tf.nn.moments(tmp, [0,1,2])
            scaler = 1.0 / tf.sqrt(var + 1e-6)
            V = tf.assign(V, V * scaler)
            b = tf.assign(b, -mean * scaler)
        x = tf.nn.conv2d(x, V, [1] + stride + [1], pad) + tf.reshape(b, [1,1,1,num_filters])
        return x


@add_arg_scope
def dense(x, num_units, init_scale=1., counters={}, init=False, **kwargs):
    ''' fully connected layer '''
    name = get_name('dense', counters)
    initdist = "uniform"
    with tf.variable_scope(name):
        in_channels = int(x.get_shape()[-1])
        fan_in = in_channels
        stdv = math.sqrt(1.0 / fan_in)
        if initdist == "uniform":
            V_initializer = tf.random_uniform_initializer(minval = -stdv, maxval = stdv)
            b_initializer = tf.random_uniform_initializer(minval = -stdv, maxval = stdv)
        elif initdist == "normal":
            V_initializer = tf.random_normal_initializer(stddev = stdv)
            b_initializer = tf.random_normal_initializer(stddev = stdv)
        else:
            raise ValueError(initdist)
        V = tf.get_variable("V", [in_channels, num_units],
                initializer = V_initializer,
                dtype = tf.float32)
        b = tf.get_variable("b", [num_units],
                initializer = b_initializer,
                dtype = tf.float32)
        if init:
            tmp = tf.matmul(x, V) + tf.reshape(b, [1, num_units])
            mean, var = tf.nn.moments(tmp, [0])
            scaler = 1.0 / tf.sqrt(var + 1e-6)
            V = tf.assign(V, V * scaler)
            b = tf.assign(b, -mean * scaler)
        x = tf.matmul(x, V) + tf.reshape(b, [1, num_units])
        return x

@add_arg_scope
def activate(x, activation, **kwargs):
    if activation == None:
        return x
    elif activation == "elu":
        return tf.nn.elu(x)
    elif activation == "relu":
        return tf.nn.relu(x)
    else:
        raise NotImplemented(activation)


def nin(x, num_units):
    """ a network in network layer (1x1 CONV) """
    return conv2d(x, num_units, filter_size = [1,1])


def downsample(x, num_units):
    return conv2d(x, num_units, stride = [2, 2])


def upsample(x, num_units, method = "subpixel"):
    if method == "conv_transposed":
        return deconv2d(x, num_units, stride = [2, 2])
    elif method == "subpixel":
        x = conv2d(x, 4*num_units)
        x = tf.depth_to_space(x, 2)
        return x
    elif method == "nearest_neighbor":
        bs,h,w,c = x.shape.as_list()
        x = tf.image.resize_images(x, [2*h,2*w], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return x
    else:
        raise NotImplemented(method)


@add_arg_scope
def residual_block(x, a = None, conv=conv2d, init=False, dropout_p=0.0, **kwargs):
    """Slight variation of original."""
    xs = int_shape(x)
    num_filters = xs[-1]

    residual = x
    if a is not None:
        a = nin(activate(a), num_filters)
        residual = tf.concat([residual, a], axis = -1)
    residual = activate(residual)
    residual = tf.nn.dropout(residual, keep_prob = 1.0 - dropout_p)
    residual = conv(residual, num_filters)

    return x + residual


def make_linear_var(
        step,
        start, end,
        start_value, end_value,
        clip_min = 0.0, clip_max = 1.0):
    """linear from (a, alpha) to (b, beta), i.e.
    (beta - alpha)/(b - a) * (x - a) + alpha"""
    linear = (
            (end_value - start_value) /
            (end - start) *
            (tf.cast(step, tf.float32) - start) + start_value)
    return tf.clip_by_value(linear, clip_min, clip_max)


def split_groups(x, bs = 2):
    return tf.split(tf.space_to_depth(x, bs), bs**2, axis = 3)


def merge_groups(xs, bs = 2):
    return tf.depth_to_space(tf.concat(xs, axis = 3), bs)


class FullLatentDistribution(object):
    def __init__(self, parameters, dim, stochastic = True):
        self.parameters = parameters
        self.dim = dim
        self.stochastic = stochastic

        ps = self.parameters.shape.as_list()
        if len(ps) != 2:
            assert len(ps) == 4
            assert ps[1] == ps[2] == 1
            self.expand_dims = True
            self.parameters = tf.squeeze(self.parameters, axis = [1,2])
            ps = self.parameters.shape.as_list()
        else:
            self.expand_dims = False

        assert len(ps) == 2
        self.batch_size = ps[0]

        event_dim = self.dim
        n_L_parameters = (event_dim*(event_dim+1))//2

        size_splits = [event_dim, n_L_parameters]

        self.mean, self.L = tf.split(self.parameters, size_splits, axis = 1)
        # L is Cholesky parameterization
        self.L = tf.contrib.distributions.fill_triangular(self.L)
        # make sure diagonal entries are positive by parameterizing them
        # logarithmically
        diag_L = tf.linalg.diag_part(self.L)
        self.log_diag_L = diag_L # keep for later computation of logdet
        diag_L = tf.exp(diag_L)
        # scale down then set diags
        row_weights = np.array([np.sqrt(i+1) for i in range(event_dim)])
        row_weights = np.reshape(row_weights, [1, event_dim, 1])
        self.L = self.L / row_weights
        self.L = tf.linalg.set_diag(self.L, diag_L)
        self.Sigma = tf.matmul(self.L, self.L, transpose_b = True) # L times L^t

        ms = self.mean.shape.as_list()
        self.event_axes = list(range(1, len(ms)))
        self.event_shape = ms[1:]
        assert len(self.event_shape) == 1, self.event_shape

    @staticmethod
    def n_parameters(dim):
        return dim + (dim*(dim+1))//2

    def sample(self, noise_level = 1.0):
        if not self.stochastic:
            out = self.mean
        else:
            eps = noise_level*tf.random_normal([self.batch_size, self.dim, 1])
            eps = tf.matmul(self.L, eps)
            eps = tf.squeeze(eps, axis = -1)
            out = self.mean + eps
        if self.expand_dims:
            out = tf.expand_dims(out, axis = 1)
            out = tf.expand_dims(out, axis = 1)
        return out

    def kl(self, other = None):
        if other is not None:
            raise NotImplemented("Only KL to standard normal is implemented.")

        delta = tf.square(self.mean)
        diag_covar = tf.reduce_sum(
                tf.square(self.L),
                axis = 2)
        logdet = 2.0 * self.log_diag_L

        kl = 0.5*tf.reduce_sum(
                diag_covar
                - 1.0
                + delta
                - logdet,
                axis = self.event_axes)
        kl = tf.reduce_mean(kl)
        return kl
