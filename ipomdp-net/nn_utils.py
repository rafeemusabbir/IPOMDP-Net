import tensorflow as tf
import numpy as np


def conv_layer(input_data,
               kernel_size,
               num_filter,
               name,
               w_mean=0.0,
               w_std=None,
               dtype=tf.float32,
               add_bias=True,
               strides=(1, 1, 1, 1),
               padding='SAME'):
    """
    This function creates variables and operator for the convolutional layer.
    The initializer is set to follow a truncated normal distribution.
    :param input_data: the tensor input
    :param kernel_size: the size of kernel
    :param num_filter: the number of convolutional filters
    :param name: the variable name for kernel weights or biases
    :param w_mean: the mean value of initializer for kernel weights
    :param w_std: the standard error of initializer for kernel weights
    :param dtype: the data type of the kernel weights
    :param add_bias: whether adding bias in convolutional layer or not
    :param strides: the convolutional strides, match TF
    :param padding: the padding, match TF
    :return: (tensor) output
    """
    input_size = int(input_data.get_shape()[3])

    if w_std is None:
        w_std = 1.0 / np.sqrt(float(input_size * kernel_size * kernel_size))

    kernel = tf.get_variable(name='w_'+name,
                             shape=[kernel_size, kernel_size,
                                    input_size, num_filter],
                             initializer=tf.truncated_normal_initializer(
                                 mean=w_mean,
                                 stddev=w_std,
                                 dtype=dtype),
                             dtype=dtype)

    output = tf.nn.conv2d(input=input_data,
                          filter=kernel,
                          strides=strides,
                          padding=padding)

    if add_bias:
        biases = tf.get_variable(name='b_'+name,
                                 shape=[num_filter],
                                 dtype=dtype,
                                 initializer=tf.constant_initializer(value=0.0))
        output = tf.nn.bias_add(output, biases)

    return output


def linear_layer(input_data,
                 output_size,
                 name,
                 w_mean=0.0,
                 w_std=None,
                 dtype=tf.float32):
    """

    :param input_data:
    :param output_size:
    :param name:
    :param w_mean:
    :param w_std:
    :param dtype
    :return: (tensor) output
    """
    input_size = 1

    if len(input_data.get_shape().as_list()) > 1:
        input_size = input_data.get_shape().as_list()[1]

    if w_std is None:
        w_std = 1.0 / np.sqrt(float(np.prod(input_size)))

    w = tf.get_variable(name='w_'+name,
                        shape=[input_size, output_size],
                        initializer=tf.truncated_normal_initializer(
                            mean=w_mean,
                            stddev=w_std,
                            dtype=dtype))
    b = tf.get_variable(name='b_'+name,
                        shape=[output_size],
                        initializer=tf.constant_initializer(value=0.0))

    output = tf.matmul(input_data, w) + b

    return output


def conv_layers(input_data, conv_params, names, **kwargs):
    """
    This function builds convolutional layers from a list of descriptions
    Each single description is again a Python list:
    [KERNEL SIZE, NUMBER of HIDDEN FILTERS, TYPE of ACTIVATION FUNCTION]
    :param input_data:
    :param conv_params:
    :param names:
    :param kwargs:
    :return:
    """
    output = input_data

    for layer in range(conv_params.shape[0]):
        kernel_size = int(conv_params[layer][0])
        hidden_filter_num = int(conv_params[layer][1])

        if isinstance(names, list):
            name = names[layer]
        else:
            name = names + "_%d" % layer

        output = conv_layer(input_data=output,
                            kernel_size=kernel_size,
                            num_filter=hidden_filter_num,
                            name=name, **kwargs)
        output = activate(input_data=output,
                          name=conv_params[layer][2])

    return output


def fc_layers(input_data, fc_params, names, **kwargs):
    """

    :param input_data:
    :param conv_params:
    :param names:
    :param kwargs:
    :return:
    """
    output = input_data

    for layer in range(fc_params.shape[0]):
        fc_size = int(fc_params[layer][0])

        if isinstance(names, list):
            name = names[layer]
        else:
            name = names + "_%d" % layer

        output = linear_layer(input_data=output,
                              output_size=fc_size,
                              name=name,
                              **kwargs)
        output = activate(input_data=output,
                          name=fc_params[layer][-1])
    return output


def activate(input_data, name):
    """
    This function applies a certain activation function on the input tensor.
    :param input_data:
    :param name:
    :return: (tensor) output
    """
    output = input_data
    if name in ["linear", "lin", "l"]:
        pass
    elif name in ["relu", "r"]:
        output = tf.nn.relu(output)
    elif name in ["sigmoid", "sig", "s"]:
        output = tf.nn.sigmoid(output)
    elif name in ["tanh", "t"]:
        output = tf.nn.tanh(output)
    elif name in ["softmax", "smax", "sm"]:
        output = tf.nn.softmax(output)
    else:
        raise NotImplementedError

    return output
