# Copyright 2018 Changan Wang

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import math

_BATCH_NORM_DECAY = 0.9
_BATCH_NORM_EPSILON = 1e-5
_USE_FUSED_BN = True

################################################################################
# Convenience functions for building the ResNet model.
################################################################################
def batch_norm(inputs, training, data_format, name=None):
    """Performs a batch normalization using a standard set of parameters."""
    # We set fused=True for a significant performance boost. See
    # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
    return tf.layers.batch_normalization(
        inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
        momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
        scale=True, training=training, name=name, fused=_USE_FUSED_BN)


def fixed_padding(inputs, kernel_size, data_format):
    """Pads the input along the spatial dimensions independently of input size.

    Args:
      inputs: A tensor of size [batch, channels, height_in, width_in] or
        [batch, height_in, width_in, channels] depending on data_format.
      kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                   Should be a positive integer.
      data_format: The input format ('channels_last' or 'channels_first').

    Returns:
      A tensor with the same format as the input with the data either intact
      (if kernel_size == 1) or padded (if kernel_size > 1).
    """
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    if data_format == 'channels_first':
        padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                      [pad_beg, pad_end], [pad_beg, pad_end]])
    else:
        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                      [pad_beg, pad_end], [0, 0]])
    return padded_inputs


def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format, kernel_initializer=tf.glorot_uniform_initializer, name=None):
    """Strided 2-D convolution with explicit padding."""
    # The padding is consistent and is based only on `kernel_size`, not on the
    # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
    if strides > 1:
        inputs = fixed_padding(inputs, kernel_size, data_format)

    return tf.layers.conv2d(
                inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
                padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
                kernel_initializer=kernel_initializer(),
                data_format=data_format, name=name)


################################################################################
# ResNet block definitions.
################################################################################
def _bottleneck_block_v1(inputs, filters, training, projection_shortcut,
                         strides, data_format):
    """A single block for ResNet v1, with a bottleneck.

    Similar to _building_block_v1(), except using the "bottleneck" blocks
    described in:
      Convolution then batch normalization then ReLU as described by:
        Deep Residual Learning for Image Recognition
        https://arxiv.org/pdf/1512.03385.pdf
        by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.

    Args:
      inputs: A tensor of size [batch, channels, height_in, width_in] or
        [batch, height_in, width_in, channels] depending on data_format.
      filters: The number of filters for the convolutions.
      training: A Boolean for whether the model is in training or inference
        mode. Needed for batch normalization.
      projection_shortcut: The function to use for projection shortcuts
        (typically a 1x1 convolution when downsampling the input).
      strides: The block's stride. If greater than 1, this block will ultimately
        downsample the input.
      data_format: The input format ('channels_last' or 'channels_first').

    Returns:
      The output tensor of the block; shape should match inputs.
    """
    shortcut = inputs

    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)
        shortcut = batch_norm(inputs=shortcut, training=training,
                              data_format=data_format)

    inputs = conv2d_fixed_padding(
                inputs=inputs, filters=filters, kernel_size=1, strides=1,
                data_format=data_format)
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)

    inputs = conv2d_fixed_padding(
                inputs=inputs, filters=filters, kernel_size=3, strides=strides,
                data_format=data_format)
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)

    inputs = conv2d_fixed_padding(
                inputs=inputs, filters=4 * filters, kernel_size=1, strides=1,
                data_format=data_format)
    inputs = batch_norm(inputs, training, data_format)
    inputs += shortcut
    inputs = tf.nn.relu(inputs)

    return inputs

def block_layer(inputs, filters, bottleneck, block_fn, blocks, strides,
                training, name, data_format):
    """Creates one layer of blocks for the ResNet model.

    Args:
      inputs: A tensor of size [batch, channels, height_in, width_in] or
        [batch, height_in, width_in, channels] depending on data_format.
      filters: The number of filters for the first convolution of the layer.
      bottleneck: Is the block created a bottleneck block.
      block_fn: The block to use within the model, either `building_block` or
        `bottleneck_block`.
      blocks: The number of blocks contained in the layer.
      strides: The stride to use for the first convolution of the layer. If
        greater than 1, this layer will ultimately downsample the input.
      training: Either True or False, whether we are currently training the
        model. Needed for batch norm.
      name: A string name for the tensor output of the block layer.
      data_format: The input format ('channels_last' or 'channels_first').

    Returns:
      The output tensor of the block layer.
    """

    # Bottleneck blocks end with 4x the number of filters as they start with
    filters_out = filters * 4 if bottleneck else filters

    def projection_shortcut(inputs):
        return conv2d_fixed_padding(
            inputs=inputs, filters=filters_out, kernel_size=1, strides=strides,
            data_format=data_format)

    # Only the first block per block_layer uses projection_shortcut and strides
    inputs = block_fn(inputs, filters, training, projection_shortcut, strides,
                      data_format)

    for _ in range(1, blocks):
        inputs = block_fn(inputs, filters, training, None, 1, data_format)

    return tf.identity(inputs, name)

# input image order: BGR, range [0-255]
# mean_value: 104, 117, 123
# only subtract mean is used
def constant_xavier_initializer(shape, group, dtype=tf.float32, uniform=True):
    """Initializer function."""
    if not dtype.is_floating:
      raise TypeError('Cannot create initializer for non-floating point type.')
    # Estimating fan_in and fan_out is not possible to do perfectly, but we try.
    # This is the right thing for matrix multiply and convolutions.
    if shape:
      fan_in = float(shape[-2]) if len(shape) > 1 else float(shape[-1])
      fan_out = float(shape[-1])/group
    else:
      fan_in = 1.0
      fan_out = 1.0
    for dim in shape[:-2]:
      fan_in *= float(dim)
      fan_out *= float(dim)

    # Average number of inputs and output connections.
    n = (fan_in + fan_out) / 2.0
    if uniform:
      # To get stddev = math.sqrt(factor / n) need to adjust for uniform.
      limit = math.sqrt(3.0 * 1.0 / n)
      return tf.random_uniform(shape, -limit, limit, dtype, seed=None)
    else:
      # To get stddev = math.sqrt(factor / n) need to adjust for truncated.
      trunc_stddev = math.sqrt(1.3 * 1.0 / n)
      return tf.truncated_normal(shape, 0.0, trunc_stddev, dtype, seed=None)

def wrapper_initlizer(shape, dtype=None, partition_info=None):
    return constant_xavier_initializer(shape, 32, dtype)

# for root block, use dummy input_filters, e.g. 128 rather than 64 for the first block
def se_next_bottleneck_block(inputs, input_filters, name_prefix, is_training, group, data_format='channels_last', need_reduce=True, is_root=False, reduced_scale=16):
    bn_axis = -1 if data_format == 'channels_last' else 1
    strides_to_use = 1
    residuals = inputs
    if need_reduce:
        strides_to_use = 1 if is_root else 2
        #print(strides_to_use)
        proj_mapping = tf.layers.conv2d(inputs, input_filters, (1, 1), use_bias=False,
                                name=name_prefix + '_1x1_proj', strides=(strides_to_use, strides_to_use),
                                padding='valid', data_format=data_format, activation=None,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                bias_initializer=tf.zeros_initializer())
        # print(proj_mapping)
        residuals = tf.layers.batch_normalization(proj_mapping, momentum=_BATCH_NORM_DECAY,
                                name=name_prefix + '_1x1_proj/bn', axis=bn_axis,
                                epsilon=_BATCH_NORM_EPSILON, training=is_training, reuse=None, fused=_USE_FUSED_BN)
    #print(strides_to_use)
    reduced_inputs = tf.layers.conv2d(inputs, input_filters // 2, (1, 1), use_bias=False,
                            name=name_prefix + '_1x1_reduce', strides=(1, 1),
                            padding='valid', data_format=data_format, activation=None,
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            bias_initializer=tf.zeros_initializer())
    reduced_inputs_bn = tf.layers.batch_normalization(reduced_inputs, momentum=_BATCH_NORM_DECAY,
                                        name=name_prefix + '_1x1_reduce/bn', axis=bn_axis,
                                        epsilon=_BATCH_NORM_EPSILON, training=is_training, reuse=None, fused=_USE_FUSED_BN)
    reduced_inputs_relu = tf.nn.relu(reduced_inputs_bn, name=name_prefix + '_1x1_reduce/relu')

    if data_format == 'channels_first':
        reduced_inputs_relu = tf.pad(reduced_inputs_relu, paddings = [[0, 0], [0, 0], [1, 1], [1, 1]])
        weight_shape = [3, 3, reduced_inputs_relu.get_shape().as_list()[1]//group, input_filters // 2]
        if is_training:
            weight_ = tf.Variable(constant_xavier_initializer(weight_shape, group=group, dtype=tf.float32), trainable=is_training, name=name_prefix + '_3x3/kernel')
        else:
            weight_ = tf.get_variable(name_prefix + '_3x3/kernel', shape=weight_shape, initializer=wrapper_initlizer, trainable=is_training)
        weight_groups = tf.split(weight_, num_or_size_splits=group, axis=-1, name=name_prefix + '_weight_split')
        xs = tf.split(reduced_inputs_relu, num_or_size_splits=group, axis=1, name=name_prefix + '_inputs_split')
    else:
        reduced_inputs_relu = tf.pad(reduced_inputs_relu, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
        weight_shape = [3, 3, reduced_inputs_relu.get_shape().as_list()[-1]//group, input_filters // 2]
        if is_training:
            weight_ = tf.Variable(constant_xavier_initializer(weight_shape, group=group, dtype=tf.float32), trainable=is_training, name=name_prefix + '_3x3/kernel')
        else:
            weight_ = tf.get_variable(name_prefix + '_3x3/kernel', shape=weight_shape, initializer=wrapper_initlizer, trainable=is_training)
        weight_groups = tf.split(weight_, num_or_size_splits=group, axis=-1, name=name_prefix + '_weight_split')
        xs = tf.split(reduced_inputs_relu, num_or_size_splits=group, axis=-1, name=name_prefix + '_inputs_split')

    convolved = [tf.nn.convolution(x, weight, padding='VALID', strides=[strides_to_use, strides_to_use], name=name_prefix + '_group_conv',
                    data_format=('NCHW' if data_format == 'channels_first' else 'NHWC')) for (x, weight) in zip(xs, weight_groups)]

    if data_format == 'channels_first':
        conv3_inputs = tf.concat(convolved, axis=1, name=name_prefix + '_concat')
    else:
        conv3_inputs = tf.concat(convolved, axis=-1, name=name_prefix + '_concat')

    conv3_inputs_bn = tf.layers.batch_normalization(conv3_inputs, momentum=_BATCH_NORM_DECAY, name=name_prefix + '_3x3/bn',
                                        axis=bn_axis, epsilon=_BATCH_NORM_EPSILON, training=is_training, reuse=None, fused=_USE_FUSED_BN)
    conv3_inputs_relu = tf.nn.relu(conv3_inputs_bn, name=name_prefix + '_3x3/relu')


    increase_inputs = tf.layers.conv2d(conv3_inputs_relu, input_filters, (1, 1), use_bias=False,
                                name=name_prefix + '_1x1_increase', strides=(1, 1),
                                padding='valid', data_format=data_format, activation=None,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                bias_initializer=tf.zeros_initializer())
    increase_inputs_bn = tf.layers.batch_normalization(increase_inputs, momentum=_BATCH_NORM_DECAY,
                                        name=name_prefix + '_1x1_increase/bn', axis=bn_axis,
                                        epsilon=_BATCH_NORM_EPSILON, training=is_training, reuse=None, fused=_USE_FUSED_BN)

    if data_format == 'channels_first':
        pooled_inputs = tf.reduce_mean(increase_inputs_bn, [2, 3], name=name_prefix + '_global_pool', keep_dims=True)
    else:
        pooled_inputs = tf.reduce_mean(increase_inputs_bn, [1, 2], name=name_prefix + '_global_pool', keep_dims=True)

    down_inputs = tf.layers.conv2d(pooled_inputs, input_filters // reduced_scale, (1, 1), use_bias=True,
                                name=name_prefix + '_1x1_down', strides=(1, 1),
                                padding='valid', data_format=data_format, activation=None,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                bias_initializer=tf.zeros_initializer())
    down_inputs_relu = tf.nn.relu(down_inputs, name=name_prefix + '_1x1_down/relu')


    up_inputs = tf.layers.conv2d(down_inputs_relu, input_filters, (1, 1), use_bias=True,
                                name=name_prefix + '_1x1_up', strides=(1, 1),
                                padding='valid', data_format=data_format, activation=None,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                bias_initializer=tf.zeros_initializer())
    prob_outputs = tf.nn.sigmoid(up_inputs, name=name_prefix + '_prob')

    #print(residuals, prob_outputs, increase_inputs_bn)
    rescaled_feat = tf.multiply(prob_outputs, increase_inputs_bn, name=name_prefix + '_mul')
    pre_act = tf.add(residuals, rescaled_feat, name=name_prefix + '_add')
    return tf.nn.relu(pre_act, name=name_prefix + '/relu')
    #return tf.nn.relu(residuals + prob_outputs * increase_inputs_bn, name=name_prefix + '/relu')

# the input image should in BGR order, note that this is not the common case in Tensorflow
def sext_cpn_backbone(input_image, istraining, data_format, net_depth=50, group=32):
    bn_axis = -1 if data_format == 'channels_last' else 1

    if data_format == 'channels_last':
        image_channels = tf.unstack(input_image, axis=-1)
        swaped_input_image = tf.stack([image_channels[2], image_channels[1], image_channels[0]], axis=-1)
    else:
        image_channels = tf.unstack(input_image, axis=1)
        swaped_input_image = tf.stack([image_channels[2], image_channels[1], image_channels[0]], axis=1)
    #swaped_input_image = input_image

    if net_depth not in [50, 101]:
        raise TypeError('Only ResNeXt50 or ResNeXt101 is supprted now.')

    input_depth = [256, 512, 1024, 2048] # the input depth of the the first block is dummy input
    num_units = [3, 4, 6, 3] if net_depth==50 else [3, 4, 23, 3]
    block_name_prefix = ['conv2_{}', 'conv3_{}', 'conv4_{}', 'conv5_{}']

    if data_format == 'channels_first':
        swaped_input_image = tf.pad(swaped_input_image, paddings = [[0, 0], [0, 0], [3, 3], [3, 3]])
    else:
        swaped_input_image = tf.pad(swaped_input_image, paddings = [[0, 0], [3, 3], [3, 3], [0, 0]])

    inputs_features = tf.layers.conv2d(swaped_input_image, input_depth[0]//4, (7, 7), use_bias=False,
                                name='conv1/7x7_s2', strides=(2, 2),
                                padding='valid', data_format=data_format, activation=None,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                bias_initializer=tf.zeros_initializer())
    #print(ee)

    inputs_features = tf.layers.batch_normalization(inputs_features, momentum=_BATCH_NORM_DECAY,
                                        name='conv1/7x7_s2/bn', axis=bn_axis,
                                        epsilon=_BATCH_NORM_EPSILON, training=istraining, reuse=None, fused=_USE_FUSED_BN)
    inputs_features = tf.nn.relu(inputs_features, name='conv1/relu_7x7_s2')

    inputs_features = tf.layers.max_pooling2d(inputs_features, [3, 3], [2, 2], padding='same', data_format=data_format, name='pool1/3x3_s2')

    end_points = []
    is_root = True
    for ind, num_unit in enumerate(num_units):
        need_reduce = True
        for unit_index in range(1, num_unit+1):
            inputs_features = se_next_bottleneck_block(inputs_features, input_depth[ind], block_name_prefix[ind].format(unit_index), is_training=istraining, group=group, data_format=data_format, need_reduce=need_reduce, is_root=is_root)
            need_reduce = False
            is_root = False
        end_points.append(inputs_features)

    return end_points

# input image order: BGR, range [0-255]
# mean_value: 104, 117, 123
# only subtract mean is used

# for root block, use dummy input_filters, e.g. 128 rather than 64 for the first block
def se_bottleneck_block(inputs, input_filters, name_prefix, is_training, data_format='channels_last', need_reduce=True, is_root=False, reduced_scale=16):
    bn_axis = -1 if data_format == 'channels_last' else 1
    strides_to_use = 1
    residuals = inputs
    if need_reduce:
        strides_to_use = 1 if is_root else 2
        proj_mapping = tf.layers.conv2d(inputs, input_filters * 2, (1, 1), use_bias=False,
                                name=name_prefix + '_1x1_proj', strides=(strides_to_use, strides_to_use),
                                padding='valid', data_format=data_format, activation=None,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                bias_initializer=tf.zeros_initializer())
        residuals = tf.layers.batch_normalization(proj_mapping, momentum=_BATCH_NORM_DECAY,
                                name=name_prefix + '_1x1_proj/bn', axis=bn_axis,
                                epsilon=_BATCH_NORM_EPSILON, training=is_training, reuse=None, fused=_USE_FUSED_BN)
    #print(strides_to_use)
    reduced_inputs = tf.layers.conv2d(inputs, input_filters / 2, (1, 1), use_bias=False,
                            name=name_prefix + '_1x1_reduce', strides=(strides_to_use, strides_to_use),
                            padding='valid', data_format=data_format, activation=None,
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            bias_initializer=tf.zeros_initializer())
    reduced_inputs_bn = tf.layers.batch_normalization(reduced_inputs, momentum=_BATCH_NORM_DECAY,
                                        name=name_prefix + '_1x1_reduce/bn', axis=bn_axis,
                                        epsilon=_BATCH_NORM_EPSILON, training=is_training, reuse=None, fused=_USE_FUSED_BN)
    reduced_inputs_relu = tf.nn.relu(reduced_inputs_bn, name=name_prefix + '_1x1_reduce/relu')


    conv3_inputs = tf.layers.conv2d(reduced_inputs_relu, input_filters / 2, (3, 3), use_bias=False,
                                name=name_prefix + '_3x3', strides=(1, 1),
                                padding='same', data_format=data_format, activation=None,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                bias_initializer=tf.zeros_initializer())
    conv3_inputs_bn = tf.layers.batch_normalization(conv3_inputs, momentum=_BATCH_NORM_DECAY, name=name_prefix + '_3x3/bn',
                                        axis=bn_axis, epsilon=_BATCH_NORM_EPSILON, training=is_training, reuse=None, fused=_USE_FUSED_BN)
    conv3_inputs_relu = tf.nn.relu(conv3_inputs_bn, name=name_prefix + '_3x3/relu')


    increase_inputs = tf.layers.conv2d(conv3_inputs_relu, input_filters * 2, (1, 1), use_bias=False,
                                name=name_prefix + '_1x1_increase', strides=(1, 1),
                                padding='valid', data_format=data_format, activation=None,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                bias_initializer=tf.zeros_initializer())
    increase_inputs_bn = tf.layers.batch_normalization(increase_inputs, momentum=_BATCH_NORM_DECAY,
                                        name=name_prefix + '_1x1_increase/bn', axis=bn_axis,
                                        epsilon=_BATCH_NORM_EPSILON, training=is_training, reuse=None, fused=_USE_FUSED_BN)

    if data_format == 'channels_first':
        pooled_inputs = tf.reduce_mean(increase_inputs_bn, [2, 3], name=name_prefix + '_global_pool', keep_dims=True)
    else:
        pooled_inputs = tf.reduce_mean(increase_inputs_bn, [1, 2], name=name_prefix + '_global_pool', keep_dims=True)

    down_inputs = tf.layers.conv2d(pooled_inputs, (input_filters * 2) // reduced_scale, (1, 1), use_bias=True,
                                name=name_prefix + '_1x1_down', strides=(1, 1),
                                padding='valid', data_format=data_format, activation=None,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                bias_initializer=tf.zeros_initializer())
    down_inputs_relu = tf.nn.relu(down_inputs, name=name_prefix + '_1x1_down/relu')


    up_inputs = tf.layers.conv2d(down_inputs_relu, input_filters * 2, (1, 1), use_bias=True,
                                name=name_prefix + '_1x1_up', strides=(1, 1),
                                padding='valid', data_format=data_format, activation=None,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                bias_initializer=tf.zeros_initializer())
    prob_outputs = tf.nn.sigmoid(up_inputs, name=name_prefix + '_prob')

    rescaled_feat = tf.multiply(prob_outputs, increase_inputs_bn, name=name_prefix + '_mul')
    pre_act = tf.add(residuals, rescaled_feat, name=name_prefix + '_add')
    return tf.nn.relu(pre_act, name=name_prefix + '/relu')

def se_cpn_backbone(input_image, istraining, data_format):
    bn_axis = -1 if data_format == 'channels_last' else 1

    if data_format == 'channels_last':
        image_channels = tf.unstack(input_image, axis=-1)
        swaped_input_image = tf.stack([image_channels[2], image_channels[1], image_channels[0]], axis=-1)
    else:
        image_channels = tf.unstack(input_image, axis=1)
        swaped_input_image = tf.stack([image_channels[2], image_channels[1], image_channels[0]], axis=1)

    input_depth = [128, 256, 512, 1024] # the input depth of the the first block is dummy input
    num_units = [3, 4, 6, 3]
    block_name_prefix = ['conv2_{}', 'conv3_{}', 'conv4_{}', 'conv5_{}']

    if data_format == 'channels_first':
        swaped_input_image = tf.pad(swaped_input_image, paddings = [[0, 0], [0, 0], [3, 3], [3, 3]])
    else:
        swaped_input_image = tf.pad(swaped_input_image, paddings = [[0, 0], [3, 3], [3, 3], [0, 0]])

    inputs_features = tf.layers.conv2d(swaped_input_image, input_depth[0]//2, (7, 7), use_bias=False,
                                name='conv1/7x7_s2', strides=(2, 2),
                                padding='valid', data_format=data_format, activation=None,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                bias_initializer=tf.zeros_initializer())

    inputs_features = tf.layers.batch_normalization(inputs_features, momentum=_BATCH_NORM_DECAY,
                                        name='conv1/7x7_s2/bn', axis=bn_axis,
                                        epsilon=_BATCH_NORM_EPSILON, training=istraining, reuse=None, fused=_USE_FUSED_BN)
    inputs_features = tf.nn.relu(inputs_features, name='conv1/relu_7x7_s2')

    inputs_features = tf.layers.max_pooling2d(inputs_features, [3, 3], [2, 2], padding='same', data_format=data_format, name='pool1/3x3_s2')

    end_points = []
    is_root = True
    for ind, num_unit in enumerate(num_units):
        need_reduce = True
        for unit_index in range(1, num_unit+1):
            inputs_features = se_bottleneck_block(inputs_features, input_depth[ind], block_name_prefix[ind].format(unit_index), is_training=istraining, data_format=data_format, need_reduce=need_reduce, is_root=is_root)
            need_reduce = False
            is_root = False
        end_points.append(inputs_features)

    return end_points

def global_net_bottleneck_block(inputs, filters, istraining, data_format, projection_shortcut=None, name=None):
    with tf.variable_scope(name, 'global_net_bottleneck', values=[inputs]):
        shortcut = inputs
        if projection_shortcut is not None:
            shortcut = projection_shortcut(inputs)
            shortcut = batch_norm(inputs=shortcut, training=istraining,
                                  data_format=data_format, name='batch_normalization_shortcut')

        inputs = conv2d_fixed_padding(
            inputs=inputs, filters=filters, kernel_size=1, strides=1,
            data_format=data_format, name='1x1_down')
        inputs = batch_norm(inputs, istraining, data_format, name='batch_normalization_1')
        inputs = tf.nn.relu(inputs, name='relu1')

        inputs = conv2d_fixed_padding(
            inputs=inputs, filters=filters, kernel_size=3, strides=1,
            data_format=data_format, name='3x3_conv')
        inputs = batch_norm(inputs, istraining, data_format, name='batch_normalization_2')
        inputs = tf.nn.relu(inputs, name='relu2')

        inputs = conv2d_fixed_padding(
            inputs=inputs, filters=2 * filters, kernel_size=1, strides=1,
            data_format=data_format, name='1x1_up')
        inputs = batch_norm(inputs, istraining, data_format, name='batch_normalization_3')
        inputs += shortcut
        inputs = tf.nn.relu(inputs, name='relu3')

        return inputs

def global_net_sext_bottleneck_block(inputs, input_filters, is_training, data_format, need_reduce=False, name_prefix=None, group=32, reduced_scale=16):
    with tf.variable_scope(name_prefix, 'global_net_sext_bottleneck_block', values=[inputs]):
        bn_axis = -1 if data_format == 'channels_last' else 1
        residuals = inputs
        if need_reduce:
            proj_mapping = tf.layers.conv2d(inputs, input_filters * 2, (1, 1), use_bias=False,
                                    name=name_prefix + '_1x1_proj', strides=(1, 1),
                                    padding='valid', data_format=data_format, activation=None,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    bias_initializer=tf.zeros_initializer())
            # print(proj_mapping)
            residuals = tf.layers.batch_normalization(proj_mapping, momentum=_BATCH_NORM_DECAY,
                                    name=name_prefix + '_1x1_proj/bn', axis=bn_axis,
                                    epsilon=_BATCH_NORM_EPSILON, training=is_training, reuse=None, fused=_USE_FUSED_BN)

        reduced_inputs = tf.layers.conv2d(inputs, input_filters, (1, 1), use_bias=False,
                                name=name_prefix + '_1x1_reduce', strides=(1, 1),
                                padding='valid', data_format=data_format, activation=None,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                bias_initializer=tf.zeros_initializer())
        reduced_inputs_bn = tf.layers.batch_normalization(reduced_inputs, momentum=_BATCH_NORM_DECAY,
                                            name=name_prefix + '_1x1_reduce/bn', axis=bn_axis,
                                            epsilon=_BATCH_NORM_EPSILON, training=is_training, reuse=None, fused=_USE_FUSED_BN)
        reduced_inputs_relu = tf.nn.relu(reduced_inputs_bn, name=name_prefix + '_1x1_reduce/relu')

        if data_format == 'channels_first':
            reduced_inputs_relu = tf.pad(reduced_inputs_relu, paddings = [[0, 0], [0, 0], [1, 1], [1, 1]])
            weight_shape = [3, 3, reduced_inputs_relu.get_shape().as_list()[1]//group, input_filters]
            if is_training:
                weight_ = tf.Variable(constant_xavier_initializer(weight_shape, group=group, dtype=tf.float32), trainable=is_training, name=name_prefix + '_3x3/kernel')
            else:
                weight_ = tf.get_variable(name_prefix + '_3x3/kernel', shape=weight_shape, initializer=wrapper_initlizer, trainable=is_training)
            weight_groups = tf.split(weight_, num_or_size_splits=group, axis=-1, name=name_prefix + '_weight_split')
            xs = tf.split(reduced_inputs_relu, num_or_size_splits=group, axis=1, name=name_prefix + '_inputs_split')
        else:
            reduced_inputs_relu = tf.pad(reduced_inputs_relu, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
            weight_shape = [3, 3, reduced_inputs_relu.get_shape().as_list()[-1]//group, input_filters]
            if is_training:
                weight_ = tf.Variable(constant_xavier_initializer(weight_shape, group=group, dtype=tf.float32), trainable=is_training, name=name_prefix + '_3x3/kernel')
            else:
                weight_ = tf.get_variable(name_prefix + '_3x3/kernel', shape=weight_shape, initializer=wrapper_initlizer, trainable=is_training)
            weight_groups = tf.split(weight_, num_or_size_splits=group, axis=-1, name=name_prefix + '_weight_split')
            xs = tf.split(reduced_inputs_relu, num_or_size_splits=group, axis=-1, name=name_prefix + '_inputs_split')

        convolved = [tf.nn.convolution(x, weight, padding='VALID', strides=[1, 1], name=name_prefix + '_group_conv',
                        data_format=('NCHW' if data_format == 'channels_first' else 'NHWC')) for (x, weight) in zip(xs, weight_groups)]

        if data_format == 'channels_first':
            conv3_inputs = tf.concat(convolved, axis=1, name=name_prefix + '_concat')
        else:
            conv3_inputs = tf.concat(convolved, axis=-1, name=name_prefix + '_concat')

        conv3_inputs_bn = tf.layers.batch_normalization(conv3_inputs, momentum=_BATCH_NORM_DECAY, name=name_prefix + '_3x3/bn',
                                            axis=bn_axis, epsilon=_BATCH_NORM_EPSILON, training=is_training, reuse=None, fused=_USE_FUSED_BN)
        conv3_inputs_relu = tf.nn.relu(conv3_inputs_bn, name=name_prefix + '_3x3/relu')


        increase_inputs = tf.layers.conv2d(conv3_inputs_relu, input_filters * 2, (1, 1), use_bias=False,
                                    name=name_prefix + '_1x1_increase', strides=(1, 1),
                                    padding='valid', data_format=data_format, activation=None,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    bias_initializer=tf.zeros_initializer())
        increase_inputs_bn = tf.layers.batch_normalization(increase_inputs, momentum=_BATCH_NORM_DECAY,
                                            name=name_prefix + '_1x1_increase/bn', axis=bn_axis,
                                            epsilon=_BATCH_NORM_EPSILON, training=is_training, reuse=None, fused=_USE_FUSED_BN)

        if data_format == 'channels_first':
            pooled_inputs = tf.reduce_mean(increase_inputs_bn, [2, 3], name=name_prefix + '_global_pool', keep_dims=True)
        else:
            pooled_inputs = tf.reduce_mean(increase_inputs_bn, [1, 2], name=name_prefix + '_global_pool', keep_dims=True)

        down_inputs = tf.layers.conv2d(pooled_inputs, input_filters * 2 // reduced_scale, (1, 1), use_bias=True,
                                    name=name_prefix + '_1x1_down', strides=(1, 1),
                                    padding='valid', data_format=data_format, activation=None,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    bias_initializer=tf.zeros_initializer())
        down_inputs_relu = tf.nn.relu(down_inputs, name=name_prefix + '_1x1_down/relu')


        up_inputs = tf.layers.conv2d(down_inputs_relu, input_filters * 2, (1, 1), use_bias=True,
                                    name=name_prefix + '_1x1_up', strides=(1, 1),
                                    padding='valid', data_format=data_format, activation=None,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    bias_initializer=tf.zeros_initializer())
        prob_outputs = tf.nn.sigmoid(up_inputs, name=name_prefix + '_prob')

        #print(residuals, prob_outputs, increase_inputs_bn)
        rescaled_feat = tf.multiply(prob_outputs, increase_inputs_bn, name=name_prefix + '_mul')
        pre_act = tf.add(residuals, rescaled_feat, name=name_prefix + '_add')
        return tf.nn.relu(pre_act, name=name_prefix + '/relu')

def cascaded_pyramid_net(inputs, output_channals, heatmap_size, istraining, data_format):
    #with tf.variable_scope('resnet50', 'resnet50', values=[inputs]):
    end_points = se_cpn_backbone(inputs, istraining, data_format)
    pyramid_len = len(end_points)
    up_sampling = None
    pyramid_heatmaps = []
    pyramid_laterals = []
    with tf.variable_scope('feature_pyramid', 'feature_pyramid', values=end_points):
        # top-down
        for ind, pyramid in enumerate(reversed(end_points)):
            inputs = conv2d_fixed_padding(inputs=pyramid, filters=256, kernel_size=1, strides=1,
                          data_format=data_format, kernel_initializer=tf.glorot_uniform_initializer, name='1x1_conv1_p{}'.format(pyramid_len - ind))
            lateral = tf.nn.relu(inputs, name='relu1_p{}'.format(pyramid_len - ind))
            if up_sampling is not None:
                if data_format == 'channels_first':
                    up_sampling = tf.transpose(up_sampling, [0, 2, 3, 1], name='trans_p{}'.format(pyramid_len - ind))
                up_sampling = tf.image.resize_bilinear(up_sampling, tf.shape(up_sampling)[-3:-1] * 2, name='upsample_p{}'.format(pyramid_len - ind))
                if data_format == 'channels_first':
                    up_sampling = tf.transpose(up_sampling, [0, 3, 1, 2], name='trans_inv_p{}'.format(pyramid_len - ind))
                up_sampling = conv2d_fixed_padding(inputs=up_sampling, filters=256, kernel_size=1, strides=1,
                          data_format=data_format, kernel_initializer=tf.glorot_uniform_initializer, name='up_conv_p{}'.format(pyramid_len - ind))
                up_sampling = lateral + up_sampling
                lateral = up_sampling
            else:
                up_sampling = lateral

            pyramid_laterals.append(lateral)

            lateral = conv2d_fixed_padding(inputs=lateral, filters=256, kernel_size=1, strides=1,
                          data_format=data_format, kernel_initializer=tf.glorot_uniform_initializer, name='1x1_conv2_p{}'.format(pyramid_len - ind))
            lateral = tf.nn.relu(lateral, name='relu2_p{}'.format(pyramid_len - ind))

            outputs = conv2d_fixed_padding(inputs=lateral, filters=output_channals, kernel_size=3, strides=1,
                          data_format=data_format, kernel_initializer=tf.glorot_uniform_initializer, name='conv_heatmap_p{}'.format(pyramid_len - ind))
            if data_format == 'channels_first':
                outputs = tf.transpose(outputs, [0, 2, 3, 1], name='output_trans_p{}'.format(pyramid_len - ind))
            outputs = tf.image.resize_bilinear(outputs, [heatmap_size, heatmap_size], name='heatmap_p{}'.format(pyramid_len - ind))
            if data_format == 'channels_first':
                outputs = tf.transpose(outputs, [0, 3, 1, 2], name='heatmap_trans_inv_p{}'.format(pyramid_len - ind))
            pyramid_heatmaps.append(outputs)
    with tf.variable_scope('global_net', 'global_net', values=pyramid_laterals):
        global_pyramids = []
        for ind, lateral in enumerate(pyramid_laterals):
            inputs = lateral
            for bottleneck_ind in range(pyramid_len - ind - 1):
                inputs = global_net_bottleneck_block(inputs, 128, istraining, data_format, name='global_net_bottleneck_{}_p{}'.format(bottleneck_ind, pyramid_len - ind))

            #if ind < pyramid_len - 1:
                # resize back to the output heatmap size
            if data_format == 'channels_first':
                outputs = tf.transpose(inputs, [0, 2, 3, 1], name='global_output_trans_p{}'.format(pyramid_len - ind))
            else:
                outputs = inputs
            outputs = tf.image.resize_bilinear(outputs, [heatmap_size, heatmap_size], name='global_heatmap_p{}'.format(pyramid_len - ind))
            if data_format == 'channels_first':
                outputs = tf.transpose(outputs, [0, 3, 1, 2], name='global_heatmap_trans_inv_p{}'.format(pyramid_len - ind))
            # else:
            #     outputs = tf.identity(inputs, 'global_heatmap_p{}'.format(pyramid_len - ind))

            global_pyramids.append(outputs)

        concat_pyramids = tf.concat(global_pyramids, 1 if data_format == 'channels_first' else 3, name='concat')

        def projection_shortcut(inputs):
            return conv2d_fixed_padding(inputs=inputs, filters=256, kernel_size=1, strides=1, data_format=data_format, name='shortcut')

        outputs = global_net_bottleneck_block(concat_pyramids, 128, istraining, data_format, projection_shortcut=projection_shortcut, name='global_concat_bottleneck')
        outputs = conv2d_fixed_padding(inputs=outputs, filters=output_channals, kernel_size=3, strides=1,
                          data_format=data_format, kernel_initializer=tf.glorot_uniform_initializer, name='conv_heatmap')


    return pyramid_heatmaps + [outputs]

def xt_cascaded_pyramid_net(inputs, output_channals, heatmap_size, istraining, data_format, net_depth=50):
    #with tf.variable_scope('resnet50', 'resnet50', values=[inputs]):
    end_points = sext_cpn_backbone(inputs, istraining, data_format, net_depth=net_depth)
    pyramid_len = len(end_points)
    up_sampling = None
    pyramid_heatmaps = []
    pyramid_laterals = []
    with tf.variable_scope('feature_pyramid', 'feature_pyramid', values=end_points):
        # top-down
        for ind, pyramid in enumerate(reversed(end_points)):
            inputs = conv2d_fixed_padding(inputs=pyramid, filters=256, kernel_size=1, strides=1,
                          data_format=data_format, kernel_initializer=tf.glorot_uniform_initializer, name='1x1_conv1_p{}'.format(pyramid_len - ind))
            lateral = tf.nn.relu(inputs, name='relu1_p{}'.format(pyramid_len - ind))
            if up_sampling is not None:
                if data_format == 'channels_first':
                    up_sampling = tf.transpose(up_sampling, [0, 2, 3, 1], name='trans_p{}'.format(pyramid_len - ind))
                up_sampling = tf.image.resize_bilinear(up_sampling, tf.shape(up_sampling)[-3:-1] * 2, name='upsample_p{}'.format(pyramid_len - ind))
                if data_format == 'channels_first':
                    up_sampling = tf.transpose(up_sampling, [0, 3, 1, 2], name='trans_inv_p{}'.format(pyramid_len - ind))
                up_sampling = conv2d_fixed_padding(inputs=up_sampling, filters=256, kernel_size=1, strides=1,
                          data_format=data_format, kernel_initializer=tf.glorot_uniform_initializer, name='up_conv_p{}'.format(pyramid_len - ind))
                up_sampling = lateral + up_sampling
                lateral = up_sampling
            else:
                up_sampling = lateral

            pyramid_laterals.append(lateral)

            lateral = conv2d_fixed_padding(inputs=lateral, filters=256, kernel_size=1, strides=1,
                          data_format=data_format, kernel_initializer=tf.glorot_uniform_initializer, name='1x1_conv2_p{}'.format(pyramid_len - ind))
            lateral = tf.nn.relu(lateral, name='relu2_p{}'.format(pyramid_len - ind))

            outputs = conv2d_fixed_padding(inputs=lateral, filters=output_channals, kernel_size=3, strides=1,
                          data_format=data_format, kernel_initializer=tf.glorot_uniform_initializer, name='conv_heatmap_p{}'.format(pyramid_len - ind))
            if data_format == 'channels_first':
                outputs = tf.transpose(outputs, [0, 2, 3, 1], name='output_trans_p{}'.format(pyramid_len - ind))
            outputs = tf.image.resize_bilinear(outputs, [heatmap_size, heatmap_size], name='heatmap_p{}'.format(pyramid_len - ind))
            if data_format == 'channels_first':
                outputs = tf.transpose(outputs, [0, 3, 1, 2], name='heatmap_trans_inv_p{}'.format(pyramid_len - ind))
            pyramid_heatmaps.append(outputs)
    with tf.variable_scope('global_net', 'global_net', values=pyramid_laterals):
        global_pyramids = []
        for ind, lateral in enumerate(pyramid_laterals):
            inputs = lateral
            for bottleneck_ind in range(pyramid_len - ind - 1):
                inputs = global_net_bottleneck_block(inputs, 128, istraining, data_format, name='global_net_bottleneck_{}_p{}'.format(bottleneck_ind, pyramid_len - ind))

            #if ind < pyramid_len - 1:
                # resize back to the output heatmap size
            if data_format == 'channels_first':
                outputs = tf.transpose(inputs, [0, 2, 3, 1], name='global_output_trans_p{}'.format(pyramid_len - ind))
            else:
                outputs = inputs
            outputs = tf.image.resize_bilinear(outputs, [heatmap_size, heatmap_size], name='global_heatmap_p{}'.format(pyramid_len - ind))
            if data_format == 'channels_first':
                outputs = tf.transpose(outputs, [0, 3, 1, 2], name='global_heatmap_trans_inv_p{}'.format(pyramid_len - ind))
            # else:
            #     outputs = tf.identity(inputs, 'global_heatmap_p{}'.format(pyramid_len - ind))

            global_pyramids.append(outputs)

        concat_pyramids = tf.concat(global_pyramids, 1 if data_format == 'channels_first' else 3, name='concat')

        def projection_shortcut(inputs):
            return conv2d_fixed_padding(inputs=inputs, filters=256, kernel_size=1, strides=1, data_format=data_format, name='shortcut')

        outputs = global_net_bottleneck_block(concat_pyramids, 128, istraining, data_format, projection_shortcut=projection_shortcut, name='global_concat_bottleneck')
        outputs = conv2d_fixed_padding(inputs=outputs, filters=output_channals, kernel_size=3, strides=1,
                          data_format=data_format, kernel_initializer=tf.glorot_uniform_initializer, name='conv_heatmap')


    return pyramid_heatmaps + [outputs]

def head_xt_cascaded_pyramid_net(inputs, output_channals, heatmap_size, istraining, data_format):
    #with tf.variable_scope('resnet50', 'resnet50', values=[inputs]):
    end_points = sext_cpn_backbone(inputs, istraining, data_format)
    pyramid_len = len(end_points)
    up_sampling = None
    pyramid_heatmaps = []
    pyramid_laterals = []
    with tf.variable_scope('feature_pyramid', 'feature_pyramid', values=end_points):
        # top-down
        for ind, pyramid in enumerate(reversed(end_points)):
            inputs = conv2d_fixed_padding(inputs=pyramid, filters=256, kernel_size=1, strides=1,
                          data_format=data_format, kernel_initializer=tf.glorot_uniform_initializer, name='1x1_conv1_p{}'.format(pyramid_len - ind))
            lateral = tf.nn.relu(inputs, name='relu1_p{}'.format(pyramid_len - ind))
            if up_sampling is not None:
                if data_format == 'channels_first':
                    up_sampling = tf.transpose(up_sampling, [0, 2, 3, 1], name='trans_p{}'.format(pyramid_len - ind))
                up_sampling = tf.image.resize_bilinear(up_sampling, tf.shape(up_sampling)[-3:-1] * 2, name='upsample_p{}'.format(pyramid_len - ind))
                if data_format == 'channels_first':
                    up_sampling = tf.transpose(up_sampling, [0, 3, 1, 2], name='trans_inv_p{}'.format(pyramid_len - ind))
                up_sampling = conv2d_fixed_padding(inputs=up_sampling, filters=256, kernel_size=1, strides=1,
                          data_format=data_format, kernel_initializer=tf.glorot_uniform_initializer, name='up_conv_p{}'.format(pyramid_len - ind))
                up_sampling = lateral + up_sampling
                lateral = up_sampling
            else:
                up_sampling = lateral

            pyramid_laterals.append(lateral)

            lateral = conv2d_fixed_padding(inputs=lateral, filters=256, kernel_size=1, strides=1,
                          data_format=data_format, kernel_initializer=tf.glorot_uniform_initializer, name='1x1_conv2_p{}'.format(pyramid_len - ind))
            lateral = tf.nn.relu(lateral, name='relu2_p{}'.format(pyramid_len - ind))

            outputs = conv2d_fixed_padding(inputs=lateral, filters=output_channals, kernel_size=3, strides=1,
                          data_format=data_format, kernel_initializer=tf.glorot_uniform_initializer, name='conv_heatmap_p{}'.format(pyramid_len - ind))
            if data_format == 'channels_first':
                outputs = tf.transpose(outputs, [0, 2, 3, 1], name='output_trans_p{}'.format(pyramid_len - ind))
            outputs = tf.image.resize_bilinear(outputs, [heatmap_size, heatmap_size], name='heatmap_p{}'.format(pyramid_len - ind))
            if data_format == 'channels_first':
                outputs = tf.transpose(outputs, [0, 3, 1, 2], name='heatmap_trans_inv_p{}'.format(pyramid_len - ind))
            pyramid_heatmaps.append(outputs)
    with tf.variable_scope('global_net', 'global_net', values=pyramid_laterals):
        global_pyramids = []
        for ind, lateral in enumerate(pyramid_laterals):
            inputs = lateral
            for bottleneck_ind in range(pyramid_len - ind - 1):
                inputs = global_net_sext_bottleneck_block(inputs, 128, istraining, data_format, name_prefix='global_net_bottleneck_{}_p{}'.format(bottleneck_ind, pyramid_len - ind))

            #if ind < pyramid_len - 1:
                # resize back to the output heatmap size
            if data_format == 'channels_first':
                outputs = tf.transpose(inputs, [0, 2, 3, 1], name='global_output_trans_p{}'.format(pyramid_len - ind))
            else:
                outputs = inputs
            outputs = tf.image.resize_bilinear(outputs, [heatmap_size, heatmap_size], name='global_heatmap_p{}'.format(pyramid_len - ind))
            if data_format == 'channels_first':
                outputs = tf.transpose(outputs, [0, 3, 1, 2], name='global_heatmap_trans_inv_p{}'.format(pyramid_len - ind))
            # else:
            #     outputs = tf.identity(inputs, 'global_heatmap_p{}'.format(pyramid_len - ind))

            global_pyramids.append(outputs)

        concat_pyramids = tf.concat(global_pyramids, 1 if data_format == 'channels_first' else 3, name='concat')

        outputs = global_net_sext_bottleneck_block(concat_pyramids, 128, istraining, data_format, need_reduce=True, name_prefix='global_concat_bottleneck')
        outputs = conv2d_fixed_padding(inputs=outputs, filters=output_channals, kernel_size=3, strides=1,
                          data_format=data_format, kernel_initializer=tf.glorot_uniform_initializer, name='conv_heatmap')


    return pyramid_heatmaps + [outputs]
