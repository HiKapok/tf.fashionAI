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

def _dilated_bottleneck_block_v1(inputs, filters, training, projection_shortcut, data_format):
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

    inputs = tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=3, strides=1,
                  dilation_rate=(2, 2), padding='SAME', use_bias=False,
                  kernel_initializer=tf.glorot_uniform_initializer(),
                  data_format=data_format, name=None)

    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)

    inputs = conv2d_fixed_padding(
                inputs=inputs, filters=4 * filters, kernel_size=1, strides=1,
                data_format=data_format)
    inputs = batch_norm(inputs, training, data_format)
    #print(inputs)
    inputs += shortcut
    inputs = tf.nn.relu(inputs)

    return inputs

def dilated_block_layer(inputs, filters, bottleneck, block_fn, blocks,
                training, name, data_format):
    # Bottleneck blocks end with 4x the number of filters as they start with
    filters_out = filters * 4 if bottleneck else filters

    def projection_shortcut(inputs):
        return conv2d_fixed_padding(inputs=inputs, filters=filters_out, kernel_size=1, strides=1, data_format=data_format)

    # Only the first block per block_layer uses projection_shortcut and strides
    inputs = block_fn(inputs, filters, training, projection_shortcut, data_format)

    for _ in range(1, blocks):
        inputs = block_fn(inputs, filters, training, None, data_format)

    return tf.identity(inputs, name)

def detnet_cpn_backbone(inputs, istraining, data_format):
    block_strides = [1, 2, 2]
    inputs = conv2d_fixed_padding(inputs=inputs, filters=64, kernel_size=7, strides=2, data_format=data_format, kernel_initializer=tf.glorot_uniform_initializer)
    inputs = tf.identity(inputs, 'initial_conv')

    inputs = tf.layers.max_pooling2d(inputs=inputs, pool_size=3, strides=2, padding='SAME', data_format=data_format)
    inputs = tf.identity(inputs, 'initial_max_pool')

    end_points = []
    for i, num_blocks in enumerate([3, 4, 6]):
      num_filters = 64 * (2**i)
      #with tf.variable_scope('block_{}'.format(i), 'resnet50', values=[inputs]):
      inputs = block_layer(
          inputs=inputs, filters=num_filters, bottleneck=True,
          block_fn=_bottleneck_block_v1, blocks=num_blocks,
          strides=block_strides[i], training=istraining,
          name='block_layer{}'.format(i + 1), data_format=data_format)
      end_points.append(inputs)

    #print(inputs)
    with tf.variable_scope('additional_layer', 'additional_layer', values=[inputs]):
      # conv5
      inputs = dilated_block_layer(
            inputs=inputs, filters=256, bottleneck=True,
            block_fn=_dilated_bottleneck_block_v1, blocks=3, training=istraining,
            name='block_layer{}'.format(4), data_format=data_format)
      end_points.append(inputs)
      # conv6
      inputs = dilated_block_layer(
            inputs=inputs, filters=256, bottleneck=True,
            block_fn=_dilated_bottleneck_block_v1, blocks=3, training=istraining,
            name='block_layer{}'.format(5), data_format=data_format)
      end_points.append(inputs)

    return end_points[1:]

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

def cascaded_pyramid_net(inputs, output_channals, heatmap_size, istraining, data_format):
    #with tf.variable_scope('resnet50', 'resnet50', values=[inputs]):
    end_points = detnet_cpn_backbone(inputs, istraining, data_format)
    pyramid_len = len(end_points)
    up_sampling = None
    pyramid_heatmaps = []
    pyramid_laterals = []
    with tf.variable_scope('feature_pyramid', 'feature_pyramid', values=end_points):
        # top-down
        for ind, pyramid in enumerate(reversed(end_points)):
            inputs = conv2d_fixed_padding(inputs=pyramid, filters=256, kernel_size=1, strides=1,
                          data_format=data_format, kernel_initializer=tf.glorot_uniform_initializer, name='1x1_conv1_p{}'.format(pyramid_len - ind + 1))
            lateral = tf.nn.relu(inputs, name='relu1_p{}'.format(pyramid_len - ind + 1))
            if up_sampling is not None:
                if ind > pyramid_len - 2:
                    if data_format == 'channels_first':
                        up_sampling = tf.transpose(up_sampling, [0, 2, 3, 1], name='trans_p{}'.format(pyramid_len - ind + 1))
                    up_sampling = tf.image.resize_bilinear(up_sampling, tf.shape(up_sampling)[-3:-1] * 2, name='upsample_p{}'.format(pyramid_len - ind + 1))
                    if data_format == 'channels_first':
                        up_sampling = tf.transpose(up_sampling, [0, 3, 1, 2], name='trans_inv_p{}'.format(pyramid_len - ind + 1))
                    up_sampling = conv2d_fixed_padding(inputs=up_sampling, filters=256, kernel_size=1, strides=1,
                              data_format=data_format, kernel_initializer=tf.glorot_uniform_initializer, name='up_conv_p{}'.format(pyramid_len - ind + 1))
                up_sampling = lateral + up_sampling
                lateral = up_sampling
            else:
                up_sampling = lateral

            pyramid_laterals.append(lateral)

            lateral = conv2d_fixed_padding(inputs=lateral, filters=256, kernel_size=1, strides=1,
                          data_format=data_format, kernel_initializer=tf.glorot_uniform_initializer, name='1x1_conv2_p{}'.format(pyramid_len - ind + 1))
            lateral = tf.nn.relu(lateral, name='relu2_p{}'.format(pyramid_len - ind + 1))

            outputs = conv2d_fixed_padding(inputs=lateral, filters=output_channals, kernel_size=3, strides=1,
                          data_format=data_format, kernel_initializer=tf.glorot_uniform_initializer, name='conv_heatmap_p{}'.format(pyramid_len - ind + 1))
            if data_format == 'channels_first':
                outputs = tf.transpose(outputs, [0, 2, 3, 1], name='output_trans_p{}'.format(pyramid_len - ind + 1))
            outputs = tf.image.resize_bilinear(outputs, [heatmap_size, heatmap_size], name='heatmap_p{}'.format(pyramid_len - ind + 1))
            if data_format == 'channels_first':
                outputs = tf.transpose(outputs, [0, 3, 1, 2], name='heatmap_trans_inv_p{}'.format(pyramid_len - ind + 1))
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
