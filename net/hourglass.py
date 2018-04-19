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
#initializer_to_use = tf.glorot_uniform_initializer  glorot_normal_initializer
initializer_to_use = tf.glorot_uniform_initializer
conv_bn_initializer_to_use = tf.glorot_uniform_initializer#lambda : tf.truncated_normal_initializer(mean=0.0, stddev=0.005)

def batch_norm_relu(inputs, is_training, data_format, name=None):
  """Performs a batch normalization followed by a ReLU."""
  # We set fused=True for a significant performance boost. See
  # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
  with tf.variable_scope(name, 'batch_norm_relu', values=[inputs]):
    inputs = tf.layers.batch_normalization(
              inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
              momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
              scale=True, training=is_training, fused=_USE_FUSED_BN, name='batch_normalization')
    inputs = tf.nn.relu(inputs, name='relu')
    return inputs

def batch_norm(inputs, is_training, data_format, name=None):
  """Performs a batch normalization followed by a ReLU."""
  # We set fused=True for a significant performance boost. See
  # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
  with tf.variable_scope(name, 'batch_norm', values=[inputs]):
    inputs = tf.layers.batch_normalization(
              inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
              momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
              scale=True, training=is_training, fused=_USE_FUSED_BN, name='batch_normalization')
    return inputs

def fixed_padding(inputs, kernel_size, data_format):
  with tf.variable_scope('fixed_padding', values=[inputs]):
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    if data_format == 'channels_first':
      padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                      [pad_beg, pad_end], [pad_beg, pad_end]], name='padding')
    else:
      padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                      [pad_beg, pad_end], [0, 0]], name='padding')
    return padded_inputs

# this is only can be used before BN
def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format, kernel_initializer=conv_bn_initializer_to_use, name=None):
  """Strided 2-D convolution with explicit padding."""
  # The padding is consistent and is based only on `kernel_size`, not on the
  # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
  with tf.variable_scope(name, 'fix_padding_conv', values=[inputs]):
    if strides > 1:
      inputs = fixed_padding(inputs, kernel_size, data_format)

    return tf.layers.conv2d(
              inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
              padding=('same' if strides == 1 else 'valid'), use_bias=False,
              kernel_initializer=kernel_initializer(),
              data_format=data_format, name='conv2d')

def bottleneck_block_v2(inputs, in_filters, out_filters, is_training, data_format, name=None):
  with tf.variable_scope(name, 'bottleneck_block', values=[inputs]):
    shortcut = inputs
    inputs = batch_norm_relu(inputs, is_training, data_format, name='bn_relu_1')

    # The projection shortcut should come after the first batch norm and ReLU
    # since it performs a 1x1 convolution.
    # different from original hourglass
    if in_filters != out_filters:
      shortcut = conv2d_fixed_padding(
                  inputs=inputs, filters=out_filters, kernel_size=1, strides=1,
                  data_format=data_format, name='skip')

    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=out_filters//2, kernel_size=1, strides=1,
        data_format=data_format, name='1x1_down')
    inputs = batch_norm_relu(inputs, is_training, data_format, name='bn_relu_2')

    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=out_filters//2, kernel_size=3, strides=1,
        data_format=data_format, name='3x3_conv')
    inputs = batch_norm_relu(inputs, is_training, data_format, name='bn_relu_3')

    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=out_filters, kernel_size=1, strides=1,
        data_format=data_format, name='1x1_up')

    return tf.add(inputs, shortcut, name='elem_add')

def bottleneck_block_v1(inputs, in_filters, out_filters, is_training, data_format, name=None):
  with tf.variable_scope(name, 'bottleneck_block_v1', values=[inputs]):
    shortcut = inputs
    if in_filters != out_filters:
      shortcut = conv2d_fixed_padding(
                  inputs=shortcut, filters=out_filters, kernel_size=1, strides=1,
                  data_format=data_format, name='skip')
      shortcut = batch_norm(shortcut, is_training, data_format, name='skip_bn')

    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=out_filters//2, kernel_size=1, strides=1,
        data_format=data_format, name='1x1_down')
    inputs = batch_norm_relu(inputs, is_training, data_format, name='bn_relu_1')

    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=out_filters//2, kernel_size=3, strides=1,
        data_format=data_format, name='3x3_conv')
    inputs = batch_norm_relu(inputs, is_training, data_format, name='bn_relu_2')

    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=out_filters, kernel_size=1, strides=1,
        data_format=data_format, name='1x1_up')
    inputs = batch_norm(inputs, is_training, data_format, name='up_bn')

    return tf.nn.relu(tf.add(inputs, shortcut, name='elem_add'), name='relu')


bottleneck_block = bottleneck_block_v2

def dozen_bottleneck_blocks(inputs, in_filters, out_filters, num_modules, is_training, data_format, name=None):
  for m in range(num_modules):
    inputs = bottleneck_block(inputs, in_filters, out_filters, is_training, data_format, name=None if name is None else name.format(m))

  return inputs

def hourglass(inputs, filters, is_training, data_format, deep_index=1, num_modules=1, name=None):
  with tf.variable_scope(name, 'hourglass_unit', values=[inputs]):
    upchannal1 = dozen_bottleneck_blocks(inputs, filters, filters, num_modules, is_training, data_format, name='up_{}')
    downchannal1 = tf.layers.max_pooling2d(inputs=inputs, pool_size=2, strides=2, padding='valid', data_format=data_format, name='down_pool')
    downchannal1 = dozen_bottleneck_blocks(downchannal1, filters, filters, num_modules, is_training, data_format, name='down1_{}')

    if deep_index > 1:
      downchannal2 = hourglass(downchannal1, filters, is_training, data_format, deep_index=deep_index-1, num_modules=num_modules, name='inner_{}'.format(deep_index))
    else:
      downchannal2 = dozen_bottleneck_blocks(downchannal1, filters, filters, num_modules, is_training, data_format, name='down2_{}')

    downchannal3 = dozen_bottleneck_blocks(downchannal2, filters, filters, num_modules, is_training, data_format, name='down3_{}')

    if data_format == 'channels_first':
        downchannal3 = tf.transpose(downchannal3, [0, 2, 3, 1], name='trans')
    input_shape = tf.shape(downchannal3)[-3:-1] * 2
    upchannal2 = tf.image.resize_bilinear(downchannal3, input_shape, name='resize')
    if data_format == 'channels_first':
      upchannal2 = tf.transpose(upchannal2, [0, 3, 1, 2], name='trans_inv')

    return tf.add(upchannal1, upchannal2, name='elem_add')

def create_model(inputs, num_stack, feat_channals, output_channals, num_modules, is_training, data_format):
  with tf.variable_scope('precede', values=[inputs]):
    inputs = conv2d_fixed_padding(inputs=inputs, filters=64, kernel_size=7, strides=2,
              data_format=data_format, kernel_initializer=conv_bn_initializer_to_use, name='conv_7x7')
    inputs = batch_norm_relu(inputs, is_training, data_format, name='inputs_bn')

    inputs = bottleneck_block(inputs, 64, 128, is_training, data_format, name='residual1')
    inputs = tf.layers.max_pooling2d(inputs=inputs, pool_size=2, strides=2, padding='valid',
                data_format=data_format, name='pool')

    inputs = bottleneck_block(inputs, 128, 128, is_training, data_format, name='residual2')
    inputs = bottleneck_block(inputs, 128, feat_channals, is_training, data_format, name='residual3')

  #return [inputs]
  hg_inputs = inputs
  outputs_list = []
  for stack_index in range(num_stack):
    hg = hourglass(hg_inputs, feat_channals, is_training, data_format, deep_index=4, num_modules=num_modules, name='stack_{}/hg'.format(stack_index))

    hg = dozen_bottleneck_blocks(hg, feat_channals, feat_channals, num_modules, is_training, data_format, name='stack_{}/'.format(stack_index) + 'output_{}')

    # produce prediction
    output_scores = conv2d_fixed_padding(inputs=hg, filters=feat_channals, kernel_size=1, strides=1, data_format=data_format, name='stack_{}/output_1x1'.format(stack_index))
    #outputs_list.append(output_scores)
    output_scores = batch_norm_relu(output_scores, is_training, data_format, name='stack_{}/output_bn'.format(stack_index))

    # produce heatmap from prediction
    # use variable_scope to help model resotre name filter
    heatmap = tf.layers.conv2d(inputs=output_scores, filters=output_channals, kernel_size=1,
                                strides=1, padding='same', use_bias=True, activation=None,
                                kernel_initializer=initializer_to_use(),
                                bias_initializer=tf.zeros_initializer(),
                                data_format=data_format,
                                name='hg_heatmap/stack_{}/heatmap_1x1'.format(stack_index))


    outputs_list.append(heatmap)
    # no remap conv for the last hourglass
    if stack_index < num_stack - 1:
      output_scores_ = tf.layers.conv2d(inputs=output_scores, filters=feat_channals, kernel_size=1,
                          strides=1, padding='same', use_bias=True, activation=None,
                          kernel_initializer=initializer_to_use(),
                          bias_initializer=tf.zeros_initializer(),
                          data_format=data_format,
                          name='stack_{}/remap_outputs'.format(stack_index))
      # use variable_scope to help model resotre name filter
      heatmap_ = tf.layers.conv2d(inputs=heatmap, filters=feat_channals, kernel_size=1,
                        strides=1, padding='same', use_bias=True, activation=None,
                        kernel_initializer=initializer_to_use(),
                        bias_initializer=tf.zeros_initializer(),
                        data_format=data_format,
                        name='hg_heatmap/stack_{}/remap_heatmap'.format(stack_index))

      # next hourglass inputs
      fused_heatmap = tf.add(output_scores_, heatmap_, 'stack_{}/fused_heatmap'.format(stack_index))
      hg_inputs = tf.add(hg_inputs, fused_heatmap, 'stack_{}/next_inputs'.format(stack_index))
      #hg_inputs = hg_inputs + output_scores_ + heatmap_

  return outputs_list




