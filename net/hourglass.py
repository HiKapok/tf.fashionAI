from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5
_USE_FUSED_BN = True
#initializer_to_use = tf.glorot_uniform_initializer
initializer_to_use = tf.glorot_normal_initializer
conv_bn_initializer_to_use = tf.glorot_normal_initializer#lambda : tf.truncated_normal_initializer(mean=0.0, stddev=0.005)

def batch_norm_relu(inputs, is_training, data_format, name=None):
  """Performs a batch normalization followed by a ReLU."""
  # We set fused=True for a significant performance boost. See
  # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
  inputs = tf.layers.batch_normalization(
            inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
            momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
            scale=True, training=is_training, fused=_USE_FUSED_BN, name=name)
  inputs = tf.nn.relu(inputs, name=name + '/relu' if name is not None else None)
  return inputs

def batch_norm(inputs, is_training, data_format, name=None):
  """Performs a batch normalization followed by a ReLU."""
  # We set fused=True for a significant performance boost. See
  # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
  inputs = tf.layers.batch_normalization(
            inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
            momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
            scale=True, training=is_training, fused=_USE_FUSED_BN, name=name)
  return inputs

def fixed_padding(inputs, kernel_size, data_format):
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

# this is only can be used before BN
def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format, kernel_initializer=conv_bn_initializer_to_use, name=None):
  """Strided 2-D convolution with explicit padding."""
  # The padding is consistent and is based only on `kernel_size`, not on the
  # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
  if strides > 1:
    inputs = fixed_padding(inputs, kernel_size, data_format)

  return tf.layers.conv2d(
            inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
            padding=('same' if strides == 1 else 'valid'), use_bias=False,
            kernel_initializer=kernel_initializer(),
            data_format=data_format, name=name)


def bottleneck_block(inputs, in_filters, out_filters, is_training, data_format, name=None):
  shortcut = inputs
  inputs = batch_norm_relu(inputs, is_training, data_format, name=None if name is None else name+'_bn1')

  # The projection shortcut should come after the first batch norm and ReLU
  # since it performs a 1x1 convolution.
  # different from original hourglass
  if in_filters != out_filters:
    shortcut = conv2d_fixed_padding(
                inputs=inputs, filters=out_filters, kernel_size=1, strides=1,
                data_format=data_format, name=None if name is None else name+'_skip')

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=out_filters//2, kernel_size=1, strides=1,
      data_format=data_format, name=None if name is None else name+'_1x1_down')
  inputs = batch_norm_relu(inputs, is_training, data_format, name=None if name is None else name+'_bn3')

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=out_filters//2, kernel_size=3, strides=1,
      data_format=data_format, name=None if name is None else name+'_3x3_conv')
  inputs = batch_norm_relu(inputs, is_training, data_format, name=None if name is None else name+'_bn3')

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=out_filters, kernel_size=1, strides=1,
      data_format=data_format, name=None if name is None else name+'_1x1_up')

  return inputs + shortcut

def dozen_bottleneck_blocks(inputs, in_filters, out_filters, num_modules, is_training, data_format, name=None):
  for m in range(num_modules):
    inputs = bottleneck_block(inputs, in_filters, out_filters, is_training, data_format, name=None if name is None else name.format(m))

  return inputs

def hourglass(inputs, filters, is_training, data_format, deep_index=1, num_modules=1, name=None):
  upchannal1 = dozen_bottleneck_blocks(inputs, filters, filters, num_modules, is_training, data_format, name=None if name is None else name+'_up_{}')
  # upchannal1 = inputs
  # for m in range(num_modules):
  #   upchannal1 = bottleneck_block(upchannal1, filters, filters, is_training, data_format, name=None if name is None else name+'_up_{}'.format(m))

  downchannal1 = tf.layers.max_pooling2d(inputs=inputs, pool_size=2, strides=2, padding='valid',
          data_format=data_format, name=None if name is None else name+'_down_pool')

  downchannal1 = dozen_bottleneck_blocks(downchannal1, filters, filters, num_modules, is_training, data_format, name=None if name is None else name+'_down1_{}')
  # for m in range(num_modules):
  #   downchannal1 = bottleneck_block(downchannal1, filters, filters, is_training, data_format, name=None if name is None else name+'_down1_{}'.format(m))

  if deep_index > 1:
    downchannal2 = hourglass(downchannal1, filters, is_training, data_format, deep_index=deep_index-1, num_modules=num_modules, name=None if name is None else name+'_inner_{}'.format(deep_index))
  else:
    downchannal2 = dozen_bottleneck_blocks(downchannal1, filters, filters, num_modules, is_training, data_format, name=None if name is None else name+'_down2_{}')
    # downchannal2 = downchannal1
    # for m in range(num_modules):
    #   downchannal2 = bottleneck_block(downchannal2, filters, filters, is_training, data_format, name=None if name is None else name+'_down2_{}'.format(m))

  downchannal3 = dozen_bottleneck_blocks(downchannal2, filters, filters, num_modules, is_training, data_format, name=None if name is None else name+'_down3_{}')
  # downchannal3 = downchannal2
  # for m in range(num_modules):
  #   downchannal3 = bottleneck_block(downchannal3, filters, filters, is_training, data_format, name=None if name is None else name+'_down3_{}'.format(m))

  if data_format == 'channels_first':
      downchannal3 = tf.transpose(downchannal3, [0, 2, 3, 1])

  input_shape = tf.shape(downchannal3)[-3:-1]
  upchannal2 = tf.image.resize_images(downchannal3, input_shape * 2, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  if data_format == 'channels_first':
    upchannal2 = tf.transpose(upchannal2, [0, 3, 1, 2])

  return upchannal1 + upchannal2

def create_model(inputs, num_stack, feat_channals, output_channals, num_modules, is_training, data_format, name):
  inputs = conv2d_fixed_padding(inputs=inputs, filters=64, kernel_size=7, strides=2,
            data_format=data_format, kernel_initializer=conv_bn_initializer_to_use, name='precede_7x7')
  inputs = batch_norm_relu(inputs, is_training, data_format, name='precede_bn')

  inputs = bottleneck_block(inputs, 64, 128, is_training, data_format, name='precede/residual1')
  inputs = tf.layers.max_pooling2d(inputs=inputs, pool_size=2, strides=2, padding='valid',
              data_format=data_format, name='precede_pool')

  inputs = bottleneck_block(inputs, 128, 128, is_training, data_format, name='precede/residual2')
  inputs = bottleneck_block(inputs, 128, feat_channals, is_training, data_format, name='precede/residual3')

  hg_inputs = inputs
  outputs_list = []
  for stack_index in range(num_stack):
    hg = hourglass(hg_inputs, feat_channals, is_training, data_format, deep_index=4, num_modules=num_modules, name='stack_{}/hg'.format(stack_index))

    hg = dozen_bottleneck_blocks(hg, feat_channals, feat_channals, num_modules, is_training, data_format, name=None if name is None else 'stack_{}/'.format(stack_index) + 'output_{}')
    # for m in range(num_modules):
    #   hg = bottleneck_block(hg, feat_channals, feat_channals, is_training, data_format, name='stack_{}/output_{}'.format(stack_index, m))

    # produce prediction
    output_scores = conv2d_fixed_padding(inputs=hg, filters=feat_channals, kernel_size=1, strides=1, data_format=data_format, name='stack_{}/output_1x1'.format(stack_index))
    output_scores = batch_norm_relu(output_scores, is_training, data_format, name='stack_{}/output_bn'.format(stack_index))

    # produce heatmap from prediction
    # use variable_scope to help model resotre name filter
    with tf.variable_scope('hg_heatmap', default_name=None, values=[output_scores], reuse=tf.AUTO_REUSE):
      heatmap = tf.layers.conv2d(inputs=output_scores, filters=output_channals, kernel_size=1,
                                strides=1, padding='same', use_bias=True, activation=None,
                                kernel_initializer=initializer_to_use(),
                                bias_initializer=tf.zeros_initializer(),
                                data_format=data_format,
                                name='stack_{}/heatmap_1x1'.format(stack_index))


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
      with tf.variable_scope('hg_heatmap', default_name=None, values=[heatmap], reuse=tf.AUTO_REUSE):
        heatmap_ = tf.layers.conv2d(inputs=heatmap, filters=feat_channals, kernel_size=1,
                        strides=1, padding='same', use_bias=True, activation=None,
                        kernel_initializer=initializer_to_use(),
                        bias_initializer=tf.zeros_initializer(),
                        data_format=data_format,
                        name='stack_{}/remap_heatmap'.format(stack_index))

      # next hourglass inputs
      hg_inputs = hg_inputs + output_scores_ + heatmap_

  return outputs_list




