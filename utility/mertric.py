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

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope

import tensorflow as tf

def metric_variable(shape, dtype, validate_shape=True, name=None):
  """Create variable in `GraphKeys.(LOCAL|METRIC_VARIABLES`) collections."""

  return variable_scope.variable(
      lambda: array_ops.zeros(shape, dtype),
      trainable=False,
      collections=[
          ops.GraphKeys.LOCAL_VARIABLES #, ops.GraphKeys.METRIC_VARIABLES
      ],
      validate_shape=validate_shape,
      name=name)

def _safe_div(numerator, denominator, name):
  """Divides two tensors element-wise, returning 0 if the denominator is <= 0.

  Args:
    numerator: A real `Tensor`.
    denominator: A real `Tensor`, with dtype matching `numerator`.
    name: Name for the returned op.

  Returns:
    0 if `denominator` <= 0, else `numerator` / `denominator`
  """
  t = math_ops.truediv(numerator, denominator)
  zero = array_ops.zeros_like(t, dtype=denominator.dtype)
  condition = math_ops.greater(denominator, zero)
  zero = math_ops.cast(zero, t.dtype)
  return array_ops.where(condition, t, zero, name=name)

def normalized_error(targets, predictions, norm_value, visible, isvalid,
             bacth_size, num_keypoint, heatmap_size,
             train_image_size, clip_at_zero=True, name=None):

  with variable_scope.variable_scope(name, 'normalized_error', (targets, predictions, norm_value, visible, isvalid, bacth_size, num_keypoint, train_image_size, heatmap_size)):

    total = metric_variable([], dtypes.float32, name='total')
    count = metric_variable([], dtypes.float32, name='count')

    targets, predictions = tf.reshape(targets, [bacth_size, num_keypoint, -1]), tf.reshape(predictions, [bacth_size, num_keypoint, -1])

    pred_max = tf.reduce_max(predictions, axis=-1)
    pred_indices = tf.argmax(predictions, axis=-1)
    pred_x, pred_y = tf.floormod(pred_indices, heatmap_size) * train_image_size / heatmap_size, tf.floordiv(pred_indices, heatmap_size) * train_image_size / heatmap_size
    pred_x, pred_y = tf.cast(pred_x, tf.float32), tf.cast(pred_y, tf.float32)
    if clip_at_zero:
      pred_x, pred_y =  pred_x * tf.cast(pred_max>0, tf.float32), pred_y * tf.cast(pred_max>0, tf.float32)

    gt_indices = tf.argmax(targets, axis=-1)
    gt_x, gt_y = tf.floormod(gt_indices, heatmap_size) * train_image_size / heatmap_size, tf.floordiv(gt_indices, heatmap_size) * train_image_size / heatmap_size

    gt_x, gt_y = tf.cast(gt_x, tf.float32), tf.cast(gt_y, tf.float32)
    #print(gt_x,gt_y,pred_x,pred_y)
    #print(norm_value)
    #print(gt_x)
    #print(pred_x)
    dist = _safe_div(tf.pow(tf.pow(gt_x - pred_x, 2.) + tf.pow(gt_y - pred_y, 2.), .5), tf.expand_dims(norm_value, -1), 'norm_dist')

    #print(visible, isvalid)

    #dist = tf.cond(tf.equal(tf.shape(visible)[-1], tf.shape(isvalid)[-1]), lambda : tf.boolean_mask(dist, tf.logical_and(visible>0, isvalid>0)), lambda : dist)
    #print(dist)
    dist = tf.boolean_mask(dist, tf.logical_and(visible>0, isvalid>0))
    #dist = dist * tf.cast(tf.logical_and(visible>0, isvalid>0), tf.float32)

    update_total_op = state_ops.assign(total, math_ops.reduce_sum(dist))#assign_add #assign
    update_count_op = state_ops.assign(count, tf.cast(tf.shape(dist)[0], tf.float32))#assign_add #assign

    mean_t = _safe_div(total, count, 'value')
    update_op = _safe_div(update_total_op, update_count_op, 'update_op')

    return mean_t, update_op


