# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Provides utilities to preprocess images.

The preprocessing steps for VGG were introduced in the following technical
report:

  Very Deep Convolutional Networks For Large-Scale Image Recognition
  Karen Simonyan and Andrew Zisserman
  arXiv technical report, 2015
  PDF: http://arxiv.org/pdf/1409.1556.pdf
  ILSVRC 2014 Slides: http://www.robots.ox.ac.uk/~karen/pdf/ILSVRC_2014.pdf
  CC-BY-4.0

More information can be obtained from the VGG website:
www.robots.ox.ac.uk/~vgg/research/very_deep/
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

import math
import os

import config

slim = tf.contrib.slim

# _R_MEAN = 123.68
# _G_MEAN = 116.78
# _B_MEAN = 103.94
_R_MEAN = 171.04
_G_MEAN = 162.98
_B_MEAN = 159.89
# std: 62.37, 64.64, 64.36

# [171.04052664596992, 162.98214001911154, 159.88648003318914]
# [62.370313796103616, 64.64434475667025, 64.35966787914904]

_RESIZE_SIDE_MIN = 256
_RESIZE_SIDE_MAX = 512


def apply_with_random_selector(x, func, num_cases):
  """Computes func(x, sel), with sel sampled from [0...num_cases-1].

  Args:
    x: input Tensor.
    func: Python function to apply.
    num_cases: Python int32, number of cases to sample sel from.

  Returns:
    The result of func(x, sel), where func receives the value of the
    selector as a python integer, but sel is sampled dynamically.
  """
  sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
  # Pass the real x only to one of the func calls.
  return control_flow_ops.merge([
      func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
      for case in range(num_cases)])[0]

def distort_color(image, color_ordering=0, fast_mode=True, scope=None):
  """Distort the color of a Tensor image.

  Each color distortion is non-commutative and thus ordering of the color ops
  matters. Ideally we would randomly permute the ordering of the color ops.
  Rather then adding that level of complication, we select a distinct ordering
  of color ops for each preprocessing thread.

  Args:
    image: 3-D Tensor containing single image in [0, 1].
    color_ordering: Python int, a type of distortion (valid values: 0-3).
    fast_mode: Avoids slower ops (random_hue and random_contrast)
    scope: Optional scope for name_scope.
  Returns:
    3-D Tensor color-distorted image on range [0, 1]
  Raises:
    ValueError: if color_ordering not in [0, 3]
  """
  with tf.name_scope(scope, 'distort_color', [image]):
    if fast_mode:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      else:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
    else:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
      elif color_ordering == 1:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
      elif color_ordering == 2:
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      elif color_ordering == 3:
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
      else:
        raise ValueError('color_ordering must be in [0, 3]')

    # The random_* ops do not necessarily clamp.
    return tf.clip_by_value(image, 0.0, 1.0)

def distort_color_v0(image, color_ordering=0, fast_mode=True, scope=None):
  """Distort the color of a Tensor image.

  Each color distortion is non-commutative and thus ordering of the color ops
  matters. Ideally we would randomly permute the ordering of the color ops.
  Rather then adding that level of complication, we select a distinct ordering
  of color ops for each preprocessing thread.

  Args:
    image: 3-D Tensor containing single image in [0, 1].
    color_ordering: Python int, a type of distortion (valid values: 0-3).
    fast_mode: Avoids slower ops (random_hue and random_contrast)
    scope: Optional scope for name_scope.
  Returns:
    3-D Tensor color-distorted image on range [0, 1]
  Raises:
    ValueError: if color_ordering not in [0, 3]
  """
  with tf.name_scope(scope, 'distort_color', [image]):
    if fast_mode:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      else:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32.)
    else:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
      elif color_ordering == 1:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
      elif color_ordering == 2:
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_brightness(image, max_delta=32.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      elif color_ordering == 3:
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32.)
      else:
        raise ValueError('color_ordering must be in [0, 3]')

    # The random_* ops do not necessarily clamp.
    return tf.clip_by_value(image, 0.0, 255.0)


def _crop(image, offset_height, offset_width, crop_height, crop_width):
  """Crops the given image using the provided offsets and sizes.

  Note that the method doesn't assume we know the input image size but it does
  assume we know the input image rank.

  Args:
    image: an image of shape [height, width, channels].
    offset_height: a scalar tensor indicating the height offset.
    offset_width: a scalar tensor indicating the width offset.
    crop_height: the height of the cropped image.
    crop_width: the width of the cropped image.

  Returns:
    the cropped (and resized) image.

  Raises:
    InvalidArgumentError: if the rank is not 3 or if the image dimensions are
      less than the crop size.
  """
  original_shape = tf.shape(image)

  rank_assertion = tf.Assert(
      tf.equal(tf.rank(image), 3),
      ['Rank of image must be equal to 3.'])
  with tf.control_dependencies([rank_assertion]):
    cropped_shape = tf.stack([crop_height, crop_width, original_shape[2]])

  size_assertion = tf.Assert(
      tf.logical_and(
          tf.greater_equal(original_shape[0], crop_height),
          tf.greater_equal(original_shape[1], crop_width)),
      ['Crop size greater than the image size.'])

  offsets = tf.to_int32(tf.stack([offset_height, offset_width, 0]))

  # Use tf.slice instead of crop_to_bounding box as it accepts tensors to
  # define the crop size.
  with tf.control_dependencies([size_assertion]):
    image = tf.slice(image, offsets, cropped_shape)
  return tf.reshape(image, cropped_shape)


def _random_crop(image_list, crop_height, crop_width):
  """Crops the given list of images.

  The function applies the same crop to each image in the list. This can be
  effectively applied when there are multiple image inputs of the same
  dimension such as:

    image, depths, normals = _random_crop([image, depths, normals], 120, 150)

  Args:
    image_list: a list of image tensors of the same dimension but possibly
      varying channel.
    crop_height: the new height.
    crop_width: the new width.

  Returns:
    the image_list with cropped images.

  Raises:
    ValueError: if there are multiple image inputs provided with different size
      or the images are smaller than the crop dimensions.
  """
  if not image_list:
    raise ValueError('Empty image_list.')

  # Compute the rank assertions.
  rank_assertions = []
  for i in range(len(image_list)):
    image_rank = tf.rank(image_list[i])
    rank_assert = tf.Assert(
        tf.equal(image_rank, 3),
        ['Wrong rank for tensor  %s [expected] [actual]',
         image_list[i].name, 3, image_rank])
    rank_assertions.append(rank_assert)

  with tf.control_dependencies([rank_assertions[0]]):
    image_shape = tf.shape(image_list[0])
  image_height = image_shape[0]
  image_width = image_shape[1]
  crop_size_assert = tf.Assert(
      tf.logical_and(
          tf.greater_equal(image_height, crop_height),
          tf.greater_equal(image_width, crop_width)),
      ['Crop size greater than the image size.'])

  asserts = [rank_assertions[0], crop_size_assert]

  for i in range(1, len(image_list)):
    image = image_list[i]
    asserts.append(rank_assertions[i])
    with tf.control_dependencies([rank_assertions[i]]):
      shape = tf.shape(image)
    height = shape[0]
    width = shape[1]

    height_assert = tf.Assert(
        tf.equal(height, image_height),
        ['Wrong height for tensor %s [expected][actual]',
         image.name, height, image_height])
    width_assert = tf.Assert(
        tf.equal(width, image_width),
        ['Wrong width for tensor %s [expected][actual]',
         image.name, width, image_width])
    asserts.extend([height_assert, width_assert])

  # Create a random bounding box.
  #
  # Use tf.random_uniform and not numpy.random.rand as doing the former would
  # generate random numbers at graph eval time, unlike the latter which
  # generates random numbers at graph definition time.
  with tf.control_dependencies(asserts):
    max_offset_height = tf.reshape(image_height - crop_height + 1, [])
  with tf.control_dependencies(asserts):
    max_offset_width = tf.reshape(image_width - crop_width + 1, [])
  offset_height = tf.random_uniform(
      [], maxval=max_offset_height, dtype=tf.int32)
  offset_width = tf.random_uniform(
      [], maxval=max_offset_width, dtype=tf.int32)

  return [_crop(image, offset_height, offset_width,
                crop_height, crop_width) for image in image_list]


def _central_crop(image_list, crop_height, crop_width):
  """Performs central crops of the given image list.

  Args:
    image_list: a list of image tensors of the same dimension but possibly
      varying channel.
    crop_height: the height of the image following the crop.
    crop_width: the width of the image following the crop.

  Returns:
    the list of cropped images.
  """
  outputs = []
  for image in image_list:
    image_height = tf.shape(image)[0]
    image_width = tf.shape(image)[1]

    offset_height = (image_height - crop_height) / 2
    offset_width = (image_width - crop_width) / 2

    outputs.append(_crop(image, offset_height, offset_width,
                         crop_height, crop_width))
  return outputs


def _mean_image_subtraction(image, means):
  """Subtracts the given means from each image channel.

  For example:
    means = [123.68, 116.779, 103.939]
    image = _mean_image_subtraction(image, means)

  Note that the rank of `image` must be known.

  Args:
    image: a tensor of size [height, width, C].
    means: a C-vector of values to subtract from each channel.

  Returns:
    the centered image.

  Raises:
    ValueError: If the rank of `image` is unknown, if `image` has a rank other
      than three or if the number of channels in `image` doesn't match the
      number of values in `means`.
  """
  if image.get_shape().ndims != 3:
    raise ValueError('Input must be of size [height, width, C>0]')
  num_channels = image.get_shape().as_list()[-1]
  if len(means) != num_channels:
    raise ValueError('len(means) must match the number of channels')

  channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
  for i in range(num_channels):
    channels[i] -= means[i]
  return tf.concat(axis=2, values=channels)

def unwhiten_image(image):
  means=[_R_MEAN, _G_MEAN, _B_MEAN]
  num_channels = image.get_shape().as_list()[-1]
  channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
  for i in range(num_channels):
    channels[i] += means[i]
  return tf.concat(axis=2, values=channels)

def _smallest_size_at_least(height, width, smallest_side):
  """Computes new shape with the smallest side equal to `smallest_side`.

  Computes new shape with the smallest side equal to `smallest_side` while
  preserving the original aspect ratio.

  Args:
    height: an int32 scalar tensor indicating the current height.
    width: an int32 scalar tensor indicating the current width.
    smallest_side: A python integer or scalar `Tensor` indicating the size of
      the smallest side after resize.

  Returns:
    new_height: an int32 scalar tensor indicating the new height.
    new_width: and int32 scalar tensor indicating the new width.
  """
  smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

  height = tf.to_float(height)
  width = tf.to_float(width)
  smallest_side = tf.to_float(smallest_side)

  scale = tf.cond(tf.greater(height, width),
                  lambda: smallest_side / width,
                  lambda: smallest_side / height)
  new_height = tf.to_int32(tf.rint(height * scale))
  new_width = tf.to_int32(tf.rint(width * scale))
  return new_height, new_width


def _aspect_preserving_resize(image, smallest_side):
  """Resize images preserving the original aspect ratio.

  Args:
    image: A 3-D image `Tensor`.
    smallest_side: A python integer or scalar `Tensor` indicating the size of
      the smallest side after resize.

  Returns:
    resized_image: A 3-D tensor containing the resized image.
  """
  smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

  shape = tf.shape(image)
  height = shape[0]
  width = shape[1]
  new_height, new_width = _smallest_size_at_least(height, width, smallest_side)
  image = tf.expand_dims(image, 0)
  resized_image = tf.image.resize_bilinear(image, [new_height, new_width],
                                           align_corners=False)
  resized_image = tf.squeeze(resized_image)
  resized_image.set_shape([None, None, 3])
  return resized_image

def distorted_bounding_box_crop(image,
                                bbox,
                                min_object_covered=1.0,
                                aspect_ratio_range=(0.75, 1.33),
                                area_range=(0.45, 1.0),#(0.05, 1.0),
                                max_attempts=100,
                                scope=None):
  """Generates cropped_image using a one of the bboxes randomly distorted.

  See `tf.image.sample_distorted_bounding_box` for more documentation.

  Args:
    image: 3-D Tensor of image (it will be converted to floats in [0, 1]).
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged
      as [ymin, xmin, ymax, xmax]. If num_boxes is 0 then it would use the whole
      image.
    min_object_covered: An optional `float`. Defaults to `0.1`. The cropped
      area of the image must contain at least this fraction of any bounding box
      supplied.
    aspect_ratio_range: An optional list of `floats`. The cropped area of the
      image must have an aspect ratio = width / height within this range.
    area_range: An optional list of `floats`. The cropped area of the image
      must contain a fraction of the supplied image within in this range.
    max_attempts: An optional `int`. Number of attempts at generating a cropped
      region of the image of the specified constraints. After `max_attempts`
      failures, return the entire image.
    scope: Optional scope for name_scope.
  Returns:
    A tuple, a 3-D Tensor cropped_image and the distorted bbox
  """
  with tf.name_scope(scope, 'distorted_bounding_box_crop', [image, bbox]):
    # Each bounding box has shape [1, num_boxes, box coords] and
    # the coordinates are ordered [ymin, xmin, ymax, xmax].

    # A large fraction of image datasets contain a human-annotated bounding
    # box delineating the region of the image containing the object of interest.
    # We choose to create a new bounding box for the object which is a randomly
    # distorted version of the human-annotated bounding box that obeys an
    # allowed range of aspect ratios, sizes and overlap with the human-annotated
    # bounding box. If no box is supplied, then we assume the bounding box is
    # the entire image.
    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
        tf.shape(image),
        bounding_boxes=bbox,
        min_object_covered=min_object_covered,
        aspect_ratio_range=aspect_ratio_range,
        area_range=area_range,
        max_attempts=max_attempts,
        use_image_if_no_bounding_boxes=True)
    bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box

    # Crop the image to the specified bounding box.
    cropped_image = tf.slice(image, bbox_begin, bbox_size)
    return cropped_image, distort_bbox

if config.DEBUG:
  from scipy.misc import imread, imsave, imshow, imresize
  def save_image_with_heatmap(image, heatmap, indR, indG, indB, shape, heatmap_size):
      if not hasattr(save_image_with_heatmap, "counter"):
          save_image_with_heatmap.counter = 0  # it doesn't exist yet, so initialize it
      save_image_with_heatmap.counter += 1

      img_to_save = np.array(image.tolist())
      #print(img_to_save)

      #mean = [_R_MEAN, _G_MEAN, _B_MEAN]
      #img_to_save += np.array(mean, dtype=img_to_save.dtype)
      #print(img_to_save.shape, heatmap.shape)

      img_to_save = img_to_save.astype(np.uint8)
      file_name = 'raw_{}.jpg'.format(save_image_with_heatmap.counter)
      imsave(os.path.join(config.DEBUG_DIR, file_name), img_to_save)
      #print(heatmap.shape)
      heatmap_all = (np.sum(heatmap, axis=0) * 255.).astype(np.uint8)
      file_name = 'heatmap_{}.jpg'.format(save_image_with_heatmap.counter)
      imsave(os.path.join(config.DEBUG_DIR, file_name), heatmap_all)

      # heatmap0 = (np.sum(heatmap[np.arange(0, np.shape(heatmap)[0], 2), ...], axis=0) * 255.).astype(np.uint8)
      # heatmap1 = (np.sum(heatmap[np.arange(1, np.shape(heatmap)[0], 2), ...], axis=0) * 255.).astype(np.uint8)

      heatmap0 = (np.sum(heatmap[indR, ...], axis=0) * 255.).astype(np.uint8)
      heatmap1 = (np.sum(heatmap[indG, ...], axis=0) * 255.).astype(np.uint8)
      heatmap2 = (np.sum(heatmap[indB, ...], axis=0) * 255.).astype(np.uint8) if len(indB) > 0 else np.zeros((heatmap_size, heatmap_size), dtype=np.float32)

      heatmap0 = imresize(heatmap0, shape, interp='lanczos')
      heatmap1 = imresize(heatmap1, shape, interp='lanczos')
      heatmap2 = imresize(heatmap2, shape, interp='lanczos')

      img_to_save = img_to_save/2.
      img_to_save[:,:,0] = np.clip((img_to_save[:,:,0] + heatmap0 + heatmap2), 0, 255)
      img_to_save[:,:,1] = np.clip((img_to_save[:,:,1] + heatmap1 + heatmap2), 0, 255)
      #img_to_save[:,:,2] = np.clip((img_to_save[:,:,2]/4. + heatmap2), 0, 255)
      file_name = 'with_heatmap_{}.jpg'.format(save_image_with_heatmap.counter)
      imsave(os.path.join(config.DEBUG_DIR, file_name), img_to_save.astype(np.uint8))
      # for num_pt in range(heatmap.shape[0]):
      #   heatmap_to_save = np.array(heatmap[num_pt]) * 255.
      #   heatmap_to_save = heatmap_to_save.astype(np.uint8)
      #   file_name = '{}_{}.jpg'.format(save_image_with_heatmap.counter, num_pt)
      #   imsave(os.path.join(config.DEBUG_DIR, file_name), heatmap_to_save)
      return save_image_with_heatmap.counter
  def _save_image(image):
      if not hasattr(save_image_with_heatmap, "counter"):
          save_image_with_heatmap.counter = 0  # it doesn't exist yet, so initialize it
      save_image_with_heatmap.counter += 1

      img_to_save = np.array(image.tolist())

      img_to_save = img_to_save.astype(np.uint8)
      file_name = 'raw_{}.jpg'.format(save_image_with_heatmap.counter)
      imsave(os.path.join(config.DEBUG_DIR, file_name), img_to_save)

      return save_image_with_heatmap.counter

def np_draw_labelmap(pt, heatmap_sigma, heatmap_size, type='Gaussian'):
    # Draw a 2D gaussian
    # Adopted from https://github.com/anewell/pose-hg-train/blob/master/src/pypose/draw.py
    img = np.zeros((heatmap_size, heatmap_size), dtype=np.float32)
    if pt[0] < 1 or pt[1] < 1:
        return (img, 0)

    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - 3 * heatmap_sigma), int(pt[1] - 3 * heatmap_sigma)]
    br = [int(pt[0] + 3 * heatmap_sigma + 1), int(pt[1] + 3 * heatmap_sigma + 1)]
    if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or
            br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        return (img, 0)

    # Generate gaussian
    size = 6 * heatmap_sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    if type == 'Gaussian':
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * heatmap_sigma ** 2))
    elif type == 'Cauchy':
        g = heatmap_sigma / (((x - x0) ** 2 + (y - y0) ** 2 + heatmap_sigma ** 2) ** 1.5)


    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return (img, 1)

def draw_labelmap(x, y, heatmap_sigma, heatmap_size):
  heatmap, isvalid = tf.map_fn(lambda pt : tf.py_func(np_draw_labelmap, [pt, heatmap_sigma, heatmap_size], [tf.float32, tf.int64], stateful=True),
                    tf.stack([x, y], axis=-1),
                    dtype=[tf.float32, tf.int64], parallel_iterations=10,
                    back_prop=False, swap_memory=False, infer_shape=True)
  heatmap.set_shape([x.get_shape().as_list()[0], heatmap_size, heatmap_size])
  isvalid.set_shape([x.get_shape().as_list()[0]])
  return heatmap, isvalid

# def get_suitable_scale(angles, image_height, image_width, x, y):
#   rotate_matrix = tf.contrib.image.angles_to_projective_transforms(angles, image_height, image_width)

#   flaten_rotate_matrix = tf.squeeze(rotate_matrix)
#   a0, a1, a2, b0, b1, b2 = flaten_rotate_matrix[0], \
#                             flaten_rotate_matrix[1], \
#                             flaten_rotate_matrix[2], \
#                             flaten_rotate_matrix[3], \
#                             flaten_rotate_matrix[4], \
#                             flaten_rotate_matrix[5]

#   normalizor = a1 * b0 - a0 * b1 + 1e-8

#   new_x = -(b1 * x - a1 * y - b1 * a2 + a1 * b2)/normalizor
#   new_y = (b0 * x - a0 * y - a2 * b0 + a0 * b2)/normalizor

#   valid_x = tf.boolean_mask(new_x, x > 0.)
#   valid_y = tf.boolean_mask(new_y, y > 0.)

#   min_x = tf.reduce_min(valid_x, axis=-1)
#   max_x = tf.reduce_max(valid_x, axis=-1)
#   min_y = tf.reduce_min(valid_y, axis=-1)
#   max_y = tf.reduce_max(valid_y, axis=-1)

#   return tf.maximum(max_x - min_x, 0.)

def get_projective_transforms(angles, image_height, image_width, x, y, name=None):
  """Returns projective transform(s) for the given angle(s).
  Args:
    angles: A scalar angle to rotate all images by, or (for batches of images)
        a vector with an angle to rotate each image in the batch. The rank must
        be statically known (the shape is not `TensorShape(None)`.
    image_height: Height of the image(s) to be transformed.
    image_width: Width of the image(s) to be transformed.
  Returns:
    A tensor of shape (num_images, 8). Projective transforms which can be given
      to `tf.contrib.image.transform`.
  """
  with tf.name_scope(name, "get_projective_transforms"):
    angle_or_angles = tf.convert_to_tensor(angles, name="angles", dtype=tf.float32)
    if len(angle_or_angles.get_shape()) == 0:  # pylint: disable=g-explicit-length-test
      angles = angle_or_angles[None]
    elif len(angle_or_angles.get_shape()) == 1:
      angles = angle_or_angles
    else:
      raise TypeError("Angles should have rank 0 or 1.")

    valid_x = tf.boolean_mask(x, x > 0.)
    valid_y = tf.boolean_mask(y, y > 0.)

    min_x = tf.reduce_min(valid_x, axis=-1)
    max_x = tf.reduce_max(valid_x, axis=-1)
    min_y = tf.reduce_min(valid_y, axis=-1)
    max_y = tf.reduce_max(valid_y, axis=-1)
    center_x = (min_x + max_x)/2.
    center_y = (min_y + max_y)/2.

    # map the center of all keypoints to the center of the transformed image
    x_offset = center_x - (tf.cos(angles) * image_width / 2. - tf.sin(angles) * image_height / 2.)
    y_offset = center_y - (tf.sin(angles) * image_width / 2. + tf.cos(angles) * image_height / 2.)

    # x_offset = ((image_width - 1) - (tf.cos(angles) *
    #                                  (image_width - 1) - tf.sin(angles) *
    #                                  (image_height - 1))) / 2.0
    # y_offset = ((image_height - 1) - (tf.sin(angles) *
    #                                   (image_width - 1) + tf.cos(angles) *
    #                                   (image_height - 1))) / 2.0
    num_angles = tf.shape(angles)[0]
    return tf.concat(
        values=[
            tf.cos(angles)[:, None],
            -tf.sin(angles)[:, None],
            x_offset[:, None],
            tf.sin(angles)[:, None],
            tf.cos(angles)[:, None],
            y_offset[:, None],
            tf.zeros((num_angles, 2), tf.float32),
        ],
        axis=1)


# single image only
def rotate_all(images, angles, x, y, interpolation="NEAREST"):
  """Rotate image(s) by the passed angle(s) in radians.
  Args:
    images: A tensor of shape (num_images, num_rows, num_columns, num_channels)
       (NHWC), (num_rows, num_columns, num_channels) (HWC), or
       (num_rows, num_columns) (HW).
    angles: A scalar angle to rotate all images by, or (if images has rank 4)
       a vector of length num_images, with an angle for each image in the batch.
    interpolation: Interpolation mode. Supported values: "NEAREST", "BILINEAR".
  Returns:
    Image(s) with the same type and shape as `images`, rotated by the given
    angle(s). Empty space due to the rotation will be filled with zeros.
  Raises:
    TypeError: If `image` is an invalid type.
  """
  image_or_images = tf.convert_to_tensor(images, name="images")
  if len(image_or_images.get_shape()) == 2:
    images = image_or_images[None, :, :, None]
  elif len(image_or_images.get_shape()) == 3:
    images = image_or_images[None, :, :, :]
  elif len(image_or_images.get_shape()) == 4:
    images = image_or_images
  else:
    raise TypeError("Images should have rank between 2 and 4.")

  image_height = tf.cast(tf.shape(images)[1], tf.float32)[None]
  image_width = tf.cast(tf.shape(images)[2], tf.float32)[None]

  rotate_matrix = get_projective_transforms(angles, image_height, image_width, x, y)

  flaten_rotate_matrix = tf.squeeze(rotate_matrix)
  a0, a1, a2, b0, b1, b2 = flaten_rotate_matrix[0], \
                            flaten_rotate_matrix[1], \
                            flaten_rotate_matrix[2], \
                            flaten_rotate_matrix[3], \
                            flaten_rotate_matrix[4], \
                            flaten_rotate_matrix[5]

  normalizor = a1 * b0 - a0 * b1 + 1e-8

  new_x = -(b1 * x - a1 * y - b1 * a2 + a1 * b2)/normalizor
  new_y = (b0 * x - a0 * y - a2 * b0 + a0 * b2)/normalizor

  #new_x, new_y = new_x/tf.cast(shape[1], tf.float32), new_y/tf.cast(shape[0], tf.float32)
  output = tf.contrib.image.transform(images, rotate_matrix, interpolation=interpolation)
  if len(image_or_images.get_shape()) == 2:
    return output[0, :, :, 0], new_x, new_y
  elif len(image_or_images.get_shape()) == 3:
    return output[0, :, :, :], new_x, new_y
  else:
    return output, new_x, new_y

def rotate_augum(image, shape, fkey_x, fkey_y, bbox_border):
  # only consider valid keypoint
  x_mask = (fkey_x > 0.)
  y_mask = (fkey_y > 0.)
  # backup the input image, keypoint, recover when the transformed image contains no keypoint
  bak_fkey_x, bak_fkey_y, bak_image = fkey_x/tf.cast(shape[1], tf.float32), fkey_y/tf.cast(shape[0], tf.float32), image
  # do rotate for image and all point, and use these new point to crop later
  # transform keypoint and image
  image, fkey_x, fkey_y = tf.cond(tf.random_uniform([1], minval=0., maxval=1., dtype=tf.float32)[0] < 0.4, lambda: rotate_all(image, tf.random_uniform([1], minval=-3.14/6., maxval=3.14/6., dtype=tf.float32)[0], fkey_x, fkey_y), lambda: (image, fkey_x, fkey_y))
  #image = tf.Print(image,[fkey_x, fkey_y])
  # normalize keypoint coord
  fkey_x, fkey_y = fkey_x/tf.cast(shape[1], tf.float32), fkey_y/tf.cast(shape[0], tf.float32)
  # mask all invalid keypoints after rotate, get mask for range 0-1
  x_mask_ = tf.logical_and(fkey_x > 0., fkey_x < 1.)
  y_mask_ = tf.logical_and(fkey_y > 0., fkey_y < 1.)
  # AND to get final valid mask
  x_mask = tf.logical_and(x_mask, x_mask_)
  y_mask = tf.logical_and(y_mask, y_mask_)
  # make these point negtive
  fkey_x = fkey_x * tf.cast(x_mask, tf.float32) + (tf.cast(x_mask, tf.float32) - 1.)
  fkey_y = fkey_y * tf.cast(y_mask, tf.float32) + (tf.cast(y_mask, tf.float32) - 1.)
  # no valid keypoint pair, then rollback
  new_image, new_fkey_x, new_fkey_y = tf.cond(tf.count_nonzero(tf.logical_and(x_mask, y_mask)) > 0, lambda : (image, fkey_x, fkey_y), lambda : (bak_image, bak_fkey_x, bak_fkey_y))

  valid_x = tf.boolean_mask(new_fkey_x, new_fkey_x > 0.)
  valid_y = tf.boolean_mask(new_fkey_y, new_fkey_y > 0.)

  # the region contains all keypoint
  min_x = tf.maximum(tf.reduce_min(valid_x, axis=-1) - bbox_border / tf.cast(shape[0], tf.float32), 0.)
  max_x = tf.minimum(tf.reduce_max(valid_x, axis=-1) + bbox_border / tf.cast(shape[0], tf.float32), 1.)
  min_y = tf.maximum(tf.reduce_min(valid_y, axis=-1) - bbox_border / tf.cast(shape[1], tf.float32), 0.)
  max_y = tf.minimum(tf.reduce_max(valid_y, axis=-1) + bbox_border / tf.cast(shape[1], tf.float32), 1.)

  return new_image, new_fkey_x, new_fkey_y, tf.reshape(tf.stack([min_y, min_x, max_y, max_x], axis=-1), [1, 1, 4])

def preprocess_for_train(image,
                         classid,
                         shape,
                         output_height,
                         output_width,
                         key_x, key_y, key_v, norm_table,
                         data_format,
                         category,
                         bbox_border, heatmap_sigma, heatmap_size,
                         return_keypoints=False,
                         resize_side_min=_RESIZE_SIDE_MIN,
                         resize_side_max=_RESIZE_SIDE_MAX,
                         fast_mode=False,
                         scope=None,
                         add_image_summaries=True):
  """Preprocesses the given image for training.

  Note that the actual resizing scale is sampled from
    [`resize_size_min`, `resize_size_max`].

  Args:
    image: A `Tensor` representing an image of arbitrary size.
    output_height: The height of the image after preprocessing.
    output_width: The width of the image after preprocessing.
    resize_side_min: The lower bound for the smallest side of the image for
      aspect-preserving resizing.
    resize_side_max: The upper bound for the smallest side of the image for
      aspect-preserving resizing.

  Returns:
    A preprocessed image.
  """
  with tf.name_scope(scope, 'vgg_distort_image', [image, output_height, output_width]):
    orig_dtype = image.dtype
    if orig_dtype != tf.float32:
      image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    # Randomly distort the colors. There are 1 or 4 ways to do it.
    num_distort_cases = 1 if fast_mode else 4
    distorted_image = apply_with_random_selector(image,
                                              lambda x, ordering: distort_color(x, ordering, fast_mode),
                                              num_cases=num_distort_cases)
    distorted_image = tf.to_float(tf.image.convert_image_dtype(distorted_image, orig_dtype, saturate=True))
    if add_image_summaries:
      tf.summary.image('color_distorted_image', tf.cast(tf.expand_dims(distorted_image, 0), tf.uint8))

    normarlized_image = _mean_image_subtraction(distorted_image, [_R_MEAN, _G_MEAN, _B_MEAN])

    fkey_x, fkey_y = tf.cast(key_x, tf.float32), tf.cast(key_y, tf.float32)
    #print(fkey_x, fkey_y)
    # rotate transform, with bbox contains the clothes region
    image, fkey_x, fkey_y, bbox = rotate_augum(normarlized_image, shape, fkey_x, fkey_y, bbox_border)

    distorted_image, distorted_bbox = distorted_bounding_box_crop(image, bbox)
    #distorted_image, distorted_bbox = image, tf.reshape(tf.stack([0., 0., 1., 1.], axis=-1), [1, 1, 4])

    distorted_bbox = tf.squeeze(distorted_bbox)
    fkey_x = fkey_x - distorted_bbox[1]# * tf.cast(x_mask, tf.float32)
    fkey_y = fkey_y - distorted_bbox[0]# * tf.cast(y_mask, tf.float32)

    outside_x = (fkey_x >= distorted_bbox[3])
    outside_y = (fkey_y >= distorted_bbox[2])

    fkey_x = fkey_x - tf.cast(outside_x, tf.float32)
    fkey_y = fkey_y - tf.cast(outside_y, tf.float32)

    fkey_x = fkey_x / (distorted_bbox[3] - distorted_bbox[1])
    fkey_y = fkey_y / (distorted_bbox[2] - distorted_bbox[0])

    # Restore the shape since the dynamic slice based upon the bbox_size loses
    # the third dimension.
    distorted_image.set_shape([None, None, 3])

    if add_image_summaries:
      tf.summary.image('cropped_image', tf.expand_dims(distorted_image, 0))

    # We select only 1 case for fast_mode bilinear.
    num_resize_cases = 1 if fast_mode else 4
    distorted_image = apply_with_random_selector(
        distorted_image,
        lambda x, method: tf.image.resize_images(x, [output_height, output_width], method),
        num_cases=num_resize_cases)
    distorted_image.set_shape([output_height, output_width, 3])
    #fkey_x = tf.Print(fkey_x,[fkey_x,fkey_y])
    #print(heatmap_size)
    #fkey_x = tf.Print(fkey_x,[fkey_x])
    #fkey_y = tf.Print(fkey_y,[fkey_y])
    ikey_x = tf.cast(tf.round(fkey_x * heatmap_size), tf.int64)
    ikey_y = tf.cast(tf.round(fkey_y * heatmap_size), tf.int64)

    gather_ind = config.left_right_remap[category]

    if add_image_summaries:
      tf.summary.image('cropped_resized_image', tf.expand_dims(distorted_image, 0))

    # when do flip_left_right we should also swap the left and right keypoint
    distorted_image, new_key_x, new_key_y, new_key_v = tf.cond(tf.random_uniform([1], minval=0., maxval=1., dtype=tf.float32)[0] < 0.5, lambda: (tf.image.flip_left_right(distorted_image), heatmap_size - tf.gather(ikey_x, gather_ind), tf.gather(ikey_y, gather_ind), tf.gather(key_v, gather_ind)), lambda: (distorted_image, ikey_x, ikey_y, key_v))

    # new_key_x = tf.Print(new_key_x,[new_key_x])
    # new_key_y = tf.Print(new_key_y,[new_key_y])
    #new_key_x = tf.Print(new_key_x,[tf.shape(new_key_x)])
    targets, isvalid = draw_labelmap(new_key_x, new_key_y, heatmap_sigma, heatmap_size)
    #norm_gather_ind_ = config.normalize_point_ind_by_id[classid]

    norm_gather_ind = tf.stack([norm_table[0].lookup(classid), norm_table[1].lookup(classid)], axis=-1)

    scale_x_ = tf.cast(output_width, tf.float32)/tf.cast(shape[1], tf.float32)
    scale_y_ = tf.cast(output_height, tf.float32)/tf.cast(shape[0], tf.float32)
    scale_x = tf.cast(output_width, tf.float32)/tf.cast(heatmap_size, tf.float32)
    scale_y = tf.cast(output_height, tf.float32)/tf.cast(heatmap_size, tf.float32)
    # if the two point used for calculate norm factor missing, then we use original point
    norm_x, norm_y = tf.cond(tf.reduce_sum(tf.gather(isvalid, norm_gather_ind)) < 2,
                        lambda: (tf.cast(tf.gather(key_x, norm_gather_ind), tf.float32) * scale_x_,
                                tf.cast(tf.gather(key_y, norm_gather_ind), tf.float32) * scale_y_),
                        lambda:(tf.cast(tf.gather(new_key_x, norm_gather_ind), tf.float32) * scale_x,
                                tf.cast(tf.gather(new_key_y, norm_gather_ind), tf.float32) * scale_y))

    norm_x, norm_y = tf.squeeze(norm_x), tf.squeeze(norm_y)

    norm_value = tf.pow(tf.pow(norm_x[0] - norm_x[1], 2.) + tf.pow(norm_y[0] - norm_y[1], 2.), .5)
    #targets = draw_labelmap(new_key_x, new_key_y) * tf.expand_dims(tf.expand_dims(tf.cast(tf.clip_by_value(new_key_v, 0, 1), tf.float32), -1), -1)

    if config.DEBUG:
      save_image_op = tf.py_func(save_image_with_heatmap,
                                  [unwhiten_image(distorted_image), targets,
                                  config.left_right_group_map[category][0],
                                  config.left_right_group_map[category][1],
                                  config.left_right_group_map[category][2],
                                  [output_height, output_width],
                                  heatmap_size],
                                  tf.int64, stateful=True)
      with tf.control_dependencies([save_image_op]):
        distorted_image = distorted_image/255.
    else:
      distorted_image = distorted_image/255.
    if data_format == 'NCHW':
      distorted_image = tf.transpose(distorted_image, perm=(2, 0, 1))

    if not return_keypoints:
      return distorted_image, targets, new_key_v, isvalid, norm_value
    else:
      return distorted_image, targets, new_key_x, new_key_y, new_key_v, isvalid, norm_value


def preprocess_for_train_v0(image,
                           classid,
                           shape,
                           output_height,
                           output_width,
                           key_x, key_y, key_v, norm_table,
                           data_format,
                           category,
                           bbox_border, heatmap_sigma, heatmap_size,
                           return_keypoints=False,
                           resize_side_min=_RESIZE_SIDE_MIN,
                           resize_side_max=_RESIZE_SIDE_MAX,
                           fast_mode=True,
                           scope=None,
                           add_image_summaries=True):
  """Preprocesses the given image for training.

  Note that the actual resizing scale is sampled from
    [`resize_size_min`, `resize_size_max`].

  Args:
    image: A `Tensor` representing an image of arbitrary size.
    output_height: The height of the image after preprocessing.
    output_width: The width of the image after preprocessing.
    resize_side_min: The lower bound for the smallest side of the image for
      aspect-preserving resizing.
    resize_side_max: The upper bound for the smallest side of the image for
      aspect-preserving resizing.

  Returns:
    A preprocessed image.
  """
  with tf.name_scope(scope, 'vgg_distort_image', [image, output_height, output_width]):
    fkey_x, fkey_y = tf.cast(key_x, tf.float32), tf.cast(key_y, tf.float32)
    #print(fkey_x, fkey_y)
    # rotate transform, with bbox contains the clothes region
    image, fkey_x, fkey_y, bbox = rotate_augum(image, shape, fkey_x, fkey_y, bbox_border)

    distorted_image, distorted_bbox = distorted_bounding_box_crop(image, bbox)
    #distorted_image, distorted_bbox = image, tf.reshape(tf.stack([0., 0., 1., 1.], axis=-1), [1, 1, 4])

    distorted_bbox = tf.squeeze(distorted_bbox)
    fkey_x = fkey_x - distorted_bbox[1]# * tf.cast(x_mask, tf.float32)
    fkey_y = fkey_y - distorted_bbox[0]# * tf.cast(y_mask, tf.float32)

    outside_x = (fkey_x >= distorted_bbox[3])
    outside_y = (fkey_y >= distorted_bbox[2])

    fkey_x = fkey_x - tf.cast(outside_x, tf.float32)
    fkey_y = fkey_y - tf.cast(outside_y, tf.float32)

    fkey_x = fkey_x / (distorted_bbox[3] - distorted_bbox[1])
    fkey_y = fkey_y / (distorted_bbox[2] - distorted_bbox[0])

    # Restore the shape since the dynamic slice based upon the bbox_size loses
    # the third dimension.
    distorted_image.set_shape([None, None, 3])

    if add_image_summaries:
      tf.summary.image('cropped_image', tf.expand_dims(distorted_image, 0))

    # We select only 1 case for fast_mode bilinear.
    num_resize_cases = 1 if fast_mode else 4
    distorted_image = apply_with_random_selector(
        distorted_image,
        lambda x, method: tf.image.resize_images(x, [output_height, output_width], method),
        num_cases=num_resize_cases)
    distorted_image.set_shape([output_height, output_width, 3])
    #fkey_x = tf.Print(fkey_x,[fkey_x,fkey_y])
    #print(heatmap_size)
    #fkey_x = tf.Print(fkey_x,[fkey_x])
    #fkey_y = tf.Print(fkey_y,[fkey_y])
    ikey_x = tf.cast(tf.round(fkey_x * heatmap_size), tf.int64)
    ikey_y = tf.cast(tf.round(fkey_y * heatmap_size), tf.int64)

    gather_ind = config.left_right_remap[category]

    if add_image_summaries:
      tf.summary.image('cropped_resized_image', tf.expand_dims(distorted_image, 0))

    # when do flip_left_right we should also swap the left and right keypoint
    distorted_image, new_key_x, new_key_y, new_key_v = tf.cond(tf.random_uniform([1], minval=0., maxval=1., dtype=tf.float32)[0] < 0.5, lambda: (tf.image.flip_left_right(distorted_image), heatmap_size - tf.gather(ikey_x, gather_ind), tf.gather(ikey_y, gather_ind), tf.gather(key_v, gather_ind)), lambda: (distorted_image, ikey_x, ikey_y, key_v))

    distorted_image = tf.to_float(distorted_image)

    # Randomly distort the colors. There are 1 or 4 ways to do it.
    num_distort_cases = 1 if fast_mode else 4
    distorted_image = apply_with_random_selector(distorted_image,
                                              lambda x, ordering: distort_color_v0(x, ordering, fast_mode),
                                              num_cases=num_distort_cases)

    if add_image_summaries:
      tf.summary.image('final_distorted_image', tf.cast(tf.expand_dims(distorted_image, 0), tf.uint8))

    # new_key_x = tf.Print(new_key_x,[new_key_x])
    # new_key_y = tf.Print(new_key_y,[new_key_y])
    #new_key_x = tf.Print(new_key_x,[tf.shape(new_key_x)])
    targets, isvalid = draw_labelmap(new_key_x, new_key_y, heatmap_sigma, heatmap_size)
    #norm_gather_ind_ = config.normalize_point_ind_by_id[classid]

    norm_gather_ind = tf.stack([norm_table[0].lookup(classid), norm_table[1].lookup(classid)], axis=-1)

    scale_x_ = tf.cast(output_width, tf.float32)/tf.cast(shape[1], tf.float32)
    scale_y_ = tf.cast(output_height, tf.float32)/tf.cast(shape[0], tf.float32)
    scale_x = tf.cast(output_width, tf.float32)/tf.cast(heatmap_size, tf.float32)
    scale_y = tf.cast(output_height, tf.float32)/tf.cast(heatmap_size, tf.float32)
    # if the two point used for calculate norm factor missing, then we use original point
    norm_x, norm_y = tf.cond(tf.reduce_sum(tf.gather(isvalid, norm_gather_ind)) < 2,
                        lambda: (tf.cast(tf.gather(key_x, norm_gather_ind), tf.float32) * scale_x_,
                                tf.cast(tf.gather(key_y, norm_gather_ind), tf.float32) * scale_y_),
                        lambda:(tf.cast(tf.gather(new_key_x, norm_gather_ind), tf.float32) * scale_x,
                                tf.cast(tf.gather(new_key_y, norm_gather_ind), tf.float32) * scale_y))

    norm_x, norm_y = tf.squeeze(norm_x), tf.squeeze(norm_y)

    norm_value = tf.pow(tf.pow(norm_x[0] - norm_x[1], 2.) + tf.pow(norm_y[0] - norm_y[1], 2.), .5)
    #targets = draw_labelmap(new_key_x, new_key_y) * tf.expand_dims(tf.expand_dims(tf.cast(tf.clip_by_value(new_key_v, 0, 1), tf.float32), -1), -1)

    if config.DEBUG:
      save_image_op = tf.py_func(save_image_with_heatmap,
                                  [distorted_image, targets,
                                  config.left_right_group_map[category][0],
                                  config.left_right_group_map[category][1],
                                  config.left_right_group_map[category][2],
                                  [output_height, output_width],
                                  heatmap_size],
                                  tf.int64, stateful=True)
      with tf.control_dependencies([save_image_op]):
        normarlized_image = _mean_image_subtraction(distorted_image, [_R_MEAN, _G_MEAN, _B_MEAN])
    else:
      normarlized_image = _mean_image_subtraction(distorted_image, [_R_MEAN, _G_MEAN, _B_MEAN])
    if data_format == 'NCHW':
      normarlized_image = tf.transpose(normarlized_image, perm=(2, 0, 1))

    return normarlized_image/255., targets, new_key_v, isvalid, norm_value

def preprocess_for_eval(image, classid, shape, output_height, output_width, key_x, key_y, key_v, norm_table, data_format, category, bbox_border, heatmap_sigma, heatmap_size, resize_side, scope=None):
  """Preprocesses the given image for evaluation.

  Args:
    image: A `Tensor` representing an image of arbitrary size.
    output_height: The height of the image after preprocessing.
    output_width: The width of the image after preprocessing.
    resize_side: The smallest side of the image for aspect-preserving resizing.

  Returns:
    A preprocessed image.
  """
  with tf.name_scope(scope, 'vgg_eval_image', [image, output_height, output_width]):
    # Crop the central region of the image with an area containing 87.5% of
    # the original image.
    fkey_x, fkey_y = tf.cast(key_x, tf.float32)/tf.cast(shape[1], tf.float32), tf.cast(key_y, tf.float32)/tf.cast(shape[0], tf.float32)
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, [output_height, output_width], align_corners=False)
    image = tf.squeeze(image, [0])
    image.set_shape([output_height, output_width, 3])
    image = tf.to_float(image)

    ikey_x = tf.cast(tf.round(fkey_x * heatmap_size), tf.int64)
    ikey_y = tf.cast(tf.round(fkey_y * heatmap_size), tf.int64)

    targets, isvalid = draw_labelmap(ikey_x, ikey_y, heatmap_sigma, heatmap_size)

    norm_gather_ind = tf.stack([norm_table[0].lookup(classid), norm_table[1].lookup(classid)], axis=-1)

    key_x = tf.cast(tf.round(fkey_x * output_width), tf.int64)
    key_y = tf.cast(tf.round(fkey_y * output_height), tf.int64)

    norm_x, norm_y = tf.cast(tf.gather(key_x, norm_gather_ind), tf.float32), tf.cast(tf.gather(key_y, norm_gather_ind), tf.float32)
    norm_x, norm_y = tf.squeeze(norm_x), tf.squeeze(norm_y)
    norm_value = tf.pow(tf.pow(norm_x[0] - norm_x[1], 2.) + tf.pow(norm_y[0] - norm_y[1], 2.), .5)

    if config.DEBUG:
      save_image_op = tf.py_func(save_image_with_heatmap,
                                  [image, targets,
                                  config.left_right_group_map[category][0],
                                  config.left_right_group_map[category][1],
                                  config.left_right_group_map[category][2],
                                  [output_height, output_width],
                                  heatmap_size],
                                  tf.int64, stateful=True)
      with tf.control_dependencies([save_image_op]):
        normarlized_image = _mean_image_subtraction(image, [_R_MEAN, _G_MEAN, _B_MEAN])
    else:
      normarlized_image = _mean_image_subtraction(image, [_R_MEAN, _G_MEAN, _B_MEAN])

    if data_format == 'NCHW':
      normarlized_image = tf.transpose(normarlized_image, perm=(2, 0, 1))
    return normarlized_image/255., targets, key_v, isvalid, norm_value


def preprocess_for_test_v0(image, shape, output_height, output_width, data_format='NCHW', bbox_border=25., heatmap_sigma=1., heatmap_size=64, scope=None):
  """Preprocesses the given image for evaluation.

  Args:
    image: A `Tensor` representing an image of arbitrary size.
    output_height: The height of the image after preprocessing.
    output_width: The width of the image after preprocessing.

  Returns:
    A preprocessed image.
  """
  with tf.name_scope(scope, 'vgg_test_image', [image, output_height, output_width]):
    # Crop the central region of the image with an area containing 87.5% of
    # the original image.

    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, [output_height, output_width], align_corners=False)
    image = tf.squeeze(image, [0])
    image.set_shape([output_height, output_width, 3])
    image = tf.to_float(image)

    normarlized_image = _mean_image_subtraction(image, [_R_MEAN, _G_MEAN, _B_MEAN])
    if data_format == 'NCHW':
      normarlized_image = tf.transpose(normarlized_image, perm=(2, 0, 1))
    return normarlized_image/255.

def preprocess_for_test(image, file_name, shape, output_height, output_width, data_format='NCHW', bbox_border=25., heatmap_sigma=1., heatmap_size=64, pred_df=None, scope=None):
  """Preprocesses the given image for evaluation.

  Args:
    image: A `Tensor` representing an image of arbitrary size.
    output_height: The height of the image after preprocessing.
    output_width: The width of the image after preprocessing.

  Returns:
    A preprocessed image.
  """
  with tf.name_scope(scope, 'vgg_test_image', [image, output_height, output_width]):
    # Crop the central region of the image with an area containing 87.5% of
    # the original image.

    if pred_df is not None:
      xmin, ymin, xmax, ymax  = [table_.lookup(file_name) for table_ in pred_df]
      #xmin, ymin, xmax, ymax = [tf.to_float(b) for b in bbox_cord]
      #xmin = tf.Print(xmin, [file_name, xmin, ymin, xmax, ymax], summarize=500)
      height, width, channals = tf.unstack(shape, axis=0)
      xmin, ymin, xmax, ymax = xmin - 100, ymin - 80, xmax + 100, ymax + 80

      xmin, ymin, xmax, ymax = tf.clip_by_value(xmin, 0, width[0]-1), tf.clip_by_value(ymin, 0, height[0]-1), \
                              tf.clip_by_value(xmax, 0, width[0]-1), tf.clip_by_value(ymax, 0, height[0]-1)

      bbox_h = ymax - ymin
      bbox_w = xmax - xmin
      areas = bbox_h * bbox_w

      offsets=tf.stack([xmin, ymin], axis=0)
      crop_shape = tf.stack([bbox_h, bbox_w, channals[0]], axis=0)

      ymin, xmin, bbox_h, bbox_w = tf.cast(ymin, tf.int32), tf.cast(xmin, tf.int32), tf.cast(bbox_h, tf.int32), tf.cast(bbox_w, tf.int32)
      crop_image = tf.image.crop_to_bounding_box(image, ymin, xmin, bbox_h, bbox_w)

      image, shape, offsets = tf.cond(areas > 0, lambda : (crop_image, crop_shape, offsets),
                                      lambda : (image, shape, tf.constant([0, 0], tf.int64)))
      offsets.set_shape([2])
      shape.set_shape([3])
    else:
      offsets = tf.constant([0, 0], tf.int64)

    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, [output_height, output_width], align_corners=False)
    image = tf.squeeze(image, [0])
    image.set_shape([output_height, output_width, 3])

    if config.DEBUG:
      save_image_op = tf.py_func(_save_image,
                                  [image],
                                  tf.int64, stateful=True)
      image = tf.Print(image, [save_image_op])

    image = tf.to_float(image)
    normarlized_image = _mean_image_subtraction(image, [_R_MEAN, _G_MEAN, _B_MEAN])
    if data_format == 'NCHW':
      normarlized_image = tf.transpose(normarlized_image, perm=(2, 0, 1))
    return normarlized_image/255., shape, offsets

def preprocess_for_test_raw_output(image, output_height, output_width, data_format='NCHW', scope=None):
  """Preprocesses the given image for evaluation.

  Args:
    image: A `Tensor` representing an image of arbitrary size.
    output_height: The height of the image after preprocessing.
    output_width: The width of the image after preprocessing.

  Returns:
    A preprocessed image.
  """
  with tf.name_scope(scope, 'vgg_test_image_raw_output', [image, output_height, output_width]):
    # Crop the central region of the image with an area containing 87.5% of
    # the original image.
    image = tf.image.resize_bilinear(image, [output_height, output_width], align_corners=False)
    image = tf.squeeze(image, [0])
    image.set_shape([output_height, output_width, 3])

    if config.DEBUG:
      save_image_op = tf.py_func(_save_image,
                                  [image],
                                  tf.int64, stateful=True)
      image = tf.Print(image, [save_image_op])

    image = tf.to_float(image)
    normarlized_image = _mean_image_subtraction(image, [_R_MEAN, _G_MEAN, _B_MEAN])
    if data_format == 'NCHW':
      normarlized_image = tf.transpose(normarlized_image, perm=(2, 0, 1))
    return tf.expand_dims(normarlized_image/255., 0)

def preprocess_image(image, classid, shape, output_height, output_width,
                    key_x, key_y, key_v, norm_table,
                    is_training=False,
                    data_format='NCHW',
                    category='*',
                    bbox_border=25., heatmap_sigma=1., heatmap_size=64,
                    return_keypoints=False,
                    resize_side_min=_RESIZE_SIDE_MIN,
                    resize_side_max=_RESIZE_SIDE_MAX):
  """Preprocesses the given image.

  Args:
    image: A `Tensor` representing an image of arbitrary size.
    output_height: The height of the image after preprocessing.
    output_width: The width of the image after preprocessing.
    is_training: `True` if we're preprocessing the image for training and
      `False` otherwise.
    resize_side_min: The lower bound for the smallest side of the image for
      aspect-preserving resizing. If `is_training` is `False`, then this value
      is used for rescaling.
    resize_side_max: The upper bound for the smallest side of the image for
      aspect-preserving resizing. If `is_training` is `False`, this value is
      ignored. Otherwise, the resize side is sampled from
        [resize_size_min, resize_size_max].

  Returns:
    A preprocessed image.
  """
  if is_training:
    return preprocess_for_train(image, classid, shape, output_height, output_width, key_x, key_y, key_v, norm_table, data_format,
                              category, bbox_border, heatmap_sigma, heatmap_size, return_keypoints, resize_side_min, resize_side_max)
  else:
    return preprocess_for_eval(image, classid, shape, output_height, output_width, key_x, key_y, key_v, norm_table, data_format,
                              category, bbox_border, heatmap_sigma, heatmap_size, min(output_height, output_width))
