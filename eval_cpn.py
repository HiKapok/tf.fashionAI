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

import os
import sys
import numpy as np
import pandas as pd
#from scipy.misc import imread, imsave, imshow, imresize
import tensorflow as tf

from net import cpn as cpn
from utility import train_helper

from preprocessing import preprocessing
from preprocessing import dataset
import config
#--num_readers=2 --num_preprocessing_threads=2 --data_dir=/media/disk/keypoint/tfrecords --model_to_train=all, blouse
# hardware related configuration
tf.app.flags.DEFINE_integer(
    'num_readers', 16,
    'The number of parallel readers that read data from the dataset.')
tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 48,
    'The number of threads used to create the batches.')
tf.app.flags.DEFINE_integer(
    'num_cpu_threads', 0,
    'The number of cpu cores used to train.')
tf.app.flags.DEFINE_float(
    'gpu_memory_fraction', 1., 'GPU memory fraction to use.')
# scaffold related configuration
tf.app.flags.DEFINE_string(
    'data_dir', '../Datasets/tfrecords_test',
    'The directory where the dataset input data is stored.')
tf.app.flags.DEFINE_string(
    'dataset_name', '{}_*.tfrecord', 'The pattern of the dataset name to load.')
tf.app.flags.DEFINE_string(
    'model_dir', './logs_cpn/',
    'The parent directory where the model will be stored.')
tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 10,
    'The frequency with which logs are print.')
tf.app.flags.DEFINE_integer(
    'save_summary_steps', 100,
    'The frequency with which summaries are saved, in seconds.')
# model related configuration
tf.app.flags.DEFINE_integer(
    'train_image_size', 384,
    'The size of the input image for the model to use.')
tf.app.flags.DEFINE_integer(
    'heatmap_size', 96,
    'The size of the output heatmap of the model.')
tf.app.flags.DEFINE_float(
    'heatmap_sigma', 1.,
    'The sigma of Gaussian which generate the target heatmap.')
tf.app.flags.DEFINE_float(
    'bbox_border', 25.,
    'The nearest distance of the crop border to al keypoints.')
tf.app.flags.DEFINE_string(
    'data_format', 'channels_last', # 'channels_first' or 'channels_last'
    'A flag to override the data format used in the model. channels_first '
    'provides a performance boost on GPU but is not always compatible '
    'with CPU. If left unspecified, the data format will be chosen '
    'automatically based on whether TensorFlow was built for CPU or GPU.')
tf.app.flags.DEFINE_integer(
    'tf_random_seed', 20180417, 'Random seed for TensorFlow initializers.')
# checkpoint related configuration
tf.app.flags.DEFINE_string(
    'checkpoint_path', None,
    'The path to a checkpoint from which to fine-tune.')
tf.app.flags.DEFINE_string(
    'coarse_pred_path', None,
    'The path to a pred csv file from which to crop the input image for finer prediction.')
tf.app.flags.DEFINE_boolean(
    'flip_on_test', False,
    'Wether we will average predictions of left-right fliped image.')
tf.app.flags.DEFINE_string(
    #'blouse', 'dress', 'outwear', 'skirt', 'trousers', 'all'
    'model_scope', 'blouse',
    'Model scope name used to replace the name_scope in checkpoint.')
tf.app.flags.DEFINE_boolean(
    'run_on_cloud', True,
    'Wether we will train on cloud.')
tf.app.flags.DEFINE_string(
    'model_to_eval', 'blouse, dress, outwear, skirt, trousers', #'all, blouse, dress, outwear, skirt, trousers', 'skirt, dress, outwear, trousers',
    'The sub-model to eval (comma-separated list).')

#--model_scope=blouse --checkpoint_path=./logs/blouse
FLAGS = tf.app.flags.FLAGS

def preprocessing_fn(org_image, file_name, shape):
  pd_df = None
  if FLAGS.coarse_pred_path is not None:
    if tf.gfile.Exists(FLAGS.coarse_pred_path):
      tf.logging.info('Finetuning Prediction From {}.'.format(FLAGS.coarse_pred_path))
      tf.gfile.Copy(FLAGS.coarse_pred_path, './__coarse_pred.csv', overwrite=True)
      pd_df = pd.read_csv('./__coarse_pred.csv', encoding='utf-8')

      all_filenames = []
      all_xmin = []
      all_ymin = []
      all_xmax = []
      all_ymax = []

      all_values = pd_df.values.tolist()
      for records in all_values:
        all_filenames.append(records[0].encode('utf8'))
        xmin = 2000
        ymin = 2000
        xmax = -1
        ymax = -1
        for kp in records[2:]:
          keypoint_info = kp.strip().split('_')
          if int(keypoint_info[2]) == -1:
            continue
          xmin = min(xmin, int(keypoint_info[0]))
          ymin = min(ymin, int(keypoint_info[1]))
          xmax = max(xmax, int(keypoint_info[0]))
          ymax = max(ymax, int(keypoint_info[1]))
        all_xmin.append(xmin)
        all_ymin.append(ymin)
        all_xmax.append(xmax)
        all_ymax.append(ymax)
      #print(all_filenames, all_xmin, all_ymin, all_xmax, all_ymax)
      xmin_table = tf.contrib.lookup.HashTable(tf.contrib.lookup.KeyValueTensorInitializer(tf.constant(all_filenames, dtype=tf.string), tf.constant(all_xmin, dtype=tf.int64)), -1)
      ymin_table = tf.contrib.lookup.HashTable(tf.contrib.lookup.KeyValueTensorInitializer(tf.constant(all_filenames, dtype=tf.string), tf.constant(all_ymin, dtype=tf.int64)), -1)
      xmax_table = tf.contrib.lookup.HashTable(tf.contrib.lookup.KeyValueTensorInitializer(tf.constant(all_filenames, dtype=tf.string), tf.constant(all_xmax, dtype=tf.int64)), -1)
      ymax_table = tf.contrib.lookup.HashTable(tf.contrib.lookup.KeyValueTensorInitializer(tf.constant(all_filenames, dtype=tf.string), tf.constant(all_ymax, dtype=tf.int64)), -1)
      pd_df = [xmin_table, ymin_table, xmax_table, ymax_table]
  #pred_item['file_name'].encode('utf8')

  #lnorm_table = tf.contrib.lookup.HashTable(tf.contrib.lookup.KeyValueTensorInitializer(tf.constant(config.global_norm_key, dtype=tf.int64), tf.constant(config.global_norm_lvalues, dtype=tf.int64)), 0)
  return preprocessing.preprocess_for_test(org_image, file_name, shape, FLAGS.train_image_size, FLAGS.train_image_size, data_format=('NCHW' if FLAGS.data_format=='channels_first' else 'NHWC'), bbox_border=FLAGS.bbox_border, heatmap_sigma=FLAGS.heatmap_sigma, heatmap_size=FLAGS.heatmap_size, pred_df=pd_df)
def input_pipeline(model_scope=FLAGS.model_scope):
    #preprocessing_fn = lambda org_image, shape: preprocessing.preprocess_for_test(org_image, shape, FLAGS.train_image_size, FLAGS.train_image_size, data_format=('NCHW' if FLAGS.data_format=='channels_first' else 'NHWC'), bbox_border=FLAGS.bbox_border, heatmap_sigma=FLAGS.heatmap_sigma, heatmap_size=FLAGS.heatmap_size)

    images, shape, file_name, classid, offsets = dataset.slim_test_get_split(FLAGS.data_dir, preprocessing_fn, FLAGS.num_readers, FLAGS.num_preprocessing_threads, file_pattern=FLAGS.dataset_name, category=(model_scope if 'all' not in model_scope else '*'), reader=None)

    return {'images': images, 'shape': shape, 'classid': classid, 'file_name': file_name, 'pred_offsets': offsets}

if config.PRED_DEBUG:
  from scipy.misc import imread, imsave, imshow, imresize
  def save_image_with_heatmap(image, height, width, heatmap_size, heatmap, predictions, indR, indG, indB):
      if not hasattr(save_image_with_heatmap, "counter"):
          save_image_with_heatmap.counter = 0  # it doesn't exist yet, so initialize it
      save_image_with_heatmap.counter += 1

      img_to_save = np.array(image.tolist()) + 120
      #print(img_to_save)

      img_to_save = img_to_save.astype(np.uint8)

      heatmap0 = np.sum(heatmap[indR, ...], axis=0).astype(np.uint8)
      heatmap1 = np.sum(heatmap[indG, ...], axis=0).astype(np.uint8)
      heatmap2 = np.sum(heatmap[indB, ...], axis=0).astype(np.uint8) if len(indB) > 0 else np.zeros((heatmap_size, heatmap_size), dtype=np.float32)

      img_to_save = imresize(img_to_save, (height, width), interp='lanczos')
      heatmap0 = imresize(heatmap0, (height, width), interp='lanczos')
      heatmap1 = imresize(heatmap1, (height, width), interp='lanczos')
      heatmap2 = imresize(heatmap2, (height, width), interp='lanczos')

      img_to_save = img_to_save/2
      img_to_save[:,:,0] = np.clip((img_to_save[:,:,0] + heatmap0 + heatmap2), 0, 255)
      img_to_save[:,:,1] = np.clip((img_to_save[:,:,1] + heatmap1 + heatmap2), 0, 255)
      #img_to_save[:,:,2] = np.clip((img_to_save[:,:,2]/4. + heatmap2), 0, 255)
      file_name = 'with_heatmap_{}.jpg'.format(save_image_with_heatmap.counter)
      imsave(os.path.join(config.EVAL_DEBUG_DIR, file_name), img_to_save.astype(np.uint8))

      predictions = np.array(predictions.tolist())
      #print(predictions.shape)
      for ind in range(predictions.shape[0]):
        img = predictions[ind]
        img = img - img.min()
        img *= 255.0/img.max()
        file_name = 'heatmap_{}_{}.jpg'.format(save_image_with_heatmap.counter, ind)
        imsave(os.path.join(config.EVAL_DEBUG_DIR, file_name), img.astype(np.uint8))
      return save_image_with_heatmap.counter

def gaussian_blur(inputs, inputs_filters, sigma, data_format, name=None):
    with tf.name_scope(name, "gaussian_blur", [inputs]):
        data_format_ = 'NHWC' if data_format=='channels_last' else 'NCHW'
        if data_format_ == 'NHWC':
            inputs = tf.transpose(inputs, [0, 2, 3, 1])
        ksize = int(6 * sigma + 1.)
        x = tf.expand_dims(tf.range(ksize, delta=1, dtype=tf.float32), axis=1)
        y = tf.transpose(x, [1, 0])
        kernel_matrix = tf.exp(- ((x - ksize/2.) ** 2 + (y - ksize/2.) ** 2) / (2 * sigma ** 2))
        #print(kernel_matrix)
        kernel_filter = tf.reshape(kernel_matrix, [ksize, ksize, 1, 1])
        kernel_filter = tf.tile(kernel_filter, [1, 1, inputs_filters, 1])
        #kernel_filter = tf.transpose(kernel_filter, [1, 0, 2, 3])
        outputs = tf.nn.depthwise_conv2d(inputs, kernel_filter, strides=[1, 1, 1, 1], padding='SAME', data_format=data_format_, name='blur')
        if data_format_ == 'NHWC':
            outputs = tf.transpose(outputs, [0, 3, 1, 2])
        return outputs

def get_keypoint(image, predictions, heatmap_size, height, width, category, clip_at_zero=True, data_format='channels_last', name=None):
    # expand_border = 10

    # pad_pred = tf.pad(predictions, tf.constant([[0, 0], [0, 0], [expand_border, expand_border], [expand_border, expand_border]]),
    #               mode='CONSTANT', name='pred_padding', constant_values=0)

    # blur_pred = gaussian_blur(pad_pred, config.class_num_joints[category], 3.5, 'channels_first', 'pred_blur')

    # predictions = tf.slice(blur_pred, [0, 0, expand_border, expand_border], [1, config.class_num_joints[category], heatmap_size, heatmap_size])
    predictions = tf.reshape(predictions, [1, -1, heatmap_size*heatmap_size])

    pred_max = tf.reduce_max(predictions, axis=-1)
    pred_max_indices = tf.argmax(predictions, axis=-1)
    pred_max_x, pred_max_y = tf.cast(tf.floormod(pred_max_indices, heatmap_size), tf.float32), tf.cast(tf.floordiv(pred_max_indices, heatmap_size), tf.float32)
    # mask the max elements to zero
    mask_predictions = predictions * tf.one_hot(pred_max_indices, heatmap_size*heatmap_size, on_value=0., off_value=1., dtype=tf.float32)
    # get the second max prediction
    pred_next_max = tf.reduce_max(mask_predictions, axis=-1)
    pred_next_max_indices = tf.argmax(mask_predictions, axis=-1)
    pred_next_max_x, pred_next_max_y = tf.cast(tf.floormod(pred_next_max_indices, heatmap_size), tf.float32), tf.cast(tf.floordiv(pred_next_max_indices, heatmap_size), tf.float32)

    dist = tf.pow(tf.pow(pred_next_max_x - pred_max_x, 2.) + tf.pow(pred_next_max_y - pred_max_y, 2.), .5)

    pred_x = tf.where(dist < 1e-3, pred_max_x, pred_max_x + (pred_next_max_x - pred_max_x) * 0.25 / dist)
    pred_y = tf.where(dist < 1e-3, pred_max_y, pred_max_y + (pred_next_max_y - pred_max_y) * 0.25 / dist)

    pred_indices_ = tf.squeeze(tf.cast(pred_x, tf.int64) + tf.cast(pred_y, tf.int64) * heatmap_size)

    width, height = tf.cast(width, tf.float32), tf.cast(height, tf.float32)
    width_ratio, height_ratio = width / tf.cast(heatmap_size, tf.float32), height / tf.cast(heatmap_size, tf.float32)

    pred_x, pred_y = pred_x * width_ratio, pred_y * height_ratio
    #pred_x, pred_y = pred_x * width_ratio + width_ratio/2., pred_y * height_ratio + height_ratio/2.

    if clip_at_zero:
      pred_x, pred_y =  pred_x * tf.cast(pred_max>0, tf.float32), pred_y * tf.cast(pred_max>0, tf.float32)
      pred_x = pred_x * tf.cast(pred_max>0, tf.float32) + tf.cast(pred_max<=0, tf.float32) * (width / 2.)
      pred_y = pred_y * tf.cast(pred_max>0, tf.float32) + tf.cast(pred_max<=0, tf.float32) * (height / 2.)

    if config.PRED_DEBUG:
      image_ = tf.squeeze(image) * 255.
      pred_heatmap = tf.one_hot(pred_indices_, heatmap_size*heatmap_size, on_value=255, off_value=0, axis=-1, dtype=tf.int32)

      pred_heatmap = tf.reshape(pred_heatmap, [-1, heatmap_size, heatmap_size])
      if data_format == 'channels_first':
        image_ = tf.transpose(image_, perm=(1, 2, 0))
      save_image_op = tf.py_func(save_image_with_heatmap,
                                  [image_, height, width,
                                  heatmap_size,
                                  pred_heatmap,
                                  tf.reshape(predictions, [-1, heatmap_size, heatmap_size]),
                                  config.left_right_group_map[category][0],
                                  config.left_right_group_map[category][1],
                                  config.left_right_group_map[category][2]],
                                  tf.int64, stateful=True)
      with tf.control_dependencies([save_image_op]):
        pred_x, pred_y = pred_x * 1., pred_y * 1.
    return pred_x, pred_y

def get_keypoint_v0(image, predictions, heatmap_size, height, width, category, clip_at_zero=True, data_format='channels_last', name=None):
    predictions = tf.reshape(predictions, [1, -1, heatmap_size*heatmap_size])

    pred_max = tf.reduce_max(predictions, axis=-1)
    pred_indices = tf.argmax(predictions, axis=-1)
    pred_x, pred_y = tf.cast(tf.floormod(pred_indices, heatmap_size), tf.float32), tf.cast(tf.floordiv(pred_indices, heatmap_size), tf.float32)

    width, height = tf.cast(width, tf.float32), tf.cast(height, tf.float32)
    pred_x, pred_y = pred_x * width / tf.cast(heatmap_size, tf.float32), pred_y * height / tf.cast(heatmap_size, tf.float32)

    if clip_at_zero:
      pred_x, pred_y =  pred_x * tf.cast(pred_max>0, tf.float32), pred_y * tf.cast(pred_max>0, tf.float32)
      pred_x = pred_x * tf.cast(pred_max>0, tf.float32) + tf.cast(pred_max<=0, tf.float32) * (width / 2.)
      pred_y = pred_y * tf.cast(pred_max>0, tf.float32) + tf.cast(pred_max<=0, tf.float32) * (height / 2.)

    if config.PRED_DEBUG:
      pred_indices_ = tf.squeeze(pred_indices)
      image_ = tf.squeeze(image) * 255.
      pred_heatmap = tf.one_hot(pred_indices_, heatmap_size*heatmap_size, on_value=255, off_value=0, axis=-1, dtype=tf.int32)

      pred_heatmap = tf.reshape(pred_heatmap, [-1, heatmap_size, heatmap_size])
      if data_format == 'channels_first':
        image_ = tf.transpose(image_, perm=(1, 2, 0))
      save_image_op = tf.py_func(save_image_with_heatmap,
                                  [image_, height, width,
                                  heatmap_size,
                                  pred_heatmap,
                                  tf.reshape(predictions, [-1, heatmap_size, heatmap_size]),
                                  config.left_right_group_map[category][0],
                                  config.left_right_group_map[category][1],
                                  config.left_right_group_map[category][2]],
                                  tf.int64, stateful=True)
      with tf.control_dependencies([save_image_op]):
        pred_x, pred_y = pred_x * 1., pred_y * 1.
    return pred_x, pred_y

def keypoint_model_fn(features, labels, mode, params):
    #print(features)
    shape = features['shape']
    classid = features['classid']
    pred_offsets = tf.to_float(features['pred_offsets'])
    file_name = features['file_name']
    features = features['images']

    file_name = tf.identity(file_name, name='current_file')

    if not params['flip_on_test']:
        with tf.variable_scope(params['model_scope'], default_name=None, values=[features], reuse=tf.AUTO_REUSE):
            pred_outputs = cpn.cascaded_pyramid_net(features, config.class_num_joints[(params['model_scope'] if 'all' not in params['model_scope'] else '*')], params['heatmap_size'], (mode == tf.estimator.ModeKeys.TRAIN), params['data_format'])
        if params['data_format'] == 'channels_last':
            pred_outputs = [tf.transpose(pred_outputs[ind], [0, 3, 1, 2], name='outputs_trans_{}'.format(ind)) for ind in list(range(len(pred_outputs)))]
    else:
        # test augumentation on the fly
        if params['data_format'] == 'channels_last':
            double_features = tf.reshape(tf.stack([features, tf.map_fn(tf.image.flip_left_right, features, back_prop=False)], axis = 1), [-1, params['train_image_size'], params['train_image_size'], 3])
        else:
            double_features = tf.reshape(tf.stack([features, tf.transpose(tf.map_fn(tf.image.flip_left_right, tf.transpose(features, [0, 2, 3, 1], name='nchw2nhwc'), back_prop=False), [0, 3, 1, 2], name='nhwc2nchw')], axis = 1), [-1, 3, params['train_image_size'], params['train_image_size']])

        num_joints = config.class_num_joints[(params['model_scope'] if 'all' not in params['model_scope'] else '*')]
        with tf.variable_scope(params['model_scope'], default_name=None, values=[double_features], reuse=tf.AUTO_REUSE):
            pred_outputs = cpn.cascaded_pyramid_net(double_features, config.class_num_joints[(params['model_scope'] if 'all' not in params['model_scope'] else '*')], params['heatmap_size'], (mode == tf.estimator.ModeKeys.TRAIN), params['data_format'])

        if params['data_format'] == 'channels_last':
            pred_outputs = [tf.transpose(pred_outputs[ind], [0, 3, 1, 2], name='outputs_trans_{}'.format(ind)) for ind in list(range(len(pred_outputs)))]
        # [[0, 0, 0, ..], [1, 1, 1, ...], ...]
        row_indices = tf.tile(tf.reshape(tf.range(tf.shape(double_features)[0]), [-1, 1]), [1, num_joints])
        # [[0, 1, 2, ...], [1, 0, 2, ...], [0, 1, 2], [1, 0, 2], ...]
        col_indices = tf.reshape(tf.tile(tf.reshape(tf.stack([tf.range(num_joints), tf.constant(config.left_right_remap[(params['model_scope'] if 'all' not in params['model_scope'] else '*')])], axis=0), [-1]), [tf.shape(features)[0]]), [-1, num_joints])
        # [[[0, 0], [0, 1], [0, 2], ...], [[1, 1], [1, 0], [1, 2], ...], [[2, 0], [2, 1], [2, 2], ...], ...]
        flip_indices=tf.stack([row_indices, col_indices], axis=-1)

        #flip_indices = tf.Print(flip_indices, [flip_indices], summarize=500)
        pred_outputs = [tf.gather_nd(pred_outputs[ind], flip_indices, name='gather_nd_{}'.format(ind)) for ind in list(range(len(pred_outputs)))]

        def cond_flip(heatmap_ind):
            return tf.cond(heatmap_ind[1] < 1, lambda : heatmap_ind[0], lambda : tf.transpose(tf.image.flip_left_right(tf.transpose(heatmap_ind[0], [1, 2, 0], name='pred_nchw2nhwc')), [2, 0, 1], name='pred_nhwc2nchw'))
        # all the heatmap of the fliped image should also be fliped back
        pred_outputs = [tf.map_fn(cond_flip, [pred_outputs[ind], tf.tile(tf.reshape(tf.range(2), [-1]), [tf.shape(features)[0]])], dtype=tf.float32, parallel_iterations=10, back_prop=True, swap_memory=False, infer_shape=True, name='map_fn_{}'.format(ind)) for ind in list(range(len(pred_outputs)))]
        # average predictions of left_reight_fliped image
        segment_indices = tf.reshape(tf.tile(tf.reshape(tf.range(tf.shape(features)[0]), [-1, 1]), [1, 2]), [-1])
        pred_outputs = [tf.segment_mean(pred_outputs[ind], segment_indices, name='segment_mean_{}'.format(ind)) for ind in list(range(len(pred_outputs)))]

    pred_x, pred_y = get_keypoint(features, pred_outputs[-1], params['heatmap_size'], shape[0][0], shape[0][1], (params['model_scope'] if 'all' not in params['model_scope'] else '*'), clip_at_zero=True, data_format=params['data_format'])

    predictions = {'pred_x': pred_x + pred_offsets[:, 0], 'pred_y': pred_y + pred_offsets[:, 1], 'file_name': file_name}

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
                              mode=mode,
                              predictions=predictions,
                              loss=None, train_op=None)
    else:
        raise ValueError('Only "PREDICT" mode is supported.')

def parse_comma_list(args):
    return [float(s.strip()) for s in args.split(',')]

def eval_each(model_fn, model_dir, model_scope, run_config):
    fashionAI = tf.estimator.Estimator(
        model_fn=model_fn, model_dir=model_dir, config=run_config,
        params={
            'train_image_size': FLAGS.train_image_size,
            'heatmap_size': FLAGS.heatmap_size,
            'data_format': FLAGS.data_format,
            'model_scope': model_scope,
            'flip_on_test': FLAGS.flip_on_test,
        })
    #tf.logging.info('params recv: %s', FLAGS.flag_values_dict())

    tensors_to_log = {
        'cur_file': 'current_file'
    }

    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=FLAGS.log_every_n_steps, formatter=lambda dicts: ', '.join(['%s=%s' % (k, v) for k, v in dicts.items()]))
    tf.logging.info('Starting to predict model {}.'.format(model_scope))
    pred_results = fashionAI.predict(input_fn=lambda : input_pipeline(model_scope), hooks=[logging_hook], checkpoint_path=train_helper.get_latest_checkpoint_for_evaluate_(model_dir, model_dir))
    #tf.logging.info()
    return list(pred_results)

def main(_):
    # Using the Winograd non-fused algorithms provides a small performance boost.
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = FLAGS.gpu_memory_fraction)
    sess_config = tf.ConfigProto(allow_soft_placement = True, log_device_placement = False, intra_op_parallelism_threads = FLAGS.num_cpu_threads, inter_op_parallelism_threads = FLAGS.num_cpu_threads, gpu_options = gpu_options)

    # Set up a RunConfig to only save checkpoints once per training cycle.
    run_config = tf.estimator.RunConfig().replace(
                                        save_checkpoints_secs=None).replace(
                                        save_checkpoints_steps=None).replace(
                                        save_summary_steps=FLAGS.save_summary_steps).replace(
                                        keep_checkpoint_max=5).replace(
                                        tf_random_seed=FLAGS.tf_random_seed).replace(
                                        log_step_count_steps=FLAGS.log_every_n_steps).replace(
                                        session_config=sess_config)

    model_to_eval = [s.strip() for s in FLAGS.model_to_eval.split(',')]
    for m in model_to_eval:
        if m == '': continue
        pred_results = eval_each(keypoint_model_fn, os.path.join(FLAGS.model_dir, m), m, run_config)
        #print(pred_results)
        # collect result
        df = pd.DataFrame(columns=['image_id', 'image_category'] + config.all_keys)
        cur_record = 0
        gloabl2local_ind = dict(zip(config.class2global_ind_map[m], list(range(len(config.class2global_ind_map[m]))) ))
        #print(gloabl2local_ind)
        for pred_item in pred_results:
            temp_list = []
            index = 0
            x = pred_item['pred_x'].tolist()
            y = pred_item['pred_y'].tolist()
            filename = pred_item['file_name'].decode('utf8')
            for ind in list(range(config.class_num_joints['*'])):
                if ind in gloabl2local_ind:
                    temp_list.append('{}_{}_1'.format(round(x[gloabl2local_ind[ind]]), round(y[gloabl2local_ind[ind]])))
                else:
                    temp_list.append('-1_-1_-1')
            #Images/blouse/ab669925e96490ec698af976586f0b2f.jpg
            df.loc[cur_record] = [filename, m] + temp_list
            cur_record = cur_record + 1
        df.to_csv('./{}.csv'.format(m), encoding='utf-8', index=False)

    # merge dataframe
    df_list = [pd.read_csv('./{}.csv'.format(model_to_eval[0]), encoding='utf-8')]
    for m in model_to_eval[1:]:
        if m == '': continue
        df_list.append(pd.read_csv('./{}.csv'.format(m), encoding='utf-8'))
    pd.concat(df_list, ignore_index=True).to_csv('./sub.csv', encoding='utf-8', index=False)

    if FLAGS.run_on_cloud:
        tf.gfile.Copy('./sub.csv', os.path.join(FLAGS.model_dir, 'sub.csv'), overwrite=True)

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
