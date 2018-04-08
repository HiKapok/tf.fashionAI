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
#from scipy.misc import imread, imsave, imshow, imresize
import tensorflow as tf

from net import hourglass as hg
from utility import train_helper

from preprocessing import preprocessing
from preprocessing import dataset
import config

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
    'data_dir', '/media/rs/0E06CD1706CD0127/Kapok/Chi/Datasets/tfrecords_test',
    'The directory where the dataset input data is stored.')
tf.app.flags.DEFINE_string(
    'dataset_name', '{}_*.tfrecord', 'The pattern of the dataset name to load.')
tf.app.flags.DEFINE_string(
    'model_dir', './eval_logs/',
    'The parent directory where the model will be stored.')
tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 10,
    'The frequency with which logs are print.')
tf.app.flags.DEFINE_integer(
    'save_summary_steps', 100,
    'The frequency with which summaries are saved, in seconds.')
# model related configuration
tf.app.flags.DEFINE_integer(
    'train_image_size', 256,
    'The size of the input image for the model to use.')
tf.app.flags.DEFINE_integer(
    'heatmap_size', 64,
    'The size of the output heatmap of the model.')
tf.app.flags.DEFINE_float(
    'heatmap_sigma', 1.,
    'The sigma of Gaussian which generate the target heatmap.')
tf.app.flags.DEFINE_integer('feats_channals', 256, 'Number of features in the hourglass.')
tf.app.flags.DEFINE_integer('num_stacks', 8, 'Number of hourglasses to stack.')#8
tf.app.flags.DEFINE_integer('num_modules', 1, 'Number of residual modules at each location in the hourglass.')
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
    'tf_random_seed', 20180406, 'Random seed for TensorFlow initializers.')
# checkpoint related configuration
tf.app.flags.DEFINE_string(
    'checkpoint_path', None,
    'The path to a checkpoint from which to fine-tune.')
tf.app.flags.DEFINE_string(
    #'blouse', 'dress', 'outwear', 'skirt', 'trousers', 'all'
    'model_scope', 'all',
    'Model scope name used to replace the name_scope in checkpoint.')
tf.app.flags.DEFINE_boolean(
    'run_on_cloud', True,
    'Wether we will train on cloud.')
#--model_scope=blouse --checkpoint_path=./logs/blouse
FLAGS = tf.app.flags.FLAGS

def input_pipeline():
    preprocessing_fn = lambda org_image, shape: preprocessing.preprocess_for_test(org_image, shape, FLAGS.train_image_size, FLAGS.train_image_size, data_format=('NCHW' if FLAGS.data_format=='channels_first' else 'NHWC'), bbox_border=FLAGS.bbox_border, heatmap_sigma=FLAGS.heatmap_sigma, heatmap_size=FLAGS.heatmap_size)

    images, shape, file_name, classid = dataset.slim_test_get_split(FLAGS.data_dir, preprocessing_fn, FLAGS.num_readers, FLAGS.num_preprocessing_threads, file_pattern=FLAGS.dataset_name, category=(FLAGS.model_scope if 'all' not in FLAGS.model_scope else '*'), reader=None)

    return {'images': images, 'shape': shape, 'classid': classid, 'file_name': file_name}

if config.PRED_DEBUG:
  from scipy.misc import imread, imsave, imshow, imresize
  def save_image_with_heatmap(image, height, width, heatmap_size, heatmap, predictions, indR, indG, indB):
      if not hasattr(save_image_with_heatmap, "counter"):
          save_image_with_heatmap.counter = 0  # it doesn't exist yet, so initialize it
      save_image_with_heatmap.counter += 1

      img_to_save = np.array(image.tolist()) + 128
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

def get_keypoint(image, predictions, heatmap_size, height, width, category, clip_at_zero=True, data_format='channels_last', name=None):
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
    file_name = features['file_name']
    features = features['images']

    file_name = tf.identity(file_name, name='current_file')

    with tf.variable_scope(params['model_scope'], default_name=None, values=[features], reuse=tf.AUTO_REUSE):
        pred_outputs = hg.create_model(features, params['num_stacks'], params['feats_channals'],
                            config.class_num_joints[(params['model_scope'] if 'all' not in params['model_scope'] else '*')], params['num_modules'],
                            (mode == tf.estimator.ModeKeys.TRAIN), params['data_format'], name='fashionAI')

    if params['data_format'] == 'channels_last':
        pred_outputs = [tf.transpose(_, [0, 3, 1, 2]) for _ in pred_outputs]

    pred_x, pred_y = get_keypoint(features, pred_outputs[-1], params['heatmap_size'], shape[0][0], shape[0][1], (params['model_scope'] if 'all' not in params['model_scope'] else '*'), clip_at_zero=True, data_format=params['data_format'])

    predictions = {'pred_x': pred_x, 'pred_y': pred_y, 'file_name': file_name}

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
                              mode=mode,
                              predictions=predictions,
                              loss=None, train_op=None)
    else:
        raise ValueError('Only "PREDICT" mode is supported.')

def parse_comma_list(args):
    return [float(s.strip()) for s in args.split(',')]

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

    fashionAI = tf.estimator.Estimator(
        model_fn=keypoint_model_fn, model_dir=FLAGS.model_dir, config=run_config,
        params={
            'train_image_size': FLAGS.train_image_size,
            'heatmap_size': FLAGS.heatmap_size,
            'feats_channals': FLAGS.feats_channals,
            'num_stacks': FLAGS.num_stacks,
            'num_modules': FLAGS.num_modules,
            'data_format': FLAGS.data_format,
            'model_scope': FLAGS.model_scope,
        })
    tf.logging.info('params recv: %s', FLAGS.flag_values_dict())

    tensors_to_log = {
        'cur_file': 'current_file'
    }

    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=FLAGS.log_every_n_steps, formatter=lambda dicts: ', '.join(['%s=%s' % (k, v) for k, v in dicts.items()]))
    tf.logging.info('Starting to predict.')
    pred_results = fashionAI.predict(input_fn=input_pipeline, hooks=[logging_hook], checkpoint_path=train_helper.get_latest_checkpoint_for_evaluate(FLAGS))
    tf.logging.info(list(pred_results))

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
