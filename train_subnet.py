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

#from scipy.misc import imread, imsave, imshow, imresize
import tensorflow as tf

from net import hourglass as hg
from utility import train_helper
from utility import mertric

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
    'data_dir', '/media/rs/0E06CD1706CD0127/Kapok/Chi/Datasets/tfrecords',
    'The directory where the dataset input data is stored.')
tf.app.flags.DEFINE_string(
    'dataset_name', '{}_????', 'The pattern of the dataset name to load.')
tf.app.flags.DEFINE_string(
    'model_dir', './logs/',
    'The parent directory where the model will be stored.')
tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 10,
    'The frequency with which logs are print.')
tf.app.flags.DEFINE_integer(
    'save_summary_steps', 100,
    'The frequency with which summaries are saved, in seconds.')
tf.app.flags.DEFINE_integer(
    'save_checkpoints_secs', 3600,
    'The frequency with which the model is saved, in seconds.')
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
tf.app.flags.DEFINE_integer(
    'train_epochs', 5,
    'The number of epochs to use for training.')
tf.app.flags.DEFINE_integer(
    'epochs_per_eval', 1,
    'The number of training epochs to run between evaluations.')
tf.app.flags.DEFINE_integer(
    'batch_size', 6,
    'Batch size for training and evaluation.')
tf.app.flags.DEFINE_string(
    'data_format', 'channels_first', # 'channels_first' or 'channels_last'
    'A flag to override the data format used in the model. channels_first '
    'provides a performance boost on GPU but is not always compatible '
    'with CPU. If left unspecified, the data format will be chosen '
    'automatically based on whether TensorFlow was built for CPU or GPU.')
# optimizer related configuration
tf.app.flags.DEFINE_integer(
    'tf_random_seed', 20180406, 'Random seed for TensorFlow initializers.')
tf.app.flags.DEFINE_float(
    'weight_decay', 0.00000, 'The weight decay on the model weights.')
tf.app.flags.DEFINE_float(
    'mse_weight', 1.0, 'The weight decay on the model weights.')
tf.app.flags.DEFINE_float(
    'momentum', 0.9,
    'The momentum for the MomentumOptimizer and RMSPropOptimizer.')
tf.app.flags.DEFINE_float('learning_rate', 2.5e-4, 'Initial learning rate.')
tf.app.flags.DEFINE_float(
    'end_learning_rate', 0.00001,
    'The minimal end learning rate used by a polynomial decay learning rate.')
# for learning rate piecewise_constant decay
tf.app.flags.DEFINE_string(
    'decay_boundaries', '2, 3',
    'Learning rate decay boundaries by global_step (comma-separated list).')
tf.app.flags.DEFINE_string(
    'lr_decay_factors', '1, 0.5, 0.1',
    'The values of learning_rate decay factor for each segment between boundaries (comma-separated list).')
# checkpoint related configuration
tf.app.flags.DEFINE_string(
    'checkpoint_path', None,
    'The path to a checkpoint from which to fine-tune.')
tf.app.flags.DEFINE_string(
    'checkpoint_model_scope', 'all',
    'Model scope in the checkpoint. None if the same as the trained model.')
tf.app.flags.DEFINE_string(
    #'blouse', 'dress', 'outwear', 'skirt', 'trousers', 'all'
    'model_scope', 'all',
    'Model scope name used to replace the name_scope in checkpoint.')
tf.app.flags.DEFINE_string(
    'checkpoint_exclude_scopes', None,
    'Comma-separated list of scopes of variables to exclude when restoring from a checkpoint.')
tf.app.flags.DEFINE_boolean(
    'ignore_missing_vars', True,
    'When restoring a checkpoint would ignore missing variables.')
tf.app.flags.DEFINE_boolean(
    'run_on_cloud', True,
    'Wether we will train on cloud.')

FLAGS = tf.app.flags.FLAGS

def input_pipeline(is_training=True, num_epochs=FLAGS.epochs_per_eval):
    if 'all' in FLAGS.model_scope:
        lnorm_table = tf.contrib.lookup.HashTable(tf.contrib.lookup.KeyValueTensorInitializer(tf.constant(config.global_norm_key, dtype=tf.int64),
                                                                tf.constant(config.global_norm_lvalues, dtype=tf.int64)), 0)
        rnorm_table = tf.contrib.lookup.HashTable(tf.contrib.lookup.KeyValueTensorInitializer(tf.constant(config.global_norm_key, dtype=tf.int64),
                                                                tf.constant(config.global_norm_rvalues, dtype=tf.int64)), 1)
    else:
        lnorm_table = tf.contrib.lookup.HashTable(tf.contrib.lookup.KeyValueTensorInitializer(tf.constant(config.local_norm_key, dtype=tf.int64),
                                                                tf.constant(config.local_norm_lvalues, dtype=tf.int64)), 0)
        rnorm_table = tf.contrib.lookup.HashTable(tf.contrib.lookup.KeyValueTensorInitializer(tf.constant(config.local_norm_key, dtype=tf.int64),
                                                                tf.constant(config.local_norm_rvalues, dtype=tf.int64)), 1)

    preprocessing_fn = lambda org_image, classid, shape, key_x, key_y, key_v: preprocessing.preprocess_image(org_image, classid, shape, FLAGS.train_image_size, FLAGS.train_image_size, key_x, key_y, key_v, (lnorm_table, rnorm_table), is_training=is_training, data_format=('NCHW' if FLAGS.data_format=='channels_first' else 'NHWC'), category=(FLAGS.model_scope if 'all' not in FLAGS.model_scope else '*'))

    images, shape, classid, targets, key_v, isvalid, norm_value = dataset.slim_get_split(FLAGS.data_dir, preprocessing_fn, FLAGS.batch_size, FLAGS.num_readers, FLAGS.num_preprocessing_threads, num_epochs=num_epochs, is_training=is_training, file_pattern='{}_????', category=(FLAGS.model_scope if 'all' not in FLAGS.model_scope else '*'), reader=None)

    return images, {'targets': targets, 'key_v': key_v, 'shape': shape, 'classid': classid, 'isvalid': isvalid, 'norm_value': norm_value}

def keypoint_model_fn(features, labels, mode, params):
    targets = labels['targets']
    shape = labels['shape']
    classid = labels['classid']
    key_v = labels['key_v']
    isvalid = labels['isvalid']
    norm_value = labels['norm_value']

    with tf.variable_scope(params['model_scope'], default_name=None, values=[features], reuse=tf.AUTO_REUSE):
        pred_outputs = hg.create_model(features, params['num_stacks'], params['feats_channals'],
                            config.class_num_joints[(params['model_scope'] if 'all' not in params['model_scope'] else '*')], params['num_modules'],
                            (mode == tf.estimator.ModeKeys.TRAIN), params['data_format'], name='fashionAI')

    if params['data_format'] == 'channels_last':
        pred_outputs = [tf.transpose(_, [0, 3, 1, 2]) for _ in pred_outputs]

    score_map = pred_outputs[-1]

    ne_mertric = mertric.normalized_error(targets, score_map, norm_value, key_v, isvalid,
                             params['batch_size'],
                             config.class_num_joints[(params['model_scope'] if 'all' not in params['model_scope'] else '*')],
                             params['heatmap_size'],
                             params['train_image_size'])

    last_pred_mse = tf.metrics.mean_squared_error(score_map, targets,
                                weights=1.0 / params['batch_size'],
                                name='last_pred_mse')
    metrics = {'normalized_error': ne_mertric}
    predictions = {'normalized_error': ne_mertric[1]}
    ne_mertric = tf.identity(ne_mertric[1], name='ne_mertric')
    #print(ne_mertric)

    mse_loss_list = []
    for pred in pred_outputs:
        mse_loss_list.append(tf.losses.mean_squared_error(targets, pred,
                            weights=1.0 / params['batch_size'],
                            loss_collection=None,#tf.GraphKeys.LOSSES,
                            reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE))

    mse_loss = tf.multiply(params['mse_weight'], tf.add_n(mse_loss_list), name='mse_loss')
    tf.summary.scalar('mse', mse_loss)
    tf.losses.add_loss(mse_loss)

    # Add weight decay to the loss. We exclude the batch norm variables because
    # doing so leads to a small improvement in accuracy.
    loss = mse_loss + params['weight_decay'] * tf.add_n(
                              [tf.nn.l2_loss(v) for v in tf.trainable_variables()
                               if 'batch_normalization' not in v.name])
    total_loss = tf.identity(loss, name='total_loss')
    tf.summary.scalar('loss', total_loss)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, predictions=predictions, eval_metric_ops=metrics)

    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()

        lr_values = [params['learning_rate'] * decay for decay in params['lr_decay_factors']]
        learning_rate = tf.train.piecewise_constant(tf.cast(global_step, tf.int32),
                                                    [int(float(ep)*params['steps_per_epoch']) for ep in params['decay_boundaries']],
                                                    lr_values)
        truncated_learning_rate = tf.maximum(learning_rate, tf.constant(params['end_learning_rate'], dtype=learning_rate.dtype), name='learning_rate')
        tf.summary.scalar('lr', truncated_learning_rate)

        optimizer = tf.train.MomentumOptimizer(learning_rate=truncated_learning_rate,
                                                momentum=params['momentum'])

        # Batch norm requires update_ops to be added as a train_op dependency.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step)
    else:
        train_op = None

    return tf.estimator.EstimatorSpec(
                          mode=mode,
                          predictions=predictions,
                          loss=loss,
                          train_op=train_op,
                          eval_metric_ops=metrics,
                          scaffold=tf.train.Scaffold(init_fn=train_helper.get_init_fn_for_scaffold(FLAGS)))

def parse_comma_list(args):
    return [float(s.strip()) for s in args.split(',')]

def main(_):
    # Using the Winograd non-fused algorithms provides a small performance boost.
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = FLAGS.gpu_memory_fraction)
    sess_config = tf.ConfigProto(allow_soft_placement = True, log_device_placement = False, intra_op_parallelism_threads = FLAGS.num_cpu_threads, inter_op_parallelism_threads = FLAGS.num_cpu_threads, gpu_options = gpu_options)

    # Set up a RunConfig to only save checkpoints once per training cycle.
    run_config = tf.estimator.RunConfig().replace(
                                        save_checkpoints_secs=FLAGS.save_checkpoints_secs).replace(
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
            'steps_per_epoch': config.split_size[(FLAGS.model_scope if 'all' not in FLAGS.model_scope else '*')]['train'] // FLAGS.batch_size,
            'batch_size': FLAGS.batch_size,
            'weight_decay': FLAGS.weight_decay,
            'mse_weight': FLAGS.mse_weight,
            'momentum': FLAGS.momentum,
            'learning_rate': FLAGS.learning_rate,
            'end_learning_rate': FLAGS.end_learning_rate,
            'decay_boundaries': parse_comma_list(FLAGS.decay_boundaries),
            'lr_decay_factors': parse_comma_list(FLAGS.lr_decay_factors),
        })
    tf.logging.info('params recv: %s', FLAGS.flag_values_dict())
    for _ in range(FLAGS.train_epochs // FLAGS.epochs_per_eval):
        tensors_to_log = {
            'lr': 'learning_rate',
            'loss': 'total_loss',
            'mse': 'mse_loss',
            'ne': 'ne_mertric',
        }

        logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=FLAGS.log_every_n_steps, formatter=lambda dicts: ', '.join(['%s=%.6f' % (k, v) for k, v in dicts.items()]))

        tf.logging.info('Starting a training cycle.')
        fashionAI.train(input_fn=lambda : input_pipeline(True), hooks=[logging_hook])

        tf.logging.info('Starting to evaluate.')
        eval_results = fashionAI.evaluate(input_fn=lambda : input_pipeline(False, 1))
        tf.logging.info(eval_results)

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
