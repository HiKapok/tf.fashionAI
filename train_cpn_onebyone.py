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

from net import cpn as cpn
from utility import train_helper
from utility import mertric

from preprocessing import preprocessing
from preprocessing import dataset
import config

# hardware related configuration
tf.app.flags.DEFINE_integer(
    'num_readers', 16,#16
    'The number of parallel readers that read data from the dataset.')
tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 48,#48
    'The number of threads used to create the batches.')
tf.app.flags.DEFINE_integer(
    'num_cpu_threads', 0,
    'The number of cpu cores used to train.')
tf.app.flags.DEFINE_float(
    'gpu_memory_fraction', 1., 'GPU memory fraction to use.')
# scaffold related configuration
tf.app.flags.DEFINE_string(
    'data_dir', '../Datasets/tfrecords',#'/media/rs/0E06CD1706CD0127/Kapok/Chi/Datasets/tfrecords',
    'The directory where the dataset input data is stored.')
tf.app.flags.DEFINE_string(
    'dataset_name', '{}_????', 'The pattern of the dataset name to load.')
tf.app.flags.DEFINE_string(
    'model_dir', './logs_cpn/',
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
tf.app.flags.DEFINE_integer(
    'train_epochs', 50,
    'The number of epochs to use for training.')
tf.app.flags.DEFINE_integer(
    'epochs_per_eval', 20,
    'The number of training epochs to run between evaluations.')
tf.app.flags.DEFINE_integer(
    'batch_size', 10,
    'Batch size for training and evaluation.')
tf.app.flags.DEFINE_boolean(
    'use_ohkm', True,
    'Wether we will use the ohkm for hard keypoints.')
tf.app.flags.DEFINE_string(
    'data_format', 'channels_first', # 'channels_first' or 'channels_last'
    'A flag to override the data format used in the model. channels_first '
    'provides a performance boost on GPU but is not always compatible '
    'with CPU. If left unspecified, the data format will be chosen '
    'automatically based on whether TensorFlow was built for CPU or GPU.')
# optimizer related configuration
tf.app.flags.DEFINE_integer(
    'tf_random_seed', 20180417, 'Random seed for TensorFlow initializers.')
tf.app.flags.DEFINE_float(
    'weight_decay', 1e-5, 'The weight decay on the model weights.')
tf.app.flags.DEFINE_float(
    'mse_weight', 1., 'The weight decay on the model weights.')
tf.app.flags.DEFINE_float(
    'momentum', 0.9,
    'The momentum for the MomentumOptimizer and RMSPropOptimizer.')
tf.app.flags.DEFINE_float('learning_rate', 1e-4, 'Initial learning rate.')#1e-3
tf.app.flags.DEFINE_float(
    'end_learning_rate', 0.000001,
    'The minimal end learning rate used by a polynomial decay learning rate.')
tf.app.flags.DEFINE_float(
    'warmup_learning_rate', 0.00001,
    'The start warm-up learning rate to avoid NAN.')
tf.app.flags.DEFINE_integer(
    'warmup_steps', 100,
    'The total steps to warm-up.')
# for learning rate piecewise_constant decay
tf.app.flags.DEFINE_string(
    'decay_boundaries', '2, 3',
    'Learning rate decay boundaries by global_step (comma-separated list).')
tf.app.flags.DEFINE_string(
    'lr_decay_factors', '1, 0.5, 0.1',
    'The values of learning_rate decay factor for each segment between boundaries (comma-separated list).')
# checkpoint related configuration
tf.app.flags.DEFINE_string(
    'checkpoint_path', './model/resnet50',
    'The path to a checkpoint from which to fine-tune.')
tf.app.flags.DEFINE_string(
    'checkpoint_model_scope', '',
    'Model scope in the checkpoint. None if the same as the trained model.')
tf.app.flags.DEFINE_string(
    #'blouse', 'dress', 'outwear', 'skirt', 'trousers', 'all'
    'model_scope', None,
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
tf.app.flags.DEFINE_string(
    'cloud_checkpoint_path', 'resnet50',
    'The path to a checkpoint from which to fine-tune.')
tf.app.flags.DEFINE_boolean(
    'seq_train', False,
    'Wether we will train a sequence model.')
tf.app.flags.DEFINE_string(
    'model_to_train', 'blouse, dress, outwear, skirt, trousers', #'all, blouse, dress, outwear, skirt, trousers', 'skirt, dress, outwear, trousers',
    'The sub-model to train (comma-separated list).')

FLAGS = tf.app.flags.FLAGS
#--model_scope=blouse --checkpoint_path=./logs/all --data_format=channels_last --batch_size=1
def input_pipeline(is_training=True, model_scope=FLAGS.model_scope, num_epochs=FLAGS.epochs_per_eval):
    if 'all' in model_scope:
        lnorm_table = tf.contrib.lookup.HashTable(tf.contrib.lookup.KeyValueTensorInitializer(tf.constant(config.global_norm_key, dtype=tf.int64),
                                                                tf.constant(config.global_norm_lvalues, dtype=tf.int64)), 0)
        rnorm_table = tf.contrib.lookup.HashTable(tf.contrib.lookup.KeyValueTensorInitializer(tf.constant(config.global_norm_key, dtype=tf.int64),
                                                                tf.constant(config.global_norm_rvalues, dtype=tf.int64)), 1)
    else:
        lnorm_table = tf.contrib.lookup.HashTable(tf.contrib.lookup.KeyValueTensorInitializer(tf.constant(config.local_norm_key, dtype=tf.int64),
                                                                tf.constant(config.local_norm_lvalues, dtype=tf.int64)), 0)
        rnorm_table = tf.contrib.lookup.HashTable(tf.contrib.lookup.KeyValueTensorInitializer(tf.constant(config.local_norm_key, dtype=tf.int64),
                                                                tf.constant(config.local_norm_rvalues, dtype=tf.int64)), 1)

    preprocessing_fn = lambda org_image, classid, shape, key_x, key_y, key_v: preprocessing.preprocess_image(org_image, classid, shape, FLAGS.train_image_size, FLAGS.train_image_size, key_x, key_y, key_v, (lnorm_table, rnorm_table), is_training=is_training, data_format=('NCHW' if FLAGS.data_format=='channels_first' else 'NHWC'), category=(model_scope if 'all' not in model_scope else '*'), bbox_border=FLAGS.bbox_border, heatmap_sigma=FLAGS.heatmap_sigma, heatmap_size=FLAGS.heatmap_size)

    images, shape, classid, targets, key_v, isvalid, norm_value = dataset.slim_get_split(FLAGS.data_dir, preprocessing_fn, FLAGS.batch_size, FLAGS.num_readers, FLAGS.num_preprocessing_threads, num_epochs=num_epochs, is_training=is_training, file_pattern=FLAGS.dataset_name, category=(model_scope if 'all' not in model_scope else '*'), reader=None)

    return images, {'targets': targets, 'key_v': key_v, 'shape': shape, 'classid': classid, 'isvalid': isvalid, 'norm_value': norm_value}

if config.PRED_DEBUG:
  from scipy.misc import imread, imsave, imshow, imresize
  def save_image_with_heatmap(image, height, width, heatmap_size, targets, pred_heatmap, indR, indG, indB):
      if not hasattr(save_image_with_heatmap, "counter"):
          save_image_with_heatmap.counter = 0  # it doesn't exist yet, so initialize it
      save_image_with_heatmap.counter += 1

      img_to_save = np.array(image.tolist()) + 128
      #print(img_to_save.shape)

      img_to_save = img_to_save.astype(np.uint8)

      heatmap0 = np.sum(targets[indR, ...], axis=0).astype(np.uint8)
      heatmap1 = np.sum(targets[indG, ...], axis=0).astype(np.uint8)
      heatmap2 = np.sum(targets[indB, ...], axis=0).astype(np.uint8) if len(indB) > 0 else np.zeros((heatmap_size, heatmap_size), dtype=np.float32)

      img_to_save = imresize(img_to_save, (height, width), interp='lanczos')
      heatmap0 = imresize(heatmap0, (height, width), interp='lanczos')
      heatmap1 = imresize(heatmap1, (height, width), interp='lanczos')
      heatmap2 = imresize(heatmap2, (height, width), interp='lanczos')

      img_to_save = img_to_save/2
      img_to_save[:,:,0] = np.clip((img_to_save[:,:,0] + heatmap0 + heatmap2), 0, 255)
      img_to_save[:,:,1] = np.clip((img_to_save[:,:,1] + heatmap1 + heatmap2), 0, 255)
      #img_to_save[:,:,2] = np.clip((img_to_save[:,:,2]/4. + heatmap2), 0, 255)
      file_name = 'targets_{}.jpg'.format(save_image_with_heatmap.counter)
      imsave(os.path.join(config.DEBUG_DIR, file_name), img_to_save.astype(np.uint8))

      pred_heatmap = np.array(pred_heatmap.tolist())
      #print(pred_heatmap.shape)
      for ind in range(pred_heatmap.shape[0]):
        img = pred_heatmap[ind]
        img = img - img.min()
        img *= 255.0/img.max()
        file_name = 'heatmap_{}_{}.jpg'.format(save_image_with_heatmap.counter, ind)
        imsave(os.path.join(config.DEBUG_DIR, file_name), img.astype(np.uint8))
      return save_image_with_heatmap.counter

def get_keypoint(image, targets, predictions, heatmap_size, height, width, category, clip_at_zero=True, data_format='channels_last', name=None):
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
      pred_heatmap = tf.one_hot(pred_indices_, heatmap_size*heatmap_size, on_value=1., off_value=0., axis=-1, dtype=tf.float32)

      pred_heatmap = tf.reshape(pred_heatmap, [-1, heatmap_size, heatmap_size])
      if data_format == 'channels_first':
        image_ = tf.transpose(image_, perm=(1, 2, 0))
      save_image_op = tf.py_func(save_image_with_heatmap,
                                  [image_, height, width,
                                  heatmap_size,
                                  tf.reshape(pred_heatmap * 255., [-1, heatmap_size, heatmap_size]),
                                  tf.reshape(predictions, [-1, heatmap_size, heatmap_size]),
                                  config.left_right_group_map[category][0],
                                  config.left_right_group_map[category][1],
                                  config.left_right_group_map[category][2]],
                                  tf.int64, stateful=True)
      with tf.control_dependencies([save_image_op]):
        pred_x, pred_y = pred_x * 1., pred_y * 1.
    return pred_x, pred_y

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

def keypoint_model_fn(features, labels, mode, params):
    targets = labels['targets']
    shape = labels['shape']
    classid = labels['classid']
    key_v = labels['key_v']
    isvalid = labels['isvalid']
    norm_value = labels['norm_value']

    cur_batch_size = tf.shape(features)[0]
    #features= tf.ones_like(features)

    with tf.variable_scope(params['model_scope'], default_name=None, values=[features], reuse=tf.AUTO_REUSE):
        pred_outputs = cpn.cascaded_pyramid_net(features, config.class_num_joints[(params['model_scope'] if 'all' not in params['model_scope'] else '*')], params['heatmap_size'], (mode == tf.estimator.ModeKeys.TRAIN), params['data_format'])

    #print(pred_outputs)

    if params['data_format'] == 'channels_last':
        pred_outputs = [tf.transpose(pred_outputs[ind], [0, 3, 1, 2], name='outputs_trans_{}'.format(ind)) for ind in list(range(len(pred_outputs)))]

    score_map = pred_outputs[-1]

    pred_x, pred_y = get_keypoint(features, targets, score_map, params['heatmap_size'], params['train_image_size'], params['train_image_size'], (params['model_scope'] if 'all' not in params['model_scope'] else '*'), clip_at_zero=True, data_format=params['data_format'])

    # this is important!!!
    targets = 255. * targets
    blur_list = [1., 1.37, 1.73, 2.4, None]#[1., 1.5, 2., 3., None]
    #blur_list = [None, None, None, None, None]

    targets_list = []
    for sigma in blur_list:
        if sigma is None:
            targets_list.append(targets)
        else:
            # always channels first foe targets
            targets_list.append(gaussian_blur(targets, config.class_num_joints[(params['model_scope'] if 'all' not in params['model_scope'] else '*')], sigma, params['data_format'], 'blur_{}'.format(sigma)))

    # print(key_v)
    #targets = tf.reshape(255.*tf.one_hot(tf.ones_like(key_v,tf.int64)*(params['heatmap_size']*params['heatmap_size']//2+params['heatmap_size']), params['heatmap_size']*params['heatmap_size']), [cur_batch_size,-1,params['heatmap_size'],params['heatmap_size']])
    #norm_value = tf.ones_like(norm_value)
    # score_map = tf.reshape(tf.one_hot(tf.ones_like(key_v,tf.int64)*(31*64+31), params['heatmap_size']*params['heatmap_size']), [cur_batch_size,-1,params['heatmap_size'],params['heatmap_size']])

    #with tf.control_dependencies([pred_x, pred_y]):
    ne_mertric = mertric.normalized_error(targets, score_map, norm_value, key_v, isvalid,
                             cur_batch_size,
                             config.class_num_joints[(params['model_scope'] if 'all' not in params['model_scope'] else '*')],
                             params['heatmap_size'],
                             params['train_image_size'])

    # last_pred_mse = tf.metrics.mean_squared_error(score_map, targets,
    #                             weights=1.0 / tf.cast(cur_batch_size, tf.float32),
    #                             name='last_pred_mse')
    # filter all invisible keypoint maybe better for this task
    # all_visible = tf.logical_and(key_v>0, isvalid>0)
    # targets_list = [tf.boolean_mask(targets_list[ind], all_visible) for ind in list(range(len(targets_list)))]
    # pred_outputs = [tf.boolean_mask(pred_outputs[ind], all_visible, name='boolean_mask_{}'.format(ind)) for ind in list(range(len(pred_outputs)))]
    all_visible = tf.expand_dims(tf.expand_dims(tf.cast(tf.logical_and(key_v>0, isvalid>0), tf.float32), axis=-1), axis=-1)
    targets_list = [targets_list[ind] * all_visible for ind in list(range(len(targets_list)))]
    pred_outputs = [pred_outputs[ind] * all_visible for ind in list(range(len(pred_outputs)))]

    sq_diff = tf.reduce_sum(tf.squared_difference(targets, pred_outputs[-1]), axis=-1)
    last_pred_mse = tf.metrics.mean_absolute_error(sq_diff, tf.zeros_like(sq_diff), name='last_pred_mse')

    metrics = {'normalized_error': ne_mertric, 'last_pred_mse':last_pred_mse}
    predictions = {'normalized_error': ne_mertric[1]}
    ne_mertric = tf.identity(ne_mertric[1], name='ne_mertric')

    base_learning_rate = params['learning_rate']
    mse_loss_list = []
    if params['use_ohkm']:
        base_learning_rate = 1. * base_learning_rate
        for pred_ind in list(range(len(pred_outputs) - 1)):
            mse_loss_list.append(0.5 * tf.losses.mean_squared_error(targets_list[pred_ind], pred_outputs[pred_ind],
                                weights=1.0 / tf.cast(cur_batch_size, tf.float32),
                                scope='loss_{}'.format(pred_ind),
                                loss_collection=None,#tf.GraphKeys.LOSSES,
                                # mean all elements of all pixels in all batch
                                reduction=tf.losses.Reduction.MEAN))# SUM, SUM_OVER_BATCH_SIZE, default mean by all elements

        temp_loss = tf.reduce_mean(tf.reshape(tf.losses.mean_squared_error(targets_list[-1], pred_outputs[-1], weights=1.0, loss_collection=None, reduction=tf.losses.Reduction.NONE), [cur_batch_size, config.class_num_joints[(params['model_scope'] if 'all' not in params['model_scope'] else '*')], -1]), axis=-1)

        num_topk = config.class_num_joints[(params['model_scope'] if 'all' not in params['model_scope'] else '*')] // 2
        gather_col = tf.nn.top_k(temp_loss, k=num_topk, sorted=True)[1]
        gather_row = tf.reshape(tf.tile(tf.reshape(tf.range(cur_batch_size), [-1, 1]), [1, num_topk]), [-1, 1])
        gather_indcies = tf.stop_gradient(tf.stack([gather_row, tf.reshape(gather_col, [-1, 1])], axis=-1))

        select_targets = tf.gather_nd(targets_list[-1], gather_indcies)
        select_heatmap = tf.gather_nd(pred_outputs[-1], gather_indcies)

        mse_loss_list.append(tf.losses.mean_squared_error(select_targets, select_heatmap,
                                weights=1.0 / tf.cast(cur_batch_size, tf.float32),
                                scope='loss_{}'.format(len(pred_outputs) - 1),
                                loss_collection=None,#tf.GraphKeys.LOSSES,
                                # mean all elements of all pixels in all batch
                                reduction=tf.losses.Reduction.MEAN))
    else:
        for pred_ind in list(range(len(pred_outputs))):
            mse_loss_list.append(tf.losses.mean_squared_error(targets_list[pred_ind], pred_outputs[pred_ind],
                                weights=1.0 / tf.cast(cur_batch_size, tf.float32),
                                scope='loss_{}'.format(pred_ind),
                                loss_collection=None,#tf.GraphKeys.LOSSES,
                                # mean all elements of all pixels in all batch
                                reduction=tf.losses.Reduction.MEAN))# SUM, SUM_OVER_BATCH_SIZE, default mean by all elements

    mse_loss = tf.multiply(params['mse_weight'], tf.add_n(mse_loss_list), name='mse_loss')
    tf.summary.scalar('mse', mse_loss)
    tf.losses.add_loss(mse_loss)

    # bce_loss_list = []
    # for pred_ind in list(range(len(pred_outputs))):
    #     bce_loss_list.append(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred_outputs[pred_ind], labels=targets_list[pred_ind]/255., name='loss_{}'.format(pred_ind)), name='loss_mean_{}'.format(pred_ind)))

    # mse_loss = tf.multiply(params['mse_weight'] / params['num_stacks'], tf.add_n(bce_loss_list), name='mse_loss')
    # tf.summary.scalar('mse', mse_loss)
    # tf.losses.add_loss(mse_loss)

    # Add weight decay to the loss. We exclude the batch norm variables because
    # doing so leads to a small improvement in accuracy.
    loss = mse_loss + params['weight_decay'] * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'batch_normalization' not in v.name])
    total_loss = tf.identity(loss, name='total_loss')
    tf.summary.scalar('loss', total_loss)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, predictions=predictions, eval_metric_ops=metrics)

    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()

        lr_values = [params['warmup_learning_rate']] + [base_learning_rate * decay for decay in params['lr_decay_factors']]
        learning_rate = tf.train.piecewise_constant(tf.cast(global_step, tf.int32),
                                                    [params['warmup_steps']] + [int(float(ep)*params['steps_per_epoch']) for ep in params['decay_boundaries']],
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
                          scaffold=tf.train.Scaffold(init_fn=train_helper.get_init_fn_for_scaffold_(params['checkpoint_path'], params['model_dir'], params['checkpoint_exclude_scopes'], params['model_scope'], params['checkpoint_model_scope'], params['ignore_missing_vars'])))

def parse_comma_list(args):
    return [float(s.strip()) for s in args.split(',')]

def sub_loop(model_fn, model_scope, model_dir, run_config, train_epochs, epochs_per_eval, lr_decay_factors, decay_boundaries, checkpoint_path=None, checkpoint_exclude_scopes='', checkpoint_model_scope='', ignore_missing_vars=True):
    steps_per_epoch = config.split_size[(model_scope if 'all' not in model_scope else '*')]['train'] // FLAGS.batch_size
    fashionAI = tf.estimator.Estimator(
        model_fn=model_fn, model_dir=model_dir, config=run_config,
        params={
            'checkpoint_path': checkpoint_path,
            'model_dir': model_dir,
            'checkpoint_exclude_scopes': checkpoint_exclude_scopes,
            'model_scope': model_scope,
            'checkpoint_model_scope': checkpoint_model_scope,
            'ignore_missing_vars': ignore_missing_vars,
            'train_image_size': FLAGS.train_image_size,
            'heatmap_size': FLAGS.heatmap_size,
            'data_format': FLAGS.data_format,
            'steps_per_epoch': steps_per_epoch,
            'use_ohkm': FLAGS.use_ohkm,
            'batch_size': FLAGS.batch_size,
            'weight_decay': FLAGS.weight_decay,
            'mse_weight': FLAGS.mse_weight,
            'momentum': FLAGS.momentum,
            'learning_rate': FLAGS.learning_rate,
            'end_learning_rate': FLAGS.end_learning_rate,
            'warmup_learning_rate': FLAGS.warmup_learning_rate,
            'warmup_steps': FLAGS.warmup_steps,
            'decay_boundaries': parse_comma_list(decay_boundaries),
            'lr_decay_factors': parse_comma_list(lr_decay_factors),
        })

    tf.gfile.MakeDirs(model_dir)
    tf.logging.info('Starting to train model {}.'.format(model_scope))
    for _ in range(train_epochs // epochs_per_eval):
        tensors_to_log = {
            'lr': 'learning_rate',
            'loss': 'total_loss',
            'mse': 'mse_loss',
            'ne': 'ne_mertric',
        }

        logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=FLAGS.log_every_n_steps, formatter=lambda dicts: '{}:'.format(model_scope) + (', '.join(['%s=%.6f' % (k, v) for k, v in dicts.items()])))

        tf.logging.info('Starting a training cycle.')
        fashionAI.train(input_fn=lambda : input_pipeline(True, model_scope, epochs_per_eval), hooks=[logging_hook], max_steps=(steps_per_epoch*train_epochs))

        tf.logging.info('Starting to evaluate.')
        eval_results = fashionAI.evaluate(input_fn=lambda : input_pipeline(False, model_scope, 1))
        tf.logging.info(eval_results)
    tf.logging.info('Finished model {}.'.format(model_scope))

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

    if FLAGS.seq_train:
        detail_params = {
            'all': {
                'model_dir' : os.path.join(FLAGS.model_dir, 'all'),
                'train_epochs': 6,
                'epochs_per_eval': 4,
                'lr_decay_factors': '1, 0.5, 0.1',
                'decay_boundaries': '3, 4',
                'model_scope': 'all',
                'checkpoint_path': None,
                'checkpoint_model_scope': '',
                'checkpoint_exclude_scopes': '',
                'ignore_missing_vars': True,
            },
            'blouse': {
                'model_dir' : os.path.join(FLAGS.model_dir, 'blouse'),
                'train_epochs': 50,
                'epochs_per_eval': 30,
                'lr_decay_factors': '1, 0.5, 0.1',
                'decay_boundaries': '15, 30',
                'model_scope': 'blouse',
                'checkpoint_path': os.path.join(FLAGS.model_dir, 'all'),
                'checkpoint_model_scope': 'all',
                'checkpoint_exclude_scopes': 'blouse/feature_pyramid/conv_heatmap, blouse/global_net/conv_heatmap',
                'ignore_missing_vars': True,
            },
            'dress': {
                'model_dir' : os.path.join(FLAGS.model_dir, 'dress'),
                'train_epochs': 50,
                'epochs_per_eval': 30,
                'lr_decay_factors': '1, 0.5, 0.1',
                'decay_boundaries': '15, 30',
                'model_scope': 'dress',
                'checkpoint_path': os.path.join(FLAGS.model_dir, 'all'),
                'checkpoint_model_scope': 'all',
                'checkpoint_exclude_scopes': 'dress/feature_pyramid/conv_heatmap, dress/global_net/conv_heatmap',
                'ignore_missing_vars': True,
            },
            'outwear': {
                'model_dir' : os.path.join(FLAGS.model_dir, 'outwear'),
                'train_epochs': 50,
                'epochs_per_eval': 30,
                'lr_decay_factors': '1, 0.5, 0.1',
                'decay_boundaries': '15, 30',
                'model_scope': 'outwear',
                'checkpoint_path': os.path.join(FLAGS.model_dir, 'all'),
                'checkpoint_model_scope': 'all',
                'checkpoint_exclude_scopes': 'outwear/feature_pyramid/conv_heatmap, outwear/global_net/conv_heatmap',
                'ignore_missing_vars': True,
            },
            'skirt': {
                'model_dir' : os.path.join(FLAGS.model_dir, 'skirt'),
                'train_epochs': 50,
                'epochs_per_eval': 30,
                'lr_decay_factors': '1, 0.5, 0.1',
                'decay_boundaries': '15, 30',
                'model_scope': 'skirt',
                'checkpoint_path': os.path.join(FLAGS.model_dir, 'all'),
                'checkpoint_model_scope': 'all',
                'checkpoint_exclude_scopes': 'skirt/feature_pyramid/conv_heatmap, skirt/global_net/conv_heatmap',
                'ignore_missing_vars': True,
            },
            'trousers': {
                'model_dir' : os.path.join(FLAGS.model_dir, 'trousers'),
                'train_epochs': 50,
                'epochs_per_eval': 30,
                'lr_decay_factors': '1, 0.5, 0.1',
                'decay_boundaries': '15, 30',
                'model_scope': 'trousers',
                'checkpoint_path': os.path.join(FLAGS.model_dir, 'all'),
                'checkpoint_model_scope': 'all',
                'checkpoint_exclude_scopes': 'trousers/feature_pyramid/conv_heatmap, trousers/global_net/conv_heatmap',
                'ignore_missing_vars': True,
            },
        }
    else:
        detail_params = {
            'blouse': {
                'model_dir' : os.path.join(FLAGS.model_dir, 'blouse'),
                'train_epochs': 28,
                'epochs_per_eval': 7,
                'lr_decay_factors': '1, 0.5, 0.1',
                'decay_boundaries': '10, 20',
                'model_scope': 'blouse',
                'checkpoint_path': os.path.join(FLAGS.data_dir, FLAGS.cloud_checkpoint_path) if FLAGS.run_on_cloud else FLAGS.checkpoint_path,
                'checkpoint_model_scope': '',
                'checkpoint_exclude_scopes': 'blouse/feature_pyramid, blouse/global_net',
                'ignore_missing_vars': True,
            },
            'dress': {
                'model_dir' : os.path.join(FLAGS.model_dir, 'dress'),
                'train_epochs': 28,
                'epochs_per_eval': 7,
                'lr_decay_factors': '1, 0.5, 0.1',
                'decay_boundaries': '10, 20',
                'model_scope': 'dress',
                'checkpoint_path': os.path.join(FLAGS.data_dir, FLAGS.cloud_checkpoint_path) if FLAGS.run_on_cloud else FLAGS.checkpoint_path,
                'checkpoint_model_scope': '',
                'checkpoint_exclude_scopes': 'dress/feature_pyramid, dress/global_net',
                'ignore_missing_vars': True,
            },
            'outwear': {
                'model_dir' : os.path.join(FLAGS.model_dir, 'outwear'),
                'train_epochs': 28,
                'epochs_per_eval': 7,
                'lr_decay_factors': '1, 0.5, 0.1',
                'decay_boundaries': '10, 20',
                'model_scope': 'outwear',
                'checkpoint_path': os.path.join(FLAGS.data_dir, FLAGS.cloud_checkpoint_path) if FLAGS.run_on_cloud else FLAGS.checkpoint_path,
                'checkpoint_model_scope': '',
                'checkpoint_exclude_scopes': 'outwear/feature_pyramid, outwear/global_net',
                'ignore_missing_vars': True,
            },
            'skirt': {
                'model_dir' : os.path.join(FLAGS.model_dir, 'skirt'),
                'train_epochs': 28,
                'epochs_per_eval': 7,
                'lr_decay_factors': '1, 0.5, 0.1',
                'decay_boundaries': '10, 20',
                'model_scope': 'skirt',
                'checkpoint_path': os.path.join(FLAGS.data_dir, FLAGS.cloud_checkpoint_path) if FLAGS.run_on_cloud else FLAGS.checkpoint_path,
                'checkpoint_model_scope': '',
                'checkpoint_exclude_scopes': 'skirt/feature_pyramid, skirt/global_net',
                'ignore_missing_vars': True,
            },
            'trousers': {
                'model_dir' : os.path.join(FLAGS.model_dir, 'trousers'),
                'train_epochs': 28,
                'epochs_per_eval': 7,
                'lr_decay_factors': '1, 0.5, 0.1',
                'decay_boundaries': '10, 20',
                'model_scope': 'trousers',
                'checkpoint_path': os.path.join(FLAGS.data_dir, FLAGS.cloud_checkpoint_path) if FLAGS.run_on_cloud else FLAGS.checkpoint_path,
                'checkpoint_model_scope': '',
                'checkpoint_exclude_scopes': 'trousers/feature_pyramid, trousers/global_net',
                'ignore_missing_vars': True,
            },
        }
    model_to_train = [s.strip() for s in FLAGS.model_to_train.split(',')]

    # import datetime
    # import time
    # while True:
    #     time.sleep(1600)
    #     if '8' in datetime.datetime.now().time().strftime('%H'):
    #         break

    for m in model_to_train:
        sub_loop(keypoint_model_fn, m, detail_params[m]['model_dir'], run_config, detail_params[m]['train_epochs'], detail_params[m]['epochs_per_eval'], detail_params[m]['lr_decay_factors'], detail_params[m]['decay_boundaries'], detail_params[m]['checkpoint_path'], detail_params[m]['checkpoint_exclude_scopes'], detail_params[m]['checkpoint_model_scope'], detail_params[m]['ignore_missing_vars'])

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
