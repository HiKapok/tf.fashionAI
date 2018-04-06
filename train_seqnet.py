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

import tensorflow as tf
import config

# scaffold related configuration
tf.app.flags.DEFINE_string(
    'data_dir', '/media/rs/0E06CD1706CD0127/Kapok/Chi/Datasets/tfrecords',
    'The directory where the dataset input data is stored.')
tf.app.flags.DEFINE_string(
    'dataset_name', '{}_????', 'The pattern of the dataset name to load.')
tf.app.flags.DEFINE_string(
    #'blouse', 'dress', 'outwear', 'skirt', 'trousers', '*'
    'dataset_split_name', 'blouse', 'The name of the train/test split.')
tf.app.flags.DEFINE_string(
    'model_dir', './logs/',
    'The parent directory where the model will be stored.')
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
    'checkpoint_model_scope', None,
    'Model scope in the checkpoint. None if the same as the trained model.')
tf.app.flags.DEFINE_string(
    #'blouse', 'dress', 'outwear', 'skirt', 'trousers', 'all'
    'model_scope', 'all',
    'Model scope name used to replace the name_scope in checkpoint.')
tf.app.flags.DEFINE_string(
    'checkpoint_exclude_scopes', None,#'all/hg_heatmap',#
    'Comma-separated list of scopes of variables to exclude when restoring from a checkpoint.')
tf.app.flags.DEFINE_boolean(
    'run_on_cloud', True,
    'Wether we will train on cloud.')

FLAGS = tf.app.flags.FLAGS

total_params = {
    '-data_dir': FLAGS.data_dir,
    '-dataset_name': FLAGS.dataset_name,
    #'blouse', 'dress', 'outwear', 'skirt', 'trousers', '*'
    '-model_dir': FLAGS.model_dir,
    '-save_checkpoints_secs': FLAGS.save_checkpoints_secs,
    '-train_image_size': FLAGS.train_image_size,
    '-heatmap_size': FLAGS.heatmap_size,
    '-heatmap_sigma': FLAGS.heatmap_sigma,
    '-feats_channals': FLAGS.feats_channals,
    '-num_stacks': FLAGS.num_stacks,
    '-num_modules': FLAGS.num_modules,
    '-bbox_border': FLAGS.bbox_border,
    '-train_epochs': FLAGS.train_epochs,
    '-epochs_per_eval': FLAGS.epochs_per_eval,
    '-batch_size': FLAGS.batch_size,
    '-data_format': FLAGS.data_format,
    '-tf_random_seed': FLAGS.tf_random_seed,
    '-weight_decay': FLAGS.weight_decay,
    '-mse_weight': FLAGS.mse_weight,
    '-momentum': FLAGS.momentum,
    '-learning_rate': FLAGS.learning_rate,
    '-end_learning_rate': FLAGS.end_learning_rate,
    '-decay_boundaries': FLAGS.decay_boundaries,
    '-lr_decay_factors': FLAGS.lr_decay_factors,
    '-checkpoint_path': FLAGS.checkpoint_path,
    '-checkpoint_model_scope': FLAGS.checkpoint_model_scope,
    '-model_scope': FLAGS.model_scope,
    '-checkpoint_exclude_scopes': FLAGS.checkpoint_exclude_scopes,
    '-run_on_cloud': FLAGS.run_on_cloud
    }

detail_params = {
    'all': {
        'model_dir' : os.path.join(FLAGS.model_dir, 'all'),
        'train_epochs': 5,
        'epochs_per_eval': 1,
        'decay_boundaries': '2, 3',
        'model_scope': 'all',
    },
    'blouse': {
        'model_dir' : os.path.join(FLAGS.model_dir, 'blouse'),
        'train_epochs': 40,
        'epochs_per_eval': 5,
        'decay_boundaries': '10, 25',
        'model_scope': 'blouse',
        'checkpoint_path': os.path.join(FLAGS.model_dir, 'all'),
        'checkpoint_model_scope': 'all',
        'checkpoint_exclude_scopes': 'all/hg_heatmap',
    },
    'dress': {
        'model_dir' : os.path.join(FLAGS.model_dir, 'dress'),
        'train_epochs': 40,
        'epochs_per_eval': 5,
        'decay_boundaries': '10, 25',
        'model_scope': 'dress',
        'checkpoint_path': os.path.join(FLAGS.model_dir, 'all'),
        'checkpoint_model_scope': 'all',
        'checkpoint_exclude_scopes': 'all/hg_heatmap',
    },
    'outwear': {
        'model_dir' : os.path.join(FLAGS.model_dir, 'outwear'),
        'train_epochs': 40,
        'epochs_per_eval': 5,
        'decay_boundaries': '10, 25',
        'model_scope': 'outwear',
        'checkpoint_path': os.path.join(FLAGS.model_dir, 'all'),
        'checkpoint_model_scope': 'all',
        'checkpoint_exclude_scopes': 'all/hg_heatmap',
    },
    'skirt': {
        'model_dir' : os.path.join(FLAGS.model_dir, 'skirt'),
        'train_epochs': 40,
        'epochs_per_eval': 5,
        'decay_boundaries': '10, 25',
        'model_scope': 'skirt',
        'checkpoint_path': os.path.join(FLAGS.model_dir, 'all'),
        'checkpoint_model_scope': 'all',
        'checkpoint_exclude_scopes': 'all/hg_heatmap',
    },
    'trousers': {
        'model_dir' : os.path.join(FLAGS.model_dir, 'trousers'),
        'train_epochs': 40,
        'epochs_per_eval': 5,
        'decay_boundaries': '10, 25',
        'model_scope': 'trousers',
        'checkpoint_path': os.path.join(FLAGS.model_dir, 'all'),
        'checkpoint_model_scope': 'all',
        'checkpoint_exclude_scopes': 'all/hg_heatmap',
    },
}

def parse_comma_list(args):
    return [float(s.strip()) for s in args.split(',')]

def main(_):
    import subprocess
    import copy

    all_category = ['all'] + config.CATEGORIES
    for cat in all_category:
        tf.gfile.MakeDirs(os.path.join(FLAGS.model_dir, cat))

    for cat in all_category:
        temp_params = copy.deepcopy(total_params)
        for k, v in total_params.items():
            if k[1:] in detail_params[cat]:
                temp_params[k] = detail_params[cat][k[1:]]

        params_str = []
        for k, v in temp_params.items():
            if v is not None:
                params_str.append(k)
                params_str.append(str(v))
        print('params send: ', params_str)
        train_process = subprocess.Popen(['python', './train_subnet.py'] + params_str, stdout=subprocess.PIPE, cwd=os.getcwd())
        output, _ = train_process.communicate()
        print(output)

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
