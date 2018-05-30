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

import os

import sys; sys.path.insert(0, ".")
import config

slim = tf.contrib.slim

# blouse_0000.tfrecord
# {}_????_val.tfrecord
#category = *
def slim_get_split(dataset_dir, image_preprocessing_fn, batch_size, num_readers, num_preprocessing_threads, num_epochs=None, is_training=True, category='blouse', file_pattern='{}_????', reader=None, return_keypoints=False):
    # Allowing None in the signature so that dataset_factory can use the default.
    if reader is None:
        reader = tf.TFRecordReader

    num_joints = config.class_num_joints[category]

    suffix = '.tfrecord' if is_training else '_val.tfrecord'
    file_pattern = file_pattern.format(category) + suffix
    # Features in Pascal VOC TFRecords.
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/filename': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/height': tf.FixedLenFeature([1], tf.int64),
        'image/width': tf.FixedLenFeature([1], tf.int64),
        'image/channels': tf.FixedLenFeature([1], tf.int64),
        'image/classid': tf.FixedLenFeature([1], tf.int64),
        'image/keypoint/x': tf.VarLenFeature(dtype=tf.int64),
        'image/keypoint/y': tf.VarLenFeature(dtype=tf.int64),
        'image/keypoint/v': tf.VarLenFeature(dtype=tf.int64),
        'image/keypoint/id': tf.VarLenFeature(dtype=tf.int64),
        'image/keypoint/gid': tf.VarLenFeature(dtype=tf.int64),
    }
    items_to_handlers = {
        'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
        'height': slim.tfexample_decoder.Tensor('image/height'),
        'width': slim.tfexample_decoder.Tensor('image/width'),
        'channels': slim.tfexample_decoder.Tensor('image/channels'),
        'classid': slim.tfexample_decoder.Tensor('image/classid'),
        'keypoint/x': slim.tfexample_decoder.Tensor('image/keypoint/x'),
        'keypoint/y': slim.tfexample_decoder.Tensor('image/keypoint/y'),
        'keypoint/v': slim.tfexample_decoder.Tensor('image/keypoint/v'),
        'keypoint/id': slim.tfexample_decoder.Tensor('image/keypoint/id'),
        'keypoint/gid': slim.tfexample_decoder.Tensor('image/keypoint/gid'),
    }
    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)

    input_source = os.path.join(dataset_dir, file_pattern)
    dataset = slim.dataset.Dataset(
                data_sources=input_source,
                reader=reader,
                decoder=decoder,
                num_samples=config.split_size[category]['train' if is_training else 'val'],#dataset_inspect.count_split_examples(dataset_dir, file_prefix='sacw_'),
                items_to_descriptions=None,
                num_classes=num_joints,
                labels_to_names=None)

    with tf.name_scope('dataset_data_provider'):
        provider = slim.dataset_data_provider.DatasetDataProvider(
                                                        dataset,
                                                        num_readers=num_readers,
                                                        common_queue_capacity=32 * batch_size,
                                                        common_queue_min=8 * batch_size,
                                                        shuffle=True,
                                                        num_epochs=num_epochs)

    [org_image, height, width, channels, classid, key_x, key_y, key_v, key_id, key_gid] = provider.get(['image', 'height',
                                                                                            'width', 'channels',
                                                                                            'classid', 'keypoint/x',
                                                                                            'keypoint/y', 'keypoint/v',
                                                                                            'keypoint/id', 'keypoint/gid'])


    gather_ind = config.class2global_ind_map[category]

    key_x, key_y, key_v, key_id, key_gid = tf.gather(key_x, gather_ind), tf.gather(key_y, gather_ind), tf.gather(key_v, gather_ind), tf.gather(key_id, gather_ind), tf.gather(key_gid, gather_ind)

    shape = tf.stack([height, width, channels], axis=0)

    if not return_keypoints:
        image, targets, new_key_v, isvalid, norm_value = image_preprocessing_fn(org_image, classid, shape, key_x, key_y, key_v)
        batch_list = [image, shape, classid, targets, new_key_v, isvalid, norm_value]
    else:
        image, targets, new_key_x, new_key_y, new_key_v, isvalid, norm_value = image_preprocessing_fn(org_image, classid, shape, key_x, key_y, key_v)
        batch_list = [image, shape, classid, targets, new_key_x, new_key_y, new_key_v, isvalid, norm_value]

    batch_input = tf.train.batch(batch_list,
                                #classid, key_x, key_y, key_v, key_id, key_gid],
                                dynamic_pad=False,#(not is_training),
                                batch_size = batch_size,
                                allow_smaller_final_batch=True,
                                num_threads = num_preprocessing_threads,
                                capacity = 64 * batch_size)
    return batch_input


def slim_test_get_split(dataset_dir, image_preprocessing_fn, num_readers, num_preprocessing_threads, category='blouse', file_pattern='{}_*.tfrecord', reader=None, dynamic_pad=False):
    # Allowing None in the signature so that dataset_factory can use the default.
    if reader is None:
        reader = tf.TFRecordReader

    num_joints = config.class_num_joints[category]
    file_pattern = file_pattern.format(category)
    # Features in Pascal VOC TFRecords.
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/filename': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/height': tf.FixedLenFeature([1], tf.int64),
        'image/width': tf.FixedLenFeature([1], tf.int64),
        'image/channels': tf.FixedLenFeature([1], tf.int64),
        'image/classid': tf.FixedLenFeature([1], tf.int64)
    }
    items_to_handlers = {
        'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
        'height': slim.tfexample_decoder.Tensor('image/height'),
        'width': slim.tfexample_decoder.Tensor('image/width'),
        'channels': slim.tfexample_decoder.Tensor('image/channels'),
        'classid': slim.tfexample_decoder.Tensor('image/classid'),
        'filename': slim.tfexample_decoder.Tensor('image/filename')
    }
    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)

    input_source = os.path.join(dataset_dir, file_pattern)
    #print(config.split_size[category]['test'])
    dataset = slim.dataset.Dataset(
                data_sources=input_source,
                reader=reader,
                decoder=decoder,
                num_samples=config.split_size[category]['test'],#dataset_inspect.count_split_examples(dataset_dir, file_prefix='sacw_'),
                items_to_descriptions=None,
                num_classes=num_joints,
                labels_to_names=None)

    with tf.name_scope('dataset_data_provider'):
        provider = slim.dataset_data_provider.DatasetDataProvider(
                                                        dataset,
                                                        num_readers=num_readers,
                                                        common_queue_capacity=32,
                                                        common_queue_min=8,
                                                        shuffle=False,
                                                        num_epochs=1)

    [org_image, height, width, channels, classid, filename] = provider.get(['image', 'height', 'width', 'channels', 'classid', 'filename'])

    shape = tf.stack([height, width, channels], axis=0)
    if image_preprocessing_fn is not None:
        image, shape, offsets = image_preprocessing_fn(org_image, filename, shape)
    else:
        image = org_image
        offsets = tf.constant([0, 0], tf.int64)

    batch_input = tf.train.batch([image, shape, filename, classid, offsets],
                                dynamic_pad = dynamic_pad,
                                batch_size = 1,
                                allow_smaller_final_batch=True,
                                num_threads = num_preprocessing_threads,
                                capacity = 64)
    return batch_input
if __name__ == '__main__':
    import preprocessing

    category='skirt'
    if '*' in category:
        lnorm_table = tf.contrib.lookup.HashTable(tf.contrib.lookup.KeyValueTensorInitializer(tf.constant(config.global_norm_key, dtype=tf.int64),
                                                                tf.constant(config.global_norm_lvalues, dtype=tf.int64)), 0)
        rnorm_table = tf.contrib.lookup.HashTable(tf.contrib.lookup.KeyValueTensorInitializer(tf.constant(config.global_norm_key, dtype=tf.int64),
                                                                tf.constant(config.global_norm_rvalues, dtype=tf.int64)), 1)
    else:
        lnorm_table = tf.contrib.lookup.HashTable(tf.contrib.lookup.KeyValueTensorInitializer(tf.constant(config.local_norm_key, dtype=tf.int64),
                                                                tf.constant(config.local_norm_lvalues, dtype=tf.int64)), 0)
        rnorm_table = tf.contrib.lookup.HashTable(tf.contrib.lookup.KeyValueTensorInitializer(tf.constant(config.local_norm_key, dtype=tf.int64),
                                                                tf.constant(config.local_norm_rvalues, dtype=tf.int64)), 1)
    preprocessing_fn = lambda org_image, classid, shape, key_x, key_y, key_v: preprocessing.preprocess_image(org_image, classid, shape, 256, 256, key_x, key_y, key_v, (lnorm_table, rnorm_table), is_training=True, category=category)
    #['blouse', 'dress', 'outwear', 'skirt', 'trousers']
    batch_input = slim_get_split(config.RECORDS_DATA_DIR, preprocessing_fn, 1, 2, 4, num_epochs=None, is_training=True, file_pattern='{}_????', category=category, reader=None)

    #preprocessing_fn = lambda org_image, classid, shape: preprocessing.preprocess_for_test(org_image, classid, shape, 256, 256)
    #batch_input = slim_test_get_split(config.TEST_RECORDS_DATA_DIR, preprocessing_fn, 2, 4)
    # Create the graph, etc.
    init_op = tf.group([tf.local_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer()])

    # Create a session for running operations in the Graph.
    sess = tf.Session()
    rotate_matrix = tf.contrib.image.angles_to_projective_transforms(1., 128,128)
    # Initialize the variables (like the epoch counter).
    sess.run(init_op)

    # Start input enqueue threads.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        while not coord.should_stop():
            # Run training steps or whatever
            print(sess.run(batch_input)[-3:])
            #print(sess.run(rotate_matrix))
            #print(sess.run(batch_input)[-2][0].decode('utf8'))

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()

    # Wait for threads to finish.
    coord.join(threads)
    sess.close()
