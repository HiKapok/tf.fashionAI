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
import config

import os
import sys
import re
import random
from scipy import misc

import numpy as np
import tensorflow as tf
import pandas as pd

# TFRecords convertion parameters.
RANDOM_SEED = 4242
SAMPLES_PER_FILES = 2500

def int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def float_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def bytes_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _process_image(filename):
    image_data = tf.gfile.FastGFile(filename, 'rb').read()
    return image_data, misc.imread(filename).shape

def _convert_to_example(image_data, shape, image_file, class_id, keypoint_x, keypoint_y, keypoint_v, keypoint_id, keypoint_global_id):
    image_format = b'JPEG'
    example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': int64_feature(shape[0]),
            'image/width': int64_feature(shape[1]),
            'image/channels': int64_feature(shape[2]),
            'image/classid': int64_feature(class_id),
            'image/keypoint/x': int64_feature(keypoint_x),
            'image/keypoint/y': int64_feature(keypoint_y),
            'image/keypoint/v': int64_feature(keypoint_v),
            'image/keypoint/id': int64_feature(keypoint_id),
            'image/keypoint/gid': int64_feature(keypoint_global_id),
            'image/format': bytes_feature(image_format),
            'image/filename': bytes_feature(image_file.encode('utf8')),
            'image/encoded': bytes_feature(image_data)}))
    return example

def _add_to_tfrecord(tfrecord_writer, image_path, image_file, class_id, keypoint_x, keypoint_y, keypoint_v, keypoint_id, keypoint_global_id):
    image_data, shape = _process_image(image_path)
    example = _convert_to_example(image_data, shape, image_file, class_id, keypoint_x, keypoint_y, keypoint_v, keypoint_id, keypoint_global_id)
    tfrecord_writer.write(example.SerializeToString())

def _test_add_to_tfrecord(tfrecord_writer, image_path, image_file, class_id):
    image_data, shape = _process_image(image_path)
    image_format = b'JPEG'
    example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': int64_feature(shape[0]),
            'image/width': int64_feature(shape[1]),
            'image/channels': int64_feature(shape[2]),
            'image/classid': int64_feature(class_id),
            'image/format': bytes_feature(image_format),
            'image/filename': bytes_feature(image_file.encode('utf8')),
            'image/encoded': bytes_feature(image_data)}))
    tfrecord_writer.write(example.SerializeToString())

def test_dataset():

    filename_queue = tf.train.string_input_producer(['/media/rs/0E06CD1706CD0127/Kapok/Chi/Datasets/tfrecords/blouse_0000.tfrecord'], num_epochs=None)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(serialized_example,
            features={
                'image/height': tf.FixedLenFeature([1], tf.int64),
                'image/width': tf.FixedLenFeature([1], tf.int64),
                'image/channels': tf.FixedLenFeature([1], tf.int64),
                'image/classid': tf.FixedLenFeature([1], tf.int64),
                'image/keypoint/x': tf.VarLenFeature(dtype=tf.int64),
                'image/keypoint/y': tf.VarLenFeature(dtype=tf.int64),
                'image/keypoint/v': tf.VarLenFeature(dtype=tf.int64),
                'image/keypoint/id': tf.VarLenFeature(dtype=tf.int64),
                'image/keypoint/gid': tf.VarLenFeature(dtype=tf.int64),
                'image/format': tf.FixedLenFeature([], tf.string, default_value='jpeg'),
                'image/filename': tf.FixedLenFeature([], tf.string, default_value=''),
                'image/encoded': tf.FixedLenFeature([], tf.string, default_value='')
            }
        )

    sess = tf.Session()
    init = tf.initialize_all_variables()
    sess.run(init)
    tf.train.start_queue_runners(sess=sess)
    eval_features = sess.run(features)
    eval_features = sess.run(features)
    eval_features = sess.run(features)
    eval_features = sess.run(features)
    eval_features = sess.run(features)
    eval_features = sess.run(features)

    print('image/height', eval_features['image/height'])
    print('image/width', eval_features['image/width'])
    print('image/channels', eval_features['image/channels'])
    print('image/classid', eval_features['image/classid'])
    print('image/keypoint/x', eval_features['image/keypoint/x'])
    print('image/keypoint/y', eval_features['image/keypoint/y'])
    print('image/keypoint/v', eval_features['image/keypoint/v'])
    print('image/keypoint/id', eval_features['image/keypoint/id'])
    print('image/keypoint/gid', eval_features['image/keypoint/gid'])
    print('image/format', eval_features['image/format'])
    print('image/filename', eval_features['image/filename'].decode('utf8'))
    #print('image/encoded', eval_features['image/encoded'])


# print(blouse_keymap)
# print(inverse_blouse_keymap)
# print(outwear_keymap)
# print(inverse_outwear_keymap)
# print(trousers_keymap)
# print(inverse_trousers_keymap)
# print(skirt_keymap)
# print(inverse_skirt_keymap)
# print(dress_keymap)
# print(inverse_dress_keymap)
# print(key2ind)
# print(inverse_key2ind)
keymap_factory = {'blouse': config.blouse_keymap,
                 'dress': config.dress_keymap,
                 'outwear': config.outwear_keymap,
                 'skirt': config.skirt_keymap,
                 'trousers': config.trousers_keymap}

def convert_train(output_dir, val_per=0.015, all_splits=config.SPLITS, file_idx_start=0):
    class_hist = {'blouse': 0,
                 'dress': 0,
                 'outwear': 0,
                 'skirt': 0,
                 'trousers': 0}

    start_file_idx = {'blouse': 5,
                 'dress': 3,
                 'outwear': 4,
                 'skirt': 4,
                 'trousers': 4}

    for cat in config.CATEGORIES:
        total_examples = 0
        # TODO: create tfrecorder writer here
        sys.stdout.write('\nprocessing category: {}...'.format(cat))
        sys.stdout.flush()
        file_idx = file_idx_start#start_file_idx[cat]
        record_idx = 0
        tf_filename = os.path.join(output_dir, '%s_%04d.tfrecord' % (cat, file_idx))
        tfrecord_writer = tf.python_io.TFRecordWriter(tf_filename)

        tf_val_filename = os.path.join(output_dir, '%s_%04d_val.tfrecord' % (cat, 0))
        val_tfrecord_writer = tf.python_io.TFRecordWriter(tf_val_filename)
        this_key_map = keymap_factory[cat]

        for split in all_splits:
            if 'test' in split: continue
            sys.stdout.write('\nprocessing split: {}...\n'.format(split))
            sys.stdout.flush()
            split_path = os.path.join(config.DATA_DIR, split)
            anna_root = os.path.join(split_path, 'Annotations')
            anna_file = os.path.join(anna_root, os.listdir(anna_root)[0])
            anna_pd = pd.read_csv(anna_file)
            anna_pd = anna_pd.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
            this_nums = len(anna_pd.index)
            total_examples += this_nums
            all_columns_name = list(anna_pd.columns)
            #print(all_columns_name)
            all_columns_name = sorted([s.strip() for s in all_columns_name[2:]])
            #print(all_columns_name)
            # print(anna_pd)
            # print(all_columns_name)
            for index, row in anna_pd.iterrows():
                sys.stdout.write('\r>> Converting image %d/%d' % (index+1, this_nums))
                sys.stdout.flush()
                category = row['image_category']
                if not (cat in category): continue
                class_hist[category] += 1
                image_file = row['image_id']
                full_file_path = os.path.join(split_path, image_file)
                #print(len(all_columns_name))
                class_id = config.category2ind[category]
                keypoint_x = []
                keypoint_y = []
                keypoint_v = []
                keypoint_id = []
                keypoint_global_id = []

                for keys in config.all_keys:
                    if keys in this_key_map:
                        keypoint_id.append(this_key_map[keys])
                    else:
                        keypoint_id.append(-1)
                    keypoint_global_id.append(config.key2ind[keys] - 1)
                    keypoint_info = row[keys].strip().split('_')
                    keypoint_x.append(int(keypoint_info[0]))
                    keypoint_y.append(int(keypoint_info[1]))
                    keypoint_v.append(int(keypoint_info[2]))
                    #print(row[keys].strip().split('_'))
                if np.random.random_sample() > val_per:
                    _add_to_tfrecord(tfrecord_writer, full_file_path, image_file, class_id, keypoint_x, keypoint_y, keypoint_v, keypoint_id, keypoint_global_id)
                else:
                    _add_to_tfrecord(val_tfrecord_writer, full_file_path, image_file, class_id, keypoint_x, keypoint_y, keypoint_v, keypoint_id, keypoint_global_id)
                record_idx += 1
                if record_idx > SAMPLES_PER_FILES:
                    record_idx = 0
                    file_idx += 1
                    tf_filename = os.path.join(output_dir, '%s_%04d.tfrecord' % (cat, file_idx))
                    tfrecord_writer.flush()
                    tfrecord_writer.close()
                    tfrecord_writer = tf.python_io.TFRecordWriter(tf_filename)
                    #print(keypoint_id)
                    #print(keypoint_global_id)
                    # print(keypoint_x)
                    # print(keypoint_y)
                    # print(keypoint_v)
                    #keymap_factory[category](full_file_path, image_file)
                    #[(col, row[col]) for col in all_columns_name]
                    #pass#print(row['c1'], row['c2'])
        val_tfrecord_writer.flush()
        val_tfrecord_writer.close()
    print('\nFinished converting the whole dataset!')
    print(class_hist, total_examples)
    return class_hist, total_examples

def convert_test(output_dir, splits=config.SPLITS):

    class_hist = {'blouse': 0,
                 'dress': 0,
                 'outwear': 0,
                 'skirt': 0,
                 'trousers': 0}

    for cat in config.CATEGORIES:
        total_examples = 0
        # TODO: create tfrecorder writer here
        sys.stdout.write('\nprocessing category: {}...'.format(cat))
        sys.stdout.flush()
        file_idx = 0
        record_idx = 0
        tf_filename = os.path.join(output_dir, '%s_%04d.tfrecord' % (cat, file_idx))
        tfrecord_writer = tf.python_io.TFRecordWriter(tf_filename)
        this_key_map = keymap_factory[cat]

        for split in splits:
            if 'train' in split: continue
            sys.stdout.write('\nprocessing split: {}...\n'.format(split))
            sys.stdout.flush()
            split_path = os.path.join(config.DATA_DIR, split)
            anna_file = os.path.join(split_path, 'test.csv')
            anna_pd = pd.read_csv(anna_file)
            this_nums = len(anna_pd.index)
            total_examples += this_nums
            for index, row in anna_pd.iterrows():
                sys.stdout.write('\r>> Converting image %d/%d' % (index+1, this_nums))
                sys.stdout.flush()
                category = row['image_category']
                if not (cat in category): continue
                class_hist[category] += 1
                image_file = row['image_id']
                full_file_path = os.path.join(split_path, image_file)
                #print(len(all_columns_name))
                class_id = config.category2ind[category]

                _test_add_to_tfrecord(tfrecord_writer, full_file_path, image_file, class_id)
                record_idx += 1
                if record_idx > SAMPLES_PER_FILES:
                    record_idx = 0
                    file_idx += 1
                    tf_filename = os.path.join(output_dir, '%s_%04d.tfrecord' % (cat, file_idx))
                    tfrecord_writer.flush()
                    tfrecord_writer.close()
                    tfrecord_writer = tf.python_io.TFRecordWriter(tf_filename)
    print('\nFinished converting the whole test dataset!')
    print(class_hist, total_examples)
    return class_hist, total_examples

def count_split_examples(split_path, file_pattern=''):
    # Count the total number of examples in all of these shard
    num_samples = 0
    tfrecords_to_count = [os.path.join(split_path, file) for file in os.listdir(split_path) if file_pattern in file]
    opts = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
    for tfrecord_file in tfrecords_to_count:
        for record in tf.python_io.tf_record_iterator(tfrecord_file):#, options = opts):
            num_samples += 1
            #print(num_samples)
    return num_samples

if __name__ == '__main__':
    np.random.seed(RANDOM_SEED)
    convert_test(config.TEST_RECORDS_STAGE2, splits=['test_1'])
    print('blouse', count_split_examples(config.TEST_RECORDS_STAGE2, file_pattern='blouse')
    , 'outwear', count_split_examples(config.TEST_RECORDS_STAGE2, file_pattern='outwear')
    , 'dress', count_split_examples(config.TEST_RECORDS_STAGE2, file_pattern='dress')
    , 'skirt', count_split_examples(config.TEST_RECORDS_STAGE2, file_pattern='skirt')
    , 'trousers', count_split_examples(config.TEST_RECORDS_STAGE2, file_pattern='trousers')
    , 'all', count_split_examples(config.TEST_RECORDS_STAGE2, file_pattern='_'))

    # os.mkdir(config.RECORDS_DATA_DIR)
    # convert_train(config.RECORDS_DATA_DIR, val_per=0.)
    # convert_train(config.RECORDS_DATA_DIR, val_per=0., all_splits=config.WARM_UP_SPLITS, file_idx_start=1000)
    # os.mkdir(config.TEST_RECORDS_DATA_DIR)
    # convert_test(config.TEST_RECORDS_DATA_DIR)
    # print('blouse', count_split_examples(config.RECORDS_DATA_DIR, file_pattern='blouse_0000_val')
    # , 'outwear', count_split_examples(config.RECORDS_DATA_DIR, file_pattern='outwear_0000_val')
    # , 'dress', count_split_examples(config.RECORDS_DATA_DIR, file_pattern='dress_0000_val')
    # , 'skirt', count_split_examples(config.RECORDS_DATA_DIR, file_pattern='skirt_0000_val')
    # , 'trousers', count_split_examples(config.RECORDS_DATA_DIR, file_pattern='trousers_0000_val')
    # , 'all', count_split_examples(config.RECORDS_DATA_DIR, file_pattern='val'))
    # test_dataset()

