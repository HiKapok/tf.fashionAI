import os

import tensorflow as tf
import numpy as np

import config

def count_split_examples(split_path, category='tfrecord', file_prefix='.tfrecord'):
    # Count the total number of examples in all of these shard
    num_samples = 0
    tfrecords_to_count = [os.path.join(split_path, file) for file in os.listdir(split_path) if file_prefix in file]
    opts = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
    for tfrecord_file in tfrecords_to_count:
        if category not in tfrecord_file: continue
        for record in tf.python_io.tf_record_iterator(tfrecord_file):#, options = opts):
            num_samples += 1
    return num_samples

if __name__ == '__main__':
    # print(count_split_examples(config.RECORDS_DATA_DIR, '*_val.tfrecord'))
    # print(count_split_examples(config.RECORDS_DATA_DIR, '_????.tfrecord'))
    # print(count_split_examples(config.TEST_RECORDS_DATA_DIR, '*.tfrecord'))

    print(count_split_examples(config.RECORDS_DATA_DIR, 'tfrecord', 'val.tfrecord'))
    print(count_split_examples(config.RECORDS_DATA_DIR, 'tfrecord', '.tfrecord') - count_split_examples(config.RECORDS_DATA_DIR, 'tfrecord', 'val.tfrecord'))
    print(count_split_examples(config.TEST_RECORDS_DATA_DIR, 'tfrecord', '.tfrecord'))
    for cat in config.CATEGORIES:
        print('count category: ', cat)
        print(count_split_examples(config.RECORDS_DATA_DIR, cat, 'val.tfrecord'))
        print(count_split_examples(config.RECORDS_DATA_DIR, cat, '.tfrecord') - count_split_examples(config.RECORDS_DATA_DIR, cat, 'val.tfrecord'))
        print(count_split_examples(config.TEST_RECORDS_DATA_DIR, cat, '.tfrecord'))
