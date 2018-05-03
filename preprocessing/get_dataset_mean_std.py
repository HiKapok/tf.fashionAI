import os
from scipy.misc import imread, imsave, imshow, imresize

import numpy as np
import tensorflow as tf

import sys; sys.path.insert(0, ".")
import config

def get_dataset_mean_std():
    all_sub_dirs = []
    for split in config.SPLITS:
        if 'test' not in split:
            for cat in config.CATEGORIES:
                all_sub_dirs.append(os.path.join(config.DATA_DIR, split, 'Images', cat))
    all_image_nums = 0
    #print(all_sub_dirs)
    means = [0., 0., 0.]
    stds = [0., 0., 0.]
    for dirs in all_sub_dirs:
        all_images = tf.gfile.Glob(os.path.join(dirs, '*.jpg'))
        for image in all_images:
            np_image = imread(image, mode='RGB')
            if len(np_image.shape) < 3 or np_image.shape[-1] != 3:
                continue
            all_image_nums += 1

            means[0] += np.mean(np_image[:, :, 0]) / 10000.
            means[1] += np.mean(np_image[:, :, 1]) / 10000.
            means[2] += np.mean(np_image[:, :, 2]) / 10000.

            stds[0] += np.std(np_image[:, :, 0]) / 10000.
            stds[1] += np.std(np_image[:, :, 1]) / 10000.
            stds[2] += np.std(np_image[:, :, 2]) / 10000.

        print([_*10000./all_image_nums for _ in means])
        print([_*10000./all_image_nums for _ in stds])
    print([_*10000./all_image_nums for _ in means])
    print([_*10000./all_image_nums for _ in stds])
    print(all_image_nums)

get_dataset_mean_std()
# [171.04052664596992, 162.98214001911154, 159.88648003318914]
# [62.370313796103616, 64.64434475667025, 64.35966787914904]
