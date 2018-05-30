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
import tensorflow as tf

import config

subs_dir = '../Submit/ensemble'
# ensemble_subs = ['cpn_320_160_1e-3_half_epoch.csv',
# 'cpn_320_160_blur_half_epoch_2e-5.csv',
# 'hg_8_256_v2_half_epoch.csv',
# 'sub_2_cpn_320_100_1e-3-half_epoch.csv',
# 'sub_2_hg_4_256_64-half_epoch.csv',
# 'sub_2_hg_8_256_64_v1-half_epoch.csv']#['cpn_2_320_160_1e-3.csv', 'sub_2_hg_4_256_64.csv', 'sub_2_cpn_320_100_1e-3.csv', 'sub_2_hg_8_256_64.csv']

ensemble_subs = ['large_seresnext_cpn_sub.csv', 'large_detnext_cpn_sub.csv']


def parse_comma_list(args):
    return [float(s.strip()) for s in args.split(',')]

def mean_ensemble():
    # all test images will be put into this dict
    all_test_items = {}
    # extract all predict items
    for sub_file in ensemble_subs:
        sub_file_path = os.path.join(subs_dir, sub_file)
        df = pd.read_csv(sub_file_path, header=0)
        #print(df.values.tolist())
        all_predict = df.values.tolist()

        for records in all_predict:
            file_id = records[0]
            preds = records[1:]
            if file_id in all_test_items:
                all_test_items[file_id].append(preds)
            else:
                all_test_items[file_id] = [preds]

    #print(all_test_items)
    cur_record = 0
    df = pd.DataFrame(columns=['image_id', 'image_category'] + config.all_keys)
    num_keypoints_plus = len(config.all_keys) + 1
    for k, v in all_test_items.items():
        #print(v)
        temp_list = []
        len_pred = len(v) * 1.
        # iterate all the predictions
        for pred_ind in range(1, num_keypoints_plus):
            pred_x, pred_y, pred_v = 0., 0., 1
            if v[0][pred_ind].strip() == '-1_-1_-1':
                temp_list.append('-1_-1_-1')
                #print(temp_list)
                continue
            for _pred in v:
                _pred_x, _pred_y, _pred_v = _pred[pred_ind].strip().split('_')
                _pred_x, _pred_y, _pred_v = float(_pred_x), float(_pred_y), int(_pred_v)
                #print(_pred_x, _pred_y)
                pred_x = pred_x + _pred_x/len_pred
                pred_y = pred_y + _pred_y/len_pred
            temp_list.append('{}_{}_{}'.format(round(pred_x), round(pred_y), pred_v))
        #print(temp_list)
            #break
        #break
        df.loc[cur_record] = [k, v[0][0]] + temp_list
        cur_record = cur_record + 1
    df.sort_values('image_id').to_csv(os.path.join(subs_dir, 'ensmeble.csv'), encoding='utf-8', index=False)

if __name__ == '__main__':
    mean_ensemble()
