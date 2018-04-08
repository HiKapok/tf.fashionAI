import os

DEBUG = False
PRED_DEBUG = False

DATA_DIR = '/media/rs/0E06CD1706CD0127/Kapok/Chi/Datasets'
RECORDS_DATA_DIR = '/media/rs/0E06CD1706CD0127/Kapok/Chi/Datasets/tfrecords'
TEST_RECORDS_DATA_DIR = '/media/rs/0E06CD1706CD0127/Kapok/Chi/Datasets/tfrecords_test'

CATEGORIES = ['blouse', 'dress', 'outwear', 'skirt', 'trousers']
SPLITS = ['test', 'train_0', 'train_1']

DEBUG_DIR = '/media/rs/0E06CD1706CD0127/Kapok/Chi/Debug'
EVAL_DEBUG_DIR = '/media/rs/0E06CD1706CD0127/Kapok/Chi/Eval_Debug'


#all_keys = sorted(list(set(blouse_key + outwear_key + trousers_key + skirt_key + dress_key)))
#print(dict(zip(all_keys, list(range(len(all_keys))))))
#print(all_keys, len(all_keys))
category2ind = dict(zip(sorted(CATEGORIES), list(range(len(CATEGORIES)))))
ind2category = dict(zip(list(range(len(CATEGORIES))), sorted(CATEGORIES)))


#（上衣、外套、连衣裙为两个腋窝点欧式距离,armpit_left|armpit_right，裤子和半身裙为两个裤头点的欧式距离, waistband_left,waistband_right--trousers, skirt）
normalize_point_ind = {
    'blouse': ([5, 6], [6, 7]),
    'outwear': ([4, 5], [6, 7]),
    'trousers': ([0, 1], [16, 17]),
    'skirt': ([0, 1], [16, 17]),
    'dress': ([5, 6], [6, 7])
}
normalize_point_ind_by_id = {}

local_norm_key = []
local_norm_lvalues = []
local_norm_rvalues = []
global_norm_key = []
global_norm_lvalues = []
global_norm_rvalues = []

for k, v in normalize_point_ind.items():
    normalize_point_ind_by_id[category2ind[k]] = v
    local_norm_key.append(category2ind[k])
    local_norm_lvalues.append(v[0][0])
    local_norm_rvalues.append(v[0][1])
    global_norm_key.append(category2ind[k])
    global_norm_lvalues.append(v[1][0])
    global_norm_rvalues.append(v[1][1])

# key2ind = {'bottom_right_in': 4, 'bottom_left_out': 3, 'waistline_left': 22, 'neckline_left': 14, 'cuff_left_out': 9, 'bottom_right_out': 5, 'waistband_left': 20, 'top_hem_right': 19, 'top_hem_left': 18, 'cuff_right_in': 10, 'armpit_left': 0, 'bottom_left_in': 2, 'cuff_left_in': 8, 'cuff_right_out': 11, 'hemline_left': 12, 'neckline_right': 15, 'shoulder_right': 17, 'hemline_right': 13, 'waistband_right': 21, 'armpit_right': 1, 'waistline_right': 23, 'shoulder_left': 16, 'center_front': 6, 'crotch': 7}
key2ind = {'neckline_left': 1, 'neckline_right': 2, 'center_front': 3, 'shoulder_left': 4, 'shoulder_right': 5, 'armpit_left': 6, 'armpit_right': 7, 'waistline_left': 8, 'waistline_right': 9, 'cuff_left_in': 10, 'cuff_left_out': 11, 'cuff_right_in': 12, 'cuff_right_out': 13, 'top_hem_left': 14, 'top_hem_right': 15, 'waistband_left': 16, 'waistband_right': 17, 'hemline_left': 18, 'hemline_right': 19, 'crotch': 20, 'bottom_left_in': 21, 'bottom_left_out': 22, 'bottom_right_in': 23, 'bottom_right_out': 24}

all_keys = ['neckline_left', 'neckline_right', 'center_front', 'shoulder_left', 'shoulder_right', 'armpit_left', 'armpit_right', 'waistline_left', 'waistline_right', 'cuff_left_in', 'cuff_left_out', 'cuff_right_in', 'cuff_right_out', 'top_hem_left', 'top_hem_right', 'waistband_left', 'waistband_right', 'hemline_left', 'hemline_right', 'crotch', 'bottom_left_in', 'bottom_left_out', 'bottom_right_in', 'bottom_right_out']

inverse_key2ind = {}
for k, v in key2ind.items():
    inverse_key2ind[v] = k


class_num_joints = {
    '*': 24,
    'blouse': 13,
    'dress': 14,
    'outwear': 7,
    'skirt': 4,
    'trousers': 15
}

# |0|1|2|3|4|
# |---|---|---|---|---|
# |neckline_left|neckline_right|shoulder_left|shoulder_right|center_front|

# |5|6|7|8|
# |---|---|---|---|
# |armpit_left|armpit_right|top_hem_left|top_hem_right|

# |9|10|11|12|
# |---|---|---|---|
# |cuff_left_in|cuff_left_out|cuff_right_in|cuff_right_out|
## Blouse 13
blouse_keymap = {'neckline_left': 0,
            'neckline_right': 1,
            'shoulder_left': 2,
            'shoulder_right': 3,
            'center_front': 4,
            'armpit_left': 5,
            'armpit_right': 6,
            'top_hem_left': 7,
            'top_hem_right': 8,
            'cuff_left_in': 9,
            'cuff_left_out': 10,
            'cuff_right_in': 11,
            'cuff_right_out': 12}

inverse_blouse_keymap = {}
for k, v in blouse_keymap.items():
    inverse_blouse_keymap[v] = k

blouse_global_ind = []
for i in range(len(inverse_blouse_keymap)):
    blouse_global_ind.append(key2ind[inverse_blouse_keymap[i]]-1)


# ## Outwear 14
## Outwear
# | 0 | 1 | 2 | 3 | 4 |
# | --- | --- | --- | --- | --- |
# | neckline_left | neckline_right | shoulder_left | shoulder_right | armpit_left |

# | 5 | 6 | 7 | 8 | 9 |
# | --- | --- | --- | --- | --- |
# | armpit_right | waistline_left | waistline_right | cuff_left_in | cuff_left_out|

# | 10 | 11 | 12 | 13 |
# | --- | --- | --- | --- |
# | cuff_right_in | cuff_right_out | top_hem_left |top_hem_right  |

outwear_keymap = {'neckline_left': 0,
            'neckline_right': 1,
            'shoulder_left': 2,
            'shoulder_right': 3,
            'armpit_left': 4,
            'armpit_right': 5,
            'waistline_left': 6,
            'waistline_right': 7,
            'cuff_left_in': 8,
            'cuff_left_out': 9,
            'cuff_right_in': 10,
            'cuff_right_out': 11,
            'top_hem_left': 12,
            'top_hem_right': 13}

inverse_outwear_keymap = {}
for k, v in outwear_keymap.items():
    inverse_outwear_keymap[v] = k

outwear_global_ind = []
for i in range(len(inverse_outwear_keymap)):
    outwear_global_ind.append(key2ind[inverse_outwear_keymap[i]]-1)

# ## Trousers 7
## Trousers
# | 0 | 1 | 2 |
# | --- | --- | --- |
# | waistband_left | waistband_right | crotch |

# |3| 4 | 5 | 6 |
# |--- | --- | --- |--- |
# | bottom_left_in | bottom_left_out | bottom_right_in | bottom_right_out |
trousers_keymap = {'waistband_left': 0,
                'waistband_right': 1,
                'crotch': 2,
                'bottom_left_in': 3,
                'bottom_left_out': 4,
                'bottom_right_in': 5,
                'bottom_right_out': 6}

inverse_trousers_keymap = {}
for k, v in trousers_keymap.items():
    inverse_trousers_keymap[v] = k

trousers_global_ind = []
for i in range(len(inverse_trousers_keymap)):
    trousers_global_ind.append(key2ind[inverse_trousers_keymap[i]]-1)


# ## Skirt 4
## Skirt
# | 0 | 1 | 2 | 3 |
# | --- | --- | --- | --- |
# | waistband_left | waistband_right | hemline_left | hemline_right |
skirt_keymap = {'waistband_left': 0,
            'waistband_right': 1,
            'hemline_left': 2,
            'hemline_right': 3}

inverse_skirt_keymap = {}
for k, v in skirt_keymap.items():
    inverse_skirt_keymap[v] = k

skirt_global_ind = []
for i in range(len(inverse_skirt_keymap)):
    skirt_global_ind.append(key2ind[inverse_skirt_keymap[i]]-1)

# ## Dress 15
## Dress
# | 0 | 1 | 2 | 3 | 4 |
# | --- | --- | --- | --- | --- |
# | neckline_left | neckline_right | shoulder_left | shoulder_right | center_front |


# | 5 | 6 | 7 | 8 | 9  |
# | --- | --- | --- | --- | --- | --- |
# | armpit_left | armpit_right |waistline_left | waistline_right | cuff_left_in|


# | 10| 11 | 12 | 13 | 14 |
# | --- | --- | --- | --- |--- |
# | cuff_left_out | cuff_right_in | cuff_right_out |hemline_left | hemline_right |
dress_keymap = {'neckline_left': 0,
            'neckline_right': 1,
            'shoulder_left': 2,
            'shoulder_right': 3,
            'center_front': 4,
            'armpit_left': 5,
            'armpit_right': 6,
            'waistline_left': 7,
            'waistline_right': 8,
            'cuff_left_in': 9,
            'cuff_left_out': 10,
            'cuff_right_in': 11,
            'cuff_right_out': 12,
            'hemline_left': 13,
            'hemline_right': 14}

inverse_dress_keymap = {}
for k, v in dress_keymap.items():
    inverse_dress_keymap[v] = k

dress_global_ind = []
for i in range(len(inverse_dress_keymap)):
    dress_global_ind.append(key2ind[inverse_dress_keymap[i]]-1)

# whick global ind is this position belongs to
class2global_ind_map = {
    '*': list(range(24)),
    'blouse': blouse_global_ind,
    'dress': dress_global_ind,
    'outwear': outwear_global_ind,
    'skirt': skirt_global_ind,
    'trousers': trousers_global_ind
}


left_right_remap = {
    '*': [1, 0, 2, 4, 3, 6, 5, 8, 7, 11, 12, 9, 10, 14, 13, 16, 15, 18, 17, 19, 22, 23, 20, 21],
    'blouse': [1, 0, 3, 2, 4, 6, 5, 8, 7, 11, 12, 9, 10],
    'outwear': [1, 0, 3, 2, 5, 4, 7, 6, 10, 11, 8, 9, 13, 12],
    'trousers': [1, 0, 2, 5, 6, 3, 4],
    'skirt': [1, 0, 3, 2],
    'dress': [1, 0, 3, 2, 4, 6, 5, 8, 7, 11, 12, 9, 10, 14, 13]
}

# left keypoint index, right keypoint index, center keypoint index
left_right_group_map = {
    '*': ([0, 3, 5, 7, 9, 10, 13, 15, 17, 20, 21],
            [1, 4, 6, 8, 11, 12, 14, 16, 18, 22, 23],
            [2, 19]),
    'blouse': ([0, 2, 5, 7, 9, 10],
                [1, 3, 6, 8, 11, 12],
                [4]),
    'outwear': ([0, 2, 4, 6, 8, 9, 12],
                [1, 3, 5, 7, 10, 11, 13],
                []),
    'trousers': ([0, 3, 4],
                [1, 5, 6],
                [2]),
    'skirt': ([0, 2],
            [1, 3],
            []),
    'dress': ([0, 2, 5, 7, 9, 10, 13],
                [1, 3, 6, 8, 11, 12, 14],
                [4])
}
# train {'blouse': 10155, 'outwear': 7734, 'dress': 7224, 'skirt': 9910, 'trousers': 9142} 220825
# test {'trousers': 1958, 'outwear': 2043, 'skirt': 1980, 'blouse': 1977, 'dress': 2038} 49980

split_size = {
            '*': {'train': 41990,
                'val': 2175,
                'test': 9996},
            'blouse': {'train': 9618,
                'val': 537,
                'test': 1977},
            'dress': {'train': 6864,
                'val': 360,
                'test': 2038},
            'outwear': {'train': 7350,
                'val': 384,
                'test': 2043},
            'skirt': {'train': 9425,
                'val': 485,
                'test': 1980},
            'trousers': {'train': 8733,
                'val': 409,
                'test': 1958},
            }

# split_size = {
#             '*': {'train': 16,
#                 'val': 16,
#                 'test': 9996},
#             'blouse': {'train': 16,
#                 'val': 16,
#                 'test': 1977},
#             'dress': {'train': 16,
#                 'val': 16,
#                 'test': 2038},
#             'outwear': {'train': 16,
#                 'val': 16,
#                 'test': 2043},
#             'skirt': {'train': 16,
#                 'val': 16,
#                 'test': 1980},
#             'trousers': {'train': 16,
#                 'val': 16,
#                 'test': 1958},
#             }
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


# Keypoints of each Category


