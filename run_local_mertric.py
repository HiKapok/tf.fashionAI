import os
import sys
import time
import numpy as np
import pandas as pd
import argparse
import math

import config as cfg

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(
    description='The Normarlized Error Mertric Calculation For FashionAI Keypoint Detection Script.')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--prediction', default='',
                    help='The path of file containing the prediction of keypoints.')
parser.add_argument('--cat', type=lambda s: s.lower() in ['True', 'true', 't', 'yes', '1'], help="whether print Normarlized Error for each catgory")
parser.add_argument('--gt', default='./stage1_testb_gt.csv',
                    help='The path of file containing the ground truth of keypoints.')

args = parser.parse_args()

def run():
    if args.prediction.strip() == '' or args.gt.strip() == '':
        parser.error('Must specify the file path of the prediction and ground truth.')

    pred_df = pd.read_csv(args.prediction, encoding='utf-8')
    gt_df = pd.read_csv(args.gt, encoding='utf-8').set_index('image_id')


    num_v = 0.
    sum_dist = 0.
    for index, row in pred_df.iterrows():
        gt = gt_df.loc[row['image_id']]
        img_cat = gt['image_category']
        gt_points = {}
        pred_points = {}

        for kp in cfg.all_keys:
            pred_kp = row[kp].strip().split('_')
            gt_kp = gt[kp].strip().split('_')
            pred_points[kp] = [int(_) for _ in pred_kp]
            gt_points[kp] = [int(_) for _ in gt_kp]

        lnorm_name, rnorm_name = cfg.normalize_point_name[img_cat]
        lnorm, rnorm = gt_points[lnorm_name][:-1], gt_points[rnorm_name][:-1]
        norm_value = math.pow(math.pow(lnorm[0] - rnorm[0], 2.) + math.pow(lnorm[1] - rnorm[1], 2.), 0.5)


        for kp in cfg.all_keys:
            if gt_points[kp][-1] == -1 or norm_value < 1e-3:
                continue
            num_v += 1.

            dist = math.pow(math.pow(pred_points[kp][0] - gt_points[kp][0], 2.) + math.pow(pred_points[kp][1] - gt_points[kp][1], 2.), 0.5)
            sum_dist += dist/norm_value

    sum_dist = sum_dist/num_v
    print(sum_dist)

def run_by_cat():
    if args.prediction.strip() == '' or args.gt.strip() == '':
        parser.error('Must specify the file path of the prediction and ground truth.')

    pred_df = pd.read_csv(args.prediction, encoding='utf-8')
    gt_df = pd.read_csv(args.gt, encoding='utf-8').set_index('image_id')

    for cat_ in cfg.CATEGORIES:
        num_v = 0.
        sum_dist = 0.
        for index, row in pred_df.iterrows():
            gt = gt_df.loc[row['image_id']]
            img_cat = gt['image_category']
            if cat_ not in img_cat:
                continue
            gt_points = {}
            pred_points = {}

            for kp in cfg.all_keys:
                pred_kp = row[kp].strip().split('_')
                gt_kp = gt[kp].strip().split('_')
                pred_points[kp] = [int(_) for _ in pred_kp]
                gt_points[kp] = [int(_) for _ in gt_kp]

            lnorm_name, rnorm_name = cfg.normalize_point_name[img_cat]
            lnorm, rnorm = gt_points[lnorm_name][:-1], gt_points[rnorm_name][:-1]
            norm_value = math.pow(math.pow(lnorm[0] - rnorm[0], 2.) + math.pow(lnorm[1] - rnorm[1], 2.), 0.5)


            for kp in cfg.all_keys:
                if gt_points[kp][-1] == -1 or norm_value < 1e-3:
                    continue
                num_v += 1.

                dist = math.pow(math.pow(pred_points[kp][0] - gt_points[kp][0], 2.) + math.pow(pred_points[kp][1] - gt_points[kp][1], 2.), 0.5)
                sum_dist += dist/norm_value

        sum_dist = sum_dist/num_v
        print('{}:'.format(cat_), sum_dist)

if __name__ == '__main__':
    if not args.cat:
        run()
    else:
        run_by_cat()
