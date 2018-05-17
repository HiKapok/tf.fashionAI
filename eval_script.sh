#! /bin/bash

# export CUDA_VISIBLE_DEVICES='0'
# source /home/kapok/pyenv35/bin/activate
# cd /media/rs/0E06CD1706CD0127/Kapok/Chi/fashionAI/Codes
python eval_all_cpn_onepass.py --run_on_cloud=False --backbone=seresnext50_cpn
python eval_all_cpn_onepass.py --run_on_cloud=False --backbone=detnext50_cpn
python eval_all_cpn_onepass.py --run_on_cloud=False --backbone=large_seresnext_cpn --train_image_size=512 --heatmap_size=128
python eval_all_cpn_onepass.py --run_on_cloud=False --backbone=large_detnext_cpn --train_image_size=512 --heatmap_size=128

# for training
python train_senet_cpn_onebyone.py --run_on_cloud=False
python train_detxt_cpn_onebyone.py --run_on_cloud=False
python train_large_xt_cpn_onebyone.py --run_on_cloud=False --backbone=detxt
python train_large_xt_cpn_onebyone.py --run_on_cloud=False --backbone=sext

