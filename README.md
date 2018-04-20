# Hourglass and CPN model in TensorFlow for 2018-FashionAI Clothes Keypoints Detection at TianChi

This repository contains codes of the re-implementent of [Stacked Hourglass Networks for Human Pose Estimation](https://arxiv.org/abs/1603.06937) and [Cascaded Pyramid Network for Multi-Person Pose Estimation](https://arxiv.org/abs/1711.07319) in TensorFlow for [FashionAI Global Challenge 2018](https://tianchi.aliyun.com/competition/introduction.htm?spm=5176.11165261.5678.1.669c2ec5y232FN&raceId=231648). The CPN(Cascaded Pyramid Network) here had two different backbones: ResNet50 or [DetNet](https://arxiv.org/abs/1804.06215). More models like [Fashion Landmark Detection in the Wild](https://arxiv.org/abs/1608.03049) will come soon.

The main goal of this competition is to detect the keypoints of the clothes image colleted from Alibaba's e-commerce platforms. There are tens of thousands images in total five categories: blouse, outwear, trousers, skirt, dress. The keypoints for each category is defined as follows.

![](demos/outline.jpg "The Keypoints for Each Category")

All the codes was writen by myself and tested under TensorFlow 1.6, Python 3.5, Ubuntu 16.04. I tried to use the latest possible TensorFlow's best practice paradigm, like [tf.estimator](https://www.tensorflow.org/api_docs/python/tf/estimator) and [tf.layers](https://www.tensorflow.org/api_docs/python/tf/layers). Almost none py_func was used in my codes to maximize the performance. Augumentations like flip, rotate, random crop, color distort were used to reduce overfit. The current performance of the model is ~0.7% in Normalized Error and got to ~50th-place in the first stage of the competition. Actually, I didn't do much experiments to find the best parameter of the models but focused on the whole framework due to time constraints. So there is still much potential to got better performance.
   
If you find it's useful to your research or competitions, any contribution or star to this repo is welcomed.

By the way, I'm looking for one computer vision related job recently. I'm very looking forward to your contact if you are interested in.

## ##
Some Detection Results:

- Cascaded Pyramid Network:
  
![](demos/cpn/blouse.jpg "blouse")
![](demos/cpn/dress.jpg "dress")
![](demos/cpn/outwear.jpg "outwear")
![](demos/cpn/skirt.jpg "skirt")
![](demos/cpn/trousers.jpg "trousers")

- Stacked Hourglass Networks:

![](demos/hg/blouse.jpg "blouse")
![](demos/hg/dress.jpg "dress")
![](demos/hg/outwear.jpg "outwear")
![](demos/hg/skirt.jpg "skirt")
![](demos/hg/trousers.jpg "trousers")

## ##
Apache License 2.0