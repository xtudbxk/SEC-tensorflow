#SEC-tensorflow version

### Introduction

This is a project which just move the [SEC-caffe](https://github.com/kolesman/SEC) to SEC-tensorflow. The SEC is referring to the approach for weakly-supervised semantic segmentation in the paper ["seed, expand and constrain: three principles for weakly-supervised image segmentation"](http://pub.ist.ac.at/~akolesnikov/files/ECCV2016/main.pdf). And here, I just use the tensorflow to implement the approach with the help of the [SEC-caffe](https://github.com/kolesman/SEC) project.

### Citing this repository

If you find this code useful in your research, please consider citing them:

> @inproceedings{kolesnikov2016seed,  
>
> ​    title={Seed, Expand and Constrain: Three Principles for Weakly-Supervised Image Segmentation},
>
> ​    author={Kolesnikov, Alexander and Lampert, Christoph H.},  
>
> ​    booktitle={European Conference on Computer Vision ({ECCV})},  
>
> ​    year={2016},  
>
> ​    organization={Springer}
>
> }

### Preparation

for using this code, you have to do something else:

##### 1. Install pydensecrf

For using the densecrf in python, we turn to the project [pydensecrf](https://github.com/lucasb-eyer/pydensecrf). And you just using the following code to install it.

> pip install pydensecrf

##### 2. Download the data and model

1. for pascal data, please referring to its [official website](http://host.robots.ox.ac.uk/pascal/VOC/). Just download it and extract in the data/ .
2. for localization_cues.pickle, please referring to [SEC-caffe](https://github.com/kolesman/SEC). And download it and extract in the data/ .
3. for the init.npy, I upload a converted file in xxx, just download it and put it in the model/ . And those weights in the file is exactly the same with the vgg16_20M.caffemodel in   [SEC-caffe](https://github.com/kolesman/SEC).

For more details, you can referring to the correspond code files or leave a message in the issue.

### Training

then, you just input the following sentence to train it.

> python SEC.py <gpu_id>

### Result

After the training, we could reach a miou of 0.487 while it is 0.517 in the paper. Maybe there is some details I don't notice, for the time being at least, I didn't go deeper in the experiment.
