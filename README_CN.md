# Yolov5 Detection Metircs Project

[English](https://github.com/Kinsue/Yolov5-Detection-Metircs-Project#a-sample-folder-structure) | 简体中文

## 项目介绍

Yolov5 Detection Metrics Project 项目针对 Yolov5 的推理结果进行计算, 得到对应的 mAP(mean Average Precision)指标


## 使用方法

本项目旨在以一种简单高效的方法评估给定的检测结果，请你跟随以下步骤来开始评估

1. [环境安装](#环境安装)
2. [准备 groundtruth 数据](#准备-groundtruth-数据)
3. [准备 detection 数据](#准备-detection-数据)
4. [准备图片数据](#准备图片数据)
5. [准备 label 数据](#准备-label-数据)

### 环境安装
```shell
# clone This Project
$ git clone https://github.com/Kinsue/Yolov5-Detection-Metircs-Project.git
```

```shell
$ conda create -n ydmp python=3.10
$ conda activate ydmp 

$ pip install -r requirements.txt 
```



### 准备 groundtruth 数据

- groundtruth 应为txt文本文件，在文件中，每一行应该以如下格式表示每一个检测框
	-  `<class_id> <x_center> <y_center> <width> <hight>`
- 数据样例

```text
4 0.42916123424598 0.44592635546535225 0.6853541938287702 0.7677587706581618
7 0.5089091699261191 0.9338938822847201 0.020860495436766623 0.009857929834734705
6 0.7105606258148631 0.03798202377500725 0.42677096914385054 0.011597564511452595
6 0.18817905258583226 0.03798202377500725 0.19730551933941765 0.011017686285879964
```



### 准备 detection 数据

- 在detection文件夹中，应该包含与对应 groundtruth 文件同名的文本文件。文件中每一行应该以如下格式表示每一个检测框
	-  `<class_id> <x_center> <y_center> <width> <hight> <confidence>`

	> **与groundtruth数据稍有不同，detection 数据应该在最后一列附加置信度数据**
- 数据样例

```txt
7 0.507611 0.933984 0.0199063 0.0101562 0.842858
6 0.186768 0.0378906 0.19555 0.0117188 0.93001
6 0.709602 0.0378906 0.423888 0.0117188 0.931442
4 0.427986 0.445703 0.685012 0.767969 0.96

```



### 准备图片数据

- 请在图片文件夹下放置与 groundtruth 同名的 jpg 文件

	> ! 当前程序仅支持 jpg 文件

### 准备 label 数据

- lable 数据应该为一个`.txt` 文件 , 文件中每一行为一个标签




### 一个数据文件夹结构样例：

```txt
input
├── DR
│   ├── 000011.txt
│   ├── 000012.txt
│   └── 000024.txt
├── GT
│   ├── 000011.txt
│   ├── 000012.txt
│   └── 000024.txt
├── image
│   ├── 000011.jpg
│   ├── 000012.jpg
│   └── 000024.jpg
└── label.txt
```



## 计算 mAP

```shell
$ cd {Directory of Project}
$ python val.py -det {path of det} -gt {path of groundtruth} -img {path of img} -l label.txt
```




## Citation

```text
@Article{electronics10030279,
AUTHOR = {Padilla, Rafael and Passos, Wesley L. and Dias, Thadeu L. B. and Netto, Sergio L. and da Silva, Eduardo A. B.},
TITLE = {A Comparative Analysis of Object Detection Metrics with a Companion Open-Source Toolkit},
JOURNAL = {Electronics},
VOLUME = {10},
YEAR = {2021},
NUMBER = {3},
ARTICLE-NUMBER = {279},
URL = {https://www.mdpi.com/2079-9292/10/3/279},
ISSN = {2079-9292},
DOI = {10.3390/electronics10030279}
}
```
