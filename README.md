# Yolov5 Detection Metircs Project

English | [简体中文](https://github.com/Kinsue/Yolov5-Detection-Metircs-Project/blob/main/README_CN.md)

## Project Introduction

Yolov5 Detection Metrics Project calculates the inference results of Yolov5 and obtains the corresponding mAP (mean Average Precision) metric.


## Usage method

This project aims to evaluate the given detection results in a simple and efficient manner. Please follow the steps below to start the evaluation.

1. [Environment](#Environment)
2. [Prepare groundtruth data](#prepare-groundtruth-data)
3. [Prepare detection data](#prepare-detection-data)
4. [Prepare image data](#prepare-image-data)
5. [Prepare label.txt](#prepare-labeltxt)

### Environment

```shell
# clone This Project
$ git clone https://github.com/Kinsue/Yolov5-Detection-Metircs-Project.git
```

```shell
$ conda create -n ydmp 
$ conda activate ydmp 

$ pip install -r requirements.txt 
```



### Prepare groundtruth data

- groundtruth should be a txt text file. In the file, each line should represent each detection box in the following format:
	-  `<class_id> <x_center> <y_center> <width> <hight>`
- Sample

```text
4 0.42916123424598 0.44592635546535225 0.6853541938287702 0.7677587706581618
7 0.5089091699261191 0.9338938822847201 0.020860495436766623 0.009857929834734705
6 0.7105606258148631 0.03798202377500725 0.42677096914385054 0.011597564511452595
6 0.18817905258583226 0.03798202377500725 0.19730551933941765 0.011017686285879964
```



### Prepare detection data

- In the detection folder, there should be text files with the same name as the corresponding groundtruth files. Each line in the file should represent a detection box in the following format.
	-  `<class_id> <x_center> <y_center> <width> <hight> <confidence>`

	> **Slightly different from the ground truth data, the detection data should have confidence scores appended in the last column.**
- Sample 

```txt
7 0.507611 0.933984 0.0199063 0.0101562 0.842858
6 0.186768 0.0378906 0.19555 0.0117188 0.93001
6 0.709602 0.0378906 0.423888 0.0117188 0.931442
4 0.427986 0.445703 0.685012 0.767969 0.96

```



### Prepare image data

- Please place jpg files with the same name as groundtruth in the image folder.

	> ! The current program only supports jpg files.

### Prepare label.txt

- The data should be a `.txt` file, with each line in the file representing a label.




### A sample folder structure：

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



## Calculate mAP

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