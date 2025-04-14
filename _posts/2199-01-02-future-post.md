---
title: 'V2E本地部署'
date: 2199-01-02
permalink: /posts/2012/08/blog-post-4/
tags:
  - cool posts
  - category1
  - category2
---

V2E的COLAB笔记本似乎由于其python版本以及其包版本的更新导致无法正常运行，所以采用虚拟机linux本地部署
报错信息:" v2e module 'numpy' has no attribute 'math'. Did you mean: 'emath'?"

首先解决这个emath的问题:在v2e.py中添加语句`import math np.math = math`目的是把np的math替换为 math以解决这个问题[参考](https://github.com/Roth-Lab/pyclone-vi/issues/39)
```python
import glob
import argparse
import importlib
import sys

import argcomplete
import cv2
import numpy as np
import os
from tempfile import TemporaryDirectory
from engineering_notation import EngNumber as eng  # only from pip
from tqdm import tqdm

import torch

import v2ecore.desktop as desktop
from v2ecore.base_synthetic_input import base_synthetic_input
from v2ecore.v2e_utils import all_images, read_image, \
    check_lowpass, v2e_quit
from v2ecore.v2e_utils import set_output_dimension
from v2ecore.v2e_utils import set_output_folder
from v2ecore.v2e_utils import ImageFolderReader
from v2ecore.v2e_args import v2e_args, write_args_info, SmartFormatter
from v2ecore.v2e_args import v2e_check_dvs_exposure_args
from v2ecore.v2e_args import NO_SLOWDOWN
from v2ecore.renderer import EventRenderer, ExposureMode
from v2ecore.slomo import SuperSloMo
from v2ecore.emulator import EventEmulator
from v2ecore.v2e_utils import inputVideoFileDialog
import logging
import time
from typing import Optional, Any

import math
np.math = math
```

想着用小一点的miniconda来弄，但是报错，然后就只能使用完全体anaconda了
```shell
bash Anaconda3-2024.10-1-Linux-x86_64.sh
 CONDA --HELP
   36  conda version
   37  source ~/.bashrc
   38  conda --help
   39  conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
   40  conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/bioconda/
   41  conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
   42  conda config --set show_channel_urls yes
   43  ls -a
```

V2E由于调用git clone 超时，只能自己下下来zip


[RuntimeError: No such operator torchvision::nms问题解决方法](https://blog.csdn.net/qq_41590635/article/details/112384718)
这个问题实际上不知道怎么解决，查看这个博客后觉得是torch版本的问题，甚至怀疑是虚拟机不能调用GPU的问题导致的代码错误，最终使用V2E说明文档中的安装对应版本的torch ，居然能conda安装！！！(官网已经把该途径下架)

[No module named 'dv_processing'](https://github.com/SensorsINI/v2e/issues/56)

[AttributeError: module 'PIL.Image' has no attribute 'ANTIALIAS'](https://blog.csdn.net/light2081/article/details/131517132)
这个的解决办法是看这个博客所说，降级Pillow包到requirements.txt里面的版本==7.1.2

[ubuntu不进入休眠状态](https://blog.csdn.net/weixin_44120025/article/details/123184263)

[VMware中Linux虚拟机设置共享文件夹与主机传输文件](https://blog.csdn.net/m0_37871461/article/details/115395589)
要是共享文件夹看不到可以试试该文件夹终端执行这行代码
```
sudo vmhgfs-fuse .host:/ /mnt/hgfs -o subtype=vmhgfs-fuse,allow_other
```

[VMware虚拟机扩容磁盘](https://blog.csdn.net/hktkfly6/article/details/123302335)

[Anaconda的卸载与安装](https://blog.csdn.net/Inochigohan/article/details/120400990)
[Anaconda安装程序向导](https://blog.csdn.net/qq_41375318/article/details/107184397)

[Anaconda环境创建](https://blog.csdn.net/mieshizhishou/article/details/140269614)


最终采用
```shell

python v2e.py -i input/tennis.mov --overwrite --timestamp_resolution=.003 --auto_timestamp_resolution=False --dvs_exposure duration 0.005 --output_folder=output/tennis --overwrite --pos_thres=.15 --neg_thres=.15 --sigma_thres=0.03 --dvs_aedat2 tennis.aedat --output_width=346 --output_height=260 --stop_time=3 --cutoff_hz=15 

```
进行验证是否正确部署，正确部署来说应该是能运行的

```shell

python v2e.py -i input/output.mp4 --output_width=640 --output_height=480


```
还有其他问题就看下面当时的的完整记录

记得每次重启机子要`conda activate v2e`

# 当时完整的部署步骤

```shell

base) doubican@DoubiCan:~/桌面$ history
    1  bash Anaconda3-2024.10-1-Linux-x86_64.sh
    2  wget Https;//mirrors.tUna.tsiNghua.edu.cn/anaconda/miniconda/miniconda-latest-linux-x86-64.sh
    3  bash Miniconda-latest-Linux-x86_64.sh
    4  enter
    5  bash Miniconda-latest-Linux-x86_64.sh
    6  source ~/.bashrc
    7  conda -help
    8  bash Miniconda-latest-Linux-x86_64.sh
    9  bash Miniconda-latest-Linux-x86.sh
   10  df -h
   11  fdisk -I
   12  fdisk -l
   13  FDISK -L
   14  fdisk -l
   15  df -h
   16  fdisk -l
   17  sudo fdisk -l
   18  root fdisk -l
   19  sudo fdisk -l
   20  sudo
   21  sudo fdisk -l
   22  fdisk /dev/sda
   23  sudo fdisk /dev/sda
   24  sudo apt-get install gparted
   25  df -h
   26  bash Anaconda3-2024.10-1-Linux-x86_64.sh
   27  source ~/.bashrc
   28  conda --help
   29  source ~/.bashrc
   30  conda version
   31  bash Anaconda3-2024.10-1-Linux-x86_64.sh
   32  -u option
   33  the -u option
   34  bash Anaconda3-2024.10-1-Linux-x86_64.sh
   35  CONDA --HELP
   36  conda version
   37  source ~/.bashrc
   38  conda --help
   39  conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
   40  conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/bioconda/
   41  conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
   42  conda config --set show_channel_urls yes
   43  ls -a
   44  conda create -n v2e python=3.10  # create a new environment
   45  conda activate v2e  # activate the environment~
   46  conda create -n v2e python=3.10
   47  conda activate v2e
   48  conda pip
   49  pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
   50  conda activate v2e
   51  pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
   52  conda activate v2e
   53  pip list
   54  git clone https://github.com/SensorsINI/v2e
   55  conda activate v2e
   56  git clone https://github.com/SensorsINI/v2e
   57  sudo apt install git
   58  git clone https://github.com/SensorsINI/v2e
   59  cd v2e
   60  sudo vim /etc/systemd/logind.conf
   61  sudo vi /etc/vmware-tools/vmware-tools.conf
   62  conda activate v2e
   63  cd v2e-master
   64  python -m pip install -e
   65  python -m pip install -e .
   66  v2e.py
   67  python -m pip install -e .
   68  v2e.py
   69  pip install dv-processing
   70  v2e.py
   71  pip install Gooey
   72  v2e.py
   73  v2e.py [--dvs128]
   74  v2e.py --dvs346
   75  pip uninstall torch
   76  pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   77  python v2e.py -i input/tennis.mov --overwrite --timestamp_resolution=.003 --auto_timestamp_resolution=False --dvs_exposure duration 0.005 --output_folder=output/tennis --overwrite --pos_thres=.15 --neg_thres=.15 --sigma_thres=0.03 --dvs_aedat2 tennis.aedat --output_width=346 --output_height=260 --stop_time=3 --cutoff_hz=15 
   78  pip3 uninstall pytorch
   79  pip uninstall pytorch
   80  pip uninstall torch
   81  pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   82  python v2e.py -i input/tennis.mov --overwrite --timestamp_resolution=.003 --auto_timestamp_resolution=False --dvs_exposure duration 0.005 --output_folder=output/tennis --overwrite --pos_thres=.15 --neg_thres=.15 --sigma_thres=0.03 --dvs_aedat2 tennis.aedat --output_width=346 --output_height=260 --stop_time=3 --cutoff_hz=15 
   83  pip install pytorch torchvision cudatoolkit=11.3 -c pytorch
   84  conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
   85  python v2e.py -i input/tennis.mov --overwrite --timestamp_resolution=.003 --auto_timestamp_resolution=False --dvs_exposure duration 0.005 --output_folder=output/tennis --overwrite --pos_thres=.15 --neg_thres=.15 --sigma_thres=0.03 --dvs_aedat2 tennis.aedat --output_width=346 --output_height=260 --stop_time=3 --cutoff_hz=15 
   86  python -m pip install -e
   87  python -m pip install -e .
   88  python v2e.py -i input/tennis.mov --overwrite --timestamp_resolution=.003 --auto_timestamp_resolution=False --dvs_exposure duration 0.005 --output_folder=output/tennis --overwrite --pos_thres=.15 --neg_thres=.15 --sigma_thres=0.03 --dvs_aedat2 tennis.aedat --output_width=346 --output_height=260 --stop_time=3 --cutoff_hz=15 
   89  pip install -r requirements.txt
   90  python v2e.py -i input/tennis.mov --overwrite --timestamp_resolution=.003 --auto_timestamp_resolution=False --dvs_exposure duration 0.005 --output_folder=output/tennis --overwrite --pos_thres=.15 --neg_thres=.15 --sigma_thres=0.03 --dvs_aedat2 tennis.aedat --output_width=346 --output_height=260 --stop_time=3 --cutoff_hz=15 
   91  pip install -r requirements.txt
   92  pip install numpy>=1.21
   93  pip install numpy==1.21
   94  python -m pip install -e .
   95  python v2e.py -i input/tennis.mov --overwrite --timestamp_resolution=.003 --auto_timestamp_resolution=False --dvs_exposure duration 0.005 --output_folder=output/tennis --overwrite --pos_thres=.15 --neg_thres=.15 --sigma_thres=0.03 --dvs_aedat2 tennis.aedat --output_width=346 --output_height=260 --stop_time=3 --cutoff_hz=15 
   96  pip install NumPy==2.1
   97  python v2e.py -i input/tennis.mov --overwrite --timestamp_resolution=.003 --auto_timestamp_resolution=False --dvs_exposure duration 0.005 --output_folder=output/tennis --overwrite --pos_thres=.15 --neg_thres=.15 --sigma_thres=0.03 --dvs_aedat2 tennis.aedat --output_width=346 --output_height=260 --stop_time=3 --cutoff_hz=15 
   98  pip install Pillow==7.1.2
   99  pip uninstall -y Pillow
  100  Y
  101  pip uninstall -y Pillow
  102  pip install Pillow==7.1.2
  103  pip install Pillow==9.5.0
  104  pip list
  105  pip install Pillow==7.1.2
  106  python v2e.py -i input/tennis.mov --overwrite --timestamp_resolution=.003 --auto_timestamp_resolution=False --dvs_exposure duration 0.005 --output_folder=output/tennis --overwrite --pos_thres=.15 --neg_thres=.15 --sigma_thres=0.03 --dvs_aedat2 tennis.aedat --output_width=346 --output_height=260 --stop_time=3 --cutoff_hz=15 
  107  python v2e.py -i input/Translating_boxes.mp4 --overwrite --timestamp_resolution=.003 --auto_timestamp_resolution=False --dvs_exposure duration 0.005 --output_folder=output/Translating_boxes --overwrite --pos_thres=.15 --neg_thres=.15 --sigma_thres=0.03 --dvs_aedat2 Translating_boxes.aedat --output_width=346 --output_height=260 --stop_time=3 --cutoff_hz=15 
  108  python v2e.py -i input/output.mp4 --overwrite --timestamp_resolution=.003 --auto_timestamp_resolution=False --dvs_exposure duration 0.005 --output_folder=output/Translating_3boxes --overwrite --pos_thres=.15 --neg_thres=.15 --sigma_thres=0.03 --dvs_aedat2 Translating_3boxes.aedat --output_width=346 --output_height=260 --stop_time=3 --cutoff_hz=15 
  109  python v2e.py -i input/output.mp4
  110  python v2e.py -i input/output.mp4 --output_width=640 --output_height=480

```

---



[新问题:如果使用`--dvs_text DVS_TEXT`参数会出现:`numpy have no attribute 'float`的错误所以需要改变包的版本](https://blog.csdn.net/qq_45934285/article/details/131120167)
```shell
pip install numpy==1.22

pip install numba==0.59.0

```

后续的补充的历史命令记录
```shell
109  python v2e.py -i input/output.mp4
  110  python v2e.py -i input/output.mp4 --output_width=640 --output_height=480
  111  history
  112  vmhgfs-fuse .host:/ /mnt/hgfs 
  113  python v2e.py -i input/Translating_boxes.mp4 --output_width=640 --output_height=480 --output_folder=output/Translating_6boxes --dvs_text DVS_TEXT
  114  HISTORY
  115  history
  116  conda activate v2e
  117  python v2e.py -i input/Translating_boxes.mp4 --output_width=640 --output_height=480 --output_folder=output/Translating_6boxes --dvs_text DVS_TEXT
  118  python v2e.py -i input/Translating_boxes.mp4 --output_width=640 --output_height=480 --output_folder=output/Translating_6boxes --overwrite --dvs_text=DVS_TEXT
  119  python v2e.py -i input/output.mp4 --overwrite --output_width=640 --output_height=480
  120  python v2e.py -i input/output.mp4 --overwrite --output_width=640 --output_height=480 --dvs_text DVS_TEXT
  121  pip list
  122  pip install numpy=1.21
  123  pip install numpy==1.21
  124  pip uninstall numpy
  125  pip install numpy==1.21
  126  pip install numpy==1.22
  127  pip install numpy==1.24
  128  python v2e.py -i input/output.mp4 --overwrite --output_width=640 --output_height=480
  129  python v2e.py -i input/output.mp4 --overwrite --output_width=640 --output_height=480 --dvs_text DVS_TEXT
  130  pip install numpy==1.21
  131  pip install numpy==1.23
  132  python v2e.py -i input/output.mp4 --overwrite --output_width=640 --output_height=480 --dvs_text DVS_TEXT
  133  pip list
  134  pip install numba==0.49.1
  135  pip install numba==0.5
  136  pip install numba==0.49
  137  pip list
  138  pip install numba==0.49.1
  139  pip install numpy==1.22
  140  pip install numba==0.49.1
  141  pip list
  142  python v2e.py -i input/output.mp4 --overwrite --output_width=640 --output_height=480 --dvs_text DVS_TEXT
  143  pip install numba==0.49.1
  144  sudo apt-get install llvm-8
  145  pip install numba==0.49.1 llvmlite==0.30.0
  146  pip install numba==0.49.1
  147  sudo apt-get install llvm-8
  148  sudo apt install llvm
  149  pip install numba==0.49.1
  150  sudo apt update
  151  sudo apt install llvm-10-dev  
  152  pip install numba==0.49.1
  153  pip install llvmlite==0.34.0 --no-cache-dir
  154  pip install llvmlite
  155  pip install numba==0.49.1
  156  pip install numba==0.51.0
  157  pip install numba==0.59.0
  158  python v2e.py -i input/output.mp4 --overwrite --output_width=640 --output_height=480 --dvs_text DVS_TEXT

```