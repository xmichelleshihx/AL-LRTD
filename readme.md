# LRTD: Long-Range Temporal Dependency based Active Learning for Surgical Workflow Recognition
Xueying Shi, Yueming Jin, Qi Dou, and Pheng-Ann Heng
## Introduction
<a href="http://fvcproductions.com"><img src="https://avatars1.githubusercontent.com/u/4284691?v=3&s=200" title="FVCproductions" alt="FVCproductions"></a>
<!-- [![FVCproductions](https://avatars1.githubusercontent.com/u/4284691?v=3&s=200)](http://fvcproductions.com) -->
The LRTD repository contains the codes of our LRTD paper. We validate our approach on a large surgical video dataset [Cholec80](http://camma.u-strasbg.fr/datasets) by performing surgical workflow recognition task. By using our LRTD based selection strategy, we can outperform other state-ofthe-art active learning methods who only consider neighbor-frame information. Using only up to **50%** of samples, our approach can exceed the performance of full-data training
## Requirements
- python 3.6.9
- torch 0.4.1
## Usage

1.  download data from [Cholec80](http://camma.u-strasbg.fr/datasets) and then split the data into 1fps using [ffmpeg](https://www.johnvansickle.com/ffmpeg/) to split the videos to image frames. 

2.  sh split_video_to_image.sh 

3.  Each time we first select data for training, each time select 10% data, so first time we use 10% data to train, next 20,...until 50% data. The selected data is stored in nonlocalselect_txt folder.

3.  run train_nolocalselect_ResNetLSTM.sh for training of ResLSTM backbone.

4.  run train_nolocalselect_ResNetLSTM_nolocalFT.sh for training of ResLSTM-Nonlocal backbone.


## Citation
If the code is helpful for your research, please cite our paper.
```
@inproceedings{shi2020lrtd,
title={LRTD: Long-Range Temporal Dependency based Active Learning for Surgical Workflow Recognition},
author={Xueying Shi, Yueming Jin, Qi Dou, and Pheng-Ann Heng},
year={2020},
booktitle={International Conference on Information Processing in Computer-Assisted Interventions (IPCAI)},
publisher={Springer}
}
```
