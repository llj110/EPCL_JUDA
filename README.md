# EPCL_JUDA

[//]: # (# Efficient Prototype Consistency in Semi-Supervised Medical Image Segmentation via Joint Uncertainty and Data Augmentation)
<div align="center">
  <h1>Efficient Prototype Consistency in Semi-Supervised Medical Image Segmentation via Joint Uncertainty and Data Augmentation</h1>
</div>

Implementation of [Towards Realistic Long-tailed Semi-supervised Learning in an Open World](https://arxiv.org).

Recently, prototype learning has emerged in semi-supervised medical image segmentation and achieved remarkable performance. However, the scarcity of labeled data limits the expressiveness of prototypes in previous methods, potentially hindering the complete representation of prototypes for class embedding. To overcome this issue, we propose an efficient prototype consistency learning via joint uncertainty quantification and data augmentation (EPCL-JUDA) to enhance the semantic expression of prototypes based on the framework of Mean-Teacher. The concatenation of original and augmented labeled data is fed into the teacher network to generate expressive prototypes. Then, a joint uncertainty quantification method is devised to optimize pseudo-labels and generate reliable prototypes for original and augmented unlabeled data separately. High-quality global prototypes for each class are formed by fusing labeled and unlabeled prototypes, optimizing the distribution of features used in consistency learning. Notably, a prototype network is proposed to reduce a high memory requirement brought by the introduction of augmented data. Extensive experiments on Left Atrium, Pancreas-CT, Type B Aortic Dissection datasets demonstrate EPCL-JUDA's superiority over previous state-of-the-art approaches, confirming the effectiveness of our framework. 

## 💥 Updates 💥
🚩 **News (2024.06.12)** We have uploaded the code EPCL_JUDA code 🥳.

## Overview 💜
This is the official code implementation project for paper **"Towards Realistic Long-tailed Semi-supervised Learning in an Open World"**. The code implementation refers
to [<img src="https://img.shields.io/badge/ECCV2022-a?style=for-the-badge&logo=github&logoColor=%23121011&label=OpenLDN&color=%23121011" height="20">](https://github.com/nayeemrizve/OpenLDN)
and [<img src="https://img.shields.io/badge/CVPR2023-a?style=for-the-badge&logo=github&logoColor=%23121011&label=ACR&color=%23121011" height="20">](https://github.com/Gank0078/ACR)
. Thanks very much
for the contribution of [<img src="https://img.shields.io/badge/ECCV2022-a?style=for-the-badge&logo=github&logoColor=%23121011&label=OpenLDN&color=%23121011" height="20">](https://github.com/nayeemrizve/OpenLDN) and [<img src="https://img.shields.io/badge/CVPR2023-a?style=for-the-badge&logo=github&logoColor=%23121011&label=ACR&color=%23121011" height="20">](https://github.com/Gank0078/ACR)
 to code structure of our paper **"Towards Realistic Long-tailed Semi-supervised Learning in an Open World"**.

## Prerequisites  💻
### Requirements and Dependencies:
Here we list our some important requirements and dependencies (Details can be found in [environment.yml](environment.yml)).
 - Linux: Ubuntu 22.04 LTS
 - GPU: RTX 4090
 - CUDA: 12.3
 - Python: 3.10
 - PyTorch: 2.1.2

### Dataset Acquisition：
* Pancreas dataset: https://wiki.cancerimagingarchive.net/display/Public/Pancreas-CT
* Left atrium dataset: http://atriaseg2018.cardiacatlas.org 
* Type B Aorta Dissection dataset: https://github.com/XiaoweiXu/Dataset_Type-B-Aortic-Dissection     

**Preprocess**: refer to the image pre-processing method in  [SASSNet](https://github.com/kleinzcy/SASSnet), [CoraNet,](https://github.com/koncle/CoraNet) and [FUSSNet](https://github.com/grant-jpg/FUSSNet) for the Pancreas dataset and Left atrium dataset. The `preprocess` folder contains the necessary code to preprocess the pancreas and TBAD dataset. It is recommended to run `pancreas_preprocess.py` and `TBAD_preprocess.py` first to preprocess the data while using the raw dataset.

**Dataset split**: The `data_lists` folder contains the information about the train-test split for all three datasets.

## Training 🚀
```shell
# LA
exp='LA'
data_dir='../../../Datasets/LA_dataset'
list_dir='../datalist/LA'
   
python train.py --exp $exp --data_dir $data_dir --list_dir $list_dir --exp $exp
```

## Citation ✨

If you find EPCL-JUDA useful in your research, please cite our work:

```bibtex

@misc{ROLSSL,
      title={Towards Realistic Long-tailed Semi-supervised Learning in an Open World}, 
      author={Yuanpeng He, Lijian Li, Tianxiang Zhan, Chi-Man Pun, Wenpin Jiao and Zhi Jin},
      year={2024},
      journal={arXiv}
}

```


## Contact 🦄

If you have any questions or suggestions, feel free to contact:

- Yuanpeng He [(heyuanpeng@stu.pku.edu.cn)](mailto:heyuanpeng@stu.pku.edu.cn)
  [![Outlook](https://img.shields.io/badge/Yuanpeng_He-0078D4?logo=microsoft-outlook&logoColor=white)](mailto:heyuanpeng@stu.pku.edu.cn)
  [![Google Scholar](https://img.shields.io/badge/Yuanpeng_He-4285F4?logo=googlescholar&logoColor=white)](https://scholar.google.com/citations?user=HaefBCQAAAAJ)
  [![ResearchGate](https://img.shields.io/badge/Yuanpeng_He-00CCBB?logo=ResearchGate&logoColor=white)](https://www.researchgate.net/profile/Yuanpeng-He)

- Lijian Li [(mc35305@umac.mo)](mailto:mc35305@umac.mo)
  [![Mail](https://img.shields.io/badge/Lijian_Li-0078D4?logo=microsoft-outlook&logoColor=white)](mailto:mc35305@umac.mo)
  [![Google Scholar](https://img.shields.io/badge/Lijian_Li-4285F4?logo=googlescholar&logoColor=white)](https://scholar.google.com/citations?user=Pe_tlDMAAAAJ)
  [![ResearchGate](https://img.shields.io/badge/Lijian_Li-00CCBB?logo=ResearchGate&logoColor=white)](https://www.researchgate.net/profile/Lijian-Li-2)
