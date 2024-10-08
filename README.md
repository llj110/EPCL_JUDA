[//]: # (# Efficient Prototype Consistency in Semi-Supervised Medical Image Segmentation via Joint Uncertainty and Data Augmentation)
<div align="center">
  <h1>Efficient Prototype Consistency in Semi-Supervised Medical Image Segmentation via Joint Uncertainty and Data Augmentation</h1>
</div>

Implementation of [Efficient Prototype Consistency in Semi-Supervised Medical Image Segmentation via Joint Uncertainty and Data Augmentation](https://arxiv.org).

Recently, prototype learning has emerged in semi-supervised medical image segmentation and achieved remarkable performance. However, the scarcity of labeled data limits the expressiveness of prototypes in previous methods, potentially hindering the complete representation of prototypes for class embedding. To overcome this issue, we propose an efficient prototype consistency learning via joint uncertainty quantification and data augmentation (EPCL-JUDA) to enhance the semantic expression of prototypes based on the framework of Mean-Teacher. The concatenation of original and augmented labeled data is fed into the teacher network to generate expressive prototypes. Then, a joint uncertainty quantification method is devised to optimize pseudo-labels and generate reliable prototypes for original and augmented unlabeled data separately. High-quality global prototypes for each class are formed by fusing labeled and unlabeled prototypes, optimizing the distribution of features used in consistency learning. Notably, a prototype network is proposed to reduce a high memory requirement brought by the introduction of augmented data. Extensive experiments on Left Atrium, Pancreas-CT, Type B Aortic Dissection datasets demonstrate EPCL-JUDA's superiority over previous state-of-the-art approaches, confirming the effectiveness of our framework. 

## 💥 Updates 💥
🚩 **News (2024.06.12)** We have uploaded the code EPCL_JUDA code 🥳.

## Overview 💜
This is the official code implementation project for paper **"Efficient Prototype Consistency in Semi-Supervised Medical Image Segmentation via Joint Uncertainty and Data Augmentation"**. The code implementation refers to UPCoL(https://github.com/VivienLu/UPCoL/tree/main). Thanks very much for the contribution of UPCoL(https://github.com/VivienLu/UPCoL/tree/main) to code structure of our paper **"Efficient Prototype Consistency in Semi-Supervised Medical Image Segmentation via Joint Uncertainty and Data Augmentation"**.

## Prerequisites  💻
### Requirements and Dependencies:
Here we list our some important requirements and dependencies.
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
   
python train_JUDA.py --exp $exp --data_dir $data_dir --list_dir $list_dir --exp $exp
```

## Citation ✨

If you find EPCL-JUDA useful in your research, please cite our work:

```bibtex

@misc{ROLSSL,
      title={Efficient Prototype Consistency in Semi-Supervised Medical Image Segmentation via Joint Uncertainty and Data Augmentation}, 
      author={Lijian Li, Yuanpeng He, Chi-Man Pun},
      year={2024},
      journal={arXiv}
}

```


## Contact 🦄
If you have any questions or suggestions, feel free to contact:
- Lijian Li [(mc35305@umac.mo)](mailto:mc35305@umac.mo)
  [![Mail](https://img.shields.io/badge/Lijian_Li-0078D4?logo=microsoft-outlook&logoColor=white)](mailto:mc35305@umac.mo)
  [![Google Scholar](https://img.shields.io/badge/Lijian_Li-4285F4?logo=googlescholar&logoColor=white)](https://scholar.google.com/citations?user=Pe_tlDMAAAAJ)
  [![ResearchGate](https://img.shields.io/badge/Lijian_Li-00CCBB?logo=ResearchGate&logoColor=white)](https://www.researchgate.net/profile/Lijian-Li-2)
  
- Yuanpeng He [(heyuanpeng@stu.pku.edu.cn)](mailto:heyuanpeng@stu.pku.edu.cn)
  [![Outlook](https://img.shields.io/badge/Yuanpeng_He-0078D4?logo=microsoft-outlook&logoColor=white)](mailto:heyuanpeng@stu.pku.edu.cn)
  [![Google Scholar](https://img.shields.io/badge/Yuanpeng_He-4285F4?logo=googlescholar&logoColor=white)](https://scholar.google.com/citations?user=HaefBCQAAAAJ)
  [![ResearchGate](https://img.shields.io/badge/Yuanpeng_He-00CCBB?logo=ResearchGate&logoColor=white)](https://www.researchgate.net/profile/Yuanpeng-He)


