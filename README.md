# Self-supervised pre-training with diffusion model for few-shot landmark detection in x-ray images
Official PyTorch implementation of the paper -> https://arxiv.org/abs/2407.18125

# Abstract
Deep neural networks have been extensively applied in the medical domain for various tasks, including image classification, segmentation, and landmark detection. However, their application is often hindered by data scarcity, both in terms of available annotations and images. This study introduces a novel application of denoising diffusion probabilistic models (DDPMs) to the landmark detection task, specifically addressing the challenge of limited annotated data in x-ray imaging. Our key innovation lies in leveraging DDPMs for self-supervised pre-training in landmark detection, a previously unexplored approach in this domain. This method enables accurate landmark detection with minimal annotated training data (as few as 50 images), surpassing both ImageNet supervised pre-training and traditional self-supervised techniques across three popular x-ray benchmark datasets. To our knowledge, this work represents the first application of diffusion models for self-supervised learning in landmark detection, which may offer a valuable pre-training approach in few-shot regimes, for mitigating data scarcity.


![ddpm_pipeline](https://github.com/user-attachments/assets/d58daec4-ed81-4b4e-aca0-4257e9149b5b)


# Getting Started
## Installation
Install python packages
```
pip install -r requirements.txt
```

## Preparing Datasets
Download the cephalometric ([link1](https://figshare.com/s/37ec464af8e81ae6ebbf), [link2](https://www.kaggle.com/datasets/c34a0ef0cd3cfd5c5afbdb30f8541e887171f19f196b1ad63790ca5b28c0ec93?select=cepha400)), hand [link](https://ipilab.usc.edu/research/baaweb/) and the chest [link](https://www.kaggle.com/datasets/nikhilpandey360/chest-xray-masks-and-labels) datasets.

Prepare datasets in the following directory structure.

- datasets
    - cephalo
        -  400_junior
            - *.txt
        -  400_senior
            - *.txt
        - jpg
            - *.jpg
    - hand
        - labels
            - all.csv # [download here](https://github.com/christianpayer/MedicalDataAugmentationTool-HeatmapRegression/blob/master/hand_xray/hand_xray_dataset/setup/all.csv)
        - jpg
            - *.jpg
    - chest
        - pngs
            - CHNCXR_*.png
        - labels
            - CHNCXR_*.txt # unzip [chest_labels.zip](https://github.com/MIRACLE-Center/YOLO_Universal_Anatomical_Landmark_Detection/blob/main/data/chest_labels.zip)
         

## Running Experiments
To run the experiments, follow these steps:
- Open a terminal.
- Navigate to the root directory of the repository.
- Make the launch_experiments.sh script executable using the following command:
  ```
  chmod +x launch_experiments.sh
  ```
- Run the launch_experiments.sh script. The script automates the process of setting up and running the desired experiments.
  ```
  ./launch_experiments.sh
  ```

# Download Pre-Trained models

All the pre-trained models used in the study are available at the following link:

[https://huggingface.co/Roberto98/X-rays_Self-Supervised_Landmark_Detection](https://huggingface.co/Roberto98/X-rays_Self-Supervised_Landmark_Detection)


In particular, it is possible to download: 
- Our DDPM pre-trained model at 6k, 8k, and 8k iterations respectively for the Chest, Cephalometric, and Hand dataset
- MocoV3 densenet161 model at 10k iterations for the Chest, Cephalometric, and Hand dataset
- SimClrV2 densenet161 model at 10k iterations for the Chest, Cephalometric, and Hand dataset
- Dino densenet161 model at 10k iterations for the Chest, Cephalometric, and Hand dataset


# Citation

Accepted at WACV (Winter Conference on Applications of Computer Vision) 2025.

### Bibtex
```
@InProceedings{Di_Via_2025_WACV,
    author    = {Di Via, Roberto and Odone, Francesca and Pastore, Vito Paolo},
    title     = {Self-Supervised Pre-Training with Diffusion Model for Few-Shot Landmark Detection in X-Ray Images},
    booktitle = {Proceedings of the Winter Conference on Applications of Computer Vision (WACV)},
    month     = {February},
    year      = {2025},
    pages     = {3886-3896}
}
```

### APA

```
Di Via, R., Odone, F., & Pastore, V. P. (2025). Self-supervised pre-training with diffusion model for few-shot landmark detection in x-ray images. IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) 2025. https://openaccess.thecvf.com/content/WACV2025/papers/Di_Via_Self-Supervised_Pre-Training_with_Diffusion_Model_for_Few-Shot_Landmark_Detection_in_WACV_2025_paper.pdf
```
