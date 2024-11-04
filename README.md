# Self-supervised pre-training with diffusion model for few-shot landmark detection in x-ray images

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
- Our DDPM pre-trained models
    - chest   [download link]()
    - cephalo [download link]()
    - hand    [download link]()
      
- MocoV3 pre-trained models
    - chest   [download link]()
    - cephalo [download link]()
    - hand    [download link]()

- SimClr2 pre-trained models
    - chest   [download link]()
    - cephalo [download link]()
    - hand    [download link]()
 
- Dino pre-trained models
    - chest   [download link]()
    - cephalo [download link]()
    - hand    [download link]()

# Citation

Accepted at WACV (Winter Conference on Applications of Computer Vision) 2025.

### Bibtex

```
@article{DiVia2024,
  author = {Di Via, R. and Odone, F. and Pastore, V. P.},
  title = {Self-supervised pre-training with diffusion model for few-shot landmark detection in x-ray images},
  year = {2024},
  journal = {arXiv},
  volume = {2407.18125},
  url = {https://arxiv.org/abs/2407.18125},
  note = {Submitted on 25 Jul 2024 (v1), last revised 29 Oct 2024 (this version, v2)}
}
```

### APA

```
Di Via, R., Odone, F., & Pastore, V. P. (2024). Self-supervised pre-training with diffusion model for few-shot landmark detection in x-ray images. ArXiv. https://arxiv.org/abs/2407.18125
```
