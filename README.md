
# Introduction
This repo contains the PyTorch implementation of our paper [Anatomical Structure Few-Shot Detection Utilizing Enhanced Human Anatomy Knowledge in Ultrasound Images].


# Contributions
We propose a novel few-shot medical object detection method in ul trasound images called TRR-CCM.

1. CCM to capture both long-range and short range dependencies of multiple anatomical structures while retaining crucial channel information.

2. TRR that encodes human anatomy knowledge as graph relations utilizing graph convolution learning the spatial topological relationships.


# Installation

## 1. Requirements
causal-conv1d == 1.1.1  
detectron2 == 0.3  
python == 3.10.15  
mamba-ssm == 1.1.1   
pytorch-cuda == 11.8   
torch-geometric == 1.5.0    
torch-scatter == 2.1.2                      
torch-sparse == 0.6.18   

## 2. Build project
According to D2DET(https://github.com/er-muyue/DeFRCN), run  

    python setup.py install

## 3. Prepare Data and Weights
Same as DeFRCN (https://github.com/er-muyue/DeFRCN), for the dataset, we use the PASCAL VOC format.

## 4. Training and Evaluation
### To train model  
    bash run.sh

# Acknowledgement
We sincerely appreciate these precious repositories [DeFRCN](https://github.com/er-muyue/DeFRCN) and [detectron2](https://github.com/facebookresearch/detectron2).