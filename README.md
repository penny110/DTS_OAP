# A Dual Teacher-Student Framework with Dual1 Perturbations for Semi-Supervised Tooth and2 Root Canal Segmentation in X-Ray Images
# Train
by Yin Pan 
### News
<05.06.2025> We released the codes;
```
### Introduction
This repository is for our paper: 'A Dual Teacher-Student Framework with Dual1 Perturbations for Semi-Supervised Tooth and2 Root Canal Segmentation in X-Ray Images'. Note that, the DTS_OAP model is named as DTS_OAP_v4_attention in our repository.
### Requirements
This experiments were conducted under consistent conditions (hardware: NVIDIA GeForce RTX 3050 GPU; software: PyTorch 1.12.1, CUDA 11.6, Python 3.8.10; random seed: 1337).
### Usage
1. Clone the repo.;
```
Please download the DTD_OAP.zip file for training, and the complete code logic can be found in the DTD_OAP.zip file.
git clone https://github.com/penny110/DTS_OAP.git
```
2. Put the data in './DTS_OAP/data';
Please send an email to the author to inquire about the data.
3. Train the model;
```
cd DTS_OAP
# e.g., for 20% labels on tooth
python ./code/train_newdata.py --dataset_name tooth --model DTS_OAP_v4_attention --labelnum 14 --gpu 0 --temperature 0.1
```
4. Test the model;
```
cd DTS_OAP
# e.g., for 20% labels on LA
python ./code/test_newdata_calculate.py --dataset_name tooth --model DTS_OAP_v4_attention --exp DTS_OAP_v4_attention --labelnum 14 --gpu 0
```
### Citation
If our DTS_OAP_v4_attention model is useful for your research, please consider citing:
### Acknowledgements:
### Questions
If any questions, feel free to contact me at '2443203471@qq.com'
