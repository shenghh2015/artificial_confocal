# Artificial confocal microscopy for deep label-free imaging

# Installation
```shell
git clone git@github.com:shenghh2015/artificial_confocal.git
cd artificial_confocal
```
## 1. Installing with conda: 
### see conda_install.md
## 2. Installing with pip: 
```
pip install -r requirement.txt
```
# Training
## prepare dataset that follow the example bead dataset, be free to see the example in codes/inspect_data.ipynb
```
datasets
   |——bead
   |    └——————bead-1
   |       └—————phase
   |         └——————z2.npy
   |         └——————z3.npy
   |         └——————...
   |       └—————fl
   |         └——————z2.npy
   |         └——————z3.npy
   |         └——————...
   |    └——————bead-2
   |       └—————phase
   |         └——————z2.npy
   |         └——————z3.npy
   |         └——————...
   |       └—————fl
   |         └——————z2.npy
   |         └——————z3.npy
   |         └——————...
   |    └——————...
```
## run the model with the train.py
``` shell
cd codes
python train.py --dataset bead
```
# Testing
## check the codes/inspect_trained_model.ipynb