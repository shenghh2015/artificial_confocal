# Artificial confocal microscopy for deep label-free imaging

## Installation with conda
```shell
git clone git@github.com:shenghh2015/artificial_confocal.git
cd artificial_confocal
```
```
conda create -n tf-eunet tensorflow-gpu=1.15 cudatoolkit=10.0 pip
conda activate tf-eunet
conda install ipython
pip install scipy sklearn matplotlib natsort jupyterlab keras_applications>=1.0.7 image-classifiers==1.0.0 efficientnet==1.0.0 segmentation-models==1.0.1 albumentations==0.3.0
pip install 'h5py==2.10.0' --force-reinstall
```

## Dataset
### The hierarchical structure of a dataset is shown as below. A example dataset (bead) is in the artificial_confocal/dataset. The example of inspecting sample in the dataset is shown in codes/inspect_data.ipynb. The bead dataset for the paper is much more then the example one.
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
## Training
### Run the model with the train.py
``` shell
cd codes
python train.py --dataset bead
```
## Testing
### The example of testing the trained model is shown in codes/inspect_trained_model.ipynb

## Citation
### The codes and data provided here are from the following paper. Please cite the paper if you use the codes.

```
@article{chen2023artificial,
  title={Artificial confocal microscopy for deep label-free imaging},
  author={Chen, Xi and Kandel, Mikhail E and He, Shenghua and Hu, Chenfei and Lee, Young Jae and Sullivan, Kathryn and Tracy, Gregory and Chung, Hee Jung and Kong, Hyun Joon and Anastasio, Mark and Popescu, Gabriel},
  journal={Nature Photonics},
  pages={1--9},
  year={2023}
}
```

## Acknowledgement
The implementation code is built up based on [segmentatioin models](https://github.com/qubvel/segmentation_models). Great thanks for their wonderful works.