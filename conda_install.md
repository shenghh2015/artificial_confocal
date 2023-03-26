conda create -n tf-eunet tensorflow-gpu=1.15 cudatoolkit=10.0 pip
conda activate tf-eunet
conda install ipython
pip install scipy sklearn matplotlib natsort jupyterlab keras_applications>=1.0.7 image-classifiers==1.0.0 efficientnet==1.0.0 segmentation-models==1.0.1 albumentations==0.3.0
pip install 'h5py==2.10.0' --force-reinstall

Reference:
https://docs.anaconda.com/anaconda/user-guide/tasks/tensorflow/
https://fmorenovr.medium.com/install-conda-and-set-up-a-tensorflow-1-15-cuda-10-0-environment-on-ubuntu-windows-2a18097e6a98