import os
import numpy as np
from natsort import natsorted
import tensorflow as tf
import albumentations as A

class Dataset:
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        phase_dir (str): path to phase image folder
        fl_dir    (str): path to flourescent image folder
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    """
    
    def __init__(
            self, 
            dataset_dir,
            sample_file_name = 'train.txt',
            z_range = [0,250],
            scale   = 1.,
            augmentation=None, 
            preprocessing=None,
    ):
        
        self.dataset_dir  = dataset_dir
        
        all_sample_names = os.listdir(self.dataset_dir)
        self.sample_names = []
        # read sample names in [dataset_dir]/train.txt or [dataset_dir]/valid.txt
        with open(self.dataset_dir + f'/{sample_file_name}', 'r+') as f:
            for line in f.readlines():
                sample_name = line.strip()
                if sample_name in all_sample_names: 
                    self.sample_names.append(sample_name)
        
        # obtain phase image samples and the corresponding flourescent ground truth 
        self.phase_fps = []
        self.fl_fps    = []
        self.ids       = []
        
        z_set = ['z{}'.format(z) for z in range(z_range[0], z_range[1])]
        for sample_name in self.sample_names:
            phase_dir = os.path.join(self.dataset_dir, sample_name, 'phase')
            fl_dir    = os.path.join(self.dataset_dir, sample_name, 'fl')
            phase_fns = natsorted(os.listdir(phase_dir))
            fl_fns    = natsorted(os.listdir(fl_dir))
            for fn in phase_fns:
                slice_flag = fn.split('_')[-1].split('.')[0]
                if fn in fl_fns and slice_flag in z_set:
                    self.phase_fps.append(os.path.join(phase_dir, fn))
                    self.fl_fps.append(os.path.join(fl_dir, fn))
                    self.ids.append(f'{sample_name}_{fn}')
                    
        self.augmentation  = augmentation
        self.preprocessing = preprocessing
        self.scale         = scale
    
    def __getitem__(self, i):
        
        # load an training or valid image sample
        image = np.load(self.phase_fps[i]) * 255.
        mask  = np.expand_dims(np.load(self.fl_fps[i]) * self.scale , axis = -1)

        # apply augmentation
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask
        
    def __len__(self):
        return len(self.ids)
    

class Dataloder(tf.keras.utils.Sequence):
    """Load data from dataset and form batches
    
    Args:
        dataset: instance of Dataset class for image loading and preprocessing.
        batch_size: Integet number of images in batch.
        shuffle: Boolean, if `True` shuffle image indexes each epoch.
    """
    
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))

        self.on_epoch_end()

    def __getitem__(self, i):
        
        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])
        
        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]
        
        return (batch[0], batch[1])
    
    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size
    
    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)


# define augmentation process
def get_augmentation(dim, is_train = True):
    
    if is_train:
        transform = [
            A.HorizontalFlip(p=0.5),
            A.PadIfNeeded(min_height=dim, min_width=dim, always_apply=True, border_mode=0),
            A.RandomCrop(height=dim, width=dim, always_apply=True),
        ]
    else:
        transform = [
            A.PadIfNeeded(dim, dim)
        ]
        
    return A.Compose(transform)

# define preprocessing function
def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    """
    _transform = [
        A.Lambda(image=preprocessing_fn),
    ]
    return A.Compose(_transform)