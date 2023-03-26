'''
    train models on bead dataset
    Author: Shenghua He
    DateL   3.25.2023
'''
import os
import sys
import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
import argparse
from natsort import natsorted
import cnn_models as sm
from cnn_models import Unet

from dataset import (Dataset, Dataloder, 
                     get_augmentation, get_preprocessing)

# tensorflow pearson
def pearson(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x-mx, y-my
    r_num = K.sum(tf.multiply(xm,ym))
    r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
    r = r_num / r_den
    return r

# numpy pearson
def numpy_pearson(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = x.mean()
    my = y.mean()
    xm, ym = x-mx, y-my
    r_num = np.sum(xm * ym)
    r_den = np.sqrt(np.sum(np.square(xm)) * np.sum(np.square(ym)))
    r = r_num / r_den
    return r

# numpy PSNR
def psnr(y_true, y_pred):
    import math
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)
    mse = np.mean((y_true - y_pred)**2)
    if mse == 0: return float('inf')
    else: return 20 * math.log10(255.0 / math.sqrt(mse))

def get_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu",        type=str, default   = '0')
    parser.add_argument("--backbone",   type=str, default   = 'efficientnetb0')
    parser.add_argument("--dataset",    type=str, default   = 'bead')
    parser.add_argument("--zmax",       type=int, default   = 150)
    parser.add_argument("--epoch",      type=int, default   = 100)
    parser.add_argument("--scale",      type=float, default = 100.) # scale pixel value in ground truth up to 100 
    parser.add_argument("--batch_size", type=int, default   = 32)
    parser.add_argument("--lr",         type=float, default = 5e-4)
    parser.add_argument("--decay",      type=float, default = 0.8)
    args = parser.parse_args()

    print(args)
    return

def main():
    


    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    dataset_name = args.dataset
    root_dir     = os.path.abspath('../')
    dataset_dir  = os.path.join(root_dir, f'dataset/{dataset_name}')

    train_dim    = 128   # train image resolution: train_dim x train_dim
    valid_dim    = 128   # valid image resolution: valid_dim x valid_dim

    # preprocessing configation
    preprocess_input = sm.get_preprocessing(args.backbone)

    # train dataset loader
    train_dataset    = Dataset(dataset_dir, 'train.txt',
                            z_range       = [0, 255],    # the max range of slice indices z direction
                            scale         = args.scale,
                            augmentation  = get_augmentation(dim = train_dim, 
                                                            is_train = True),
                            preprocessing = get_preprocessing(preprocess_input))
    train_dataloader = Dataloder(train_dataset, batch_size=args.batch_size, shuffle=True)

    # valid dataset loader
    valid_dataset    = Dataset(dataset_dir, 'valid.txt',
                                z_range       = [0, 255],   # the max range of slice indices in z direction
                                scale         = args.scale,
                                augmentation  = get_augmentation(dim = valid_dim, 
                                                                is_train = False),
                                preprocessing = get_preprocessing(preprocess_input))
    valid_dataloader = Dataloder(valid_dataset, batch_size=1, shuffle=False)

    # E-Unet
    output_channels = 1 if len(train_dataloader[0][1].shape) == 2 else train_dataloader[0][1].shape[2] # output channels
    model    = Unet(args.backbone, encoder_weights='imagenet', classes=1, activation='relu')           # model: U-net + EfficientNet
    optim    = tf.keras.optimizers.Adam(args.lr)                                                       # optimizer
    loss     = tf.keras.losses.MSE                                                                     # loss function
    metrics = [sm.metrics.PSNR(max_val=args.scale), pearson]                                           # metrics
    model.compile(optim, loss, metrics)                                                                # training set up

    # training at image level
    model_name = f'{dataset_name}_dim-{args.dim}_{args.backbone}'
    model_dir  = os.path.join(root_dir, f'results/{model_name}')
    os.makedirs(model_dir, exist_ok = True)
    callbacks = [
            tf.keras.callbacks.ModelCheckpoint(model_dir+'/model.h5', 
                                            monitor='val_pearson', save_weights_only=True, 
                                            save_best_only=True, mode='max'),
            tf.keras.callbacks.ReduceLROnPlateau(factor=args.decay),
    ]

    history = model.fit_generator(
        train_dataloader,
        steps_per_epoch  = len(train_dataloader), 
        epochs           = args.epoch, 
        callbacks        = callbacks, 
        validation_data  = valid_dataloader, 
        validation_steps = len(valid_dataloader),
    )

    # validation at stack level
    model_weights = natsorted([weight_file for weight_file in os.listdir(model_dir) if 'model' in weight_file and weight_file.endswith('.h5')])
    model.load_weights(model_dir + '/' + model_weights[-1])

    gt_fls = [val_sample[1] for val_sample in valid_dataloader]
    gt_fls = np.concatenate(gt_fls).squeeze() / args.scale * 255

    pr_fls = model.predict(valid_dataloader)
    pr_fls = pr_fls.squeeze() / args.scale * 255.

    # compute PSNR
    psnr_val   = psnr(gt_fls, pr_fls)
    # compute personr's coefficient
    pearsonr_val = numpy_pearson(gt_fls, pr_fls)
    print('psnr: {:.2f}, personr: {:.2f}'.format(psnr_val, pearsonr_val))

if __name__ == '__main__':
    main()