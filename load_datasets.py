'''
Data loading functions.
'''
from os import listdir

import numpy as np
import torchvision as tv
from PIL import Image
from wilds import get_dataset



def load_cifar10c(root_dir='data', corruption_type='fog', severity=1):
    '''
    Load CIFAR-10-C dataset.
    '''
    cifar10 = tv.datasets.CIFAR10(root=f'{root_dir}/CIFAR10', train=False, download=True)
    data_clean = cifar10.data.reshape((10000, 32*32*3))
    data_corrupt = np.load(f'{root_dir}/CIFAR10-C/{corruption_type}.npy')\
                    .reshape((50000, 32*32*3))[10000*(severity-1):10000*severity, :]
    labels = np.load(f'{root_dir}/CIFAR10-C/labels.npy')[10000*(severity-1):10000*severity]

    x_s = data_clean[(labels == 1) | (labels == 2), :].astype(float)
    y_s = labels[(labels == 1) | (labels == 2)].astype(float) - 1
    x_t = data_corrupt[(labels == 1) | (labels == 2), :][1000:, :].astype(float)
    y_t = y_s[1000:]

    return x_s.T, y_s.reshape((len(y_s), 1)), x_t.T, y_t.reshape((len(y_t), 1))


def load_tinyimagenetc(root_dir='data', corruption_type='fog', severity=1):
    '''
    Load TinyImageNet-C dataset.
    '''
    classes = [f for f in listdir(f'{root_dir}/TinyImageNet/train/')]
    classes_1 = classes[:10]
    classes_2 = classes[10:20]

    # Load source images from super-class 1
    photos_1_s = np.zeros((0, 64, 64, 3))
    for label in classes_1:
        root_dir_name = f'{root_dir}/TinyImageNet/train/{label}/images/'
        photo_names = listdir(root_dir_name)
        photos_path = [root_dir_name + path for path in photo_names]
        photo_list = []
        for x in [np.array(Image.open(fname)) for fname in photos_path]:
            if np.shape(x) == (64, 64, 3):
                photo_list.append(x)
        photos_1_s = np.concatenate((photos_1_s, np.array(photo_list)), axis=0)

    # Load source images from super-class 2
    photos_2_s = np.zeros((0, 64, 64, 3))
    for label in classes_2:
        root_dir_name = f'{root_dir}/TinyImageNet/train/{label}/images/'
        photo_names = listdir(root_dir_name)
        photos_path = [root_dir_name + path for path in photo_names]
        photo_list = []
        for x in [np.array(Image.open(fname)) for fname in photos_path]:
            if np.shape(x) == (64, 64, 3):
                photo_list.append(x)
        photos_2_s = np.concatenate((photos_2_s, np.array(photo_list)), axis=0)

    # Load target images from class 1
    photos_1_t = np.zeros((0,64,64,3))
    for label in classes_1:
        root_dir_name = f'{root_dir}Tiny-ImageNet-C/{corruption_type}/{severity}/{label}/'
        photo_names = listdir(root_dir_name)
        photos_path = [root_dir_name + path for path in photo_names]
        photo_list = []
        for x in [np.array(Image.open(fname)) for fname in photos_path]:
            if np.shape(x) == (64, 64, 3):
                photo_list.append(x) 
        photos_1_t = np.concatenate((photos_1_t, np.array(photo_list)), axis=0)

    # Load target images from class 2
    photos_2_t = np.zeros((0,64,64,3))
    for label in classes_2:
        root_dir_name = f'{root_dir}Tiny-ImageNet-C/{corruption_type}/{severity}/{label}/'
        photo_names = listdir(root_dir_name)
        photos_path = [root_dir_name + path for path in photo_names]
        photo_list = []
        for x in [np.array(Image.open(fname)) for fname in photos_path]:
            if np.shape(x) == (64, 64, 3):
                photo_list.append(x) 
        photos_2_t = np.concatenate((photos_2_t,np.array(photo_list)), axis=0)

    # downscale
    photos_1_s = photos_1_s[:, ::2, ::2 ,:]
    photos_2_s = photos_2_s[:, ::2, ::2 ,:]
    photos_1_t = photos_1_t[:, ::2, ::2 ,:]
    photos_2_t = photos_2_t[:, ::2, ::2 ,:]

    # reshape
    photos_1_s = photos_1_s.reshape((np.shape(photos_1_s))[0], 32*32*3)
    photos_2_s = photos_2_s.reshape((np.shape(photos_2_s))[0], 32*32*3)
    photos_1_t = photos_1_t.reshape((np.shape(photos_1_t))[0], 32*32*3)
    photos_2_t = photos_2_t.reshape((np.shape(photos_2_t))[0], 32*32*3)
  
    # stack
    x_s = np.vstack((photos_1_s[:1000], photos_2_s[:1000])).T
    x_t = np.vstack((photos_1_t, photos_2_t)).T
    y_s = np.vstack((np.zeros((1000,1)), np.ones((1000,1))))
    y_t = np.vstack((np.zeros((500,1)), np.ones((500,1))))
    
    return x_s, y_s, x_t, y_t   

def load_camelyon17(root_dir='data'):
    '''
    Load Camelyon17 dataset.
    '''
    data = get_dataset(dataset='camelyon17', root_root_dir=root_dir)
    x_s = []
    y_s = []
    x_t = []
    y_t = []

    for i, _ in enumerate(data._y_array):
        if data._metadata_df['center'][i] == 0:
            x_s.append(np.asarray(data.get_input(i))[32:64, 32:64, :])
            y_s.append(data._y_array[i])
        if data._metadata_df['center'][i] == 2:
            x_t.append(np.asarray(data.get_input(i))[32:64, 32:64, :])
            y_t.append(data._y_array[i])
    
    x_s = np.stack(x_s, axis=0).reshape(len(x_s), 32*32*3)
    y_s = np.array(y_s).reshape(len(x_s), 1)
    x_t = np.stack(x_t, axis=0).reshape(len(x_t), 32*32*3)
    y_t = np.array(y_t).reshape(len(x_t), 1)

    return x_s.T, y_s, x_t.T, y_t
