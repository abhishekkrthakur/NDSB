# Matt's Solution

import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import LabelBinarizer

import pandas as pd



import glob
import os


classes = np.asarray([d.split('/')[-1] for d in glob.glob('../data/train/*')])
def load_filenames(base_dir, class_name):
    #print [f.split('/')[3] for f in glob.glob(os.path.join(base_dir, class_name, '*.jpg'))]
    return [f.split('/')[-1] for f in glob.glob(os.path.join(base_dir, class_name, '*.jpg'))]

def count_files(set_name, class_name):
    return len(load_filenames(set_name, class_name))

num_files = sum(count_files('../data/train', c) for c in classes)
num_classes = len(classes)
print classes
print 'Number of classes:', num_classes
print 'Number of images:', num_files


IMG_SIZE = 96
IMG_DIM = IMG_SIZE, IMG_SIZE

from skimage.filter import threshold_adaptive
from skimage.exposure import equalize_adapthist
from skimage.restoration import denoise_tv_chambolle, denoise_bilateral
from skimage.filter import sobel
#from skimage.io import imread
import cv2
from skimage.transform import resize
from skimage.filter.rank import median
from skimage.morphology import disk

def pre_process(img):
    #img = denoise_bilateral(img, sigma_range=0.1)
    #img = sobel(img)
    #img = median(img, disk(1))
    img = resize(img, IMG_DIM)
    return (1.0 - img).astype(np.float32)

def load_train_data():
    set_name = '../data/train'
    for class_name in classes:
        files_in_class = load_filenames(set_name, class_name)
        for f in files_in_class:
            path = os.path.join(set_name, class_name, f)
            img = cv2.imread(path)
            yield class_name, pre_process(img)
           
def load_train():
    train = []
    train_label = []
    for i, (label, img) in enumerate(load_train_data()):
        train.append(img)
        train_label.append(label)
        if i % 2000 == 0:
            print i,

    train = np.asarray(train)
    train_label = np.asarray(train_label)
    
    return train, train_label

train, train_label = load_train()

idx = np.arange(len(train))
np.random.shuffle(idx)
X = (train.reshape(-1, 1, IMG_SIZE, IMG_SIZE)).astype(np.float32)[idx]
y = train_label[idx]

print X.shape, X.dtype, X.max(), X.min(), y



