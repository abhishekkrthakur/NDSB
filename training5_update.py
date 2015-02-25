import theano
import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import LabelBinarizer
from lasagne import layers
from lasagne.updates import nesterov_momentum
from lasagne.nonlinearities import softmax
from nolearn.lasagne import NeuralNet
Conv2DLayer = layers.cuda_convnet.Conv2DCCLayer
MaxPool2DLayer = layers.cuda_convnet.MaxPool2DCCLayer
import pandas as pd
import cPickle as pickle
import lasagne
import sys

#### Always specify and check these
sys.setrecursionlimit(100000)
IMG_SIZE = 96
IMG_DIM = IMG_SIZE, IMG_SIZE
DATABASE_FOLDER = '/upb/departments/pc2/scratch/thakur/'
NETWORK = 'network5.pickle'
SUBMISSION_FILE = 'submission_5.csv'

DUMP_LABEL = 'ndsb_label_dump'+ str(IMG_SIZE) +'.pkl'
DUMP_TRAIN = 'ndsb_training_data_dump'+ str(IMG_SIZE) +'.pkl'
DUMP_TEST = 'ndsb_test_data_dump'+ str(IMG_SIZE) +'.pkl'

LOAD_DUMP = False
TRAIN = False
TEST = True
UPDATE = True

sample = pd.read_csv(DATABASE_FOLDER + 'sampleSubmission.csv')
columns = list(sample.columns)[1:]

def float32(k):
    return np.cast['float32'](k)

class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)

class EarlyStopping(object):
    def __init__(self, patience=100):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_epoch = train_history[-1]['epoch']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = [w.get_value() for w in nn.get_all_params()]
        elif self.best_valid_epoch + self.patience < current_epoch:
            print("Early stopping.")
            print("Best valid loss was {:.6f} at epoch {}.".format(
                self.best_valid, self.best_valid_epoch))
            nn.load_weights_from(self.best_weights)
            raise StopIteration()

import glob
import os

classes = np.asarray([d.split('/')[-1] for d in glob.glob(DATABASE_FOLDER + 'train/*')])
def load_filenames(base_dir, class_name):
    return [f.split('/')[-1] for f in glob.glob(os.path.join(base_dir, class_name, '*.jpg'))]

def count_files(set_name, class_name):
    return len(load_filenames(set_name, class_name))

num_files = sum(count_files(DATABASE_FOLDER + 'train', c) for c in classes)
num_classes = len(classes)

print 'Number of classes:', num_classes
print 'Number of images:', num_files

from skimage.filter import threshold_adaptive
from skimage.exposure import equalize_adapthist
from skimage.restoration import denoise_tv_chambolle, denoise_bilateral
from skimage.filter import sobel
from skimage.io import imread
from skimage.transform import resize
from skimage.filter.rank import median
from skimage.morphology import disk

def pre_process(img):
    img = resize(img, IMG_DIM)
    return (1-img).astype(np.float32)

def load_train_data():
    set_name = DATABASE_FOLDER + 'train'
    for class_name in classes:
        files_in_class = load_filenames(set_name, class_name)
        for f in files_in_class:
            path = os.path.join(set_name, class_name, f)
            img = imread(path)
            yield class_name, pre_process(img)
           
def load_train():
    train = []
    train_label = []
    for i, (label, img) in enumerate(load_train_data()):
        train.append(img)
        train_label.append(label)
        if i % 2000 == 0:
            print i

    train = np.asarray(train)
    train_label = np.asarray(train_label)
    
    return train, train_label

if TRAIN:
    if LOAD_DUMP:
        train = pickle.load(open(DATABASE_FOLDER + DUMP_TRAIN, 'rb'))
        train_label = pickle.load(open(DATABASE_FOLDER + DUMP_LABEL, 'rb'))
    else:
        train, train_label = load_train()
        pickle.dump(train, open(DATABASE_FOLDER + DUMP_TRAIN, 'wb'), -1)
        pickle.dump(train_label, open(DATABASE_FOLDER + DUMP_LABEL, 'wb'), -1)

    idx = np.arange(len(train))
    np.random.shuffle(idx)
    X = (train.reshape(-1, 1, IMG_SIZE, IMG_SIZE)).astype(np.float32)[idx]
    y = train_label[idx]
    print X.shape, X.dtype, X.max(), X.min(), y

from os.path import basename

def load_test_data():
    for f in glob.glob(DATABASE_FOLDER + 'test/*.jpg'):
        img = imread(f)
        yield pre_process(img)

def load_test():
    data = np.zeros((len(sample), IMG_SIZE*IMG_SIZE))
    for i, img in enumerate(load_test_data()):
        data[i,:] = img.astype(np.float32).reshape(-1)
        if i % 1000 == 0:
            print i
    return data

def test_ids():
    ids = []
    for f in glob.glob(DATABASE_FOLDER + 'test/*.jpg'):
        idx = basename(f).split('.jpg')[0]
        ids.append(idx)
    return ids

from random import random
import skimage.transform
def rotate(img, angle):
    return skimage.transform.rotate(img, angle, resize=False)


from nolearn.lasagne import BatchIterator

class FlipBatchIterator(BatchIterator):
    def transform(self, Xb, yb):
        Xb = Xb.copy()
        Xb, yb = super(FlipBatchIterator, self).transform(Xb, yb)
        bs = Xb.shape[0]
        indices = np.random.choice(bs, bs / 2, replace=False)
        Xb[indices] = Xb[indices, :, :, ::-1]
        indices = np.random.choice(bs, bs / 2, replace=False)
        Xb[indices] = Xb[indices, :, ::-1, :]
        angles = [45, 90, 135, 180, 225, 270, 315]
        for i in range(bs):
            Xb[i][0] = rotate(Xb[i][0], np.random.choice(angles))
        return Xb, yb


net2 = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv1', Conv2DLayer),
        ('pool1', MaxPool2DLayer),
        ('dropout1', layers.DropoutLayer),

        ('conv2', Conv2DLayer),
        ('pool2', MaxPool2DLayer),
        ('dropout2', layers.DropoutLayer),

        #('conv3', Conv2DLayer),
        #('pool3', MaxPool2DLayer),
        #('dropout3', layers.DropoutLayer),

        ('conv4', Conv2DLayer),
        ('pool4', MaxPool2DLayer),
        ('dropout4', layers.DropoutLayer),

        # ('conv5', Conv2DLayer),
        # ('pool5', MaxPool2DLayer),
        # ('dropout5', layers.DropoutLayer),

        # ('conv6', Conv2DLayer),
        # ('pool6', MaxPool2DLayer),
        # ('dropout6', layers.DropoutLayer),

        # ('conv7', Conv2DLayer),
        # #('pool7', MaxPool2DLayer),
        # ('dropout7', layers.DropoutLayer),

        # ('conv8', Conv2DLayer),
        # ('pool8', MaxPool2DLayer),
        # ('dropout8', layers.DropoutLayer),

        ('hidden1', layers.DenseLayer),
        ('dropout1h', layers.DropoutLayer),
        ('hidden2', layers.DenseLayer),
        ('dropout2h', layers.DropoutLayer),

        ('output', layers.DenseLayer),
        ],
    input_shape=(None, 1, IMG_SIZE, IMG_SIZE),

    conv1_num_filters=128, 
    conv1_filter_size=(3, 3), 
    conv1_pad=1, 
    conv1_nonlinearity=lasagne.nonlinearities.rectify, 
    pool1_ds=(2, 2),
    dropout1_p=0.25,

    conv2_num_filters=128, 
    conv2_filter_size=(3, 3), 
    conv2_pad=1, 
    conv2_nonlinearity=lasagne.nonlinearities.rectify,
    pool2_ds=(2,2),
    dropout2_p=0.25,

    #conv3_num_filters=256, 
    #conv3_filter_size=(3, 3),
    #conv3_pad=1, 
    #conv3_nonlinearity=lasagne.nonlinearities.rectify, 
    #pool3_ds=(2, 2),
    #dropout3_p=0.25,

    conv4_num_filters=256, 
    conv4_filter_size=(3, 3), 
    conv4_pad=1, 
    conv4_nonlinearity=lasagne.nonlinearities.rectify, 
    pool4_ds=(2, 2),
    dropout4_p=0.25,

    # conv5_num_filters=512, 
    # conv5_filter_size=(3, 3), 
    # conv5_pad=1, 
    # conv5_nonlinearity=lasagne.nonlinearities.rectify,
    # pool5_ds=(2,2),
    # dropout5_p=0.50,

    # conv6_num_filters=512, 
    # conv6_filter_size=(3, 3),
    # conv6_pad=1, 
    # conv6_nonlinearity=lasagne.nonlinearities.rectify, 
    # pool6_ds=(2, 2),
    # dropout6_p=0.25,

    # conv7_num_filters=1024, 
    # conv7_filter_size=(3, 3),
    # #conv7_pad=1, 
    # conv7_nonlinearity=lasagne.nonlinearities.rectify, 
    # #pool7_ds=(2, 2),
    # dropout7_p=0.50,

    # conv8_num_filters=1024, 
    # conv8_filter_size=(3, 3), 
    # #conv8_pad=1, 
    # conv8_nonlinearity=lasagne.nonlinearities.rectify, 
    # #pool8_ds=(2, 2),
    # dropout8_p=0.50,


    hidden1_num_units=1024,
    hidden1_nonlinearity=lasagne.nonlinearities.rectify,

    dropout1h_p=0.5,

    hidden2_num_units=1024,
    hidden2_nonlinearity=lasagne.nonlinearities.rectify, 

    dropout2h_p=0.5,

    # hidden3_num_units=1024,
    # hidden3_nonlinearity=lasagne.nonlinearities.rectify, 

    # dropout3h_p=0.5,

    # hidden4_num_units=1024,
    # hidden4_nonlinearity=lasagne.nonlinearities.rectify, 

    # dropout4h_p=0.5,

    output_num_units=121, 
    output_nonlinearity=lasagne.nonlinearities.softmax,

    regression=False,
    #loss=lasagne.objectives.multinomial_nll,
    use_label_encoder=True,
    batch_iterator_train=FlipBatchIterator(batch_size=256),
    
    on_epoch_finished=[
        #AdjustVariable('update_learning_rate', start=0.01, stop=0.0001),
        #AdjustVariable('update_momentum', start=0.9, stop=0.999),
        EarlyStopping(patience=20),
        ],
    max_epochs=100,
    verbose=2,
    update_learning_rate=theano.shared(float32(0.007)),
    update_momentum=theano.shared(float32(0.9)),
    test_size=0.1
    )


if TRAIN:
    if UPDATE:
        with open(DATABASE_FOLDER + NETWORK, 'rb') as f:
            net2 = pickle.load(f)  
    net2.fit(X, y)
    with open(DATABASE_FOLDER + NETWORK, 'wb') as f:
    	pickle.dump(net2, f, -1)

def predict():
    with open(DATABASE_FOLDER + NETWORK, 'rb') as f:
        model = pickle.load(f)
    if LOAD_DUMP:
        testdata = pickle.load(open(DATABASE_FOLDER + DUMP_TEST, 'rb'))
    else:
        testdata = load_test()
        pickle.dump(testdata, open(DATABASE_FOLDER + DUMP_TEST, 'wb'), -1)

    predictions = np.zeros((testdata.shape[0], 121))
    for i in range(0, testdata.shape[0], 4075):
        preds = model.predict_proba(testdata[i:i+4075,:].astype(np.float32).reshape(-1, 1, IMG_SIZE, IMG_SIZE))
        predictions[i:i+4075,:] = preds
        print "Done: ", i+4075

    angles = [45, 90, 135, 180, 225, 270, 315]
    for angle in angles:
        for k in range(testdata.shape[0]):
            testdata[k,:] = rotate(testdata[k,:].reshape(IMG_SIZE, IMG_SIZE)).reshape(-1)
        for i in range(0, testdata.shape[0], 4075):
            preds = model.predict_proba(testdata[i:i+4075,:].astype(np.float32).reshape(-1, 1, IMG_SIZE, IMG_SIZE))
            predictions[i:i+4075,:] += preds
            print angle, i+4075

    return model, predictions*1./8.

if TEST:
    model, y_test = predict()
    idx = [str(i) + '.jpg' for i in test_ids()]
    print "creating submission file"
    preds = pd.DataFrame(y_test, index = idx, columns=model.enc_.classes_)
    preds.to_csv(DATABASE_FOLDER + SUBMISSION_FILE, index_label = 'image')
