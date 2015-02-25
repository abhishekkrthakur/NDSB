import cPickle as pickle
import cv2
from datetime import datetime
import glob
import os
import sys

import numpy as np
from lasagne import layers
import lasagne
from nolearn.lasagne import BatchIterator
from nolearn.lasagne import NeuralNet
import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer, StandardScaler
import theano

try:
    import pylearn2
except ImportError:
    Conv2DLayer = layers.Conv2DLayer
    MaxPool2DLayer = layers.MaxPool2DLayer
else:  # Use faster (GPU-only) Conv2DCCLayer only if it's available
    Conv2DLayer = layers.cuda_convnet.Conv2DCCLayer
    MaxPool2DLayer = layers.cuda_convnet.MaxPool2DCCLayer

sys.setrecursionlimit(10000)
np.random.seed(42)
IMAGE_SIZE = 48
sample = pd.read_csv('data/sampleSubmission.csv')
columns = list(sample.columns)[1:]

def rotateImage(image, angle):
  image_center = tuple(np.array(image.shape)/2)
  rot_mat = cv2.getRotationMatrix2D(image_center,angle,1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape,flags=cv2.INTER_LINEAR)
  return result

def float32(k):
	return np.cast['float32'](k)

def load_training_data():
	label = []
	count = 1
	i = 0
	traindata = np.zeros((30336, IMAGE_SIZE*IMAGE_SIZE))
	for column in columns:
		path = 'data/train/' + str(column) + '/'
		jpeg_files = glob.glob(path + '*.jpg')
		print path
		for jpeg_file in jpeg_files:
			img = cv2.imread(jpeg_file, cv2.CV_LOAD_IMAGE_GRAYSCALE)
			img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
			img = img.reshape(-1)
			traindata[i, :] = img
			label.append(count)
			i += 1
		count += 1
	traindata = traindata / 255.
	traindata = traindata.astype(np.float32)
	label = np.array(label)
	label = label.astype(np.int32)
	traindata, label = shuffle(traindata, label, random_state=42)
	traindata = traindata.reshape(-1, 1, IMAGE_SIZE, IMAGE_SIZE)
	return traindata, label


def load_test_data():
	testdata = np.zeros((len(sample), IMAGE_SIZE*IMAGE_SIZE))
	path = 'data/test/'
	test_files = sample.image.values
	i = 0
	for test_file in test_files:
		img = cv2.imread(path + test_file, cv2.CV_LOAD_IMAGE_GRAYSCALE)
		img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
		img = img.reshape(-1)
		testdata[i,:] = img
		i += 1 
	testdata = testdata / 255.
	testdata = testdata.astype(np.float32)
	return testdata	

class FlipBatchIterator(BatchIterator):

    def transform(self, Xb, yb):
        Xb, yb = super(FlipBatchIterator, self).transform(Xb, yb)

        # Flip half of the images in this batch at random:
        bs = Xb.shape[0]
        indices = np.random.choice(bs, bs / 2, replace=False)
        Xb[indices] = Xb[indices, :, :, ::-1]

        return Xb, yb

class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = np.cast['float32'](self.ls[epoch - 1])
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

net = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv1', Conv2DLayer),
        ('pool1', MaxPool2DLayer),
        ('conv2', Conv2DLayer),
        ('conv3', Conv2DLayer),
        ('pool3', MaxPool2DLayer),
        ('hidden1', layers.DenseLayer),
        ('dropout1', layers.DropoutLayer),
        ('hidden2', layers.DenseLayer),
		('dropout2', layers.DropoutLayer),
        ('output', layers.DenseLayer),
        ],
    input_shape=(None, 1, IMAGE_SIZE, IMAGE_SIZE),

    conv1_num_filters=96, 
    conv1_filter_size=(5, 5), 
    conv1_pad=2, 
    conv1_strides=(4,4), 
    conv1_nonlinearity=lasagne.nonlinearities.rectify, 

    pool1_ds=(3, 3),
    pool1_strides=(2,2),

    conv2_num_filters=128, 
    conv2_filter_size=(3, 3), 
    conv2_pad=2, 
    conv2_nonlinearity=lasagne.nonlinearities.rectify,

    conv3_num_filters=128, 
    conv3_filter_size=(3, 3),
    conv3_pad=1, 
    conv3_nonlinearity=lasagne.nonlinearities.rectify, 

    pool3_ds=(3, 3),
    pool3_strides=(2,2),

    hidden1_num_units=512,
    hidden1_nonlinearity=lasagne.nonlinearities.rectify,

    dropout1_p=0.5,

    hidden2_num_units=512,
    hidden2_nonlinearity=lasagne.nonlinearities.rectify, 

    dropout2_p=0.5,

    output_num_units=121, 
    output_nonlinearity=lasagne.nonlinearities.softmax,

    update_learning_rate=theano.shared(float32(0.01)),
    update_momentum=theano.shared(float32(0.9)),

    regression=False,
    #loss=lasagne.objectives.multinomial_nll,
    use_label_encoder=True,
    batch_iterator_train=FlipBatchIterator(batch_size=256),
    
    on_epoch_finished=[
        AdjustVariable('update_learning_rate', start=0.01, stop=0.0001),
        AdjustVariable('update_momentum', start=0.9, stop=0.999),
        EarlyStopping(patience=20),
        ],
    max_epochs=500,
    verbose=2,
    )

def fit():
    X, y = load_training_data()
    net.fit(X, y)
    with open('net.pickle', 'wb') as f:
        pickle.dump(net, f, -1)

def predict():
    with open('net.pickle', 'rb') as f:
        model = pickle.load(f)

    testdata = load_test_data()
    predictions = np.zeros((testdata.shape[0], 121))
    for i in range(0, testdata.shape[0], 4075):
    	preds = model.predict_proba(testdata[i:i+4075,:].reshape(-1, 1, IMAGE_SIZE, IMAGE_SIZE))
    	predictions[i:i+4075,:] = preds
    	print "Done: ", i+4075
    return predictions

def create_submission(predictions):
	print "creating submission file"
	for i, feat in enumerate(columns):
		sample[feat] = predictions[:,i]
	sample.to_csv('submission.csv', index = False)

if __name__ == '__main__':
	fit()
	predictions = predict()
	create_submission(predictions)

