from nolearn.lasagne import NeuralNet
from utils import *

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
    loss=lasagne.objectives.multinomial_nll,
    use_label_encoder=True,

    on_epoch_finished=[
        AdjustVariable('update_learning_rate', start=0.01, stop=0.0001),
        AdjustVariable('update_momentum', start=0.9, stop=0.999),
        EarlyStopping(patience=200),
        ],
    max_epochs=100,
    verbose=2,
    )