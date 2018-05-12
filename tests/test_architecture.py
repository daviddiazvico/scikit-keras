"""
Tests.

@author: David Diaz Vico
@license: MIT
"""

from keras import backend as K
import numpy as np
from sklearn.datasets import load_iris, load_diabetes, load_digits

from skkeras.architecture import Straight
from skkeras.base import FFClassifier, FFRegressor


np.random.seed(0)


###############################################################################
#  Architecture tests
###############################################################################


def test_pcp():
    """Tests PCP."""
    data = load_iris()
    predictor = FFClassifier()
    assert isinstance(predictor, FFClassifier)
    predictor.fit(data.data, data.target, epochs=1)
    config = predictor.model_.get_config()
    assert len(config['layers']) == 2
    assert config['layers'][0]['class_name'] == 'InputLayer'
    assert config['layers'][1]['class_name'] == 'Dense'


def test_mlp():
    """Tests MLP."""
    data = load_iris()
    predictor = FFClassifier(architecture=Straight(dense_units=(10,)))
    assert isinstance(predictor, FFClassifier)
    predictor.fit(data.data, data.target, epochs=1)
    config = predictor.model_.get_config()
    assert len(config['layers']) == 3
    assert config['layers'][0]['class_name'] == 'InputLayer'
    assert config['layers'][1]['class_name'] == 'Dense'
    assert config['layers'][2]['class_name'] == 'Dense'
    transformations = predictor.transform(data.data)
    assert transformations.shape == (len(data.data), 10)


def test_batchnormalization():
    """Tests batch normalization."""
    data = load_iris()
    predictor = FFClassifier(architecture=Straight(batchnormalization=True,
                                                   dense_units=(10,)))
    assert isinstance(predictor, FFClassifier)
    predictor.fit(data.data, data.target, epochs=1)
    config = predictor.model_.get_config()
    assert len(config['layers']) == 4
    assert config['layers'][0]['class_name'] == 'InputLayer'
    assert config['layers'][1]['class_name'] == 'BatchNormalization'
    assert config['layers'][2]['class_name'] == 'Dense'
    assert config['layers'][3]['class_name'] == 'Dense'


def test_dropout():
    """Tests dropout."""
    data = load_iris()
    predictor = FFClassifier(architecture=Straight(dense_units=(10,),
                                                   dropout_rate=0.1))
    assert isinstance(predictor, FFClassifier)
    predictor.fit(data.data, data.target, epochs=1)
    config = predictor.model_.get_config()
    assert len(config['layers']) == 4
    assert config['layers'][0]['class_name'] == 'InputLayer'
    assert config['layers'][1]['class_name'] == 'Dense'
    assert config['layers'][2]['class_name'] == 'Dropout'
    assert config['layers'][3]['class_name'] == 'Dense'


def test_cnn():
    """Tests CNN."""
    data = load_digits()
    data.data = data.data.reshape([data.data.shape[0], 1, 8, 8]) / 16.0
    K.set_image_data_format('channels_first')
    predictor = FFClassifier(architecture=Straight(convolution_filters=(1,),
                                                   convolution_kernel_size=((2, 2),)))
    assert isinstance(predictor, FFClassifier)
    predictor.fit(data.data, data.target, epochs=1)
    config = predictor.model_.get_config()
    assert len(predictor.model_.get_config()['layers']) == 4
    assert config['layers'][0]['class_name'] == 'InputLayer'
    assert config['layers'][1]['class_name'] == 'Conv2D'
    assert config['layers'][2]['class_name'] == 'Flatten'
    assert config['layers'][3]['class_name'] == 'Dense'


def test_cnnpool():
    """Tests CNN + pooling."""
    data = load_digits()
    data.data = data.data.reshape([data.data.shape[0], 1, 8, 8]) / 16.0
    K.set_image_data_format('channels_first')
    predictor = FFClassifier(architecture=Straight(convolution_filters=(1,),
                                                   convolution_kernel_size=((2, 2),),
                                                   pooling_pool_size=((1, 1),)))
    assert isinstance(predictor, FFClassifier)
    predictor.fit(data.data, data.target, epochs=1)
    config = predictor.model_.get_config()
    assert len(predictor.model_.get_config()['layers']) == 5
    assert config['layers'][0]['class_name'] == 'InputLayer'
    assert config['layers'][1]['class_name'] == 'Conv2D'
    assert config['layers'][2]['class_name'] == 'MaxPooling2D'
    assert config['layers'][3]['class_name'] == 'Flatten'
    assert config['layers'][4]['class_name'] == 'Dense'


def test_cnnmlp():
    """Tests CNN + MLP."""
    data = load_digits()
    data.data = data.data.reshape([data.data.shape[0], 1, 8, 8]) / 16.0
    K.set_image_data_format('channels_first')
    predictor = FFClassifier(architecture=Straight(convolution_filters=(1,),
                                                   convolution_kernel_size=((2, 2),),
                                                   dense_units=(10,)))
    assert isinstance(predictor, FFClassifier)
    predictor.fit(data.data, data.target, epochs=1)
    config = predictor.model_.get_config()
    assert len(predictor.model_.get_config()['layers']) == 5
    assert config['layers'][0]['class_name'] == 'InputLayer'
    assert config['layers'][1]['class_name'] == 'Conv2D'
    assert config['layers'][2]['class_name'] == 'Flatten'
    assert config['layers'][3]['class_name'] == 'Dense'
    assert config['layers'][4]['class_name'] == 'Dense'


def test_rnn():
    """Tests RNN."""
    data = load_diabetes()
    predictor = FFRegressor(architecture=Straight(recurrent_units=(10,)),
                            window=3)
    assert isinstance(predictor, FFRegressor)
    predictor.fit(data.data, data.target, epochs=1)
    config = predictor.model_.get_config()
    assert len(predictor.model_.get_config()['layers']) == 3
    assert config['layers'][0]['class_name'] == 'InputLayer'
    assert config['layers'][1]['class_name'] == 'LSTM'
    assert config['layers'][2]['class_name'] == 'Dense'


def test_rnnmlp():
    """Tests RNN + MLP."""
    data = load_diabetes()
    predictor = FFRegressor(architecture=Straight(recurrent_units=(10,),
                                                  dense_units=(10,)),
                            window=3)
    assert isinstance(predictor, FFRegressor)
    predictor.fit(data.data, data.target, epochs=1)
    config = predictor.model_.get_config()
    assert len(predictor.model_.get_config()['layers']) == 4
    assert config['layers'][0]['class_name'] == 'InputLayer'
    assert config['layers'][1]['class_name'] == 'LSTM'
    assert config['layers'][2]['class_name'] == 'Dense'
    assert config['layers'][3]['class_name'] == 'Dense'


def test_cnnrnn():
    """Tests CNN + RNN."""
    data = load_digits()
    data.data = data.data.reshape([data.data.shape[0], 1, 8, 8]) / 16.0
    K.set_image_data_format('channels_first')
    predictor = FFClassifier(architecture=Straight(convolution_filters=(1,),
                                                   convolution_kernel_size=((2, 2),),
                                                   recurrent_units=(10,)),
                             window=3)
    assert isinstance(predictor, FFClassifier)
    predictor.fit(data.data, data.target, epochs=1)
    config = predictor.model_.get_config()
    assert len(predictor.model_.get_config()['layers']) == 5
    assert config['layers'][0]['class_name'] == 'InputLayer'
    assert config['layers'][1]['class_name'] == 'TimeDistributed'
    assert config['layers'][1]['config']['layer']['class_name'] == 'Conv2D'
    assert config['layers'][2]['class_name'] == 'TimeDistributed'
    assert config['layers'][2]['config']['layer']['class_name'] == 'Flatten'
    assert config['layers'][3]['class_name'] == 'LSTM'
    assert config['layers'][4]['class_name'] == 'Dense'


def test_cnnrnnmlp():
    """Tests CNN + RNN + MLP."""
    data = load_digits()
    data.data = data.data.reshape([data.data.shape[0], 1, 8, 8]) / 16.0
    K.set_image_data_format('channels_first')
    predictor = FFClassifier(architecture=Straight(convolution_filters=(1,),
                                                   convolution_kernel_size=((2, 2),),
                                                   recurrent_units=(10,),
                                                   dense_units=(10,)),
                             window=3)
    assert isinstance(predictor, FFClassifier)
    predictor.fit(data.data, data.target, epochs=1)
    config = predictor.model_.get_config()
    assert len(predictor.model_.get_config()['layers']) == 6
    assert config['layers'][0]['class_name'] == 'InputLayer'
    assert config['layers'][1]['class_name'] == 'TimeDistributed'
    assert config['layers'][1]['config']['layer']['class_name'] == 'Conv2D'
    assert config['layers'][2]['class_name'] == 'TimeDistributed'
    assert config['layers'][2]['config']['layer']['class_name'] == 'Flatten'
    assert config['layers'][3]['class_name'] == 'LSTM'
    assert config['layers'][4]['class_name'] == 'Dense'
    assert config['layers'][5]['class_name'] == 'Dense'


###############################################################################
#  Regularizer test
###############################################################################


def test_regularizer():
    """Tests regularizer."""
    data = load_digits()
    data.data = data.data.reshape([data.data.shape[0], 1, 8, 8]) / 16.0
    K.set_image_data_format('channels_first')
    for l1, l2 in ((0.1, None), (None, 0.1), (0.1, 0.1)):
        predictor = FFClassifier(architecture=Straight(convolution_filters=(1,),
                                                       convolution_kernel_size=((2, 2),),
                                                       pooling_pool_size=((1, 1),),
                                                       recurrent_units=(10,),
                                                       dense_units=(10,),
                                                       batchnormalization=True,
                                                       kernel_regularizer_l1=l1,
                                                       kernel_regularizer_l2=l2,
                                                       bias_regularizer_l1=l1,
                                                       bias_regularizer_l2=l2,
                                                       activity_regularizer_l1=l1,
                                                       activity_regularizer_l2=l2,
                                                       recurrent_regularizer_l1=l1,
                                                       recurrent_regularizer_l2=l2,
                                                       beta_regularizer_l1=l1,
                                                       beta_regularizer_l2=l2,
                                                       gamma_regularizer_l1=l1,
                                                       gamma_regularizer_l2=l2),
                                 window=3)
        assert isinstance(predictor, FFClassifier)
        predictor.fit(data.data, data.target, epochs=1)
        config = predictor.model_.get_config()
        assert all(regularizer
                   in config['layers'][1]['config']['layer']['config']
                   for regularizer in ('kernel_regularizer', 'bias_regularizer',
                                       'activity_regularizer'))
        assert all(regularizer in config['layers'][4]['config']
                   for regularizer in ('kernel_regularizer',
                                       'recurrent_regularizer',
                                       'bias_regularizer',
                                       'activity_regularizer'))
        assert all(regularizer in config['layers'][5]['config']
                   for regularizer in ('beta_regularizer', 'gamma_regularizer'))
        assert all(regularizer in config['layers'][6]['config']
                   for regularizer in ('kernel_regularizer', 'bias_regularizer',
                                       'activity_regularizer'))
        assert all(regularizer in config['layers'][7]['config']
                   for regularizer in ('kernel_regularizer', 'bias_regularizer',
                                       'activity_regularizer'))
