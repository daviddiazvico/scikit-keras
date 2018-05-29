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


def check_architecture(estimator, layer_types):
    """Checks architecture."""
    layers = estimator.get_config()['layers']
    assert len(layers) == len(layer_types)
    for layer, layer_type in zip(layers, layer_types):
        assert layer['class_name'] == layer_type


def test_architecture():
    """Tests architecture."""
    digits = load_digits()
    digits.data = digits.data.reshape([digits.data.shape[0], 1, 8, 8]) / 16.0
    K.set_image_data_format('channels_first')
    datasets = {'pcp': load_iris(), 'mlp': load_iris(),
                'batchnormalization': load_iris(), 'dropout': load_iris(),
                'cnn': digits, 'cnnpool': digits, 'cnnmlp': digits,
                'rnn': load_diabetes(), 'rnnmlp': load_diabetes(),
                'cnnrnn': digits, 'cnnrnnmlp': digits}
    estimators = {'pcp': FFClassifier(),
                  'mlp': FFClassifier(architecture=Straight(dense_units=(10,))),
                  'batchnormalization': FFClassifier(architecture=Straight(batchnormalization=True,
                                                                           dense_units=(10,))),
                  'dropout': FFClassifier(architecture=Straight(dense_units=(10,),
                                                                dropout_rate=0.1)),
                  'cnn': FFClassifier(architecture=Straight(convolution_filters=(1,),
                                                            convolution_kernel_size=((2, 2),))),
                  'cnnpool': FFClassifier(architecture=Straight(convolution_filters=(1,),
                                                                convolution_kernel_size=((2, 2),),
                                                                pooling_pool_size=((1, 1),))),
                  'cnnmlp': FFClassifier(architecture=Straight(convolution_filters=(1,),
                                                               convolution_kernel_size=((2, 2),),
                                                               dense_units=(10,))),
                  'rnn': FFRegressor(architecture=Straight(recurrent_units=(10,)),
                                                           window=3),
                  'rnnmlp': FFRegressor(architecture=Straight(recurrent_units=(10,),
                                                              dense_units=(10,)),
                                        window=3),
                  'cnnrnn': FFClassifier(architecture=Straight(convolution_filters=(1,),
                                                               convolution_kernel_size=((2, 2),),
                                                               recurrent_units=(10,)),
                                         window=3),
                  'cnnrnnmlp': FFClassifier(architecture=Straight(convolution_filters=(1,),
                                                                  convolution_kernel_size=((2, 2),),
                                                                  recurrent_units=(10,),
                                                                  dense_units=(10,)),
                                            window=3)}
    layers = {'pcp': ['InputLayer', 'Dense'],
              'mlp': ['InputLayer', 'Dense', 'Dense'],
              'batchnormalization': ['InputLayer', 'BatchNormalization',
                                     'Dense', 'Dense'],
              'dropout': ['InputLayer', 'Dense', 'Dropout', 'Dense'],
              'cnn': ['InputLayer', 'Conv2D', 'Flatten', 'Dense'],
              'cnnpool': ['InputLayer', 'Conv2D', 'MaxPooling2D', 'Flatten',
                          'Dense'],
              'cnnmlp': ['InputLayer', 'Conv2D', 'Flatten', 'Dense', 'Dense'],
              'rnn': ['InputLayer', 'LSTM', 'Dense'],
              'rnnmlp': ['InputLayer', 'LSTM', 'Dense', 'Dense'],
              'cnnrnn': ['InputLayer', 'TimeDistributed', 'TimeDistributed',
                         'LSTM', 'Dense'],
              'cnnrnnmlp': ['InputLayer', 'TimeDistributed', 'TimeDistributed',
                            'LSTM', 'Dense', 'Dense']}
    for test in ['pcp', 'mlp', 'batchnormalization', 'dropout', 'cnn',
                 'cnnpool', 'cnnmlp', 'rnn', 'rnnmlp', 'cnnrnn', 'cnnrnnmlp']:
        print(test) #######
        data = datasets[test]
        estimator = estimators[test]
        layer_types = layers[test]
        estimator.fit(data.data, data.target, epochs=1)
        check_architecture(estimator.model_, layer_types)
        estimator.predict(data.data)
        estimator.transform(data.data)
        estimator.score(data.data, data.target)


###############################################################################
#  Regularizer test
###############################################################################


def test_regularizer():
    """Tests regularizer."""
    data = load_digits()
    data.data = data.data.reshape([data.data.shape[0], 1, 8, 8]) / 16.0
    K.set_image_data_format('channels_first')
    for l1, l2 in ((0.1, None), (None, 0.1), (0.1, 0.1)):
        estimator = FFClassifier(architecture=Straight(convolution_filters=(1,),
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
        assert isinstance(estimator, FFClassifier)
        estimator.fit(data.data, data.target, epochs=1)
        config = estimator.model_.get_config()
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
