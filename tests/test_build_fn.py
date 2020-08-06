from keras import backend as K
import numpy as np
from sklearn.datasets import load_iris, load_diabetes, load_digits
# from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from skkeras.scikit_learn import KerasClassifier, KerasRegressor

from skkeras.build_fn import build_fn_classifier, build_fn_regressor


def time_series(X, y=None, window=None, return_sequences=False):
    """Time series transformation.

    Transform X, y tensors to time series tensors.

    Parameters
    ----------
    X: numpy array of shape [n_samples, n_features]
       Training set.
    y: numpy array of shape [n_samples]
       Target values.
    window: integer, default=None
            Time window length.
    return_sequences: boolean, default=False
                      Whether to return the last output in the output sequence,
                      or the full sequence.

    Returns
    -------
    Time series tensors.

    """
    if window is not None:
        X = np.array([X[i:i + window] for i in range(X.shape[0] - window + 1)])
        if y is not None:
            y = np.array([y[i:i + window] for i in range(y.shape[0] - window + 1)]
                         ) if return_sequences else np.array(y[window - 1:])
    return X, y


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
    diabetes_ts = load_diabetes()
    diabetes_ts.data, diabetes_ts.target = time_series(
        diabetes_ts.data, y=diabetes_ts.target, window=3)
    digits_ts = load_digits()
    digits_ts.data = digits_ts.data.reshape(
        [digits_ts.data.shape[0], 1, 8, 8]) / 16.0
    digits_ts.data, digits_ts.target = time_series(
        digits_ts.data, y=digits_ts.target, window=3)
    datasets = {'pcp': load_iris(), 'mlp': load_iris(),
                'batchnormalization': load_iris(), 'dropout': load_iris(),
                'cnn': digits, 'cnnpool': digits, 'cnnmlp': digits,
                'rnn': diabetes_ts, 'rnnmlp': diabetes_ts,
                'cnnrnn': digits_ts, 'cnnrnnmlp': digits_ts}
    estimators = {'pcp': KerasClassifier(build_fn_classifier),
                  'mlp': KerasClassifier(build_fn_classifier,
                                         dense_units=[10]),
                  'batchnormalization': KerasClassifier(build_fn_classifier,
                                                        batchnormalization=True,
                                                        dense_units=[10]),
                  'dropout': KerasClassifier(build_fn_classifier,
                                             dense_units=[10],
                                             dropout_rate=0.1),
                  'cnn': KerasClassifier(build_fn_classifier,
                                         convolution_filters=(1,),
                                         convolution_kernel_size=[(2, 2)]),
                  'cnnpool': KerasClassifier(build_fn_classifier,
                                             convolution_filters=[1],
                                             convolution_kernel_size=[
                                                 (2, 2)],
                                             pooling_pool_size=[(1, 1)]),
                  'cnnmlp': KerasClassifier(build_fn_classifier,
                                            convolution_filters=[1],
                                            convolution_kernel_size=[
                                                (2, 2)],
                                            dense_units=[10]),
                  'rnn': KerasRegressor(build_fn_regressor,
                                        recurrent_units=[10]),
                  'rnnmlp': KerasRegressor(build_fn_regressor,
                                           recurrent_units=[10],
                                           dense_units=[10]),
                  'cnnrnn': KerasClassifier(build_fn_classifier,
                                            convolution_filters=[1],
                                            convolution_kernel_size=[
                                                (2, 2)],
                                            recurrent_units=[10]),
                  'cnnrnnmlp': KerasClassifier(build_fn_classifier,
                                               convolution_filters=[1],
                                               convolution_kernel_size=[
                                                   (2, 2)],
                                               recurrent_units=[10],
                                               dense_units=[10])}
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
        data = datasets[test]
        estimator = estimators[test]
        layer_types = layers[test]
        estimator.fit(data.data, data.target, epochs=1)
        check_architecture(estimator.model_, layer_types)
        estimator.predict(data.data)
        estimator.score(data.data, data.target)


def test_regularized():
    """Tests regularizer."""
    data = load_digits()
    data.data = data.data.reshape([data.data.shape[0], 1, 8, 8]) / 16.0
    data.data, data.target = time_series(data.data, y=data.target, window=3)
    K.set_image_data_format('channels_first')
    for l1, l2 in ((0.1, None), (None, 0.1), (0.1, 0.1)):
        estimator = KerasClassifier(build_fn_classifier,
                                    convolution_filters=[1],
                                    convolution_kernel_size=[(2, 2)],
                                    pooling_pool_size=[(1, 1)],
                                    recurrent_units=[2],
                                    dense_units=[2],
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
                                    gamma_regularizer_l2=l2)
        assert isinstance(estimator, KerasClassifier)
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
