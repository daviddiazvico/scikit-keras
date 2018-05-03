"""
Tests.

@author: David Diaz Vico
@license: MIT
"""

from keras import backend as K
import numpy as np
import pickle
from scipy.stats import uniform
from sklearn.datasets import load_iris, load_boston, load_diabetes, load_digits
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.estimator_checks import check_estimator

from skkeras.base import FFClassifier, FFRegressor


np.random.seed(0)


###############################################################################
#  Scikit-learn integration tests
###############################################################################


#def test_sklearn():
#    """Tests general compatibility with Scikit-learn."""
#    for model in (FFClassifier, FFRegressor):
#        try:
#            K.set_epsilon(1e-12)
#            check_estimator(model(dense_units=(100, 100), epochs=1000,
#                                  early_stopping=False))
#        except AssertionError as error:
#            assert 'Not equal to tolerance rtol=1e-07, atol=1e-09' in str(error)


def test_pipeline():
    """Tests compatibility with Scikit-learn's pipeline."""
    data = load_iris()
    predictor = Pipeline([('s', StandardScaler()),
                          ('k', FFClassifier())])
    assert isinstance(predictor.named_steps['k'], FFClassifier)
    predictor.fit(data.data, data.target, k__epochs=1)
    assert isinstance(predictor.named_steps['k'], FFClassifier)
    preds = predictor.predict(data.data)
    assert isinstance(preds, np.ndarray)
    score = predictor.score(data.data, data.target)
    assert isinstance(score, float)


def test_hyperparametersearchcv():
    """Tests compatibility with Scikit-learn's hyperparameter search CV."""
    data = load_iris()
    for search, space in ((GridSearchCV,
                           {'output_kernel_regularizer_l2': [0.0, 0.5, 1.0]}),
                          (RandomizedSearchCV,
                           {'output_kernel_regularizer_l2': uniform(0.0,
                                                                    1.0)})):
        predictor = search(FFClassifier(epochs=1, validation_split=0.1), space)
        assert isinstance(predictor, search)
        predictor.fit(data.data, data.target)
        assert isinstance(predictor.best_estimator_, FFClassifier)
        preds = predictor.predict(data.data)
        assert isinstance(preds, np.ndarray)
        score = predictor.score(data.data, data.target)
        assert isinstance(score, float)


def test_ensemble():
    """Tests compatibility with Scikit-learn's ensembles."""
    data = load_iris()
    for ensemble in (BaggingClassifier, AdaBoostClassifier):
        predictor = ensemble(base_estimator=FFClassifier(epochs=1),
                             n_estimators=3)
        assert isinstance(predictor, ensemble)
        predictor.fit(data.data, data.target)
        assert len(predictor.estimators_) == 3
        preds = predictor.predict(data.data)
        assert isinstance(preds, np.ndarray)
        score = predictor.score(data.data, data.target)
        assert isinstance(score, float)


###############################################################################
#  Serialization test
###############################################################################


def test_serialization():
    """Tests serialization capability."""
    data = load_iris()
    predictor = FFClassifier()
    predictor.fit(data.data, data.target, epochs=1)
    serialized_predictor = pickle.dumps(predictor)
    deserialized_predictor = pickle.loads(serialized_predictor)
    assert isinstance(deserialized_predictor, FFClassifier)
    preds = deserialized_predictor.predict(data.data)
    assert isinstance(preds, np.ndarray)
    score = deserialized_predictor.score(data.data, data.target)
    assert isinstance(score, float)


###############################################################################
#  Class tests
###############################################################################


def test_class():
    """Tests class."""
    for dataset, model in ((load_iris, FFClassifier),
                           (load_boston, FFRegressor)):
        data = dataset()
        predictor = model()
        assert isinstance(predictor, model)
        predictor.fit(data.data, data.target, epochs=1)
        assert isinstance(predictor, model)
        preds = predictor.predict(data.data)
        assert isinstance(preds, np.ndarray)
        score = predictor.score(data.data, data.target)
        assert isinstance(score, float)


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
    predictor = FFClassifier(dense_units=(10,))
    assert isinstance(predictor, FFClassifier)
    predictor.fit(data.data, data.target, epochs=1)
    config = predictor.model_.get_config()
    assert len(config['layers']) == 3
    assert config['layers'][0]['class_name'] == 'InputLayer'
    assert config['layers'][1]['class_name'] == 'Dense'
    assert config['layers'][2]['class_name'] == 'Dense'


def test_batchnormalization():
    """Tests batch normalization."""
    data = load_iris()
    predictor = FFClassifier(batchnormalization=True, dense_units=(10,))
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
    predictor = FFClassifier(dense_units=(10,), dropout_rate=0.1)
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
    predictor = FFClassifier(convolution_filters=(1,),
                             convolution_kernel_size=((2, 2),))
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
    predictor = FFClassifier(convolution_filters=(1,),
                             convolution_kernel_size=((2, 2),),
                             pooling_pool_size=((1, 1),))
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
    predictor = FFClassifier(convolution_filters=(1,),
                             convolution_kernel_size=((2, 2),),
                             dense_units=(10,))
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
    predictor = FFRegressor(recurrent_window=3, recurrent_units=(10,))
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
    predictor = FFRegressor(recurrent_window=3, recurrent_units=(10,),
                            dense_units=(10,))
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
    predictor = FFClassifier(convolution_filters=(1,),
                             convolution_kernel_size=((2, 2),),
                             recurrent_window=3, recurrent_units=(10,))
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
    predictor = FFClassifier(convolution_filters=(1,),
                             convolution_kernel_size=((2, 2),),
                             recurrent_window=3, recurrent_units=(10,),
                             dense_units=(10,))
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
        predictor = FFClassifier(convolution_filters=(1,),
                                 convolution_kernel_size=((2, 2),),
                                 convolution_kernel_regularizer_l1=l1,
                                 convolution_kernel_regularizer_l2=l2,
                                 convolution_bias_regularizer_l1=l1,
                                 convolution_bias_regularizer_l2=l2,
                                 convolution_activity_regularizer_l1=l1,
                                 convolution_activity_regularizer_l2=l2,
                                 pooling_pool_size=((1, 1),),
                                 recurrent_window=3,
                                 recurrent_units=(10,),
                                 recurrent_kernel_regularizer_l1=l1,
                                 recurrent_kernel_regularizer_l2=l2,
                                 recurrent_recurrent_regularizer_l1=l1,
                                 recurrent_recurrent_regularizer_l2=l2,
                                 recurrent_bias_regularizer_l1=l1,
                                 recurrent_bias_regularizer_l2=l2,
                                 recurrent_activity_regularizer_l1=l1,
                                 recurrent_activity_regularizer_l2=l2,
                                 dense_units=(10,),
                                 dense_kernel_regularizer_l1=l1,
                                 dense_kernel_regularizer_l2=l2,
                                 dense_bias_regularizer_l1=l1,
                                 dense_bias_regularizer_l2=l2,
                                 dense_activity_regularizer_l1=l1,
                                 dense_activity_regularizer_l2=l2,
                                 batchnormalization=True,
                                 batchnormalization_beta_regularizer_l1=l1,
                                 batchnormalization_beta_regularizer_l2=l2,
                                 batchnormalization_gamma_regularizer_l1=l1,
                                 batchnormalization_gamma_regularizer_l2=l2,
                                 output_kernel_regularizer_l1=l1,
                                 output_kernel_regularizer_l2=l2,
                                 output_bias_regularizer_l1=l1,
                                 output_bias_regularizer_l2=l2,
                                 output_activity_regularizer_l1=l1,
                                 output_activity_regularizer_l2=l2)
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


###############################################################################
#  Solver test
###############################################################################


def test_solver():
    """Tests solver."""
    data = load_iris()
    predictor = FFClassifier()
    for solver in ('adam', 'sgd', 'rmsprop', 'adagrad', 'adadelta', 'adamax',
                   'nadam'):
        predictor.fit(data.data, data.target, solver=solver, epochs=10)
        assert isinstance(predictor, FFClassifier)
