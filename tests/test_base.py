"""
Tests.

@author: David Diaz Vico
@license: MIT
"""

from keras import backend as K
import numpy as np
import pickle
from scipy.stats import uniform
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import load_iris, load_boston
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


#def test_sklearn(transformer=None):
#    """Tests general compatibility with Scikit-learn."""
#    for model in (FFClassifier, FFRegressor):
#        try:
#            K.set_epsilon(1e-12)
#            check_estimator(model(transformer=transformer, epochs=1000,
#                                  early_stopping=False))
#        except AssertionError as error:
#            assert 'Not equal to tolerance rtol=1e-07, atol=1e-09' in str(error)


def test_pipeline(transformer=None):
    """Tests compatibility with Scikit-learn's pipeline."""
    data = load_iris()
    predictor = Pipeline([('s', StandardScaler()),
                          ('k', FFClassifier(transformer=transformer))])
    assert isinstance(predictor.named_steps['k'], FFClassifier)
    predictor.fit(data.data, data.target, k__epochs=1)
    assert isinstance(predictor.named_steps['k'], FFClassifier)
    preds = predictor.predict(data.data)
    assert isinstance(preds, np.ndarray)
    score = predictor.score(data.data, data.target)
    assert isinstance(score, float)


def test_hyperparametersearchcv(transformer=None):
    """Tests compatibility with Scikit-learn's hyperparameter search CV."""
    data = load_iris()
    for search, space in ((GridSearchCV,
                           {'kernel_regularizer_l2': [0.0, 0.5, 1.0]}),
                          (RandomizedSearchCV,
                           {'kernel_regularizer_l2': uniform(0.0, 1.0)})):
        predictor = search(FFClassifier(transformer=transformer, epochs=1,
                                        validation_split=0.1), space)
        assert isinstance(predictor, search)
        predictor.fit(data.data, data.target)
        assert isinstance(predictor.best_estimator_, FFClassifier)
        preds = predictor.predict(data.data)
        assert isinstance(preds, np.ndarray)
        score = predictor.score(data.data, data.target)
        assert isinstance(score, float)


def test_ensemble(transformer=None):
    """Tests compatibility with Scikit-learn's ensembles."""
    data = load_iris()
    for ensemble in (BaggingClassifier, AdaBoostClassifier):
        predictor = ensemble(base_estimator=FFClassifier(transformer=transformer,
                                                         epochs=1),
                             n_estimators=3)
        assert isinstance(predictor, ensemble)
        predictor.fit(data.data, data.target)
        assert len(predictor.estimators_) == 3
        preds = predictor.predict(data.data)
        assert isinstance(preds, np.ndarray)
        score = predictor.score(data.data, data.target)
        assert isinstance(score, float)


def test_calibratedclassifier(transformer=None):
    """Tests compatibility with Scikit-learn's calibrated classifier."""
    data = load_iris()
    predictor = CalibratedClassifierCV(base_estimator=FFClassifier(transformer=transformer,
                                                                   epochs=1))
    predictor.fit(data.data, data.target)
    assert isinstance(predictor, CalibratedClassifierCV)
    preds = predictor.predict(data.data)
    assert isinstance(preds, np.ndarray)
    probas = predictor.predict_proba(data.data)
    assert isinstance(probas, np.ndarray)
    score = predictor.score(data.data, data.target)
    assert isinstance(score, float)


###############################################################################
#  Serialization test
###############################################################################


def test_serialization(transformer=None):
    """Tests serialization capability."""
    data = load_iris()
    predictor = FFClassifier(transformer=transformer)
    predictor.fit(data.data, data.target, epochs=1)
    serialized_predictor = pickle.dumps(predictor)
    deserialized_predictor = pickle.loads(serialized_predictor)
    assert isinstance(deserialized_predictor, FFClassifier)
    preds = deserialized_predictor.predict(data.data)
    assert isinstance(preds, np.ndarray)
    score = deserialized_predictor.score(data.data, data.target)
    assert isinstance(score, float)


###############################################################################
#  FeedForward tests
###############################################################################


def test_class(transformer=None):
    """Tests class."""
    for dataset, model in ((load_iris, FFClassifier),
                           (load_boston, FFRegressor)):
        data = dataset()
        predictor = model(transformer=transformer)
        assert isinstance(predictor, model)
        predictor.fit(data.data, data.target, epochs=1)
        assert isinstance(predictor, model)
        preds = predictor.predict(data.data)
        assert isinstance(preds, np.ndarray)
        score = predictor.score(data.data, data.target)
        assert isinstance(score, float)
        transformations = predictor.transform(data.data)
        if transformer is None: assert transformations.shape == data.data.shape


###############################################################################
#  Optimizer test
###############################################################################


def test_optimizer(transformer=None):
    """Tests optimizer."""
    data = load_iris()
    predictor = FFClassifier(transformer=transformer)
    for optimizer in ('adam', 'sgd', 'rmsprop', 'adagrad', 'adadelta', 'adamax',
                      'nadam'):
        predictor.fit(data.data, data.target, optimizer=optimizer, epochs=10)
        assert isinstance(predictor, FFClassifier)


###############################################################################
#  Regularizer test
###############################################################################


def test_regularizer(transformer=None):
    """Tests regularizer."""
    data = load_iris()
    for l1, l2 in ((0.1, None), (None, 0.1), (0.1, 0.1)):
        predictor = FFClassifier(transformer=transformer,
                                 kernel_regularizer_l1=l1,
                                 kernel_regularizer_l2=l2,
                                 bias_regularizer_l1=l1, bias_regularizer_l2=l2,
                                 activity_regularizer_l1=l1,
                                 activity_regularizer_l2=l2)
        assert isinstance(predictor, FFClassifier)
        predictor.fit(data.data, data.target, epochs=1)
        config = predictor.model_.get_config()
        assert all(regularizer in config['layers'][1]['config']
                   for regularizer in ('kernel_regularizer', 'bias_regularizer',
                                       'activity_regularizer'))
