"""
Tests.

@author: David Diaz Vico
@license: MIT
"""

#from keras import backend as K
import numpy as np
import pickle
from scipy.stats import uniform
from sklearn.datasets import load_iris, load_boston
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
#from sklearn.utils.estimator_checks import check_estimator

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
#            check_estimator(model(epochs=1000, early_stopping=False))
#        except AssertionError as error:
#            assert 'Not equal to tolerance rtol=1e-07, atol=1e-09' in str(error)


def test_pipeline():
    """Tests compatibility with Scikit-learn's pipeline."""
    data = load_iris()
    predictor = Pipeline([('s', StandardScaler()), ('k', FFClassifier())])
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
                           {'kernel_regularizer_l2': [0.0, 0.5, 1.0]}),
                          (RandomizedSearchCV,
                           {'kernel_regularizer_l2': uniform(0.0, 1.0)})):
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
        transformations = predictor.transform(data.data)
        assert transformations.shape == data.data.shape


###############################################################################
#  Optimizer test
###############################################################################


def test_optimizer():
    """Tests optimizer."""
    data = load_iris()
    predictor = FFClassifier()
    for optimizer in ('adam', 'sgd', 'rmsprop', 'adagrad', 'adadelta', 'adamax',
                      'nadam'):
        predictor.fit(data.data, data.target, optimizer=optimizer, epochs=10)
        assert isinstance(predictor, FFClassifier)


###############################################################################
#  Regularizer test
###############################################################################


def test_regularizer():
    """Tests regularizer."""
    data = load_iris()
    for l1, l2 in ((0.1, None), (None, 0.1), (0.1, 0.1)):
        predictor = FFClassifier(kernel_regularizer_l1=l1,
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
