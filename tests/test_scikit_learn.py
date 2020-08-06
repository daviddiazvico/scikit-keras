import numpy as np
import pickle
import pytest
from scipy.stats import randint
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import TransformedTargetRegressor
from sklearn.datasets import load_boston, load_digits, load_iris
from sklearn.ensemble import (AdaBoostClassifier, AdaBoostRegressor,
                              BaggingClassifier, BaggingRegressor)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from keras import backend as K
from keras.layers import Activation, Concatenate, Conv2D, Dense, Flatten, Input
from keras.models import Model, Sequential
from keras.utils import to_categorical

from skkeras.scikit_learn import BaseWrapper, KerasClassifier, KerasRegressor


hidden_layer_sizes = [5]
batch_size = 32
epochs = 1
optimizer = 'adam'

np.random.seed(42)


def load_digits8x8():
    data = load_digits()
    data.data = data.data.reshape([data.data.shape[0], 1, 8, 8]) / 16.0
    K.set_image_data_format('channels_first')
    return data


def assert_predictor_works(estimator, loader):
    data = loader()
    estimator.fit(data.data, data.target)
    preds = estimator.predict(data.data)
    score = estimator.score(data.data, data.target)
    serialized_estimator = pickle.dumps(estimator)
    deserialized_estimator = pickle.loads(serialized_estimator)
    preds = deserialized_estimator.predict(data.data)
    score = deserialized_estimator.score(data.data, data.target)
    assert True


def assert_classification_works(clf):
    X, y = load_iris(return_X_y=True)
    num_classes = len(np.unique(y))
    clf.fit(X, y, sample_weight=np.ones(X.shape[0]), batch_size=batch_size,
            epochs=epochs)
    score = clf.score(X, y, batch_size=batch_size)
    assert np.isscalar(score) and np.isfinite(score)
    preds = clf.predict(X, batch_size=batch_size)
    assert preds.shape == (len(X), )
    for prediction in np.unique(preds):
        assert prediction in range(num_classes)
    proba = clf.predict_proba(X, batch_size=batch_size)
    assert proba.shape == (len(X), num_classes)
    assert np.allclose(np.sum(proba, axis=1), np.ones(len(X)))


def assert_string_classification_works(clf):
    X, y = load_iris(return_X_y=True)
    num_classes = len(np.unique(y))
    string_classes = ['cls{}'.format(x) for x in range(num_classes)]
    str_y = np.array(string_classes)[y]
    clf.fit(X, str_y, batch_size=batch_size, epochs=epochs)
    score = clf.score(X, str_y, batch_size=batch_size)
    assert np.isscalar(score) and np.isfinite(score)
    preds = clf.predict(X, batch_size=batch_size)
    assert preds.shape == (len(X), )
    for prediction in np.unique(preds):
        assert prediction in string_classes
    proba = clf.predict_proba(X, batch_size=batch_size)
    assert proba.shape == (len(X), num_classes)
    assert np.allclose(np.sum(proba, axis=1), np.ones(len(X)))


def assert_regression_works(reg):
    X, y = load_boston(return_X_y=True)
    reg.fit(X, y, batch_size=batch_size, epochs=epochs)
    score = reg.score(X, y, batch_size=batch_size)
    assert np.isscalar(score) and np.isfinite(score)
    preds = reg.predict(X, batch_size=batch_size)
    assert preds.shape == (len(X), )


def build_fn_clf(input_shape, output_shape, hidden_layer_sizes=[]):
    model = Sequential()
    for size in hidden_layer_sizes:
        model.add(Dense(size, activation='relu'))
    model.add(Dense(np.prod(output_shape, dtype=np.uint8),
                    activation='softmax'))
    model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def build_fn_regs(input_shape, output_shape, hidden_layer_sizes=[]):
    model = Sequential()
    for size in hidden_layer_sizes:
        model.add(Dense(size, activation='relu'))
    model.add(Dense(np.prod(output_shape, dtype=np.uint8)))
    model.compile(optimizer, loss='mean_squared_error')
    return model


def build_fn_clss(input_shape, output_shape, hidden_layer_sizes=[]):
    model = Sequential()
    for size in hidden_layer_sizes:
        model.add(Dense(size, activation='relu'))
    model.add(Dense(np.prod(output_shape), activation='softmax'))
    model.compile(optimizer, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def build_fn_clscs(input_shape, output_shape, hidden_layer_sizes=[]):
    model = Sequential()
    model.add(Conv2D(3, (3, 3)))
    model.add(Flatten())
    for size in hidden_layer_sizes:
        model.add(Dense(size, activation='relu'))
    model.add(Dense(np.prod(output_shape), activation='softmax'))
    model.compile(optimizer, loss='categorical_crossentropy',
                 metrics=['accuracy'])
    return model


def build_fn_clscf(input_shape, output_shape, hidden_layer_sizes=[]):
    x = Input(shape=input_shape)
    z = Conv2D(3, (3, 3))(x)
    z = Flatten()(z)
    for size in hidden_layer_sizes:
        z = Dense(size, activation='relu')(z)
    y = Dense(np.prod(output_shape), activation='softmax')(z)
    model = Model(inputs=x, outputs=y)
    model.compile(optimizer, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def build_fn_multi(input_shape, output_shape, hidden_layer_sizes=[]):
    features = Input(shape=input_shape['features'], name='features')
    class_in = Input(shape=input_shape['class_in'], name='class_in')
    z = Concatenate()([features, class_in])
    for size in hidden_layer_sizes:
        z = Dense(size, activation='relu')(z)
    onehot = Dense(np.prod(output_shape['onehot']), activation='softmax',
                   name='onehot')(z)
    class_out = Dense(np.prod(output_shape['class_out']), name='class_out')(z)
    model = Model(inputs=[features, class_in], outputs=[onehot, class_out])
    model.compile(optimizer,
                  loss={'onehot': 'categorical_crossentropy',
                        'class_out': 'mse'},
                  metrics={'onehot': 'accuracy'})
    return model


def build_fn_reg(input_shape, output_shape, hidden_layer_sizes=[]):
    model = Sequential()
    for size in hidden_layer_sizes:
        model.add(Dense(size, activation='relu'))
    model.add(Dense(np.prod(output_shape, dtype=np.uint8)))
    model.compile(optimizer=optimizer, loss='mean_absolute_error',
                  metrics=['accuracy'])
    return model


class ClassBuildFnClf(object):

    def __call__(self, input_shape, output_shape, hidden_layer_sizes=[]):
        return build_fn_clf(input_shape, output_shape,
                            hidden_layer_sizes=hidden_layer_sizes)


class InheritClassBuildFnClf(KerasClassifier):

    def __call__(self, input_shape, output_shape, hidden_layer_sizes=[]):
        return build_fn_clf(input_shape, output_shape,
                            hidden_layer_sizes=hidden_layer_sizes)


class ClassBuildFnReg(object):

    def __call__(self, input_shape, output_shape, hidden_layer_sizes=[]):
        return build_fn_reg(input_shape, output_shape,
                            hidden_layer_sizes=hidden_layer_sizes)


class InheritClassBuildFnReg(KerasRegressor):

    def __call__(self, input_shape, output_shape, hidden_layer_sizes=[]):
        return build_fn_reg(input_shape, output_shape,
                            hidden_layer_sizes=hidden_layer_sizes)


CONFIG = {'MLPRegressor': (load_boston, KerasRegressor, build_fn_regs,
                           (BaggingRegressor, AdaBoostRegressor)),
          'MLPClassifier': (load_iris, KerasClassifier, build_fn_clss,
                            (BaggingClassifier, AdaBoostClassifier)),
          'CNNClassifier': (load_digits8x8, KerasClassifier, build_fn_clscs,
                            (BaggingClassifier, AdaBoostClassifier)),
          'CNNClassifierF': (load_digits8x8, KerasClassifier, build_fn_clscf,
                             (BaggingClassifier, AdaBoostClassifier))}


def test_classify_build_fn():
    """Tests classifier."""
    clf = KerasClassifier(build_fn=build_fn_clf,
                          hidden_layer_sizes=hidden_layer_sizes,
                          batch_size=batch_size, epochs=epochs)
    assert_classification_works(clf)
    assert_string_classification_works(clf)


def test_classify_class_build_fn():
    """Tests classifier specified by class."""
    clf = KerasClassifier(build_fn=ClassBuildFnClf(),
                          hidden_layer_sizes=hidden_layer_sizes,
                          batch_size=batch_size, epochs=epochs)
    assert_classification_works(clf)
    assert_string_classification_works(clf)


def test_classify_inherit_class_build_fn():
    """Tests classifier specified by inheritance."""
    clf = InheritClassBuildFnClf(build_fn=None,
                                 hidden_layer_sizes=hidden_layer_sizes,
                                 batch_size=batch_size, epochs=epochs)
    assert_classification_works(clf)
    assert_string_classification_works(clf)


def test_regression_build_fn():
    """Tests regressor."""
    reg = KerasRegressor(build_fn=build_fn_reg,
                         hidden_layer_sizes=hidden_layer_sizes,
                         batch_size=batch_size, epochs=epochs)
    assert_regression_works(reg)


def test_regression_class_build_fn():
    """Tests regressor specified by class."""
    reg = KerasRegressor(build_fn=ClassBuildFnReg(),
                         hidden_layer_sizes=hidden_layer_sizes,
                         batch_size=batch_size, epochs=epochs)
    assert_regression_works(reg)


def test_regression_inherit_class_build_fn():
    """Tests regressor specified by inheritance."""
    reg = InheritClassBuildFnReg(build_fn=None,
                                 hidden_layer_sizes=hidden_layer_sizes,
                                 batch_size=batch_size, epochs=epochs)
    assert_regression_works(reg)


def test_regression_predict_shape_correct_num_test(num_test=1):
    """Tests regressor predictions."""
    X, y = load_boston(return_X_y=True)
    reg = KerasRegressor(build_fn=build_fn_reg,
                         hidden_layer_sizes=hidden_layer_sizes,
                         batch_size=batch_size, epochs=epochs)
    reg.fit(X, y, batch_size=batch_size, epochs=epochs)
    preds = reg.predict(X[:num_test], batch_size=batch_size)
    assert preds.shape == (num_test, )


def test_standalone():
    """Tests standalone estimator."""
    for config in ['MLPRegressor', 'MLPClassifier', 'CNNClassifier',
                   'CNNClassifierF']:
        hidden_layer_sizes = [10, 10, 10]
        loader, model, build_fn, _ = CONFIG[config]
        estimator = model(build_fn, epochs=1)
        assert_predictor_works(estimator, loader)


def test_pipeline():
    """Tests compatibility with Scikit-learn's pipeline."""
    for config in ['MLPRegressor', 'MLPClassifier']:
        loader, model, build_fn, _ = CONFIG[config]
        estimator = model(build_fn, epochs=1)
        estimator = Pipeline([('s', StandardScaler()), ('e', estimator)])
        assert_predictor_works(estimator, loader)


def test_searchcv():
    """Tests compatibility with Scikit-learn's hyperparameter search CV."""
    for config in ['MLPRegressor', 'MLPClassifier', 'CNNClassifier',
                   'CNNClassifierF']:
        loader, model, build_fn, _ = CONFIG[config]
        estimator = model(build_fn, epochs=1, validation_split=0.1)
        assert_predictor_works(GridSearchCV(estimator,
                                            {'hidden_layer_sizes': [[], [5]]}),
                                            loader)
        assert_predictor_works(RandomizedSearchCV(estimator,
                                                  {'epochs': randint(1, 5)},
                                                  n_iter=2), loader)


def test_ensemble():
    """Tests compatibility with Scikit-learn's ensembles."""
    for config in ['MLPRegressor', 'MLPClassifier']:
        loader, model, build_fn, ensembles = CONFIG[config]
        base_estimator = model(build_fn, epochs=1)
        for ensemble in ensembles:
            estimator = ensemble(base_estimator=base_estimator, n_estimators=2)
            assert_predictor_works(estimator, loader)


def test_calibratedclassifiercv():
    """Tests compatibility with Scikit-learn's calibrated classifier CV."""
    for config in ['MLPClassifier']:
        loader, _, build_fn, _ = CONFIG[config]
        base_estimator = KerasClassifier(build_fn, epochs=1)
        estimator = CalibratedClassifierCV(base_estimator=base_estimator)
        assert_predictor_works(estimator, loader)


def test_transformedtargetregressor():
    """Tests compatibility with Scikit-learn's transformed target regressor."""
    for config in ['MLPRegressor']:
        loader, _, build_fn, _ = CONFIG[config]
        base_estimator = KerasRegressor(build_fn, epochs=1)
        estimator = TransformedTargetRegressor(regressor=base_estimator,
                                               transformer=StandardScaler())
        assert_predictor_works(estimator, loader)


def test_standalone_multi():
    """Tests standalone estimator with multiple inputs and outputs."""
    estimator = BaseWrapper(build_fn_multi, epochs=1)
    data = load_iris()
    features = data.data
    klass = data.target.reshape((-1, 1)).astype(np.float32)
    onehot = to_categorical(data.target)
    estimator.fit({'features': features, 'class_in': klass},
                  {'onehot': onehot, 'class_out': klass})
    preds = estimator.predict({'features': features, 'class_in': klass})
    score = estimator.score({'features': features, 'class_in': klass},
                            {'onehot': onehot, 'class_out': klass})
    serialized_estimator = pickle.dumps(estimator)
    deserialized_estimator = pickle.loads(serialized_estimator)
    preds = deserialized_estimator.predict({'features': features,
                                            'class_in': klass})
    score = deserialized_estimator.score({'features': features,
                                          'class_in': klass},
                                         {'onehot': onehot, 'class_out': klass})


if __name__ == '__main__':
    pytest.main([__file__])
