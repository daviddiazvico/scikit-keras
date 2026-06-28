from keras import backend as K
from keras.layers import Concatenate, Conv2D, Dense, Flatten, Input
from keras.models import Model, Sequential
from keras.utils import set_random_seed, to_categorical
import numpy as np
import pickle
import pytest
from scipy.stats import randint
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import TransformedTargetRegressor
from sklearn.datasets import load_breast_cancer, load_diabetes, load_digits, load_iris, make_multilabel_classification
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor, BaggingClassifier, BaggingRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import Bunch

from skkeras.scikit_learn import BaseWrapper, KerasClassifier, KerasRegressor
from skkeras.models import Straight

hidden_layer_sizes = [5]
batch_size = 32
epochs = 1
optimizer = "adam"

np.random.seed(42)
set_random_seed(42)


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
        X = np.array([X[i : i + window] for i in range(X.shape[0] - window + 1)])
        if y is not None:
            y = (
                np.array([y[i : i + window] for i in range(y.shape[0] - window + 1)])
                if return_sequences
                else np.array(y[window - 1 :])
            )
    return X, y


def check_architecture(estimator, layer_types):
    """Checks architecture."""
    layers = estimator.get_config()["layers"]
    assert len(layers) == len(layer_types)
    for layer, layer_type in zip(layers, layer_types):
        assert layer["class_name"] == layer_type


def test_architecture():
    """Tests architecture."""
    digits = load_digits()
    digits.data = digits.data.reshape([digits.data.shape[0], 8, 8, 1]) / 16.0
    K.set_image_data_format("channels_last")
    diabetes_ts = load_diabetes()
    diabetes_ts.data, diabetes_ts.target = time_series(diabetes_ts.data, y=diabetes_ts.target, window=3)
    digits_ts = load_digits()
    digits_ts.data = digits_ts.data.reshape([digits_ts.data.shape[0], 8, 8, 1]) / 16.0
    digits_ts.data, digits_ts.target = time_series(digits_ts.data, y=digits_ts.target, window=3)
    multilabel = Bunch()
    multilabel.data, multilabel.target = make_multilabel_classification(
        n_samples=100, n_features=20, n_classes=3, n_labels=2
    )

    configurations = {
        "pcp_bin": {"dataset": load_breast_cancer(), "estimator": KerasClassifier(), "layers": ["InputLayer", "Dense"]},
        "pcp_mcls": {"dataset": load_iris(), "estimator": KerasClassifier(), "layers": ["InputLayer", "Dense"]},
        "pcp_mlbl": {"dataset": multilabel, "estimator": KerasClassifier(), "layers": ["InputLayer", "Dense"]},
        "mlp": {
            "dataset": load_iris(),
            "estimator": KerasClassifier(hidden=Straight(dense_units=[10])),
            "layers": ["InputLayer", "Dense", "Dense"],
        },
        "batchnormalization": {
            "dataset": load_iris(),
            "estimator": KerasClassifier(hidden=Straight(batchnormalization=True, dense_units=[10])),
            "layers": ["InputLayer", "BatchNormalization", "Dense", "Dense"],
        },
        "dropout": {
            "dataset": load_iris(),
            "estimator": KerasClassifier(hidden=Straight(dense_units=[10], dropout_rate=0.1)),
            "layers": ["InputLayer", "Dense", "Dropout", "Dense"],
        },
        "cnn": {
            "dataset": digits,
            "estimator": KerasClassifier(
                hidden=Straight(convolution_filters=[1], convolution_kernel_size=[(2, 2)]),
            ),
            "layers": ["InputLayer", "Conv2D", "Flatten", "Dense"],
        },
        "cnnpool": {
            "dataset": digits,
            "estimator": KerasClassifier(
                hidden=Straight(convolution_filters=[1], convolution_kernel_size=[(2, 2)], pooling_pool_size=[(1, 1)]),
            ),
            "layers": ["InputLayer", "Conv2D", "MaxPooling2D", "Flatten", "Dense"],
        },
        "cnnmlp": {
            "dataset": digits,
            "estimator": KerasClassifier(
                hidden=Straight(convolution_filters=[1], convolution_kernel_size=[(2, 2)], dense_units=[10]),
            ),
            "layers": ["InputLayer", "Conv2D", "Flatten", "Dense", "Dense"],
        },
        "rnn": {
            "dataset": diabetes_ts,
            "estimator": KerasRegressor(hidden=Straight(recurrent_units=[10])),
            "layers": ["InputLayer", "LSTM", "Dense"],
        },
        "rnnmlp": {
            "dataset": diabetes_ts,
            "estimator": KerasRegressor(hidden=Straight(recurrent_units=[10], dense_units=[10])),
            "layers": ["InputLayer", "LSTM", "Dense", "Dense"],
        },
        "cnnrnn": {
            "dataset": digits_ts,
            "estimator": KerasClassifier(
                hidden=Straight(convolution_filters=[1], convolution_kernel_size=[(2, 2)], recurrent_units=[10]),
            ),
            "layers": ["InputLayer", "TimeDistributed", "TimeDistributed", "LSTM", "Dense"],
        },
        "cnnrnnmlp": {
            "dataset": digits_ts,
            "estimator": KerasClassifier(
                hidden=Straight(
                    convolution_filters=[1], convolution_kernel_size=[(2, 2)], recurrent_units=[10], dense_units=[10]
                ),
            ),
            "layers": [
                "InputLayer",
                "TimeDistributed",
                "TimeDistributed",
                "LSTM",
                "Dense",
                "Dense",
            ],
        },
    }
    for test, config in configurations.items():
        data = config["dataset"]
        estimator = config["estimator"]
        layer_types = config["layers"]
        estimator.fit(data.data, data.target, epochs=1)
        estimator.summary()
        check_architecture(estimator.model_, layer_types)
        estimator.predict(data.data)
        estimator.score(data.data, data.target)
        if hasattr(estimator, "predict_proba"):
            estimator.predict_proba(data.data)


def load_digits8x8():
    data = load_digits()
    data.data = data.data.reshape([data.data.shape[0], 8, 8, 1]) / 16.0
    K.set_image_data_format("channels_last")
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
    clf.fit(X, y, sample_weight=np.ones(X.shape[0]), batch_size=batch_size, epochs=epochs)
    score = clf.score(X, y, batch_size=batch_size)
    assert np.isscalar(score) and np.isfinite(score)
    preds = clf.predict(X, batch_size=batch_size)
    assert preds.shape == (len(X),)
    for prediction in np.unique(preds):
        assert prediction in range(num_classes)
    proba = clf.predict_proba(X, batch_size=batch_size)
    assert proba.shape == (len(X), num_classes)
    assert np.allclose(np.sum(proba, axis=1), np.ones(len(X)))


def assert_string_classification_works(clf):
    X, y = load_iris(return_X_y=True)
    num_classes = len(np.unique(y))
    string_classes = ["cls{}".format(x) for x in range(num_classes)]
    str_y = np.array(string_classes)[y]
    clf.fit(X, str_y, batch_size=batch_size, epochs=epochs)
    score = clf.score(X, str_y, batch_size=batch_size)
    assert np.isscalar(score) and np.isfinite(score)
    preds = clf.predict(X, batch_size=batch_size)
    assert preds.shape == (len(X),)
    for prediction in np.unique(preds):
        assert prediction in string_classes
    proba = clf.predict_proba(X, batch_size=batch_size)
    assert proba.shape == (len(X), num_classes)
    assert np.allclose(np.sum(proba, axis=1), np.ones(len(X)))


def assert_regression_works(reg):
    X, y = load_diabetes(return_X_y=True)
    reg.fit(X, y, batch_size=batch_size, epochs=epochs)
    score = reg.score(X, y, batch_size=batch_size)
    assert np.isscalar(score) and np.isfinite(score)
    preds = reg.predict(X, batch_size=batch_size)
    assert preds.shape == (len(X),)


def build_fn_clf(input_shape, output_shape, hidden_layer_sizes=[]):
    model = Sequential()
    for size in hidden_layer_sizes:
        model.add(Dense(int(size), activation="relu"))
    model.add(Dense(int(np.prod(output_shape, dtype=np.uint8)), activation="softmax"))
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def build_fn_regs(input_shape, output_shape, hidden_layer_sizes=[]):
    model = Sequential()
    for size in hidden_layer_sizes:
        model.add(Dense(int(size), activation="relu"))
    model.add(Dense(int(np.prod(output_shape, dtype=np.uint8))))
    model.compile(optimizer, loss="mean_squared_error", metrics=["r2_score"])
    return model


def build_fn_clss(input_shape, output_shape, hidden_layer_sizes=[]):
    model = Sequential()
    for size in hidden_layer_sizes:
        model.add(Dense(int(size), activation="relu"))
    model.add(Dense(int(np.prod(output_shape)), activation="softmax"))
    model.compile(optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def build_fn_clscs(input_shape, output_shape, hidden_layer_sizes=[]):
    model = Sequential()
    model.add(Conv2D(3, (3, 3)))
    model.add(Flatten())
    for size in hidden_layer_sizes:
        model.add(Dense(int(size), activation="relu"))
    model.add(Dense(int(np.prod(output_shape)), activation="softmax"))
    model.compile(optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def build_fn_clscf(input_shape, output_shape, hidden_layer_sizes=[]):
    X = Input(shape=input_shape)
    z = Conv2D(3, (3, 3))(X)
    z = Flatten()(z)
    for size in hidden_layer_sizes:
        z = Dense(int(size), activation="relu")(z)
    y = Dense(int(np.prod(output_shape)), activation="softmax")(z)
    model = Model(inputs=X, outputs=y)
    model.compile(optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def build_fn_multi_dict(input_shape, output_shape, hidden_layer_sizes=[]):
    features = Input(shape=input_shape["features"], name="features")
    class_in = Input(shape=input_shape["class_in"], name="class_in")
    z = Concatenate()([features, class_in])
    for size in hidden_layer_sizes:
        z = Dense(int(size), activation="relu")(z)
    onehot = Dense(int(np.prod(output_shape["onehot"])), activation="softmax", name="onehot")(z)
    class_out = Dense(int(np.prod(output_shape["class_out"])), name="class_out")(z)
    model = Model(inputs=[features, class_in], outputs=[onehot, class_out])
    model.compile(
        optimizer,
        loss={"onehot": "categorical_crossentropy", "class_out": "mse"},
        metrics={"onehot": "accuracy"},
    )
    return model


def build_fn_multi_tuple(input_shape, output_shape, hidden_layer_sizes=[]):
    features = Input(shape=input_shape[0], name="features")
    class_in = Input(shape=input_shape[1], name="class_in")
    z = Concatenate()([features, class_in])
    for size in hidden_layer_sizes:
        z = Dense(int(size), activation="relu")(z)
    onehot = Dense(int(np.prod(output_shape[0])), activation="softmax", name="onehot")(z)
    class_out = Dense(int(np.prod(output_shape[1])), name="class_out")(z)
    model = Model(inputs=[features, class_in], outputs=[onehot, class_out])
    model.compile(
        optimizer,
        loss={"onehot": "categorical_crossentropy", "class_out": "mse"},
        metrics={"onehot": "accuracy"},
    )
    return model


def build_fn_reg(input_shape, output_shape, hidden_layer_sizes=[]):
    model = Sequential()
    for size in hidden_layer_sizes:
        model.add(Dense(int(size), activation="relu"))
    model.add(Dense(int(np.prod(output_shape, dtype=np.uint8))))
    model.compile(optimizer=optimizer, loss="mean_absolute_error", metrics=["r2_score"])
    return model


class ClassBuildFnClf(object):
    def __call__(self, input_shape, output_shape, hidden_layer_sizes=[]):
        return build_fn_clf(input_shape, output_shape, hidden_layer_sizes=hidden_layer_sizes)


class InheritClassBuildFnClf(KerasClassifier):
    def __call__(self, input_shape, output_shape, hidden_layer_sizes=[]):
        return build_fn_clf(input_shape, output_shape, hidden_layer_sizes=hidden_layer_sizes)


class ClassBuildFnReg(object):
    def __call__(self, input_shape, output_shape, hidden_layer_sizes=[]):
        return build_fn_reg(input_shape, output_shape, hidden_layer_sizes=hidden_layer_sizes)


class InheritClassBuildFnReg(KerasRegressor):
    def __call__(self, input_shape, output_shape, hidden_layer_sizes=[]):
        return build_fn_reg(input_shape, output_shape, hidden_layer_sizes=hidden_layer_sizes)


CONFIG = {
    "MLPRegressor": (
        load_diabetes,
        KerasRegressor,
        build_fn_regs,
        (BaggingRegressor, AdaBoostRegressor),
    ),
    "MLPClassifier": (
        load_iris,
        KerasClassifier,
        build_fn_clss,
        (BaggingClassifier, AdaBoostClassifier),
    ),
    "CNNClassifier": (
        load_digits8x8,
        KerasClassifier,
        build_fn_clscs,
        (BaggingClassifier, AdaBoostClassifier),
    ),
    "CNNClassifierF": (
        load_digits8x8,
        KerasClassifier,
        build_fn_clscf,
        (BaggingClassifier, AdaBoostClassifier),
    ),
}


def test_classify_build_fn():
    """Tests classifier."""
    clf = KerasClassifier(
        build_fn=build_fn_clf,
        hidden_layer_sizes=hidden_layer_sizes,
        batch_size=batch_size,
        epochs=epochs,
    )
    assert_classification_works(clf)
    assert_string_classification_works(clf)


def test_classify_class_build_fn():
    """Tests classifier specified by class."""
    clf = KerasClassifier(
        build_fn=ClassBuildFnClf(),
        hidden_layer_sizes=hidden_layer_sizes,
        batch_size=batch_size,
        epochs=epochs,
    )
    assert_classification_works(clf)
    assert_string_classification_works(clf)


def test_classify_inherit_class_build_fn():
    """Tests classifier specified by inheritance."""
    clf = InheritClassBuildFnClf(
        build_fn=None,
        hidden_layer_sizes=hidden_layer_sizes,
        batch_size=batch_size,
        epochs=epochs,
    )
    assert_classification_works(clf)
    assert_string_classification_works(clf)


def test_regression_build_fn():
    """Tests regressor."""
    reg = KerasRegressor(
        build_fn=build_fn_reg,
        hidden_layer_sizes=hidden_layer_sizes,
        batch_size=batch_size,
        epochs=epochs,
    )
    assert_regression_works(reg)


def test_regression_class_build_fn():
    """Tests regressor specified by class."""
    reg = KerasRegressor(
        build_fn=ClassBuildFnReg(),
        hidden_layer_sizes=hidden_layer_sizes,
        batch_size=batch_size,
        epochs=epochs,
    )
    assert_regression_works(reg)


def test_regression_inherit_class_build_fn():
    """Tests regressor specified by inheritance."""
    reg = InheritClassBuildFnReg(
        build_fn=None,
        hidden_layer_sizes=hidden_layer_sizes,
        batch_size=batch_size,
        epochs=epochs,
    )
    assert_regression_works(reg)


def test_regression_predict_shape_correct_num_test(num_test=1):
    """Tests regressor predictions."""
    X, y = load_diabetes(return_X_y=True)
    reg = KerasRegressor(
        build_fn=build_fn_reg,
        hidden_layer_sizes=hidden_layer_sizes,
        batch_size=batch_size,
        epochs=epochs,
    )
    reg.fit(X, y, batch_size=batch_size, epochs=epochs)
    preds = reg.predict(X[:num_test], batch_size=batch_size)
    assert preds.shape == (num_test,)


def test_standalone():
    """Tests standalone estimator."""
    for config in ["MLPRegressor", "MLPClassifier", "CNNClassifier", "CNNClassifierF"]:
        hidden_layer_sizes = [10, 10, 10]
        loader, model, build_fn, _ = CONFIG[config]
        estimator = model(build_fn, epochs=1)
        assert_predictor_works(estimator, loader)


def test_pipeline():
    """Tests compatibility with Scikit-learn's pipeline."""
    for config in ["MLPRegressor", "MLPClassifier"]:
        loader, model, build_fn, _ = CONFIG[config]
        estimator = model(build_fn, epochs=1)
        estimator = Pipeline([("s", StandardScaler()), ("e", estimator)])
        assert_predictor_works(estimator, loader)


def test_searchcv():
    """Tests compatibility with Scikit-learn's hyperparameter search CV."""
    for config in ["MLPRegressor", "MLPClassifier", "CNNClassifier", "CNNClassifierF"]:
        loader, model, build_fn, _ = CONFIG[config]
        estimator = model(build_fn, epochs=1, validation_split=0.1)
        assert_predictor_works(GridSearchCV(estimator, {"hidden_layer_sizes": [[], [5]]}), loader)
        assert_predictor_works(RandomizedSearchCV(estimator, {"epochs": randint(1, 5)}, n_iter=2), loader)


def test_ensemble():
    """Tests compatibility with Scikit-learn's ensembles."""
    for config in ["MLPRegressor", "MLPClassifier"]:
        loader, model, build_fn, ensembles = CONFIG[config]
        base_estimator = model(build_fn, epochs=1)
        for ensemble in ensembles:
            estimator = ensemble(estimator=base_estimator, n_estimators=2)
            assert_predictor_works(estimator, loader)


def test_calibratedclassifiercv():
    """Tests compatibility with Scikit-learn's calibrated classifier CV."""
    for config in ["MLPClassifier"]:
        loader, _, build_fn, _ = CONFIG[config]
        base_estimator = KerasClassifier(build_fn, epochs=1)
        estimator = CalibratedClassifierCV(estimator=base_estimator)
        assert_predictor_works(estimator, loader)


def test_transformedtargetregressor():
    """Tests compatibility with Scikit-learn's transformed target regressor."""
    for config in ["MLPRegressor"]:
        loader, _, build_fn, _ = CONFIG[config]
        base_estimator = KerasRegressor(build_fn, epochs=1)
        estimator = TransformedTargetRegressor(regressor=base_estimator, transformer=StandardScaler())
        assert_predictor_works(estimator, loader)


def test_standalone_multi_dict():
    """Tests standalone estimator with multiple inputs and outputs."""
    estimator = BaseWrapper(build_fn_multi_dict, epochs=1)
    data = load_iris()
    features = data.data
    klass = data.target.reshape((-1, 1)).astype(np.float32)
    onehot = to_categorical(data.target)
    estimator.fit(
        {"features": features, "class_in": klass},
        {"onehot": onehot, "class_out": klass},
    )
    preds = estimator.predict({"features": features, "class_in": klass})
    score = estimator.score(
        {"features": features, "class_in": klass},
        {"onehot": onehot, "class_out": klass},
    )
    serialized_estimator = pickle.dumps(estimator)
    deserialized_estimator = pickle.loads(serialized_estimator)
    preds = deserialized_estimator.predict({"features": features, "class_in": klass})
    score = deserialized_estimator.score(
        {"features": features, "class_in": klass}, {"onehot": onehot, "class_out": klass}
    )


def test_standalone_multi_tuple():
    """Tests standalone estimator with multiple inputs and outputs."""
    estimator = BaseWrapper(build_fn_multi_tuple, epochs=1)
    data = load_iris()
    features = data.data
    klass = data.target.reshape((-1, 1)).astype(np.float32)
    onehot = to_categorical(data.target)
    estimator.fit((features, klass), (onehot, klass))
    preds = estimator.predict({"features": features, "class_in": klass})
    score = estimator.score((features, klass), (onehot, klass))
    serialized_estimator = pickle.dumps(estimator)
    deserialized_estimator = pickle.loads(serialized_estimator)
    preds = deserialized_estimator.predict({"features": features, "class_in": klass})
    score = deserialized_estimator.score((features, klass), (onehot, klass))


if __name__ == "__main__":
    pytest.main([__file__])
