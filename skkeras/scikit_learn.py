"""Wrapper for using the Scikit-Learn API with Keras models."""

from inspect import signature
from keras.layers import Dense, Input
from keras.models import Model
from keras.ops import prod
from keras.utils import to_categorical
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted

_filter_args = lambda args, fn: {k: v for k, v in args.items() if k in signature(fn).parameters}


def _process_labels(labels):
    labels = np.array(labels, ndmin=1, ndmax=2)
    if len(labels.shape) == 2 and labels.shape[1] > 1:
        multilabel = True
        classes = np.arange(labels.shape[1])
    else:
        multilabel = False
        classes = np.unique(labels)
        labels = to_categorical(np.searchsorted(classes, labels))
    return labels, classes, len(classes), multilabel


def _build_fn(
    input_shape,
    output_shape,
    input_layer=Input,
    output_layer=Dense,
    output_activation=None,
    hidden=None,
    compile_kwargs={},
):
    """Build a neural network.

    Build a neural network with the specified hyper-parameters.
    Scikit-learn only supports single input and single output neural network architectures.

    Parameters
    ----------
    input_shape: tuple
        Input shape.
    output_shape: tuple
        Output shape.
    output_layer: keras function, default=Dense
        Output layer function.
    output_activation: str or None, default=None
        Activation function for the output layer.
    hidden: keras function or None, default=None
        Hidden layers function.
    compile_kwargs: keyword arguments, default={"loss": "mean_squared_error", "metrics": ["r2_score"],
        "optimizer": "adam"}
        Additional keyword arguments to be passed to the compile method.

    Returns
    -------
    Model

    """
    x = input_layer(shape=input_shape)
    z = hidden(x) if hidden else x
    y = output_layer(units=int(prod(output_shape)), activation=output_activation)(z)
    model = Model(x, y)
    kwargs = {"loss": "mse", "metrics": ["r2_score"], "optimizer": "adam"}
    kwargs.update(compile_kwargs)
    model.compile(**kwargs)
    return model


class BaseWrapper(BaseEstimator):
    """Base class for the Keras scikit-learn wrapper.

    # Arguments
        build_fn : callable function or class instance
        **kwargs : model parameters & fitting parameters

    The `build_fn` should construct, compile and return a Keras model, which
    will then be used to fit/predict. One of the following
    three values could be passed to `build_fn`:
    1. A function
    2. An instance of a class that implements the `__call__` method
    3. None. This means you implement a class that inherits from either
    `KerasClassifier` or `KerasRegressor`. The `__call__` method of the
    present class will then be treated as the default `build_fn`.

    `kwargs` takes both model parameters and fitting parameters. Legal model
    parameters are the arguments of `build_fn`. Note that like all other
    estimators in scikit-learn, `build_fn` should provide default values for
    its arguments, so that you could create the estimator without passing any
    values to `kwargs`.

    `kwargs` could also accept parameters for calling `fit`, `predict`,
    and `score` methods (e.g., `epochs`, `batch_size`).
    fitting (predicting) parameters are selected in the following order:
    1. Values passed to the dictionary arguments of `fit`, `predict` and
    `score` methods
    2. Values passed to `kwargs`
    3. The default values of the `keras.models.Model` `fit`, `predict` and
    `score` methods

    When using scikit-learn's `grid_search` API, legal tunable parameters are
    those you could pass to `kwargs`, including fitting parameters.
    In other words, you could use `grid_search` to search for the best
    `batch_size` or `epochs` as well as the model parameters.
    """

    _estimator_type = "regressor"

    def __init__(self, build_fn=_build_fn, **kwargs):
        self.build_fn = build_fn
        self.kwargs = kwargs

    def set_params(self, **params):
        """Sets the parameters of this estimator.

        # Arguments
            **params : Dictionary of parameter names mapped to their values.

        # Returns
            self
        """
        self.kwargs.update(params)
        return self

    def fit(self, X, y, sample_weight=None, **kwargs):
        """Constructs a new model with `build_fn` & fit the model to `(X, y)`.

        # Arguments
            X : Input data. It could be:
                - A Numpy array (or array-like), or a list of arrays (in case
                  the model has multiple inputs).
                - A dict mapping input names to the corresponding array/tensors,
                  if the model has named inputs.
                - None (default) if feeding from framework-native tensors.
            y : Target data. Like the input data `X`, it could be either Numpy
                array(s), framework-native tensor(s), list of Numpy arrays (if
                the model has multiple outputs) or None (default) if feeding
                from framework-native tensors.
                If output layers in the model are named, you can also pass a
                dictionary mapping output names to Numpy arrays.
            sample_weight : Numpy array of weights for the training samples, or
                a list of Numpy arrays (if the model has multiple outputs).

            **kwargs : dictionary arguments
                Legal arguments are the arguments of `Model.fit`

        # Returns
            history : object
                details about the training history at each epoch.
        """

        def _get_shape(X):
            if isinstance(X, dict):
                shape = {k: v.shape[1:] for k, v in X.items()}
            elif isinstance(X, list) or isinstance(X, tuple):
                shape = [i.shape[1:] for i in X]
            else:
                shape = X.shape[1:]
            return shape

        build = self.build_fn if self.build_fn else self.__call__
        self.kwargs.update(kwargs)
        self.model_ = build(_get_shape(X), _get_shape(y), **_filter_args(self.kwargs, build))
        return self.model_.fit(X, y, sample_weight=sample_weight, **_filter_args(self.kwargs, Model.fit))

    def predict(self, X, **kwargs):
        """Returns predictions for the given test data.

        # Arguments
            X : Input data. It could be:
                - A Numpy array (or array-like), or a list of arrays (in case
                  the model has multiple inputs).
                - A dict mapping input names to the corresponding array/tensors,
                  if the model has named inputs.
                - None (default) if feeding from framework-native tensors.
            **kwargs : dictionary arguments
                Legal arguments are the arguments of `Model.predict`.

        # Returns
            Numpy array(s) of predictions.
        """
        check_is_fitted(self, ["model_"])
        return self.model_.predict(X, **_filter_args(kwargs, Model.predict))

    def score(self, X, y, **kwargs):
        """Returns the mean loss on the given test data and labels.

        # Arguments
            X : Input data. It could be:
                - A Numpy array (or array-like), or a list of arrays (in case
                  the model has multiple inputs).
                - A dict mapping input names to the corresponding array/tensors,
                  if the model has named inputs.
                - None (default) if feeding from framework-native tensors.
            y : Target data. Like the input data `X`, it could be either Numpy
                array(s), framework-native tensor(s), list of Numpy arrays (if
                the model has multiple outputs) or None (default) if feeding
                from framework-native tensors.
                If output layers in the model are named, you can also pass a
                dictionary mapping output names to Numpy arrays.
            **kwargs : dictionary arguments
                Legal arguments are the arguments of `Model.evaluate`.

        # Returns
            Scalar test loss (if the model has a single output and no metrics)
            or list of scalars (if the model has multiple outputs and/or
            metrics). The attribute `model.metrics_names` will give you the
            display labels for the scalar outputs.
        """
        check_is_fitted(self, ["model_"])
        return self.model_.evaluate(X, y, **_filter_args(kwargs, Model.evaluate))[1]

    def summary(self, **kwargs):
        """Prints a string summary of the network.

        # Arguments
            **kwargs : dictionary arguments
                Legal arguments are the arguments of `Model.summary`.
        """
        return self.model_.summary(**_filter_args(kwargs, Model.summary))


class KerasClassifier(ClassifierMixin, BaseWrapper):
    """Implementation of the scikit-learn classifier API for Keras."""

    _estimator_type = "classifier"

    def fit(self, X, y, sample_weight=None, **kwargs):
        y, self.classes_, self.n_classes_, self.multilabel_ = _process_labels(y)
        if self.multilabel_:
            kwargs.update({"output_activation": "sigmoid", "compile_kwargs": {"loss": "ce", "metrics": ["accuracy"]}})
        elif self.n_classes_ > 2:
            kwargs.update({"output_activation": "softmax", "compile_kwargs": {"loss": "ce", "metrics": ["accuracy"]}})
        else:
            kwargs.update({"output_activation": "sigmoid", "compile_kwargs": {"loss": "bce", "metrics": ["accuracy"]}})
        return BaseWrapper.fit(self, X, y, sample_weight=sample_weight, **kwargs)

    predict_proba = BaseWrapper.predict

    def predict(self, X, **kwargs):
        return self.classes_[self.predict_proba(X, **kwargs).argmax(axis=-1)]

    def score(self, X, y, **kwargs):
        y, _, _, _ = _process_labels(y)
        return BaseWrapper.score(self, X, y, **kwargs)


class KerasRegressor(RegressorMixin, BaseWrapper):
    """Implementation of the scikit-learn regressor API for Keras."""

    def fit(self, X, y, sample_weight=None, **kwargs):
        kwargs.update({"output_activation": None, "compile_kwargs": {"loss": "mse", "metrics": ["r2_score"]}})
        return BaseWrapper.fit(self, X, y, sample_weight=sample_weight, **kwargs)

    def predict(self, X, **kwargs):
        return BaseWrapper.predict(self, X, **kwargs).squeeze(axis=-1)

    score = BaseWrapper.score
