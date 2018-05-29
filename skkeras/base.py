"""
Scikit-learn-compatible Keras models.

@author: David Diaz Vico
@license: MIT
"""

from functools import partialmethod
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Input, TimeDistributed
from keras.models import Model, load_model, save_model
from keras.optimizers import (Adadelta, Adagrad, Adam, Adamax, Nadam, RMSprop,
                              SGD)
from keras.regularizers import l1 as l1_, l2 as l2_, l1_l2 as l1_l2_
from keras.utils import to_categorical
import numpy as np
from sklearn.base import (BaseEstimator, ClassifierMixin, RegressorMixin,
                          TransformerMixin)
from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from tempfile import NamedTemporaryFile


###############################################################################
#  Keras Model serialization
#  http://zachmoshe.com/2017/04/03/pickling-keras-models.html
###############################################################################


def __getstate__(self):
    with NamedTemporaryFile(suffix='.hdf5') as handler:
        save_model(self, handler.name, overwrite=True)
        return {'model_str': handler.read()}


Model.__getstate__ = __getstate__


def __setstate__(self, state):
    with NamedTemporaryFile(suffix='.hdf5') as handler:
        handler.write(state['model_str'])
        handler.flush()
        self.__dict__ = load_model(handler.name).__dict__


Model.__setstate__ = __setstate__


###############################################################################
#  Time-series formatting
###############################################################################


def _time_series(X, y=None, window=None, return_sequences=False):
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
            y = np.array([y[i:i + window] for i in range(y.shape[0] - window + 1)]) if return_sequences else y[window - 1:]
    return X, y


###############################################################################
#  Optimizer
###############################################################################


class Optimizer():

    """Optimizer.

    Optimizer class.

    Parameters
    ----------
    optimizer: {"sgd", "rmsprop", "adagrad", "adadelta", "adam", "adamax",
               "nadam"}, default='adam'
               Optimizer
    lr: float>=0, default=0.001
        Learning rate.
    momentum: float>=0, default=0.0
              Parameter updates momentum.
    nesterov: boolean, default=False
              Whether to apply Nesterov momentum.
    decay: float>=0, default=0.0
           Learning rate decay over each update.
    rho: float>=0, default=0.9
    epsilon: float>=0, default=1e-08
             Fuzz factor.
    beta_1: float in (0, 1), default=0.9
    beta_2: float in (0, 1), default=0.999
    schedule_decay: , default=0.004

    Returns
    -------
    Optimizer

    """

    def __new__(cls, optimizer='adam', lr=0.001, momentum=0.0, nesterov=False,
                decay=0.0, rho=0.9, epsilon=1e-08, beta_1=0.9, beta_2=0.999,
                schedule_decay=0.004):
        optimizers = {'sgd':  SGD(lr=lr, momentum=momentum, decay=decay,
                                  nesterov=nesterov),
                      'rmsprop': RMSprop(lr=lr, rho=rho, epsilon=epsilon,
                                         decay=decay),
                      'adagrad': Adagrad(lr=lr, epsilon=epsilon, decay=decay),
                      'adadelta': Adadelta(lr=lr, rho=rho, epsilon=epsilon,
                                           decay=decay),
                      'adam': Adam(lr=lr, beta_1=beta_1, beta_2=beta_2,
                                   epsilon=epsilon, decay=decay),
                      'adamax': Adamax(lr=lr, beta_1=beta_1, beta_2=beta_2,
                                       epsilon=epsilon, decay=decay),
                      'nadam': Nadam(lr=lr, beta_1=beta_1, beta_2=beta_2,
                                     epsilon=epsilon,
                                     schedule_decay=schedule_decay)}
        return optimizers[optimizer]


###############################################################################
#  Regularization
###############################################################################


class Regularizer():

    """Regularizer.

    Regularizer class.

    Parameters
    ----------
    l1: float, default=None
        L1 regularization factor.
    l2: float, default=None
        L2 regularization factor.

    Returns
    -------
    Regularizer.

    """

    def __new__(cls, l1=None, l2=None):
        if (l1 is None) and (l2 is not None):
            regularizer = l2_(l=l2)
        elif (l1 is not None) and (l2 is None):
            regularizer = l1_(l=l1)
        elif (l1 is not None) and (l2 is not None):
            regularizer = l1_l2_(l1=l1, l2=l2)
        else:
            regularizer = None
        return regularizer


###############################################################################
#  Base feed-forward class
###############################################################################


class BaseFeedForward(BaseEstimator, TransformerMixin):

    """Feed-forward regressor/classifier.

    This model optimizes the MSE/categorical-crossentropy function using
    back-propagation.

    Parameters
    ----------
    architecture: keras function, default=None
                  Feature transformation.
    activation: string/function, default='linear'/'softmax'
                Activation function to use.
    use_bias: boolean, default=True
              Whether the layer uses a bias vector.
    kernel_initializer: string/function, default='glorot_uniform'
                        Initializer for the kernel weights matrix.
    bias_initializer: string/function, default='zeros'
                      Initializer for the bias vector.
    kernel_regularizer_l1: float, default=None
                           L1 regularization factor applied to the kernel
                           weights matrix.
    kernel_regularizer_l2: float, default=None
                           L2 regularization factor applied to the kernel
                           weights matrix.
    bias_regularizer_l1: float, default=None
                         L1 regularization factor applied to the bias vector.
    bias_regularizer_l2: float, default=None
                         L2 regularization factor applied to the bias vector.
    activity_regularizer_l1: float, default=None
                             L1 regularization factor applied to the output of
                             the layer.
    activity_regularizer_l2: float, default=None
                             L2 regularization factor applied to the output of
                             the layer.
    kernel_constraint: function, default=None
                       Constraint function applied to the kernel weights matrix.
    bias_constraint: function, default=None
                     Constraint function applied to the bias vector.
    optimizer: {"sgd", "rmsprop", "adagrad", "adadelta", "adam", "adamax",
               "nadam"}, default='adam'
               Optimizer
    lr: float>=0, default=0.001
        Learning rate.
    momentum: float>=0, default=0.0
              Parameter updates momentum.
    nesterov: boolean, default=False
              Whether to apply Nesterov momentum.
    decay: float>=0, default=0.0
           Learning rate decay over each update.
    rho: float>=0, default=0.9
    epsilon: float>=0, default=1e-08
             Fuzz factor.
    beta_1: float in (0, 1), default=0.9
    beta_2: float in (0, 1), default=0.999
    schedule_decay: , default=0.004
    loss: string/function, default='mse'/'categorical_crossentropy'
          Loss function.
    metrics: list, default=None
             List of metrics to be evaluated by the model during training and
             testing.
    loss_weights: list or dictionary, default=None
                  Scalar coefficients to weight the loss contributions of
                  different model outputs.
    sample_weight_mode: {"temporal", None}, default=None
                        Timestep-wise sample weighting.
    batch_size: integer, default='auto'
                Number of samples per gradient update.
    epochs: integer, default=200
            The number of times to iterate over the training data arrays.
    verbose: {0, 1, 2}, default=2
             Verbosity mode. 0=silent, 1=verbose, 2=one log line per epoch.
    early_stopping: bool, default True
                    Whether to use early stopping to terminate training when
                    validation score is not improving.
    tol: float, default 1e-4
         Tolerance for the optimization.
    patience: integer, default 2
              Number of epochs with no improvement after which training will
              be stopped.
    validation_split: float in [0, 1], default=0.1
                      Fraction of the training data to be used as validation
                      data.
    validation_data: array-like, shape ((n_samples, features_shape),
                                        (n_samples, targets_shape)),
                     default=None
                     Data on which to evaluate the loss and any model metrics at
                     the end of each epoch.
    shuffle: boolean, default=True
             Whether to shuffle the training data before each epoch.
    class_weight: dictionary, default=None
                  class indices to weights to apply to the model's loss for the
                  samples from each class during training.
    sample_weight: array-like, shape (n_samples), default=None
                   Weights to apply to the model's loss for each sample.
    initial_epoch: integer, default=0
                   Epoch at which to start training.
    window: integer, default=None
            Time window length.
    return_sequences: boolean, default=False
                      Whether to return the last output in the output sequence,
                      or the full sequence.

    """

    def __init__(self, architecture=None, activation='linear', use_bias=True,
                 kernel_initializer='glorot_uniform', bias_initializer='zeros',
                 kernel_regularizer_l1=None, kernel_regularizer_l2=None,
                 bias_regularizer_l1=None, bias_regularizer_l2=None,
                 activity_regularizer_l1=None, activity_regularizer_l2=None,
                 kernel_constraint=None, bias_constraint=None, optimizer='adam',
                 lr=0.001, momentum=0.0, nesterov=False, decay=0.0, rho=0.9,
                 epsilon=1e-08, beta_1=0.9, beta_2=0.999, schedule_decay=0.004,
                 loss='mse', metrics=None, loss_weights=None,
                 sample_weight_mode=None, batch_size='auto', epochs=200,
                 verbose=2, early_stopping=True, tol=0.0001, patience=2,
                 validation_split=0.1, validation_data=None, shuffle=True,
                 class_weight=None, sample_weight=None, initial_epoch=0,
                 window=None, return_sequences=False):
        for k, v in locals().items():
            if k != 'self': self.__dict__[k] = v

    def fit(self, X, y, optimizer=None, lr=None, momentum=None, nesterov=None,
            decay=None, rho=None, epsilon=None, beta_1=None, beta_2=None,
            schedule_decay=None, loss=None, metrics=None, loss_weights=None,
            sample_weight_mode=None, batch_size=None, epochs=None, verbose=None,
            early_stopping=None, tol=None, patience=None, validation_split=None,
            validation_data=None, shuffle=None, class_weight=None,
            sample_weight=None, initial_epoch=None):
        """Fit to data.

        Fit model to X.

        Parameters
        ----------
        X: numpy array of shape [n_samples, n_features]
           Training set.
        y: numpy array of shape [n_samples]
           Target values.
        optimizer: {"sgd", "rmsprop", "adagrad", "adadelta", "adam", "adamax",
                   "nadam"}, default='adam'
                   Optimizer
        lr: float>=0, default=0.001
            Learning rate.
        momentum: float>=0, default=0.0
                  Parameter updates momentum.
        nesterov: boolean, default=False
                  Whether to apply Nesterov momentum.
        decay: float>=0, default=0.0
               Learning rate decay over each update.
        rho: float>=0, default=0.9
        epsilon: float>=0, default=1e-08
                 Fuzz factor.
        beta_1: float in (0, 1), default=0.9
        beta_2: float in (0, 1), default=0.999
        schedule_decay: , default=0.004
        loss: string/function, default='mse'/'categorical_crossentropy'
              Loss function.
        metrics: list, default=None
                 List of metrics to be evaluated by the model during training
                 and testing.
        loss_weights: list or dictionary, default=None
                      Scalar coefficients to weight the loss contributions of
                      different model outputs.
        sample_weight_mode: {"temporal", None}, default=None
                            Timestep-wise sample weighting.
        batch_size: integer, default='auto'
                    Number of samples per gradient update.
        epochs: integer, default=200
                The number of times to iterate over the training data arrays.
        verbose: {0, 1, 2}, default=1
                 Verbosity mode. 0=silent, 1=verbose, 2=one log line per epoch.
        early_stopping: bool, default True
                        Whether to use early stopping to terminate training
                        when validation score is not improving.
        tol: float, default 1e-4
             Tolerance for the optimization.
        patience: integer, default 2
                  Number of epochs with no improvement after which training will
                  be stopped.
        validation_split: float in [0, 1], default=0.1
                          Fraction of the training data to be used as validation
                          data.
        validation_data: array-like, shape ((n_samples, features_shape),
                                            (n_samples, targets_shape)),
                         default=None
                         Data on which to evaluate the loss and any model
                         metrics at the end of each epoch.
        shuffle: boolean, default=True
                 Whether to shuffle the training data before each epoch.
        class_weight: dictionary, default=None
                      class indices to weights to apply to the model's loss for
                      the samples from each class during training.
        sample_weight: array-like, shape (n_samples), default=None
                       Weights to apply to the model's loss for each sample.
        initial_epoch: integer, default=0
                       Epoch at which to start training.

        Returns
        -------
        self

        """
        for k, v in locals().items():
            if (k != 'self') and (v is not None): self.__dict__[k] = v
        X, y = check_X_y(X, y, allow_nd=True, multi_output=True)
        if len(y.shape) == 1: y = y.reshape((len(y), 1))
        X, y = _time_series(X, y=y, window=self.window,
                            return_sequences=self.return_sequences)
        z = inputs = Input(shape=X.shape[1:])
        if self.architecture is not None: z = self.architecture(z)
        layer = Dense(int(np.prod(y.shape[1:])), activation=self.activation,
                      use_bias=self.use_bias,
                      kernel_initializer=self.kernel_initializer,
                      bias_initializer=self.bias_initializer,
                      kernel_regularizer=Regularizer(l1=self.kernel_regularizer_l1,
                                                     l2=self.kernel_regularizer_l2),
                      bias_regularizer=Regularizer(l1=self.bias_regularizer_l1,
                                                   l2=self.bias_regularizer_l2),
                      activity_regularizer=Regularizer(l1=self.activity_regularizer_l1,
                                                       l2=self.activity_regularizer_l2),
                      kernel_constraint=self.kernel_constraint,
                      bias_constraint=self.bias_constraint)
        if self.return_sequences: layer = TimeDistributed(layer)
        output = layer(z)
        optimizer = Optimizer(optimizer=self.optimizer, lr=self.lr,
                              momentum=self.momentum, nesterov=self.nesterov,
                              decay=self.decay, rho=self.rho,
                              epsilon=self.epsilon, beta_1=self.beta_1,
                              beta_2=self.beta_2,
                              schedule_decay=self.schedule_decay)
        self.model_ = Model(inputs, output)
        self.model_.compile(optimizer, self.loss, metrics=self.metrics,
                            loss_weights=self.loss_weights,
                            sample_weight_mode=self.sample_weight_mode)
        monitor = 'val_loss' if (self.validation_split > 0.0 or self.validation_data is not None) else 'loss'
        callbacks = [EarlyStopping(monitor=monitor,
                                   min_delta=self.tol, patience=self.patience)] if self.early_stopping and (self.tol > 0.0) else []
        if self.batch_size == 'auto': batch_size = min(200, len(X))
        if type(self.sample_weight) in (list, tuple): sample_weight = np.asarray(self.sample_weight)
        self.history_ = self.model_.fit(X, y, batch_size=batch_size,
                                        epochs=self.epochs,
                                        verbose=self.verbose,
                                        callbacks=callbacks,
                                        validation_split=self.validation_split,
                                        validation_data=self.validation_data,
                                        shuffle=self.shuffle,
                                        class_weight=self.class_weight,
                                        sample_weight=sample_weight,
                                        initial_epoch=self.initial_epoch)
        return self

    def predict(self, X, batch_size=32, verbose=0):
        """Predict using the trained model.

        Parameters
        ----------
        X: array-like, shape (n_samples, features_shape)
           The input data.
        batch_size: integer, default=32
                    Batch size.
        verbose: {0, 1}, default=0
                 Verbosity mode.

        Returns
        -------
        y_pred: array-like, shape (n_samples, targets_shape)
                Target predictions for X.

        """
        check_is_fitted(self, ['model_', 'history_'])
        X = check_array(X, allow_nd=True)
        X, _ = _time_series(X, y=None, window=self.window,
                            return_sequences=self.return_sequences)
        preds = self.model_.predict(X, batch_size=batch_size, verbose=verbose)
        return preds.reshape((len(preds))) if (len(preds.shape) == 2 and preds.shape[1] == 1) else preds

    def transform(self, X):
        """Transform using the trained model.

        Parameters
        ----------
        X: array-like, shape (n_samples, features_shape)
           The input data.

        Returns
        -------
        Z: array-like, shape (n_samples, last_hidden_layer_shape)
           Transformations for X.

        """
        check_is_fitted(self, ['model_', 'history_'])
        X = check_array(X, allow_nd=True)
        X, _ = _time_series(X, y=None, window=self.window,
                            return_sequences=self.return_sequences)
        propagate = K.function([self.model_.layers[0].input],
                               [self.model_.layers[-2].output])
        return propagate([X])[0]

    def score(self, X, y, sample_weight=None, metric=r2_score):
        """Return the score of the model on the data X.

        Parameters
        ----------
        X: array-like, shape (n_samples, features_shape)
           Test samples.
        y: array-like, shape (n_samples, targets_shape)
           Targets for X.
        sample_weight: array-like, shape [n_samples], default=None
                       Sample weights.
        metric: function, default=r2_score/accuracy_score
                Metric to be evaluated.

        Returns
        -------
        score: float
               r2_score/accuracy of self.predict(X) wrt. y.

        """
        check_is_fitted(self, ['model_', 'history_'])
        X, y = check_X_y(X, y, allow_nd=True, multi_output=True)
        if len(y.shape) == 1: y = y.reshape((len(y), 1))
        _, y = _time_series(X, y=y, window=self.window,
                            return_sequences=self.return_sequences)
        return metric(y, self.predict(X), sample_weight=sample_weight)


###############################################################################
#  Classifier class
###############################################################################


class FFClassifier(BaseFeedForward, ClassifierMixin):

    __doc__ = BaseFeedForward.__doc__

    __init__ = partialmethod(BaseFeedForward.__init__, activation='softmax',
                             loss='categorical_crossentropy')

    def fit(self, X, y, sample_weight=None, **kwargs):
        self.classes_ = np.unique(y)
        self.n_classes_ = self.classes_.shape[0]
        return BaseFeedForward.fit(self, X, to_categorical(y),
                                   sample_weight=sample_weight, **kwargs)
    fit.__doc__ = BaseFeedForward.fit.__doc__

    predict_proba = BaseFeedForward.predict

    def predict(self, X, **kwargs):
        return BaseFeedForward.predict(self, X, **kwargs).argmax(axis=1)
    predict.__doc__ = BaseFeedForward.predict.__doc__

    score = partialmethod(BaseFeedForward.score, metric=accuracy_score)


###############################################################################
#  Regressor class
###############################################################################


class FFRegressor(BaseFeedForward, RegressorMixin):

    __doc__ = BaseFeedForward.__doc__
