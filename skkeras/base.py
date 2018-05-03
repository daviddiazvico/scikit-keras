"""
Scikit-learn-compatible Keras models.

@author: David Diaz Vico
@license: MIT
"""

from keras.callbacks import EarlyStopping
from keras.layers import (AveragePooling1D, AveragePooling2D, AveragePooling3D,
                          BatchNormalization, Dense, Dropout, Flatten, Input,
                          Conv1D, Conv2D, Conv3D, GRU, MaxPooling1D,
                          MaxPooling2D, MaxPooling3D, LSTM, TimeDistributed)
from keras.models import Model, load_model, save_model
from keras.optimizers import (SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax,
                              Nadam)
from keras.regularizers import l1, l2, l1_l2
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


def time_series_tensor(X, window):
    return np.array([X[i:i + window] for i in range(X.shape[0] - window + 1)])


###############################################################################
#  Base feed-forward class
###############################################################################


class BaseFeedForward(BaseEstimator):

    """Feed-forward regressor/classifier.

    This model optimizes the MSE/categorical-crossentropy function using
    back-propagation.

    Parameters
    ----------
    convolution_filters: integer, default=None
                         Dimensionality of the output space.
    convolution_kernel_size: integer/tuple/list, default=None
                             Dimensionality of the convolution window.
    convolution_strides: integer/tuple/list, default=None
                         Strides of the convolution.
    convolution_padding: {"valid", "same"}, default='valid'
    convolution_dilation_rate: integer/tuple/list, default=None
                               Dilation rate to use for dilated convolution.
    convolution_activation: string/function, default=None
                            Activation function.
    convolution_use_bias: boolean, default=True
                          Whether the layer uses a bias vector.
    convolution_kernel_initializer: string/function, default='glorot_uniform'
                                    Initializer for the kernel weights matrix.
    convolution_bias_initializer: string/function, default='zeros'
                                  Initializer for the bias vector.
    convolution_kernel_regularizer_l1: float, default=None
                                       L1 regularization factor applied to the
                                       kernel weights matrix.
    convolution_kernel_regularizer_l2: float, default=None
                                       L2 regularization factor applied to the
                                       kernel weights matrix.
    convolution_bias_regularizer_l1: float, default=None
                                     L1 regularization factor applied to the
                                     bias vector.
    convolution_bias_regularizer_l2: float, default=None
                                     L2 regularization factor applied to the
                                     bias vector.
    convolution_activity_regularizer_l1: float, default=None
                                     L1 regularization factor applied to the
                                     output of the layer.
    convolution_activity_regularizer_l2: float, default=None
                                     L2 regularization factor applied to the
                                     output of the layer.
    convolution_kernel_constraint: function, default=None
                                   Constraint function applied to the kernel
                                   matrix.
    convolution_bias_constraint: function, default=None
                                 Constraint function applied to the bias vector.
    pooling_type: {"max", "average}, default='max'
    pooling_pool_size: integer/tuple/list, default=None
                       Factors by which to downscale.
    pooling_strides: integer/tuple/list, default=None
                     Strides values.
    pooling_padding: {"valid", "same"}, default='valid'
    recurrent_type: {"lstm", "gru"}, default='lstm'
    recurrent_window: integer, default=3
                      Time window length.
    recurrent_units: integer, default=None
                     Dimensionality of the output space.
    recurrent_activation: string/function, default='tanh'
                          Activation function to use.
    recurrent_recurrent_activation: string/function, default='hard_sigmoid'
                                    Activation function to use for the recurrent
                                    step.
    recurrent_use_bias: boolean, default=True
                        Whether the layer uses a bias vector.
    recurrent_kernel_initializer: string/function, default='glorot_uniform'
                                  Initializer for the kernel weights matrix.
    recurrent_recurrent_initializer: string/function, default='orthogonal'
                                     Initializer for the recurrent_kernel
                                     weights matrix.
    recurrent_bias_initializer: string/function, default='zeros'
                                Initializer for the bias vector.
    recurrent_unit_forget_bias: boolean, default=True
                                If True, add 1 to the bias of the forget gate
                                at initialization.
    recurrent_kernel_regularizer_l1: float, default=None
                                     L1 regularization factor applied to the
                                     kernel weights matrix.
    recurrent_kernel_regularizer_l2: float, default=None
                                     L2 regularization factor applied to the
                                     kernel weights matrix.
    recurrent_recurrent_regularizer_l1: float, default=None
                                        L1 regularization factor applied to the
                                        recurrent_kernel weights matrix.
    recurrent_recurrent_regularizer_l2: float, default=None
                                        L2 regularization factor applied to the
                                        recurrent_kernel weights matrix.
    recurrent_bias_regularizer_l1: float, default=None
                                   L1 regularization factor applied to the bias
                                   vector.
    recurrent_bias_regularizer_l2: float, default=None
                                   L2 regularization factor applied to the bias
                                   vector.
    recurrent_activity_regularizer_l1: float, default=None
                                       L1 regularization factor applied to the
                                       output of the layer.
    recurrent_activity_regularizer_l2: float, default=None
                                       L2 regularization factor applied to the
                                       output of the layer.
    recurrent_kernel_constraint: function, default=None
                                 Constraint function applied to the kernel
                                 weights matrix.
    recurrent_recurrent_constraint: function, default=None
                                    Constraint function applied to the
                                    recurrent_kernel weights matrix.
    recurrent_bias_constraint: function, default=None
                               Constraint function applied to the bias vector.
    recurrent_dropout: float in [0, 1], default=0.0
                       Fraction of the units to drop for the linear
                       transformation of the inputs.
    recurrent_recurrent_dropout: float in [0, 1], default=0.0
                                 Fraction of the units to drop for the linear
                                 transformation of the recurrent state.
    recurrent_return_sequences: boolean, default=False
                                Whether to return the last output in the output
                                sequence, or the full sequence.
    recurrent_go_backwards: boolean, default=False
                            If True, process the input sequence backwards and
                            return the reversed sequence.
    recurrent_stateful: boolean, default=False
                        If True, the last state for each sample at index i in a
                        batch will be used as initial state for the sample of
                        index i in the following batch.
    recurrent_unroll: boolean, default=False
                      If True, the network will be unrolled, else a symbolic
                      loop will be used.
    recurrent_implementation: {0, 1, 2}, default=0
    batchnormalization: boolean, default=False
                        Whether to perform batch normalization or not.
    batchnormalization_axis: integer, default=-1
                             The axis that should be normalized (typically the
                             features axis).
    batchnormalization_momentum: float, default=0.99
                                 Momentum for the moving average.
    batchnormalization_epsilon: float, default=0.001
                                Small float added to variance to avoid dividing
                                by zero.
    batchnormalization_center: boolean, default=True
                               If True, add offset of beta to normalized tensor.
                               If False, beta is ignored.
    batchnormalization_scale: boolean, default=True
                              If True, multiply by gamma. If False, gamma is not
                              used.
    batchnormalization_beta_initializer: string/function, default='zeros'
                                         Initializer for the beta weight.
    batchnormalization_gamma_initializer: string/function, default='ones'
                                          Initializer for the gamma weight.
    batchnormalization_moving_mean_initializer: string/function, default='zeros'
                                                Initializer for the moving mean.
    batchnormalization_moving_variance_initializer: string/function,
                                                    default='ones'
                                                    Initializer for the moving
                                                    variance.
    batchnormalization_beta_regularizer_l1: float, default=None
                                            L1 regularization factor applied to
                                            the beta weight.
    batchnormalization_beta_regularizer_l2: float, default=None
                                            L2 regularization factor applied to
                                            the beta weight.
    batchnormalization_gamma_regularizer_l1: float, default=None
                                            L1 regularization factor applied to
                                            the gamma  weight.
    batchnormalization_gamma_regularizer_l2: float, default=None
                                            L2 regularization factor applied to
                                            the gamma  weight.
    batchnormalization_beta_constraint: function, default=None
                                        Optional constraint for the beta weight.
    batchnormalization_gamma_constraint: function, default=None
                                         Optional constraint for the gamma
                                         weight.
    dense_units: integer, default=None
                 Dimensionality of the output space.
    dense_activation: string/function, default='relu'
                      Activation function to use.
    dense_use_bias: boolean, default=True
                    Whether the layer uses a bias vector.
    dense_kernel_initializer: string/function, default='he_uniform'
                              Initializer for the kernel weights matrix.
    dense_bias_initializer: string/function, default='zeros'
                            Initializer for the bias vector.
    dense_kernel_regularizer_l1: float, default=None
                                  L1 regularization factor applied to the kernel
                                  weights matrix.
    dense_kernel_regularizer_l2: float, default=None
                                  L2 regularization factor applied to the kernel
                                  weights matrix.
    dense_bias_regularizer_l1: float, default=None
                               L1 regularization factor applied to the bias
                               vector.
    dense_bias_regularizer_l2: float, default=None
                               L2 regularization factor applied to the bias
                               vector.
    dense_activity_regularizer_l1: float, default=None
                                   L1 regularization factor applied to the
                                   output of the layer.
    dense_activity_regularizer_l2: float, default=None
                                   L2 regularization factor applied to the
                                   output of the layer.
    dense_kernel_constraint: function, default=None
                             Constraint function applied to the kernel weights
                             matrix.
    dense_bias_constraint: function, default=None
                           Constraint function applied to the bias vector.
    output_activation: string/function, default='linear'/'softmax'
                       Activation function to use.
    output_use_bias: boolean, default=True
                     Whether the layer uses a bias vector.
    output_kernel_initializer: string/function, default='glorot_uniform'
                              Initializer for the kernel weights matrix.
    output_bias_initializer: string/function, default='zeros'
                            Initializer for the bias vector.
    output_kernel_regularizer_l1: float, default=None
                                  L1 regularization factor applied to the kernel
                                  weights matrix.
    output_kernel_regularizer_l2: float, default=None
                                  L2 regularization factor applied to the kernel
                                  weights matrix.
    output_bias_regularizer_l1: float, default=None
                                L1 regularization factor applied to the bias
                                vector.
    output_bias_regularizer_l2: float, default=None
                                L2 regularization factor applied to the bias
                                vector.
    output_activity_regularizer_l1: float, default=None
                                    L1 regularization factor applied to the
                                    output of the layer.
    output_activity_regularizer_l2: float, default=None
                                    L2 regularization factor applied to the
                                    output of the layer.
    output_kernel_constraint: function, default=None
                              Constraint function applied to the kernel weights
                              matrix.
    output_bias_constraint: function, default=None
                            Constraint function applied to the bias vector.
    dropout_rate: float in [0, 1], default=0.0
                  Fraction of the input units to drop.
    dropout_noise_shape: array-like, default=None
                         shape of the binary dropout mask that will be
                         multiplied with the input.
    dropout_seed: integer, default=None
                  Random seed.
    solver: {"sgd", "rmsprop", "adagrad", "adadelta", "adam", "adamax",
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

    """

    def __init__(self, convolution_filters=None, convolution_kernel_size=None,
                 convolution_strides=None, convolution_padding='valid',
                 convolution_dilation_rate=None, convolution_activation=None,
                 convolution_use_bias=True,
                 convolution_kernel_initializer='glorot_uniform',
                 convolution_bias_initializer='zeros',
                 convolution_kernel_regularizer_l1=None,
                 convolution_kernel_regularizer_l2=None,
                 convolution_bias_regularizer_l1=None,
                 convolution_bias_regularizer_l2=None,
                 convolution_activity_regularizer_l1=None,
                 convolution_activity_regularizer_l2=None,
                 convolution_kernel_constraint=None,
                 convolution_bias_constraint=None, pooling_type='max',
                 pooling_pool_size=None, pooling_strides=None,
                 pooling_padding='valid', recurrent_type='lstm',
                 recurrent_window=3, recurrent_units=None,
                 recurrent_activation='tanh',
                 recurrent_recurrent_activation='hard_sigmoid',
                 recurrent_use_bias=True,
                 recurrent_kernel_initializer='glorot_uniform',
                 recurrent_recurrent_initializer='orthogonal',
                 recurrent_bias_initializer='zeros',
                 recurrent_unit_forget_bias=True,
                 recurrent_kernel_regularizer_l1=None,
                 recurrent_kernel_regularizer_l2=None,
                 recurrent_recurrent_regularizer_l1=None,
                 recurrent_recurrent_regularizer_l2=None,
                 recurrent_bias_regularizer_l1=None,
                 recurrent_bias_regularizer_l2=None,
                 recurrent_activity_regularizer_l1=None,
                 recurrent_activity_regularizer_l2=None,
                 recurrent_kernel_constraint=None,
                 recurrent_recurrent_constraint=None,
                 recurrent_bias_constraint=None, recurrent_dropout=0.0,
                 recurrent_recurrent_dropout=0.0,
                 recurrent_return_sequences=False, recurrent_go_backwards=False,
                 recurrent_stateful=False, recurrent_unroll=False,
                 recurrent_implementation=0, batchnormalization=False,
                 batchnormalization_axis=-1, batchnormalization_momentum=0.99,
                 batchnormalization_epsilon=0.001,
                 batchnormalization_center=True, batchnormalization_scale=True,
                 batchnormalization_beta_initializer='zeros',
                 batchnormalization_gamma_initializer='ones',
                 batchnormalization_moving_mean_initializer='zeros',
                 batchnormalization_moving_variance_initializer='ones',
                 batchnormalization_beta_regularizer_l1=None,
                 batchnormalization_beta_regularizer_l2=None,
                 batchnormalization_gamma_regularizer_l1=None,
                 batchnormalization_gamma_regularizer_l2=None,
                 batchnormalization_beta_constraint=None,
                 batchnormalization_gamma_constraint=None, dense_units=None,
                 dense_activation='relu', dense_use_bias=True,
                 dense_kernel_initializer='he_uniform',
                 dense_bias_initializer='zeros',
                 dense_kernel_regularizer_l1=None,
                 dense_kernel_regularizer_l2=None,
                 dense_bias_regularizer_l1=None, dense_bias_regularizer_l2=None,
                 dense_activity_regularizer_l1=None,
                 dense_activity_regularizer_l2=None,
                 dense_kernel_constraint=None, dense_bias_constraint=None,
                 output_activation=None, output_use_bias=True,
                 output_kernel_initializer='glorot_uniform',
                 output_bias_initializer='zeros',
                 output_kernel_regularizer_l1=None,
                 output_kernel_regularizer_l2=None,
                 output_bias_regularizer_l1=None,
                 output_bias_regularizer_l2=None,
                 output_activity_regularizer_l1=None,
                 output_activity_regularizer_l2=None,
                 output_kernel_constraint=None, output_bias_constraint=None,
                 dropout_rate=0.0, dropout_noise_shape=None, dropout_seed=None,
                 solver='adam', lr=0.001, momentum=0.0, nesterov=False,
                 decay=0.0, rho=0.9, epsilon=1e-08, beta_1=0.9, beta_2=0.999,
                 schedule_decay=0.004, loss=None, metrics=None,
                 loss_weights=None, sample_weight_mode=None, batch_size='auto',
                 epochs=200, verbose=2, early_stopping=True, tol=0.0001,
                 patience=2, validation_split=0.1, validation_data=None,
                 shuffle=True, class_weight=None, sample_weight=None,
                 initial_epoch=0):
        self.convolution_filters = convolution_filters
        self.convolution_kernel_size = convolution_kernel_size
        self.convolution_strides = convolution_strides
        self.convolution_padding = convolution_padding
        self.convolution_dilation_rate = convolution_dilation_rate
        self.convolution_activation = convolution_activation
        self.convolution_use_bias = convolution_use_bias
        self.convolution_kernel_initializer = convolution_kernel_initializer
        self.convolution_bias_initializer = convolution_bias_initializer
        self.convolution_kernel_regularizer_l1 = convolution_kernel_regularizer_l1
        self.convolution_kernel_regularizer_l2 = convolution_kernel_regularizer_l2
        self.convolution_bias_regularizer_l1 = convolution_bias_regularizer_l1
        self.convolution_bias_regularizer_l2 = convolution_bias_regularizer_l2
        self.convolution_activity_regularizer_l1 = convolution_activity_regularizer_l1
        self.convolution_activity_regularizer_l2 = convolution_activity_regularizer_l2
        self.convolution_kernel_constraint = convolution_kernel_constraint
        self.convolution_bias_constraint = convolution_bias_constraint
        self.pooling_type = pooling_type
        self.pooling_pool_size = pooling_pool_size
        self.pooling_strides = pooling_strides
        self.pooling_padding = pooling_padding
        self.recurrent_type = recurrent_type
        self.recurrent_window = recurrent_window
        self.recurrent_units = recurrent_units
        self.recurrent_activation = recurrent_activation
        self.recurrent_recurrent_activation = recurrent_recurrent_activation
        self.recurrent_use_bias = recurrent_use_bias
        self.recurrent_kernel_initializer = recurrent_kernel_initializer
        self.recurrent_recurrent_initializer = recurrent_recurrent_initializer
        self.recurrent_bias_initializer = recurrent_bias_initializer
        self.recurrent_unit_forget_bias = recurrent_unit_forget_bias
        self.recurrent_kernel_regularizer_l1 = recurrent_kernel_regularizer_l1
        self.recurrent_kernel_regularizer_l2 = recurrent_kernel_regularizer_l2
        self.recurrent_recurrent_regularizer_l1 = recurrent_recurrent_regularizer_l1
        self.recurrent_recurrent_regularizer_l2 = recurrent_recurrent_regularizer_l2
        self.recurrent_bias_regularizer_l1 = recurrent_bias_regularizer_l1
        self.recurrent_bias_regularizer_l2 = recurrent_bias_regularizer_l2
        self.recurrent_activity_regularizer_l1 = recurrent_activity_regularizer_l1
        self.recurrent_activity_regularizer_l2 = recurrent_activity_regularizer_l2
        self.recurrent_kernel_constraint = recurrent_kernel_constraint
        self.recurrent_recurrent_constraint = recurrent_recurrent_constraint
        self.recurrent_bias_constraint = recurrent_bias_constraint
        self.recurrent_dropout = recurrent_dropout
        self.recurrent_recurrent_dropout = recurrent_recurrent_dropout
        self.recurrent_return_sequences = recurrent_return_sequences
        self.recurrent_go_backwards = recurrent_go_backwards
        self.recurrent_stateful = recurrent_stateful
        self.recurrent_unroll = recurrent_unroll
        self.recurrent_implementation = recurrent_implementation
        self.batchnormalization = batchnormalization
        self.batchnormalization_axis = batchnormalization_axis
        self.batchnormalization_momentum = batchnormalization_momentum
        self.batchnormalization_epsilon = batchnormalization_epsilon
        self.batchnormalization_center = batchnormalization_center
        self.batchnormalization_scale = batchnormalization_scale
        self.batchnormalization_beta_initializer = batchnormalization_beta_initializer
        self.batchnormalization_gamma_initializer = batchnormalization_gamma_initializer
        self.batchnormalization_moving_mean_initializer = batchnormalization_moving_mean_initializer
        self.batchnormalization_moving_variance_initializer = batchnormalization_moving_variance_initializer
        self.batchnormalization_beta_regularizer_l1 = batchnormalization_beta_regularizer_l1
        self.batchnormalization_beta_regularizer_l2 = batchnormalization_beta_regularizer_l2
        self.batchnormalization_gamma_regularizer_l1 = batchnormalization_gamma_regularizer_l1
        self.batchnormalization_gamma_regularizer_l2 = batchnormalization_gamma_regularizer_l2
        self.batchnormalization_beta_constraint = batchnormalization_beta_constraint
        self.batchnormalization_gamma_constraint = batchnormalization_gamma_constraint
        self.dense_units = dense_units
        self.dense_activation = dense_activation
        self.dense_use_bias = dense_use_bias
        self.dense_kernel_initializer = dense_kernel_initializer
        self.dense_bias_initializer = dense_bias_initializer
        self.dense_kernel_regularizer_l1 = dense_kernel_regularizer_l1
        self.dense_kernel_regularizer_l2 = dense_kernel_regularizer_l2
        self.dense_bias_regularizer_l1 = dense_bias_regularizer_l1
        self.dense_bias_regularizer_l2 = dense_bias_regularizer_l2
        self.dense_activity_regularizer_l1 = dense_activity_regularizer_l1
        self.dense_activity_regularizer_l2 = dense_activity_regularizer_l2
        self.dense_kernel_constraint = dense_kernel_constraint
        self.dense_bias_constraint = dense_bias_constraint
        self.output_activation = output_activation if output_activation is not None else self.output_activation
        self.output_use_bias = output_use_bias
        self.output_kernel_initializer = output_kernel_initializer
        self.output_bias_initializer = output_bias_initializer
        self.output_kernel_regularizer_l1 = output_kernel_regularizer_l1
        self.output_kernel_regularizer_l2 = output_kernel_regularizer_l2
        self.output_bias_regularizer_l1 = output_bias_regularizer_l1
        self.output_bias_regularizer_l2 = output_bias_regularizer_l2
        self.output_activity_regularizer_l1 = output_activity_regularizer_l1
        self.output_activity_regularizer_l2 = output_activity_regularizer_l2
        self.output_kernel_constraint = output_kernel_constraint
        self.output_bias_constraint = output_bias_constraint
        self.dropout_rate = dropout_rate
        self.dropout_noise_shape = dropout_noise_shape
        self.dropout_seed = dropout_seed
        self.solver = solver
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.decay = decay
        self.rho = rho
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.schedule_decay = schedule_decay
        self.loss = loss if loss is not None else self.loss
        self.metrics = metrics
        self.loss_weights = loss_weights
        self.sample_weight_mode = sample_weight_mode
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose
        self.early_stopping = early_stopping
        self.tol = tol
        self.patience = patience
        self.validation_split = validation_split
        self.validation_data = validation_data
        self.shuffle = shuffle
        self.class_weight = class_weight
        self.sample_weight = sample_weight
        self.initial_epoch = initial_epoch

    @staticmethod
    def _regularize(lambda1, lambda2):
        regularizer = {False: {False: None, True: l2(l=lambda2)},
                       True: {False: l1(l=lambda1), True: l1_l2(l1=lambda1,
                                                                l2=lambda2)}}
        return regularizer[lambda1 is not None][lambda2 is not None]

    def _convolve_and_pool(self, x, convolution_filters,
                           convolution_kernel_size, convolution_strides,
                           convolution_dilation_rate, pooling_pool_size,
                           pooling_strides, return_tensors=True,
                           return_sequences=False):
        if convolution_kernel_size is not None:
            conv = {1: Conv1D, 2: Conv2D, 3: Conv3D}
            layer = conv[len(convolution_kernel_size)](convolution_filters, convolution_kernel_size,
                                                       strides=convolution_strides,
                                                       padding=self.convolution_padding,
                                                       dilation_rate=convolution_dilation_rate,
                                                       activation=self.convolution_activation,
                                                       use_bias=self.convolution_use_bias,
                                                       kernel_initializer=self.convolution_kernel_initializer,
                                                       bias_initializer=self.convolution_bias_initializer,
                                                       kernel_regularizer=self._regularize(self.convolution_kernel_regularizer_l1,
                                                                                           self.convolution_kernel_regularizer_l2),
                                                       bias_regularizer=self._regularize(self.convolution_bias_regularizer_l1,
                                                                                         self.convolution_bias_regularizer_l2),
                                                       activity_regularizer=self._regularize(self.convolution_activity_regularizer_l1,
                                                                                             self.convolution_activity_regularizer_l2),
                                                       kernel_constraint=self.convolution_kernel_constraint,
                                                       bias_constraint=self.convolution_bias_constraint)
            layer = TimeDistributed(layer) if return_sequences else layer
            x = layer(x)
        if pooling_pool_size is not None:
            pool = {'max': {1: MaxPooling1D, 2: MaxPooling2D, 3: MaxPooling3D},
                    'average': {1: AveragePooling1D, 2: AveragePooling2D,
                                3: AveragePooling3D}}
            layer = pool[self.pooling_type][len(pooling_pool_size)](pool_size=pooling_pool_size,
                                                                    strides=pooling_strides,
                                                                    padding=self.pooling_padding)
            layer = TimeDistributed(layer) if return_sequences else layer
            x = layer(x)
        if not return_tensors:
            layer = Flatten()
            layer = TimeDistributed(layer) if return_sequences else layer
            x = layer(x)
        return x

    def _recur(self, x, units, return_sequences=True):
        recur = {'lstm': LSTM, 'gru': GRU}
        layer = recur[self.recurrent_type](units, activation=self.recurrent_activation,
                                           recurrent_activation=self.recurrent_recurrent_activation,
                                           use_bias=self.recurrent_use_bias,
                                           kernel_initializer=self.recurrent_kernel_initializer,
                                           recurrent_initializer=self.recurrent_recurrent_initializer,
                                           bias_initializer=self.recurrent_bias_initializer,
                                           unit_forget_bias=self.recurrent_unit_forget_bias,
                                           kernel_regularizer=self._regularize(self.recurrent_kernel_regularizer_l1,
                                                                              self.recurrent_kernel_regularizer_l2),
                                           recurrent_regularizer=self._regularize(self.recurrent_recurrent_regularizer_l1,
                                                                                  self.recurrent_recurrent_regularizer_l2),
                                           bias_regularizer=self._regularize(self.recurrent_bias_regularizer_l1,
                                                                             self.recurrent_bias_regularizer_l2),
                                           activity_regularizer=self._regularize(self.recurrent_activity_regularizer_l1,
                                                                                 self.recurrent_activity_regularizer_l2),
                                           kernel_constraint=self.recurrent_kernel_constraint,
                                           recurrent_constraint=self.recurrent_recurrent_constraint,
                                           bias_constraint=self.recurrent_bias_constraint,
                                           dropout=self.recurrent_dropout,
                                           recurrent_dropout=self.recurrent_recurrent_dropout,
                                           return_sequences=return_sequences,
                                           go_backwards=self.recurrent_go_backwards,
                                           stateful=self.recurrent_stateful,
                                           unroll=self.recurrent_unroll,
                                           implementation=self.recurrent_implementation)
        x = layer(x)
        return x

    def _connect(self, x, units, dropout_noise_shape=None):
        if self.batchnormalization:
            layer= BatchNormalization(axis=self.batchnormalization_axis,
                                      momentum=self.batchnormalization_momentum,
                                      epsilon=self.batchnormalization_epsilon,
                                      center=self.batchnormalization_center,
                                      scale=self.batchnormalization_scale,
                                      beta_initializer=self.batchnormalization_beta_initializer,
                                      gamma_initializer=self.batchnormalization_gamma_initializer,
                                      moving_mean_initializer=self.batchnormalization_moving_mean_initializer,
                                      moving_variance_initializer=self.batchnormalization_moving_variance_initializer,
                                      beta_regularizer=self._regularize(self.batchnormalization_beta_regularizer_l1,
                                                                        self.batchnormalization_beta_regularizer_l2),
                                      gamma_regularizer=self._regularize(self.batchnormalization_gamma_regularizer_l1,
                                                                         self.batchnormalization_gamma_regularizer_l2),
                                      beta_constraint=self.batchnormalization_beta_constraint,
                                      gamma_constraint=self.batchnormalization_gamma_constraint)
            layer = TimeDistributed(layer) if self.recurrent_return_sequences else layer
            x = layer(x)
        layer = Dense(units, activation=self.dense_activation,
                      use_bias=self.dense_use_bias,
                      kernel_initializer=self.dense_kernel_initializer,
                      bias_initializer=self.dense_bias_initializer,
                      kernel_regularizer=self._regularize(self.dense_kernel_regularizer_l1,
                                                          self.dense_kernel_regularizer_l2),
                      bias_regularizer=self._regularize(self.dense_bias_regularizer_l1,
                                                        self.dense_bias_regularizer_l2),
                      activity_regularizer=self._regularize(self.dense_activity_regularizer_l1,
                                                            self.dense_activity_regularizer_l2),
                      kernel_constraint=self.dense_kernel_constraint,
                      bias_constraint=self.dense_bias_constraint)
        layer = TimeDistributed(layer) if self.recurrent_return_sequences else layer
        x = layer(x)
        if 0.0 < self.dropout_rate < 1.0:
            layer = Dropout(self.dropout_rate, noise_shape=dropout_noise_shape,
                            seed=self.dropout_seed)
            layer = TimeDistributed(layer) if self.recurrent_return_sequences else layer
            x = layer(x)
        return x

    def _body(self, z):
        if (self.convolution_filters is not None) or (self.convolution_kernel_size is not None):
            if len(self.convolution_filters) == len(self.convolution_kernel_size):
                self.convolution_strides = [[1] * len(k) for k in self.convolution_kernel_size] if self.convolution_strides is None else self.convolution_strides
                self.convolution_dilation_rate = [[1] * len(k) for k in self.convolution_kernel_size] if self.convolution_dilation_rate is None else self.convolution_dilation_rate
                self.pooling_pool_size = [None] * len(self.convolution_filters) if self.pooling_pool_size is None else self.pooling_pool_size
                self.pooling_strides = [None] * len(self.convolution_filters) if self.pooling_strides is None else self.pooling_strides
                for i, (cf, cks, cs, cdr, pps, ps) in enumerate(zip(self.convolution_filters,
                                                                    self.convolution_kernel_size,
                                                                    self.convolution_strides,
                                                                    self.convolution_dilation_rate,
                                                                    self.pooling_pool_size,
                                                                    self.pooling_strides)):
                    z = self._convolve_and_pool(z, cf, cks, cs, cdr, pps, ps,
                                                return_tensors=i < len(self.convolution_filters) - 1,
                                                return_sequences=self.recurrent_units is not None)
        if self.recurrent_units is not None:
            for i, ru in enumerate(self.recurrent_units):
                z = self._recur(z, ru,
                                return_sequences=i < len(self.recurrent_units) - 1)
        if self.dense_units is not None:
            self.dropout_noise_shape = [None] * len(self.dense_units) if self.dropout_noise_shape is None else self.dropout_noise_shape
            for (du, dns) in zip(self.dense_units, self.dropout_noise_shape):
                z = self._connect(z, du, dropout_noise_shape=dns)
        return z

    def _model(self, X, y):
        z = inputs = Input(shape=X.shape[1:])
        z = self._body(z)
        layer = Dense(int(np.prod(y.shape[1:])),
                      activation=self.output_activation,
                      use_bias=self.output_use_bias,
                      kernel_initializer=self.output_kernel_initializer,
                      bias_initializer=self.output_bias_initializer,
                      kernel_regularizer=self._regularize(self.output_kernel_regularizer_l1,
                                                          self.output_kernel_regularizer_l2),
                      bias_regularizer=self._regularize(self.output_bias_regularizer_l1,
                                                        self.output_bias_regularizer_l2),
                      activity_regularizer=self._regularize(self.output_activity_regularizer_l1,
                                                            self.output_activity_regularizer_l2),
                      kernel_constraint=self.output_kernel_constraint,
                      bias_constraint=self.output_bias_constraint)
        layer = TimeDistributed(layer) if self.recurrent_return_sequences else layer
        output = layer(z)
        return Model(inputs, output)

    def _solver(self, solver):
        solvers = {'sgd': SGD(lr=self.lr, momentum=self.momentum,
                              decay=self.decay, nesterov=self.nesterov),
                   'rmsprop': RMSprop(lr=self.lr, rho=self.rho,
                                      epsilon=self.epsilon, decay=self.decay),
                   'adagrad': Adagrad(lr=self.lr, epsilon=self.epsilon,
                                      decay=self.decay),
                   'adadelta': Adadelta(lr=self.lr, rho=self.rho,
                                        epsilon=self.epsilon, decay=self.decay),
                   'adam': Adam(lr=self.lr, beta_1=self.beta_1,
                                beta_2=self.beta_2, epsilon=self.epsilon,
                                decay=self.decay),
                   'adamax': Adamax(lr=self.lr, beta_1=self.beta_1,
                                    beta_2=self.beta_2, epsilon=self.epsilon,
                                    decay=self.decay),
                   'nadam': Nadam(lr=self.lr, beta_1=self.beta_1,
                                  beta_2=self.beta_2, epsilon=self.epsilon,
                                  schedule_decay=self.schedule_decay)}
        return solvers[solver]

    def fit(self, X, y, solver=None, lr=None, momentum=None, nesterov=None,
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
        solver: {"sgd", "rmsprop", "adagrad", "adadelta", "adam", "adamax",
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
        self.solver = solver if solver is not None else self.solver
        self.lr = lr if lr is not None else self.lr
        self.momentum = momentum if momentum is not None else self.momentum
        self.nesterov = nesterov if nesterov is not None else self.nesterov
        self.decay = decay if decay is not None else self.decay
        self.rho = rho if rho is not None else self.rho
        self.epsilon = epsilon if epsilon is not None else self.epsilon
        self.beta_1 = beta_1 if beta_1 is not None else self.beta_1
        self.beta_2 = beta_2 if beta_2 is not None else self.beta_2
        self.schedule_decay = schedule_decay if schedule_decay is not None else self.schedule_decay
        self.loss = loss if loss is not None else self.loss
        self.metrics = metrics if metrics is not None else self.metrics
        self.loss_weights = loss_weights if loss_weights is not None else self.loss_weights
        self.sample_weight_mode = sample_weight_mode if sample_weight_mode is not None else self.sample_weight_mode
        self.batch_size = batch_size if batch_size is not None else self.batch_size
        self.epochs = epochs if epochs is not None else self.epochs
        self.verbose = verbose if verbose is not None else self.verbose
        self.early_stopping = early_stopping if early_stopping is not None else self.early_stopping
        self.tol = tol if tol is not None else self.tol
        self.patience = patience if patience is not None else self.patience
        self.validation_split = validation_split if validation_split is not None else self.validation_split
        self.validation_data = validation_data if validation_data is not None else self.validation_data
        self.shuffle = shuffle if shuffle is not None else self.shuffle
        self.class_weight = class_weight if class_weight is not None else self.class_weight
        self.sample_weight = sample_weight if sample_weight is not None else self.sample_weight
        self.initial_epoch = initial_epoch if initial_epoch is not None else self.initial_epoch
        y = y.reshape((len(y), 1)) if len(y.shape) == 1 else y
        if self.recurrent_units is not None:
            X = time_series_tensor(X, self.recurrent_window)
            y = time_series_tensor(y, self.recurrent_window) if self.recurrent_return_sequences else y[self.recurrent_window - 1:]
        X, y = check_X_y(X, y, ensure_2d=False, allow_nd=True,
                         multi_output=True)
        self.model_ = self._model(X, y)
        self.model_.compile(self._solver(self.solver), self.loss,
                            metrics=self.metrics,
                            loss_weights=self.loss_weights,
                            sample_weight_mode=self.sample_weight_mode)
        callbacks = [EarlyStopping(monitor='val_loss' if (self.validation_split > 0.0 or self.validation_data is not None) else 'loss',
                                   min_delta=self.tol, patience=self.patience)] if self.early_stopping and (self.tol > 0.0) else []
        self.history_ = self.model_.fit(X, y,
                                        batch_size=min(200, len(X)) if self.batch_size == 'auto' else self.batch_size,
                                        epochs=self.epochs,
                                        verbose=self.verbose,
                                        callbacks=callbacks,
                                        validation_split=self.validation_split,
                                        validation_data=self.validation_data,
                                        shuffle=self.shuffle,
                                        class_weight=self.class_weight,
                                        sample_weight=np.asarray(self.sample_weight) if type(self.sample_weight) in (list, tuple) else self.sample_weight,
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
        if self.recurrent_units is not None:
            X = time_series_tensor(X, self.recurrent_window)
        X = check_array(X, ensure_2d=False, allow_nd=True)
        preds = self.model_.predict(X, batch_size=batch_size, verbose=verbose)
        return preds.reshape((len(preds))) if (len(preds.shape) == 2 and preds.shape[1] == 1) else preds

    transform = predict

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
        y = y.reshape((len(y), 1)) if len(y.shape) == 1 else y
        if self.recurrent_units is not None:
            X = time_series_tensor(X, self.recurrent_window)
            y = time_series_tensor(y, self.recurrent_window) if self.recurrent_return_sequences else y[self.recurrent_window - 1:]
        X, y = check_X_y(X, y, ensure_2d=False, allow_nd=True,
                         multi_output=True)
        return metric(y, self.predict(X), sample_weight=sample_weight)


###############################################################################
#  Classifier class
###############################################################################


class FFClassifier(BaseFeedForward, ClassifierMixin, TransformerMixin):

    __doc__ = BaseFeedForward.__doc__

    output_activation = 'softmax'

    loss = 'categorical_crossentropy'

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

    transform = predict_proba

    def score(self, X, y, sample_weight=None, metric=accuracy_score):
        return BaseFeedForward.score(self, X, y, sample_weight=sample_weight,
                                     metric=metric)
    score.__doc__ = BaseFeedForward.score.__doc__


###############################################################################
#  Regressor class
###############################################################################


class FFRegressor(BaseFeedForward, RegressorMixin, TransformerMixin):

    __doc__ = BaseFeedForward.__doc__

    output_activation = 'linear'

    loss = 'mse'
