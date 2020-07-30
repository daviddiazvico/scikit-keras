"""Build functions to be used with the wrapper.
"""

from functools import wraps
from keras import backend as K
from keras.layers import (AveragePooling1D, AveragePooling2D, AveragePooling3D,
                          BatchNormalization, Dense, Dropout, Flatten, Input,
                          Conv1D, Conv2D, Conv3D, GRU, MaxPooling1D,
                          MaxPooling2D, MaxPooling3D, LSTM, TimeDistributed)
from keras.models import Model
from keras.optimizers import (Adadelta, Adagrad, Adam, Adamax, Nadam, RMSprop,
                              SGD)
from keras.regularizers import l1 as l1_, l2 as l2_, l1_l2 as l1_l2_
import numpy as np

from skkeras.scikit_learn import BaseWrapper


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
    Regularizer

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


class SingleIO():

    """Single input/output architecture.

    Single input/output architecture class.

    Parameters
    ----------
    input_shape: tuple
        Input shape.
    output_shape: tuple
        Output shape.
    hidden: keras function, default=None
        Feature transformation in the hidden layers.
    activation: string/function, default='linear'/'softmax'
        Activation function to use.
    use_bias: boolean, default=True
        Whether the layer uses a bias vector.
    kernel_initializer: string/function, default='glorot_uniform'
        Initializer for the kernel weights matrix.
    bias_initializer: string/function, default='zeros'
        Initializer for the bias vector.
    kernel_regularizer_l1: float, default=None
        L1 regularization factor applied to the kernel weights matrix.
    kernel_regularizer_l2: float, default=None
        L2 regularization factor applied to the kernel weights matrix.
    bias_regularizer_l1: float, default=None
        L1 regularization factor applied to the bias vector.
    bias_regularizer_l2: float, default=None
        L2 regularization factor applied to the bias vector.
    activity_regularizer_l1: float, default=None
        L1 regularization factor applied to the output of the layer.
    activity_regularizer_l2: float, default=None
        L2 regularization factor applied to the output of the layer.
    kernel_constraint: function, default=None
        Constraint function applied to the kernel weights matrix.
    bias_constraint: function, default=None
        Constraint function applied to the bias vector.
    return_sequences: boolean, default=False
        Whether to return the last output in the output sequence, or the full
        sequence.

    Returns
    -------
    Model

    """

    def __new__(cls, input_shape, output_shape, hidden=None,
                activation='linear', use_bias=True,
                kernel_initializer='glorot_uniform', bias_initializer='zeros',
                kernel_regularizer_l1=None, kernel_regularizer_l2=None,
                bias_regularizer_l1=None, bias_regularizer_l2=None,
                activity_regularizer_l1=None, activity_regularizer_l2=None,
                kernel_constraint=None, bias_constraint=None,
                return_sequences=False):
        z = inputs = Input(shape=input_shape)
        if hidden is not None:
            z = hidden(z)
        layer = Dense(np.prod(output_shape, dtype=np.int32),
                      activation=activation, use_bias=use_bias,
                      kernel_initializer=kernel_initializer,
                      bias_initializer=bias_initializer,
                      kernel_regularizer=Regularizer(l1=kernel_regularizer_l1,
                                                     l2=kernel_regularizer_l2),
                      bias_regularizer=Regularizer(l1=bias_regularizer_l1,
                                                   l2=bias_regularizer_l2),
                      activity_regularizer=Regularizer(l1=activity_regularizer_l1,
                                                       l2=activity_regularizer_l2),
                      kernel_constraint=kernel_constraint,
                      bias_constraint=bias_constraint)
        if return_sequences:
            layer = TimeDistributed(layer)
        output = layer(z)
        return Model(inputs, output)


class Straight():

    """Straight feed-forward hidden-layers.

    Basic straight feed-forward hidden-layer architecture.

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
    covolution_kernel_regularizer_l1: float, default=None
        L1 regularization factor applied to the kernel weights matrix.
    convolution_kernel_regularizer_l2: float, default=None
        L2 regularization factor applied to the kernel weights matrix.
    convolution_bias_regularizer_l1: float, default=None
        L1 regularization factor applied to the bias vector.
    convolution_bias_regularizer_l2: float, default=None
        L2 regularization factor applied to the bias vector.
    convolution_activity_regularizer_l1: float, default=None
        L1 regularization factor applied to the output of the layer.
    convolution_activity_regularizer_l2: float, default=None
        L2 regularization factor applied to the output of the layer.
    convolution_kernel_constraint: function, default=None
        Constraint function applied to the kernel matrix.
    convolution_bias_constraint: function, default=None
        Constraint function applied to the bias vector.
    pooling_type: {"max", "average}, default='max'
    pooling_pool_size: integer/tuple/list, default=None
        Factors by which to downscale.
    pooling_strides: integer/tuple/list, default=None
        Strides values.
    pooling_padding: {"valid", "same"}, default='valid'
    recurrent_type: {"lstm", "gru"}, default='lstm'
    recurrent_units: integer, default=None
        Dimensionality of the output space.
    recurrent_activation: string/function, default='tanh'
        Activation function to use.
    recurrent_recurrent_activation: string/function, default='hard_sigmoid'
        Activation function to use for the recurrent step.
    recurrent_use_bias: boolean, default=True
        Whether the layer uses a bias vector.
    recurrent_kernel_initializer: string/function, default='glorot_uniform'
        Initializer for the kernel weights matrix.
    recurrent_recurrent_initializer: string/function, default='orthogonal'
        Initializer for the recurrent_kernel weights matrix.
    recurrent_bias_initializer: string/function, default='zeros'
        Initializer for the bias vector.
    recurrent_unit_forget_bias: boolean, default=True
        If True, add 1 to the bias of the forget gate at initialization.
    recurrent_kernel_regularizer_l1: float, default=None
        L1 regularization factor applied to the kernel weights matrix.
    recurrent_kernel_regularizer_l2: float, default=None
        L2 regularization factor applied to the kernel weights matrix.
    recurrent_bias_regularizer_l1: float, default=None
        L1 regularization factor applied to the bias vector.
    recurrent_bias_regularizer_l2: float, default=None
        L2 regularization factor applied to the bias vector.
    recurrent_activity_regularizer_l1: float, default=None
        L1 regularization factor applied to the output of the layer.
    recurrent_activity_regularizer_l2: float, default=None
        L2 regularization factor applied to the output of the layer.
    recurrent_kernel_constraint: function, default=None
        Constraint function applied to the kernel weights matrix.
    recurrent_recurrent_constraint: function, default=None
        Constraint function applied to the recurrent_kernel weights matrix.
    recurrent_bias_constraint: function, default=None
        Constraint function applied to the bias vector.
    recurrent_dropout: float in [0, 1], default=0.0
        Fraction of the units to drop for the linear transformation of the
        inputs.
    recurrent_recurrent_dropout: float in [0, 1], default=0.0
        Fraction of the units to drop for the linear transformation of the
        recurrent state.
    recurrent_go_backwards: boolean, default=False
        If True, process the input sequence backwards and return the reversed
        sequence.
    recurrent_stateful: boolean, default=False
        If True, the last state for each sample at index i in a batch will be
        used as initial state for the sample of index i in the following batch.
    recurrent_unroll: boolean, default=False
        If True, the network will be unrolled, else a symbolic loop will be
        used.
    recurrent_implementation: {0, 1, 2}, default=1
    batchnormalization: boolean, default=False
        Whether to perform batch normalization or not.
    batchnormalization_axis: integer, default=-1
        The axis that should be normalized (typically the features axis).
    batchnormalization_momentum: float, default=0.99
        Momentum for the moving average.
    batchnormalization_epsilon: float, default=0.001
        Small float added to variance to avoid dividing by zero.
    batchnormalization_center: boolean, default=True
        If True, add offset of beta to normalized tensor. If False, beta is
        ignored.
    batchnormalization_scale: boolean, default=True
        If True, multiply by gamma. If False, gamma is not used.
    batchnormalization_beta_initializer: string/function, default='zeros'
        Initializer for the beta weight.
    batchnormalization_gamma_initializer: string/function, default='ones'
        Initializer for the gamma weight.
    batchnormalization_moving_mean_initializer: string/function, default='zeros'
        Initializer for the moving mean.
    batchnormalization_moving_variance_initializer: string/function,
        default='ones'
        Initializer for the moving variance.
    batchnormalization_beta_constraint: function, default=None
        Optional constraint for the beta weight.
    batchnormalization_gamma_constraint: function, default=None
        Optional constraint for the gamma weight.
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
        L1 regularization factor applied to the kernel weights matrix.
    dense_kernel_regularizer_l2: float, default=None
        L2 regularization factor applied to the kernel weights matrix.
    dense_bias_regularizer_l1: float, default=None
        L1 regularization factor applied to the bias vector.
    dense_bias_regularizer_l2: float, default=None
        L2 regularization factor applied to the bias vector.
    dense_activity_regularizer_l1: float, default=None
        L1 regularization factor applied to the output of the layer.
    dense_activity_regularizer_l2: float, default=None
        L2 regularization factor applied to the output of the layer.
    dense_kernel_constraint: function, default=None
        Constraint function applied to the kernel weights matrix.
    dense_bias_constraint: function, default=None
        Constraint function applied to the bias vector.
    dropout_rate: float in [0, 1], default=0.0
        Fraction of the input units to drop.
    dropout_noise_shape: array-like, default=None
        shape of the binary dropout mask that will be multiplied with the input.
    dropout_seed: integer, default=None
        Random seed.
    recurrent_regularizer_l1: float, default=None
        L1 regularization factor applied to the recurrent_kernel weights matrix.
    recurrent_regularizer_l2: float, default=None
        L2 regularization factor applied to the recurrent_kernel weights matrix.
    beta_regularizer_l1: float, default=None
        L1 regularization factor applied to the beta weight.
    beta_regularizer_l2: float, default=None
        L2 regularization factor applied to the beta weight.
    gamma_regularizer_l1: float, default=None
        L1 regularization factor applied to the gamma  weight.
    gamma_regularizer_l2: float, default=None
        L2 regularization factor applied to the gamma  weight.
    return_sequences: boolean, default=False
        Whether to return the last output in the output sequence, or the full
        sequence.

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
                 recurrent_units=None, recurrent_activation='tanh',
                 recurrent_recurrent_activation='hard_sigmoid',
                 recurrent_use_bias=True,
                 recurrent_kernel_initializer='glorot_uniform',
                 recurrent_recurrent_initializer='orthogonal',
                 recurrent_bias_initializer='zeros',
                 recurrent_unit_forget_bias=True,
                 recurrent_kernel_regularizer_l1=None,
                 recurrent_kernel_regularizer_l2=None,
                 recurrent_bias_regularizer_l1=None,
                 recurrent_bias_regularizer_l2=None,
                 recurrent_activity_regularizer_l1=None,
                 recurrent_activity_regularizer_l2=None,
                 recurrent_kernel_constraint=None,
                 recurrent_recurrent_constraint=None,
                 recurrent_bias_constraint=None, recurrent_dropout=0.0,
                 recurrent_recurrent_dropout=0.0, recurrent_go_backwards=False,
                 recurrent_stateful=False, recurrent_unroll=False,
                 recurrent_implementation=1, batchnormalization=False,
                 batchnormalization_axis=-1, batchnormalization_momentum=0.99,
                 batchnormalization_epsilon=0.001,
                 batchnormalization_center=True, batchnormalization_scale=True,
                 batchnormalization_beta_initializer='zeros',
                 batchnormalization_gamma_initializer='ones',
                 batchnormalization_moving_mean_initializer='zeros',
                 batchnormalization_moving_variance_initializer='ones',
                 batchnormalization_beta_constraint=None,
                 batchnormalization_gamma_constraint=None, dense_units=None,
                 dense_activation='relu', dense_use_bias=True,
                 dense_kernel_initializer='he_uniform',
                 dense_bias_initializer='zeros',
                 recurrent_regularizer_l1=None,
                 recurrent_regularizer_l2=None, beta_regularizer_l1=None,
                 beta_regularizer_l2=None, gamma_regularizer_l1=None,
                 gamma_regularizer_l2=None,
                 dense_kernel_regularizer_l1=None,
                 dense_kernel_regularizer_l2=None, dense_bias_regularizer_l1=None,
                 dense_bias_regularizer_l2=None, dense_activity_regularizer_l1=None,
                 dense_activity_regularizer_l2=None,
                 dense_kernel_constraint=None,
                 dense_bias_constraint=None, dropout_rate=0.0,
                 dropout_noise_shape=None, dropout_seed=None,
                 return_sequences=False):
        for k, v in locals().items():
            if k != 'self':
                self.__dict__[k] = v

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
                                                       kernel_regularizer=self.convolution_kernel_regularizer,
                                                       bias_regularizer=self.convolution_bias_regularizer,
                                                       activity_regularizer=self.convolution_activity_regularizer,
                                                       kernel_constraint=self.convolution_kernel_constraint,
                                                       bias_constraint=self.convolution_bias_constraint)
            if return_sequences:
                layer = TimeDistributed(layer)
            x = layer(x)
        if pooling_pool_size is not None:
            pool = {'max': {1: MaxPooling1D, 2: MaxPooling2D, 3: MaxPooling3D},
                    'average': {1: AveragePooling1D, 2: AveragePooling2D,
                                3: AveragePooling3D}}
            layer = pool[self.pooling_type][len(pooling_pool_size)](pool_size=pooling_pool_size,
                                                                    strides=pooling_strides,
                                                                    padding=self.pooling_padding)
            if return_sequences:
                layer = TimeDistributed(layer)
            x = layer(x)
        if not return_tensors:
            layer = Flatten()
            if return_sequences:
                layer = TimeDistributed(layer)
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
                                           kernel_regularizer=self.recurrent_kernel_regularizer,
                                           recurrent_regularizer=self._recurrent_regularizer,
                                           bias_regularizer=self.recurrent_bias_regularizer,
                                           activity_regularizer=self.recurrent_activity_regularizer,
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
            layer = BatchNormalization(axis=self.batchnormalization_axis,
                                       momentum=self.batchnormalization_momentum,
                                       epsilon=self.batchnormalization_epsilon,
                                       center=self.batchnormalization_center,
                                       scale=self.batchnormalization_scale,
                                       beta_initializer=self.batchnormalization_beta_initializer,
                                       gamma_initializer=self.batchnormalization_gamma_initializer,
                                       moving_mean_initializer=self.batchnormalization_moving_mean_initializer,
                                       moving_variance_initializer=self.batchnormalization_moving_variance_initializer,
                                       beta_regularizer=self._beta_regularizer,
                                       gamma_regularizer=self._gamma_regularizer,
                                       beta_constraint=self.batchnormalization_beta_constraint,
                                       gamma_constraint=self.batchnormalization_gamma_constraint)
            if self.return_sequences:
                layer = TimeDistributed(layer)
            x = layer(x)
        layer = Dense(units, activation=self.dense_activation,
                      use_bias=self.dense_use_bias,
                      kernel_initializer=self.dense_kernel_initializer,
                      bias_initializer=self.dense_bias_initializer,
                      kernel_regularizer=self.dense_kernel_regularizer,
                      bias_regularizer=self.dense_bias_regularizer,
                      activity_regularizer=self.dense_activity_regularizer,
                      kernel_constraint=self.dense_kernel_constraint,
                      bias_constraint=self.dense_bias_constraint)
        if self.return_sequences:
            layer = TimeDistributed(layer)
        x = layer(x)
        if 0.0 < self.dropout_rate < 1.0:
            layer = Dropout(self.dropout_rate, noise_shape=dropout_noise_shape,
                            seed=self.dropout_seed)
            if self.return_sequences:
                layer = TimeDistributed(layer)
            x = layer(x)
        return x

    def __call__(self, z):
        self.convolution_kernel_regularizer = Regularizer(l1=self.convolution_kernel_regularizer_l1,
                                               l2=self.convolution_kernel_regularizer_l2)
        self.convolution_bias_regularizer = Regularizer(l1=self.convolution_bias_regularizer_l1,
                                             l2=self.convolution_bias_regularizer_l2)
        self.convolution_activity_regularizer = Regularizer(l1=self.convolution_activity_regularizer_l1,
                                                 l2=self.convolution_activity_regularizer_l2)
        self.recurrent_kernel_regularizer = Regularizer(l1=self.recurrent_kernel_regularizer_l1,
                                               l2=self.recurrent_kernel_regularizer_l2)
        self.recurrent_bias_regularizer = Regularizer(l1=self.recurrent_bias_regularizer_l1,
                                             l2=self.recurrent_bias_regularizer_l2)
        self.recurrent_activity_regularizer = Regularizer(l1=self.recurrent_activity_regularizer_l1,
                                                 l2=self.recurrent_activity_regularizer_l2)
        self.dense_kernel_regularizer = Regularizer(l1=self.dense_kernel_regularizer_l1,
                                               l2=self.dense_kernel_regularizer_l2)
        self.dense_bias_regularizer = Regularizer(l1=self.dense_bias_regularizer_l1,
                                             l2=self.dense_bias_regularizer_l2)
        self.dense_activity_regularizer = Regularizer(l1=self.dense_activity_regularizer_l1,
                                                 l2=self.dense_activity_regularizer_l2)
        self._recurrent_regularizer = Regularizer(l1=self.recurrent_regularizer_l1,
                                                  l2=self.recurrent_regularizer_l2)
        self._beta_regularizer = Regularizer(l1=self.beta_regularizer_l1,
                                             l2=self.beta_regularizer_l2)
        self._gamma_regularizer = Regularizer(l1=self.gamma_regularizer_l1,
                                              l2=self.gamma_regularizer_l2)
        if (self.convolution_filters is not None) or (self.convolution_kernel_size is not None):
            if len(self.convolution_filters) == len(self.convolution_kernel_size):
                if self.convolution_strides is None:
                    self.convolution_strides = [
                        [1] * len(k) for k in self.convolution_kernel_size]
                if self.convolution_dilation_rate is None:
                    self.convolution_dilation_rate = [
                        [1] * len(k) for k in self.convolution_kernel_size]
                if self.pooling_pool_size is None:
                    self.pooling_pool_size = [None] * \
                        len(self.convolution_filters)
                if self.pooling_strides is None:
                    self.pooling_strides = [None] * \
                        len(self.convolution_filters)
                for i, (cf, cks, cs, cdr, pps, ps) in enumerate(zip(self.convolution_filters,
                                                                    self.convolution_kernel_size,
                                                                    self.convolution_strides,
                                                                    self.convolution_dilation_rate,
                                                                    self.pooling_pool_size,
                                                                    self.pooling_strides)):
                    z = self._convolve_and_pool(z, cf, cks, cs, cdr, pps, ps,
                                                return_tensors=i < len(
                                                    self.convolution_filters) - 1,
                                                return_sequences=self.recurrent_units is not None)
        if self.recurrent_units is not None:
            for i, ru in enumerate(self.recurrent_units):
                z = self._recur(z, ru,
#                                return_sequences=i < len(self.recurrent_units) - 1)
                                return_sequences=self.return_sequences or (i < len(self.recurrent_units) - 1))
        if self.dense_units is not None:
            if self.dropout_noise_shape is None:
                self.dropout_noise_shape = [None] * len(self.dense_units)
            for (du, dns) in zip(self.dense_units, self.dropout_noise_shape):
                z = self._connect(z, du, dropout_noise_shape=dns)
        return z


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


def build_fn(input_shape, output_shape, activation='linear', use_bias=True,
             kernel_initializer='glorot_uniform', bias_initializer='zeros',
             kernel_regularizer_l1=None, kernel_regularizer_l2=None,
             bias_regularizer_l1=None, bias_regularizer_l2=None,
             activity_regularizer_l1=None, activity_regularizer_l2=None,
             kernel_constraint=None, bias_constraint=None,
             return_sequences=False, convolution_filters=None,
             convolution_kernel_size=None,
             convolution_strides=None,
             convolution_padding='valid',
             convolution_dilation_rate=None,
             convolution_activation=None,
             convolution_use_bias=True,
             convolution_kernel_initializer='glorot_uniform',
             convolution_bias_initializer='zeros',
             convolution_kernel_regularizer_l1=None,
             convolution_kernel_regularizer_l2=None, convolution_bias_regularizer_l1=None,
             convolution_bias_regularizer_l2=None,
             convolution_activity_regularizer_l1=None,
             convolution_activity_regularizer_l2=None,
             convolution_kernel_constraint=None,
             convolution_bias_constraint=None,
             pooling_type='max', pooling_pool_size=None,
             pooling_strides=None, pooling_padding='valid',
             recurrent_type='lstm', recurrent_units=None,
             recurrent_activation='tanh',
             recurrent_recurrent_activation='hard_sigmoid',
             recurrent_use_bias=True,
             recurrent_kernel_initializer='glorot_uniform',
             recurrent_recurrent_initializer='orthogonal',
             recurrent_bias_initializer='zeros',
             recurrent_unit_forget_bias=True,
             recurrent_kernel_regularizer_l1=None,
             recurrent_kernel_regularizer_l2=None, recurrent_bias_regularizer_l1=None,
             recurrent_bias_regularizer_l2=None,
             recurrent_activity_regularizer_l1=None,
             recurrent_activity_regularizer_l2=None,
             recurrent_kernel_constraint=None,
             recurrent_recurrent_constraint=None,
             recurrent_bias_constraint=None,
             recurrent_dropout=0.0,
             recurrent_recurrent_dropout=0.0,
             recurrent_go_backwards=False,
             recurrent_stateful=False, recurrent_unroll=False,
             recurrent_implementation=1, batchnormalization=False,
             batchnormalization_axis=-1,
             batchnormalization_momentum=0.99,
             batchnormalization_epsilon=0.001,
             batchnormalization_center=True,
             batchnormalization_scale=True,
             batchnormalization_beta_initializer='zeros',
             batchnormalization_gamma_initializer='ones',
             batchnormalization_moving_mean_initializer='zeros',
             batchnormalization_moving_variance_initializer='ones',
             batchnormalization_beta_constraint=None,
             batchnormalization_gamma_constraint=None,
             dense_units=None, dense_activation='relu',
             dense_use_bias=True,
             dense_kernel_initializer='he_uniform',
             dense_bias_initializer='zeros',
             dense_kernel_regularizer_l1=None,
             dense_kernel_regularizer_l2=None, dense_bias_regularizer_l1=None,
             dense_bias_regularizer_l2=None,
             dense_activity_regularizer_l1=None,
             dense_activity_regularizer_l2=None,
             recurrent_regularizer_l1=None,
             recurrent_regularizer_l2=None,
             beta_regularizer_l1=None, beta_regularizer_l2=None,
             gamma_regularizer_l1=None, gamma_regularizer_l2=None,
             dense_kernel_constraint=None,
             dense_bias_constraint=None, dropout_rate=0.0,
             dropout_noise_shape=None, dropout_seed=None,
             optimizer='adam', lr=0.001, momentum=0.0, nesterov=False,
             decay=0.0, rho=0.9, epsilon=1e-08, beta_1=0.9, beta_2=0.999,
             schedule_decay=0.004, loss=None, metrics=None, loss_weights=None,
             sample_weight_mode=None, batch_size=None, epochs=None,
             verbose=None, early_stopping=None, tol=None, patience=None,
             validation_split=None, validation_data=None, shuffle=None,
             class_weight=None, sample_weight=None, initial_epoch=None):
    """Build a neural network.

    Build a neural network with the specified hyper-parameters.

    Parameters
    ----------
    input_shape: tuple
        Input shape.
    output_shape: tuple
        Output shape.
    activation: string/function, default='linear'/'softmax'
        Activation function to use.
    use_bias: boolean, default=True
        Whether the layer uses a bias vector.
    kernel_initializer: string/function, default='glorot_uniform'
        Initializer for the kernel weights matrix.
    bias_initializer: string/function, default='zeros'
        Initializer for the bias vector.
    kernel_regularizer_l1: float, default=None
        L1 regularization factor applied to the kernel weights matrix.
    kernel_regularizer_l2: float, default=None
        L2 regularization factor applied to the kernel weights matrix.
    bias_regularizer_l1: float, default=None
        L1 regularization factor applied to the bias vector.
    bias_regularizer_l2: float, default=None
        L2 regularization factor applied to the bias vector.
    activity_regularizer_l1: float, default=None
        L1 regularization factor applied to the output of the layer.
    activity_regularizer_l2: float, default=None
        L2 regularization factor applied to the output of the layer.
    kernel_constraint: function, default=None
        Constraint function applied to the kernel weights matrix.
    bias_constraint: function, default=None
        Constraint function applied to the bias vector.
    return_sequences: boolean, default=False
        Whether to return the last output in the output sequence, or the full
        sequence.
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
    convolution_kernel_initializer: string/function,
        default='glorot_uniform'
        Initializer for the kernel weights matrix.
    convolution_bias_initializer: string/function, default='zeros'
        Initializer for the bias vector.
    convolution_kernel_constraint: function, default=None
        Constraint function applied to the kernel matrix.
    convolution_bias_constraint: function, default=None
        Constraint function applied to the bias vector.
    pooling_type: {"max", "average}, default='max'
    pooling_pool_size: integer/tuple/list, default=None
        Factors by which to downscale.
    pooling_strides: integer/tuple/list, default=None
        Strides values.
    pooling_padding: {"valid", "same"}, default='valid'
    recurrent_type: {"lstm", "gru"}, default='lstm'
    recurrent_units: integer, default=None
        Dimensionality of the output space.
    recurrent_activation: string/function, default='tanh'
        Activation function to use.
    recurrent_recurrent_activation: string/function,
        default='hard_sigmoid'
        Activation function to use for the recurrent step.
    recurrent_use_bias: boolean, default=True
        Whether the layer uses a bias vector.
    recurrent_kernel_initializer: string/function,
        default='glorot_uniform'
        Initializer for the kernel weights matrix.
    recurrent_recurrent_initializer: string/function,
        default='orthogonal'
        Initializer for the recurrent_kernel weights matrix.
    recurrent_bias_initializer: string/function, default='zeros'
        Initializer for the bias vector.
    recurrent_unit_forget_bias: boolean, default=True
        If True, add 1 to the bias of the forget gate at initialization.
    recurrent_kernel_constraint: function, default=None
        Constraint function applied to the kernel weights matrix.
    recurrent_recurrent_constraint: function, default=None
        Constraint function applied to the recurrent_kernel weights matrix.
    recurrent_bias_constraint: function, default=None
        Constraint function applied to the bias vector.
    recurrent_dropout: float in [0, 1], default=0.0
        Fraction of the units to drop for the linear transformation of the
        inputs.
    recurrent_recurrent_dropout: float in [0, 1], default=0.0
        Fraction of the units to drop for the linear transformation of the
        recurrent state.
    recurrent_return_sequences: boolean, default=False
        Whether to return the last output in the output sequence, or the full
        sequence.
    recurrent_go_backwards: boolean, default=False
        If True, process the input sequence backwards and return the reversed
        sequence.
    recurrent_stateful: boolean, default=False
        If True, the last state for each sample at index i in a batch will be
        used as initial state for the sample of index i in the following batch.
    recurrent_unroll: boolean, default=False
        If True, the network will be unrolled, else a symbolic loop will be
        used.
    recurrent_implementation: {0, 1, 2}, default=1
    batchnormalization: boolean, default=False
        Whether to perform batch normalization or not.
    batchnormalization_axis: integer, default=-1
        The axis that should be normalized (typically the features axis).
    batchnormalization_momentum: float, default=0.99
        Momentum for the moving average.
    batchnormalization_epsilon: float, default=0.001
        Small float added to variance to avoid dividing by zero.
    batchnormalization_center: boolean, default=True
        If True, add offset of beta to normalized tensor. If False, beta is
        ignored.
    batchnormalization_scale: boolean, default=True
        If True, multiply by gamma. If False, gamma is not used.
    batchnormalization_beta_initializer: string/function, default='zeros'
        Initializer for the beta weight.
    batchnormalization_gamma_initializer: string/function, default='ones'
        Initializer for the gamma weight.
    batchnormalization_moving_mean_initializer: string/function,
        default='zeros'
        Initializer for the moving mean.
    batchnormalization_moving_variance_initializer: string/function,
        default='ones'
        Initializer for the moving variance.
    batchnormalization_beta_constraint: function, default=None
        Optional constraint for the beta weight.
    batchnormalization_gamma_constraint: function, default=None
        Optional constraint for the gamma weight.
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
    dense_kernel_constraint: function, default=None
        Constraint function applied to the kernel weights matrix.
    dense_bias_constraint: function, default=None
        Constraint function applied to the bias vector.
    dropout_rate: float in [0, 1], default=0.0
        Fraction of the input units to drop.
    dropout_noise_shape: array-like, default=None
        shape of the binary dropout mask that will be multiplied with the input.
    dropout_seed: integer, default=None
        Random seed.
    kernel_regularizer_l1: float, default=None
        L1 regularization factor applied to the kernel weights matrix.
    kernel_regularizer_l2: float, default=None
        L2 regularization factor applied to the kernel weights matrix.
    bias_regularizer_l1: float, default=None
        L1 regularization factor applied to the bias vector.
    bias_regularizer_l2: float, default=None
        L2 regularization factor applied to the bias vector.
    activity_regularizer_l1: float, default=None
        L1 regularization factor applied to the output of the layer.
    activity_regularizer_l2: float, default=None
        L2 regularization factor applied to the output of the layer.
    recurrent_regularizer_l1: float, default=None
        L1 regularization factor applied to the recurrent_kernel weights matrix.
    recurrent_regularizer_l2: float, default=None
        L2 regularization factor applied to the recurrent_kernel weights matrix.
    beta_regularizer_l1: float, default=None
        L1 regularization factor applied to the beta weight.
    beta_regularizer_l2: float, default=None
        L2 regularization factor applied to the beta weight.
    gamma_regularizer_l1: float, default=None
        L1 regularization factor applied to the gamma  weight.
    gamma_regularizer_l2: float, default=None
        L2 regularization factor applied to the gamma  weight.
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
        Scalar coefficients to weight the loss contributions of different model
        outputs.
    sample_weight_mode: {"temporal", None}, default=None
        Timestep-wise sample weighting.
    batch_size: integer, default='auto'
        Number of samples per gradient update.
    epochs: integer, default=200
        The number of times to iterate over the training data arrays.
    verbose: {0, 1, 2}, default=1
        Verbosity mode. 0=silent, 1=verbose, 2=one log line per epoch.
    early_stopping: bool, default True
        Whether to use early stopping to terminate training when validation
        score is not improving.
    tol: float, default 1e-4
        Tolerance for the optimization.
    patience: integer, default 2
        Number of epochs with no improvement after which training will be
        stopped.
    validation_split: float in [0, 1], default=0.1
        Fraction of the training data to be used as validation data.
    validation_data: array-like, shape ((n_samples, features_shape),
        (n_samples, targets_shape)), default=None
        Data on which to evaluate the loss and any model metrics at the end of
        each epoch.
    shuffle: boolean, default=True
        Whether to shuffle the training data before each epoch.
    class_weight: dictionary, default=None
        class indices to weights to apply to the model's loss for the samples
        from each class during training.
    sample_weight: array-like, shape (n_samples), default=None
        Weights to apply to the model's loss for each sample.
    initial_epoch: integer, default=0
        Epoch at which to start training.

    Returns
    -------
    Model

    """
    hidden = Straight(convolution_filters=convolution_filters,
                      convolution_kernel_size=convolution_kernel_size,
                      convolution_strides=convolution_strides,
                      convolution_padding=convolution_padding,
                      convolution_dilation_rate=convolution_dilation_rate,
                      convolution_activation=convolution_activation,
                      convolution_use_bias=convolution_use_bias,
                      convolution_kernel_initializer=convolution_kernel_initializer,
                      convolution_bias_initializer=convolution_bias_initializer,
                      convolution_kernel_regularizer_l1=convolution_kernel_regularizer_l1,
                      convolution_kernel_regularizer_l2=convolution_kernel_regularizer_l2,
                      convolution_bias_regularizer_l1=convolution_bias_regularizer_l1,
                      convolution_bias_regularizer_l2=convolution_bias_regularizer_l2,
                      convolution_activity_regularizer_l1=convolution_activity_regularizer_l1,
                      convolution_activity_regularizer_l2=convolution_activity_regularizer_l2,
                      convolution_kernel_constraint=convolution_kernel_constraint,
                      convolution_bias_constraint=convolution_bias_constraint,
                      pooling_type=pooling_type,
                      pooling_pool_size=pooling_pool_size,
                      pooling_strides=pooling_strides,
                      pooling_padding=pooling_padding,
                      recurrent_type=recurrent_type,
                      recurrent_units=recurrent_units,
                      recurrent_activation=recurrent_activation,
                      recurrent_recurrent_activation=recurrent_recurrent_activation,
                      recurrent_use_bias=recurrent_use_bias,
                      recurrent_kernel_initializer=recurrent_kernel_initializer,
                      recurrent_recurrent_initializer=recurrent_recurrent_initializer,
                      recurrent_bias_initializer=recurrent_bias_initializer,
                      recurrent_unit_forget_bias=recurrent_unit_forget_bias,
                      recurrent_kernel_constraint=recurrent_kernel_constraint,
                      recurrent_kernel_regularizer_l1=recurrent_kernel_regularizer_l1,
                      recurrent_kernel_regularizer_l2=recurrent_kernel_regularizer_l2,
                      recurrent_bias_regularizer_l1=recurrent_bias_regularizer_l1,
                      recurrent_bias_regularizer_l2=recurrent_bias_regularizer_l2,
                      recurrent_activity_regularizer_l1=recurrent_activity_regularizer_l1,
                      recurrent_activity_regularizer_l2=recurrent_activity_regularizer_l2,
                      recurrent_recurrent_constraint=recurrent_recurrent_constraint,
                      recurrent_bias_constraint=recurrent_bias_constraint,
                      recurrent_dropout=recurrent_dropout,
                      recurrent_recurrent_dropout=recurrent_recurrent_dropout,
                      recurrent_go_backwards=recurrent_go_backwards,
                      recurrent_stateful=recurrent_stateful,
                      recurrent_unroll=recurrent_unroll,
                      recurrent_implementation=recurrent_implementation,
                      batchnormalization=batchnormalization,
                      batchnormalization_axis=batchnormalization_axis,
                      batchnormalization_momentum=batchnormalization_momentum,
                      batchnormalization_epsilon=batchnormalization_epsilon,
                      batchnormalization_center=batchnormalization_center,
                      batchnormalization_scale=batchnormalization_scale,
                      batchnormalization_beta_initializer=batchnormalization_beta_initializer,
                      batchnormalization_gamma_initializer=batchnormalization_gamma_initializer,
                      batchnormalization_moving_mean_initializer=batchnormalization_moving_mean_initializer,
                      batchnormalization_moving_variance_initializer=batchnormalization_moving_variance_initializer,
                      batchnormalization_beta_constraint=batchnormalization_beta_constraint,
                      batchnormalization_gamma_constraint=batchnormalization_gamma_constraint,
                      dense_units=dense_units,
                      dense_activation=dense_activation,
                      dense_use_bias=dense_use_bias,
                      dense_kernel_initializer=dense_kernel_initializer,
                      dense_bias_initializer=dense_bias_initializer,
                      dense_kernel_regularizer_l1=dense_kernel_regularizer_l1,
                      dense_kernel_regularizer_l2=dense_kernel_regularizer_l2,
                      dense_bias_regularizer_l1=dense_bias_regularizer_l1,
                      dense_bias_regularizer_l2=dense_bias_regularizer_l2,
                      dense_activity_regularizer_l1=dense_activity_regularizer_l1,
                      dense_activity_regularizer_l2=dense_activity_regularizer_l2,
                      recurrent_regularizer_l1=recurrent_regularizer_l1,
                      recurrent_regularizer_l2=recurrent_regularizer_l2,
                      beta_regularizer_l1=beta_regularizer_l1,
                      beta_regularizer_l2=beta_regularizer_l2,
                      gamma_regularizer_l1=gamma_regularizer_l1,
                      gamma_regularizer_l2=gamma_regularizer_l2,
                      dense_kernel_constraint=dense_kernel_constraint,
                      dense_bias_constraint=dense_bias_constraint,
                      dropout_rate=dropout_rate,
                      dropout_noise_shape=dropout_noise_shape,
                      dropout_seed=dropout_seed,
                     return_sequences=return_sequences)
    model = SingleIO(input_shape, output_shape, hidden=hidden,
                     activation=activation, use_bias=use_bias,
                     kernel_initializer=kernel_initializer,
                     bias_initializer=bias_initializer,
                     kernel_regularizer_l1=kernel_regularizer_l1,
                     kernel_regularizer_l2=kernel_regularizer_l2,
                     bias_regularizer_l1=bias_regularizer_l1,
                     bias_regularizer_l2=bias_regularizer_l2,
                     activity_regularizer_l1=activity_regularizer_l1,
                     activity_regularizer_l2=activity_regularizer_l2,
                     kernel_constraint=kernel_constraint,
                     bias_constraint=bias_constraint,
                     return_sequences=return_sequences)
    optimizer = Optimizer(optimizer=optimizer, lr=lr, momentum=momentum,
                          nesterov=nesterov, decay=decay, rho=rho,
                          epsilon=epsilon, beta_1=beta_1, beta_2=beta_2,
                          schedule_decay=schedule_decay)
    model.compile(optimizer, loss, metrics=metrics, loss_weights=loss_weights,
                  sample_weight_mode=sample_weight_mode)
    return model


def partial_with_signature(f, **fixed_kwargs):
    @wraps(f)
    def wrapper(*args, **kwargs):
        kwargs.update(fixed_kwargs)
        return f(*args, **kwargs)
    return wrapper


build_fn_classifier = partial_with_signature(build_fn, activation='softmax',
                                             loss='categorical_crossentropy',
                                             metrics=['accuracy'])


build_fn_regressor = partial_with_signature(build_fn, activation='linear',
                                            loss='mean_squared_error')
