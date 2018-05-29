"""
Scikit-learn-compatible Keras model architectures.

@author: David Diaz Vico
@license: MIT
"""

from keras.layers import (AveragePooling1D, AveragePooling2D, AveragePooling3D,
                          BatchNormalization, Dense, Dropout, Flatten, Input,
                          Conv1D, Conv2D, Conv3D, GRU, MaxPooling1D,
                          MaxPooling2D, MaxPooling3D, LSTM, TimeDistributed)

from .base import Regularizer


###############################################################################
#  Feed-forward architecture class
###############################################################################


class Straight:

    """Straight feed-forward architecture.

    Basic straight feed-forward model architecture.

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
    dense_kernel_constraint: function, default=None
                             Constraint function applied to the kernel weights
                             matrix.
    dense_bias_constraint: function, default=None
                           Constraint function applied to the bias vector.
    dropout_rate: float in [0, 1], default=0.0
                  Fraction of the input units to drop.
    dropout_noise_shape: array-like, default=None
                         shape of the binary dropout mask that will be
                         multiplied with the input.
    dropout_seed: integer, default=None
                  Random seed.
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
    recurrent_regularizer_l1: float, default=None
                              L1 regularization factor applied to the
                              recurrent_kernel weights matrix.
    recurrent_regularizer_l2: float, default=None
                              L2 regularization factor applied to the
                              recurrent_kernel weights matrix.
    beta_regularizer_l1: float, default=None
                         L1 regularization factor applied to the beta weight.
    beta_regularizer_l2: float, default=None
                         L2 regularization factor applied to the beta weight.
    gamma_regularizer_l1: float, default=None
                          L1 regularization factor applied to the gamma  weight.
    gamma_regularizer_l2: float, default=None
                          L2 regularization factor applied to the gamma  weight.

    """

    def __init__(self, convolution_filters=None, convolution_kernel_size=None,
                 convolution_strides=None, convolution_padding='valid',
                 convolution_dilation_rate=None, convolution_activation=None,
                 convolution_use_bias=True,
                 convolution_kernel_initializer='glorot_uniform',
                 convolution_bias_initializer='zeros',
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
                 batchnormalization_beta_constraint=None,
                 batchnormalization_gamma_constraint=None, dense_units=None,
                 dense_activation='relu', dense_use_bias=True,
                 dense_kernel_initializer='he_uniform',
                 dense_bias_initializer='zeros', kernel_regularizer_l1=None,
                 kernel_regularizer_l2=None, bias_regularizer_l1=None,
                 bias_regularizer_l2=None, activity_regularizer_l1=None,
                 activity_regularizer_l2=None, recurrent_regularizer_l1=None,
                 recurrent_regularizer_l2=None, beta_regularizer_l1=None,
                 beta_regularizer_l2=None, gamma_regularizer_l1=None,
                 gamma_regularizer_l2=None, dense_kernel_constraint=None,
                 dense_bias_constraint=None, dropout_rate=0.0,
                 dropout_noise_shape=None, dropout_seed=None):
        for k, v in locals().items():
            if k != 'self': self.__dict__[k] = v

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
                                                       kernel_regularizer=self._kernel_regularizer,
                                                       bias_regularizer=self._bias_regularizer,
                                                       activity_regularizer=self._activity_regularizer,
                                                       kernel_constraint=self.convolution_kernel_constraint,
                                                       bias_constraint=self.convolution_bias_constraint)
            if return_sequences: layer = TimeDistributed(layer)
            x = layer(x)
        if pooling_pool_size is not None:
            pool = {'max': {1: MaxPooling1D, 2: MaxPooling2D, 3: MaxPooling3D},
                    'average': {1: AveragePooling1D, 2: AveragePooling2D,
                                3: AveragePooling3D}}
            layer = pool[self.pooling_type][len(pooling_pool_size)](pool_size=pooling_pool_size,
                                                                    strides=pooling_strides,
                                                                    padding=self.pooling_padding)
            if return_sequences: layer = TimeDistributed(layer)
            x = layer(x)
        if not return_tensors:
            layer = Flatten()
            if return_sequences: layer = TimeDistributed(layer)
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
                                           kernel_regularizer=self._kernel_regularizer,
                                           recurrent_regularizer=self._recurrent_regularizer,
                                           bias_regularizer=self._bias_regularizer,
                                           activity_regularizer=self._activity_regularizer,
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
                                      beta_regularizer=self._beta_regularizer,
                                      gamma_regularizer=self._gamma_regularizer,
                                      beta_constraint=self.batchnormalization_beta_constraint,
                                      gamma_constraint=self.batchnormalization_gamma_constraint)
            if self.recurrent_return_sequences: layer = TimeDistributed(layer)
            x = layer(x)
        layer = Dense(units, activation=self.dense_activation,
                      use_bias=self.dense_use_bias,
                      kernel_initializer=self.dense_kernel_initializer,
                      bias_initializer=self.dense_bias_initializer,
                      kernel_regularizer=self._kernel_regularizer,
                      bias_regularizer=self._bias_regularizer,
                      activity_regularizer=self._activity_regularizer,
                      kernel_constraint=self.dense_kernel_constraint,
                      bias_constraint=self.dense_bias_constraint)
        if self.recurrent_return_sequences: layer = TimeDistributed(layer)
        x = layer(x)
        if 0.0 < self.dropout_rate < 1.0:
            layer = Dropout(self.dropout_rate, noise_shape=dropout_noise_shape,
                            seed=self.dropout_seed)
            if self.recurrent_return_sequences: layer = TimeDistributed(layer)
            x = layer(x)
        return x

    def __call__(self, z):
        self._kernel_regularizer = Regularizer(l1=self.kernel_regularizer_l1,
                                               l2=self.kernel_regularizer_l2)
        self._bias_regularizer = Regularizer(l1=self.bias_regularizer_l1,
                                             l2=self.bias_regularizer_l2)
        self._activity_regularizer = Regularizer(l1=self.activity_regularizer_l1,
                                                 l2=self.activity_regularizer_l2)
        self._recurrent_regularizer = Regularizer(l1=self.recurrent_regularizer_l1,
                                                  l2=self.recurrent_regularizer_l2)
        self._beta_regularizer = Regularizer(l1=self.beta_regularizer_l1,
                                             l2=self.beta_regularizer_l2)
        self._gamma_regularizer = Regularizer(l1=self.gamma_regularizer_l1,
                                              l2=self.gamma_regularizer_l2)
        if (self.convolution_filters is not None) or (self.convolution_kernel_size is not None):
            if len(self.convolution_filters) == len(self.convolution_kernel_size):
                if self.convolution_strides is None: self.convolution_strides = [[1] * len(k) for k in self.convolution_kernel_size]
                if self.convolution_dilation_rate is None: self.convolution_dilation_rate = [[1] * len(k) for k in self.convolution_kernel_size]
                if self.pooling_pool_size is None: self.pooling_pool_size = [None] * len(self.convolution_filters)
                if self.pooling_strides is None: self.pooling_strides = [None] * len(self.convolution_filters)
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
            if self.dropout_noise_shape is None: self.dropout_noise_shape = [None] * len(self.dense_units)
            for (du, dns) in zip(self.dense_units, self.dropout_noise_shape):
                z = self._connect(z, du, dropout_noise_shape=dns)
        return z
