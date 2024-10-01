#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: MIT
#
# Copyright (c) 2024, Marius Kurz
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import tensorflow as tf


class GraphReadout(tf.keras.layers.Layer):  # pylint: disable=no-member
    '''Implements a graph readout layer using TensorFlow.

    Implements a graph readout layer, also called (flat) graph pooling layer,
    that reduces the dimensionality of the graph representation to a single
    vector. The input is expected to follow loosely the form
    `[batchsize, nodes, features]`. However, the layer is agnostic to the
    specific dimensions and just performs the reduction along the dimension
    `reduction_dim`.

    Args:
        reduction_op (str): Type of reduction operation. Currently supported
            are {'mean', 'max', 'min'}.
        reduction_dim (int): Axis along which reduction should be performed.
            Should correspond to the dimension in which the indididual graph
            nodes are listed.
    '''

    def __init__(self, reduction_op='mean', reduction_dim=-2):
        super().__init__()
        self.reduction_op = reduction_op
        self._reduction_dim = reduction_dim

    @property
    def reduction_op(self) -> str:
        """Reduction operation. Either 'mean', 'max' or 'min'."""
        return self._reduction_op

    @reduction_op.setter
    def reduction_op(self, reduction_op: str):
        """Setter for the reduction operation."""
        if not reduction_op in {'mean', 'max', 'min'}:
            raise ValueError('Invalid reduction type. Only `mean`, `max` and `min` supported!')
        self._reduction_op = reduction_op

    def call(self, x: tf.Tensor):
        '''Performes reduction operation.

        Args:
            x (tf.Tensor): Performs type of reduction specified in
                initialization and along specified dimension.

        Returns:
            Reduced TF tensor missing the `reduction_dim` dimension.

        Raises:
            NotImplementedError: If the reduction operation is not supported.
        '''
        if self.reduction_op == 'mean':
            return tf.math.reduce_mean(x, axis=self._reduction_dim)
        if self.reduction_op == 'max':
            return tf.math.reduce_max(x, axis=self._reduction_dim)
        if self.reduction_op == 'min':
            return tf.math.reduce_min(x, axis=self._reduction_dim)
        raise ValueError('Invalid reduction type. Currently, only `mean`, `max` and `min` are supported!')
