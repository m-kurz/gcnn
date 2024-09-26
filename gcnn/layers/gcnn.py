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


import inspect

import numpy as np
import tensorflow as tf


class GraphConv(tf.keras.layers.Layer):
    '''Implements a Graph-Convolutional layer.

    Implements a single Graph-Convolutional layer following:

        Kipf, Thomas N., and Max Welling. "Semi-supervised classification with
        graph convolutional networks." arXiv preprint arXiv:1609.02907 (2016).

    Args:
        num_outputs (int): Integer resulting in the number of output variables
            for each node.
        A (np.ndarray): Adjacency matrix. Requires to be a square matrix,
            i.e. a two-dimensional array of dimension (N, N).
        sparse_op (bool): Indicates whether multiplying A is computed as sparse
            operation. Provides better performance and less memory footprint
            for large graphs (>1000 nodes). Might be slower for small graphs.
        kernel_initializer (str): Initializer for the trainable weight matrix.
        use_bias (bool): Indicates whether a (trainable) bias should be added
            to the output.
    '''
    def __init__(self, num_outputs, A, sparse_op=True, kernel_initializer='he_uniform', use_bias=False):
        super().__init__()
        self.num_outputs = num_outputs
        self.sparse_op = sparse_op
        self.kernel_initializer = kernel_initializer
        self.use_bias = use_bias
        # A_hat: Precompute geometry information
        self._A_hat = self._compute_A_hat(A, self.sparse_op)

    @property
    def sparse_op(self) -> bool:
        """Indicates whether multiplying A is computed as sparse operation."""
        return self._sparse_op

    @sparse_op.setter
    def sparse_op(self, sparse_op: bool):
        """Setter for the sparse operation."""
        if not isinstance(sparse_op, bool):
            raise AttributeError('Expected a boolean value for `sparse_op`!')
        self._sparse_op = sparse_op

    @property
    def kernel_initializer(self) -> str:
        """Initializer for the trainable weight matrix."""
        return self._kernel_initializer

    @kernel_initializer.setter
    def kernel_initializer(self, kernel_initializer: str):
        """Set requested initializer if it is implemented in TensorFlow."""
        initializer_list = []
        for name, obj in inspect.getmembers(tf.keras.initializers):
            if inspect.isclass(obj) and issubclass(obj, tf.keras.initializers.Initializer):
                initializer_list.append(name)
        if kernel_initializer not in initializer_list:
            raise AttributeError(f'Invalid initializer {kernel_initializer}. Available are: {initializer_list}')
        self._kernel_initializer = kernel_initializer

    def build(self, input_shape):
        '''Build layer by adding trainable weight matrix in correct size.'''
        # W: Add trainable weight_matrix
        self._kernel = self.add_weight(
                                      "kernel",
                                      shape=[int(input_shape[-1]), self.num_outputs],
                                      initializer=self.kernel_initializer,
                                      trainable=True
                                      )
        if self.use_bias:
            self._b = self.add_weight(shape=(self.num_outputs,), initializer="zeros", trainable=True)

    def call(self, x):
        '''Evaluates layer for input `x` following Eq.(8).'''
        if self.sparse_op:
            # Sparse operation has to be applied manually to each element along
            # the batch dimension, since batch dim not supported by TF.
            def sparse_matmul_A_hat(x):
                return  tf.sparse.sparse_dense_matmul(self._A_hat, x)
            if self.use_bias:
                return tf.vectorized_map(sparse_matmul_A_hat, x, warn=False) @ self._kernel + self._b
            return tf.vectorized_map(sparse_matmul_A_hat, x, warn=False) @ self._kernel
        # Otherwise, just return regular matmul
        if self.use_bias:
            return self._A_hat @ x @ self._kernel + self._b
        return self._A_hat @ x @ self._kernel

    @staticmethod
    def _compute_A_hat(A, sparse):
        '''Computes A_hat matrix from Eq.(9) relying only on graph topology.

        Precomputes the necessary matrices for computing the state update that
        rely only on the graph layout, i.e. the Adjacency Matrix A.

        Args:
            A (np.ndarray): Adjacency matrix. Requires to be a square matrix,
                i.e. a two-dimensional array of dimension (N, N).
            sparse (bool): Indicates whether the output should be a sparse
                tensor.

        Raises:
            AttributeError: if `A` is either not a numpy array, not
                of rank 2 or is not square.

        Returns:
            A_hat from Eq.(9) as a sparse tensor. A_hat comprises all
                required layout-specific information of a graph.
        '''
        # Check if the input is a NumPy array
        if not isinstance(A, np.ndarray):
            try:
                A = np.array(A)
            except Exception as e:
                raise AttributeError('Expected a numpy array!') from e
        # Check if the array has two dimensions
        if A.ndim != 2:
            raise AttributeError('Input array has more than two dimensions!')
        # Check if the matrix is square (number of rows equals number of columns)
        if A.shape[0] != A.shape[1]:
            raise AttributeError('Input array is not square!')

        # A_tilde: adding self-loops
        A_tilde = A + np.eye(A.shape[0], dtype=A.dtype)

        # D_tilde: get degree matrix already raise to exponent `-0.5`
        D_tilde = np.diag(np.power(np.sum(A_tilde,axis=1), -0.5))

        # A_hat: matmul precomputed matrices as in Eq. (9)
        A_hat = tf.convert_to_tensor(D_tilde @ A_tilde @ D_tilde)

        if sparse:
            return tf.sparse.from_dense(A_hat)
        return A_hat
