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


import unittest
import typing as t

import numpy as np
import tensorflow as tf

from gcnn.layers import GraphConv


class TestGraphConv(unittest.TestCase):
    """Unittests for the GraphConv layer."""

    def test_build_sparse(self):
        """Test the build of the sparse layer."""
        for p in range(2, 20):
            A = self._get_A_matrix(p)
            layer = GraphConv(16, A, sparse_op=True)
            self.assertTrue(layer.sparse_op)

    def test_build_dense(self):
        """Test the build of the sparse layer."""
        for p in range(2, 20):
            A = self._get_A_matrix(p)
            layer = GraphConv(16, A, sparse_op=False)
            self.assertTrue(not layer.sparse_op)

    def test_build_sparse_invalid(self):
        """Test error for invalid sparse_op."""
        with self.assertRaises(ValueError):
            GraphConv(16, self._get_A_matrix(8), sparse_op='invalid')

    def test_build_invalid_A_nonsquare(self):
        """Test error for invalid nonsquare A."""
        with self.assertRaises(ValueError):
            GraphConv(16, np.ones((2,3)))

    def test_build_invalid_A_non2d(self):
        """Test error for invalid non-2D A."""
        with self.assertRaises(ValueError):
            GraphConv(16, np.ones((2)))
        with self.assertRaises(ValueError):
            GraphConv(16, np.ones((2,2,4)))
        with self.assertRaises(ValueError):
            GraphConv(16, np.ones((2,2,4,3)))
        with self.assertRaises(ValueError):
            GraphConv(16, np.ones((2,2,4,3,5)))

    def test_build_invalid_A_type(self):
        """Test error for invalid A type."""
        with self.assertRaises(ValueError):
            GraphConv(16, 'invalid')
        with self.assertRaises(ValueError):
            GraphConv(16, 1)
        with self.assertRaises(ValueError):
            GraphConv(16, {'a': 1, 'b': 2})
        with self.assertRaises(ValueError):
            GraphConv(16, None)

    def test_kernel_initializer(self):
        """Test that TF kernel initializer are set."""
        A = self._get_A_matrix(8)
        initializers = [
            'he_uniform',
            'he_normal',
            'glorot_uniform',
            'glorot_normal',
            'zeros',
            'Identity',
        ]
        for initializer in initializers:
            layer = GraphConv(16, A, kernel_initializer=initializer)
            self.assertEqual(layer.kernel_initializer, initializer)

    def test_invalid_kernel_initializer(self):
        """Test that invalid kernel initializers raise an error."""
        A = self._get_A_matrix(8)
        with self.assertRaises(ValueError):
            GraphConv(16, A, kernel_initializer='invalid')

    def test_dense_sparse_correctness(self):
        """Test the correctness of the dense and sparse operations."""
        LATENT_DIM = 8
        BATCH_SIZE = 4
        NUM_TRIES = 4
        MAX_NODES = 30
        for n in range(2, MAX_NODES):
            # Test random A matrices and data
            for use_bias in [True, False]:
                for _ in range(NUM_TRIES):
                    A = self._get_A_matrix(n)
                    x = np.random.rand(BATCH_SIZE, n, LATENT_DIM).astype(np.float32)

                    # Build and run dense layer
                    dense_layer = GraphConv(
                            LATENT_DIM,
                            A,
                            sparse_op=False,
                            use_bias=use_bias
                    )
                    dense_layer.build(x.shape)
                    y_dense = dense_layer(x)

                    # Build and run sparse layer with dense weights
                    # (run layer before transferring weights, since weights are
                    # initialized on build, and would overwrite the dense weights)
                    sparse_layer = GraphConv(
                            LATENT_DIM,
                            A,
                            sparse_op=True,
                            use_bias=use_bias
                    )
                    y_sparse = sparse_layer(x)
                    sparse_layer.set_weights(dense_layer.get_weights())
                    y_sparse = sparse_layer(x)

                    np.testing.assert_allclose(y_dense, y_sparse, rtol=1e-6, atol=5e-7)

    @staticmethod
    def _get_A_matrix(
            n: int,
            seed: t.Optional[int] = None
        ) -> np.ndarray:
        """Create valid nxn adjacency matrix A."""
        if seed is not None:
            np.random.seed(seed)
        A = np.random.randint(low=0, high=2, size=(n,n))
        # Make sure that the diagonal is zero
        np.fill_diagonal(A, 0)
        # Make sure that the matrix is symmetric
        return (A + A.T) // 2
