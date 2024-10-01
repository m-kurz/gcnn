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

from gcnn.layers import GraphReadout


class TestGraphReadout(unittest.TestCase):
    """Unit tests for the GraphReadout layer."""

    def test_build_reduction_op(self):
        """Test the reduction operation."""
        for reduction_op in ['mean', 'max', 'min']:
            layer = GraphReadout(reduction_op=reduction_op)
            self.assertEqual(layer.reduction_op, reduction_op)

    def test_build_reduction_op_invalid(self):
        """Test the reduction operation with invalid values."""
        with self.assertRaises(ValueError):
            GraphReadout(reduction_op='invalid')

    def test_reduction_op(self):
        """Test the reduction operation."""
        NUM_TRIES = 100
        for reduction_op in ['mean', 'max', 'min']:
            for reduction_dim in [-1, -2, -3]:
                layer = GraphReadout(reduction_op=reduction_op, reduction_dim=reduction_dim)
                for _ in range(NUM_TRIES):
                    x = np.random.rand(1, 4, 8, 16)
                    y = layer(x)
                    if reduction_op == 'mean':
                        ref = np.mean(x, axis=reduction_dim)
                    if reduction_op == 'max':
                        ref = np.max(x, axis=reduction_dim)
                    if reduction_op == 'min':
                        ref = np.min(x, axis=reduction_dim)
                    np.testing.assert_allclose(y, ref, rtol=1e-6, atol=1e-7)

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
