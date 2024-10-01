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
import unittest.mock
import matplotlib.pyplot as plt

import numpy as np

from gcnn.utils import GridGraph


class TestGridGraph(unittest.TestCase):
    """Unittests for the GridGraph class."""

    def test_grid_graph_1d(self):
        """Test the creation of a grid graph."""
        for p in range(2, 20):
            dims = np.array((p,), dtype=np.int32)
            graph = GridGraph(dims)
            n_nodes = p
            self.assertEqual(graph.n_nodes, n_nodes)
            self.assertEqual(graph.n_edges, 2*(n_nodes - 1))
            np.testing.assert_array_equal(graph.dims, dims)
            np.testing.assert_array_equal(graph.adj_matrix.shape, (p, p))

    def test_grid_graph_2d(self):
        """Test the creation of a grid graph."""
        for p in range(2, 15):
            for q in range(2, 15):
                dims = np.array((p, q), dtype=np.int32)
                graph = GridGraph(dims)
                n_nodes = p * q
                self.assertEqual(graph.n_nodes, n_nodes)
                self.assertEqual(graph.n_edges, 2*(2*n_nodes - p - q))
                np.testing.assert_array_equal(graph.dims, dims)
                np.testing.assert_array_equal(graph.adj_matrix.shape, (n_nodes, n_nodes))

    def test_grid_graph_3d(self):
        """Test the creation of a grid graph."""
        for p in range(2, 8):
            for q in range(2, 8):
                for r in range(2, 8):
                    dims = np.array((p, q, r), dtype=np.int32)
                    graph = GridGraph(dims)
                    n_nodes = p * q * r
                    self.assertEqual(graph.n_nodes, n_nodes)
                    self.assertEqual(graph.n_edges, 2*(3*n_nodes - p*q - q*r - p*r))
                    np.testing.assert_array_equal(graph.dims, dims)
                    np.testing.assert_array_equal(graph.adj_matrix.shape, (n_nodes, n_nodes))

    @unittest.mock.patch('matplotlib.pyplot.show')
    def test_grid_graph_plot(self, mock_show):
        """Mocked test to check if the plot function is called."""
        dims_1d = np.array((   3,  ), dtype=np.int32)
        dims_2d = np.array((   3, 4), dtype=np.int32)
        dims_3d = np.array((2, 3, 4), dtype=np.int32)
        dims_3d_cast =     (2, 3, 4)
        for dims in [dims_1d, dims_2d, dims_3d, dims_3d_cast]:
            GridGraph.visualize_graph(dims)
            # Check that plot was called and figure exists
            assert(plt.fignum_exists(1))
            plt.close(1)
            mock_show.assert_called_once()
            mock_show.reset_mock()

    def test_grid_graph_plot_invalid_dim(self):
        """Test error for invalid dimensions."""
        with self.assertRaises(ValueError):
            dims = np.array((2,3,4,5), dtype=np.int32)
            GridGraph.visualize_graph(dims)
