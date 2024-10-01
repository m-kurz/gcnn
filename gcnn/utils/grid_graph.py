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


import numpy as np
import matplotlib.pyplot as plt


class GridGraph():
    """Class to bunch functionality for a multi-dimensional grid graph.

    This class provides static methods to create the adjacency matrix, edge
    list, and visualize the grid graph based on the input dimensions. The class
    is not designed to be actually instantiated, but rather to provide a set of
    utility functions for grid graphs.

    Methods:
        get_num_nodes: Get the total number of nodes in the grid graph.
        get_adjacency_matrix: Create an adjacency matrix for a multi-dimensional
            grid based on the input dimensions.
        get_num_edges: Get the total number of edges in the grid graph.
        get_edge_list: Get the list of (directed) edges of the grid graph.
        visualize_graph: Visualize the grid graph.

    Attributes:
        dims: List of integers representing the number of nodes in each dimension.
        adj_matrix: Adjacency matrix of grid with dim. (n_nodes, n_nodes)
        edge_list: List of (directed) edges of the grid graph.
        num_nodes: Total number of nodes in the grid graph.
        num_edges: Total number of edges in the grid graph.
    """

    def __init__(self, dims: np.ndarray):
        """Initialize the grid graph.

        Args:
            dims (np.ndarray): List or tuple of integers representing the
                number of nodes in each dimension.
        """
        self.dims = dims

        self.adj_matrix = self.get_adjacency_matrix(self.dims)
        self.edge_list = self.get_edge_list(self.dims)
        self.n_nodes = self.get_num_nodes(self.dims)
        self.n_edges = self.get_num_edges(self.dims)

    @staticmethod
    def get_num_nodes(dims: np.ndarray) -> int:
        """Get the total number of nodes in the grid graph.

        Args:
            dims (np.ndarray): List or tuple of integers representing the
                number of nodes in each dimension.

        Returns:
            (int): Total number of nodes in the grid graph.
        """
        return np.prod(dims)

    @staticmethod
    def get_adjacency_matrix(dims: np.ndarray) -> np.ndarray:
        """Create an adjacency matrix for a multi-dimensional grid based on the input dimensions.

        Args:
            dims (np.ndarray): List or tuple of integers representing the
                number of nodes in each dimension.

        Returns:
            (np.ndarray): Adjacency matrix of grid with dim. (n_nodes, n_nodes)
        """
        total_nodes = GridGraph.get_num_nodes(dims)
        adj_matrix = np.zeros((total_nodes, total_nodes), dtype=int)

        # Create edges for the adjacency matrix
        for index in np.ndindex(*dims):
            current_node = GridGraph._multi_index_to_linear_index(index, dims)

            # Check neighbors in each dimension
            for dim in range(len(dims)):
                # Check the positive neighbor
                if index[dim] + 1 < dims[dim]:
                    neighbor_index = list(index)
                    neighbor_index[dim] += 1
                    neighbor_node = GridGraph._multi_index_to_linear_index(
                            tuple(neighbor_index),
                            dims
                    )
                    adj_matrix[current_node][neighbor_node] = 1
                    adj_matrix[neighbor_node][current_node] = 1  # Undirected graph

                # Check the negative neighbor
                if index[dim] - 1 >= 0:
                    neighbor_index = list(index)
                    neighbor_index[dim] -= 1
                    neighbor_node = GridGraph._multi_index_to_linear_index(
                            tuple(neighbor_index),
                            dims
                    )
                    adj_matrix[current_node][neighbor_node] = 1
                    adj_matrix[neighbor_node][current_node] = 1  # Undirected graph

        return adj_matrix

    @staticmethod
    def get_num_edges(dims: np.ndarray) -> int:
        """Get the total number of edges in the grid graph.

        Args:
            dims (np.ndarray): List or tuple of integers representing the
                number of nodes in each dimension.

        Returns:
            (int): Total number of edges in the grid graph.
        """
        # Alternative implementation without requiring the adjacency matrix
        #n_dims = len(dims)
        #total_edges = 0
        #for i in range(n_dims):
        #    edges_in_dim = dims[i] - 1
        #    # Calculate the product of dims excluding the current one
        #    product_of_other_dims = 1
        #    for j in range(n_dims):
        #        if j != i:
        #            product_of_other_dims *= dims[j]
        #    total_edges += edges_in_dim * product_of_other_dims
        #return total_edges
        A = GridGraph.get_adjacency_matrix(dims)
        return np.sum(A)

    @staticmethod
    def get_edge_list(dims: np.ndarray) -> np.ndarray:
        """Get the list of (directed) edges of the grid graph.

        Args:
            dims (np.ndarray): List or tuple of integers representing the
                number of nodes in each dimension.

        Returns:
            A tuple of two numpy arrays representing the source and target nodes of the edges.
        """
        num_edges = GridGraph.get_num_edges(dims)
        edge_list = np.zeros((num_edges, 2), dtype=int)

        A = GridGraph.get_adjacency_matrix(dims)
        num_nodes = GridGraph.get_num_nodes(dims)

        k = 0
        for i in range(num_nodes):
            for j in range(num_nodes):
                if A[i][j] == 1:
                    edge_list[k, 0] = i
                    edge_list[k, 1] = j
                    k += 1
        return edge_list

    @staticmethod
    def visualize_graph(dims: np.ndarray):
        """Visualize the grid graph.

        Args:
            dims (np.ndarray): List or tuple of integers representing the
                number of nodes in each dimension.

        Raises:
            ValueError: If the visualization is not supported for the given
                number of dimensions.
        """
        if not isinstance(dims, np.ndarray):
            dims = np.array(dims, dtype=np.int32)

        if dims.ndim != 1:
            dims = dims.flatten()

        if dims.size not in [1, 2, 3]:
            raise ValueError(
                "Visualization only supported for 1D, 2D, or 3D grids.")

        if dims.size == 1:
            dims = np.append(dims, (1, 1))
        elif dims.size == 2:
            dims = np.append(dims, (1,))

        edge_list = GridGraph.get_edge_list(dims)
        adj_matrix = GridGraph.get_adjacency_matrix(dims)

        coordinates = []
        for index in np.ndindex(*dims):
            coordinates.append(index)

        # Extract x, y and z-coordinates separately from list
        x_coords, y_coords, z_coords = zip(*coordinates)

        # Create the plot
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        # Plot Nodes
        ax.scatter(x_coords, y_coords, z_coords, color='blue', marker='o')

        # Plot Edges
        for i in range(len(edge_list)):
            x = [coordinates[edge_list[i,0]][0],coordinates[edge_list[i,1]][0]]
            y = [coordinates[edge_list[i,0]][1],coordinates[edge_list[i,1]][1]]
            z = [coordinates[edge_list[i,0]][2],coordinates[edge_list[i,1]][2]]
            plt.plot(x, y, z, color='red')

        # Add grid, title, and labels
        ax.grid(True)
        plt.title('Grid Graph')
        ax.set_xlabel('Xi')
        ax.set_ylabel('Eta')
        ax.set_zlabel('Zeta')

        # Set limits for better visualization
        ax.set_xlim(min(x_coords) - 1, max(x_coords) + 1)
        ax.set_ylim(min(y_coords) - 1, max(y_coords) + 1)
        ax.set_zlim(min(z_coords) - 1, max(z_coords) + 1)

        # Display the plot
        plt.show()

    @staticmethod
    def _multi_index_to_linear_index(indices: np.ndarray, dims: np.ndarray):
        """Create a mapping from multi-dimensional indices to linear index.

        Args:
            indices: Tuple of integers representing the multi-dimensional
                indices.

        Returns:
            (int): Linear index corresponding to the multi-dimensional indices.
        """
        linear_index = 0
        multiplier = 1
        for i in reversed(range(len(dims))):
            linear_index += indices[i] * multiplier
            multiplier *= dims[i]
        return linear_index

    @staticmethod
    def _linear_index_to_multi_index(index: int, dims: np.ndarray) -> np.ndarray:
        """Create a mapping from linear index to multi-dimensional indices.

        Args:
            index (int): Integer representing the linear index.
            dims (np.ndarray): List or tuple of integers representing the number of nodes in

        Returns:
            (np.ndarray): Multi-dimensional indices corresponding to the linear index.
        """
        multi_index = np.zeros(len(dims), dtype=int)
        for i in reversed(range(len(dims))):
            multi_index[i] = index % dims[i]
            index //= dims[i]
        return multi_index
