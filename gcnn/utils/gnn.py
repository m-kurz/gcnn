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
import tensorflow as tf
import tensorflow_gnn as tfgnn
import matplotlib.pyplot as plt

from tensorflow_gnn.models.gcn import gcn_conv

import matplotlib
#matplotlib.use('Qt5Agg')


def get_edge_connectivity_of_grid(p):
    '''
    Computes edge connectivity for a three-dimensional grid with pxpxp nodes
    '''

    # Number of edges within a grid:
    # n_e = n_dim*(p-1)*p^(n_dim-1)
    n_dim = 3 # hard-coded currently.... #len(data.shape)-1
    n_e   = n_dim*(p-1)*np.power(p,n_dim-1)
    # Since we use directed edges, we have to use two edges for each node connection
    n_e = 2*n_e

    target_nodes=np.zeros(n_e,dtype=np.int32)
    source_nodes=np.zeros(n_e,dtype=np.int32)

    n = 0
    for i in range(p):
        for j in range(p):
            for k in range(p):
                global_idx = i+j*(p)+k*np.power(p,2)
                if i>0:
                    # i-1
                    local_idx = (i-1)+j*(p)+k*np.power(p,2)
                    source_nodes[n] = global_idx
                    target_nodes[n] =  local_idx
                    n=n+1
                if i<(p-1):
                    # i+1
                    local_idx = (i+1)+j*(p)+k*np.power(p,2)
                    source_nodes[n] = global_idx
                    target_nodes[n] =  local_idx
                    n=n+1
                if j>0:
                    # j-1
                    local_idx = i+(j-1)*(p)+k*np.power(p,2)
                    source_nodes[n] = global_idx
                    target_nodes[n] =  local_idx
                    n=n+1
                if j<(p-1):
                    # j+1
                    local_idx = i+(j+1)*(p)+k*np.power(p,2)
                    source_nodes[n] = global_idx
                    target_nodes[n] =  local_idx
                    n=n+1
                if k>0:
                    # k-1
                    local_idx = i+j*(p)+(k-1)*np.power(p,2)
                    source_nodes[n] = global_idx
                    target_nodes[n] =  local_idx
                    n=n+1
                if k<(p-1):
                    # k+1
                    local_idx = i+j*(p)+(k+1)*np.power(p,2)
                    source_nodes[n] = global_idx
                    target_nodes[n] =  local_idx
                    n=n+1

    return source_nodes, target_nodes


def visualize_graph(source_nodes,target_nodes,N):
    '''
    Visualizes the graph by plotting the edges between source_nodes and target_nodes
    '''
    n_edges = len(source_nodes)
    coordinates = []

    # Create list mapping node number to (i,j,k) tuple
    for k in range(N+1):
        for j in range(N+1):
            for i in range(N+1):
                coordinates.append((i,j,k))

    # Extract x, y and z-coordinates separately from list
    x_coords, y_coords, z_coords = zip(*coordinates)

    # Create the plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot Nodes
    ax.scatter(x_coords, y_coords, z_coords, color='blue', marker='o')

    # Plot Edges
    for i in range(n_edges):
        x = [coordinates[source_nodes[i]][0],coordinates[target_nodes[i]][0]]
        y = [coordinates[source_nodes[i]][1],coordinates[target_nodes[i]][1]]
        z = [coordinates[source_nodes[i]][2],coordinates[target_nodes[i]][2]]
        plt.plot( x,y,z, color='red')

    # Add grid, title, and labels
    ax.grid(True)
    plt.title('Element Graph')
    ax.set_xlabel('Xi')
    ax.set_ylabel('Eta')
    ax.set_zlabel('Zeta')

    # Set limits for better visualization
    ax.set_xlim(min(x_coords) - 1, max(x_coords) + 1)
    ax.set_ylim(min(y_coords) - 1, max(y_coords) + 1)
    ax.set_zlim(min(z_coords) - 1, max(z_coords) + 1)

    # Display the plot
    plt.show()
    return


def model_fn(graph_tensor_spec: tfgnn.GraphTensorSpec):
    def set_initial_node_states():

    graph = inputs = tf.keras.layers.Input(type_spec=graph_tensor_spec)
    graph = tfgnn.keras.layers.MapFeatures(node_sets_fn=set_initial_node_states)(graph)
    for i in range(num_graph_updates):
        graph = tfgnn.models.gcn(
            units=node_state_dim,
        )(graph)
    return tf.keras.Model(inputs, graph)

def callback():
    # A simplistic way to map node features to an initial state.
    def node_sets_fn(node_set, *, node_set_name):
      state_dims_by_node_set = {"author": 32, "paper": 64}  # ...and so on.
      state_dim = state_dims_by_node_set[node_set_name]
      features = node_set.features  # Immutable view.
      if features: # Concatenate and project all inputs (assumes they are floats).
        return tf.keras.layers.Dense(state_dim)(
            tf.keras.layers.Concatenate([v for _, v in sorted(features.items())]))
      else:  # There are no inputs, create an empty state.
        return tfgnn.keras.layers.MakeEmptyFeature()(node_set)
    graph = tfgnn.keras.layers.MapFeatures(node_sets_fn=node_sets_fn)(graph)
    
    # Doubles all feature values, with one callback used for all graph pieces,
    # including auxiliary ones.
    def fn(inputs, **unused_kwargs):
      return {k: tf.add(v, v) for k, v in inputs.features.items()}
    graph = tfgnn.keras.layers.MapFeatures(
        context_fn=fn, node_sets_fn=fn, edge_sets_fn=fn,
        allowed_aux_node_sets_pattern=r".*", allowed_aux_edge_sets_pattern=r".*"
    )(graph)


if __name__=="__main__":

    N=3
    nVar=5
    nVarEdge=1
    
    data=np.random.rand(N+1,N+1,N+1,nVar)
    n_gp=np.power(N+1,3)

    source_nodes, target_nodes = get_edge_connectivity_of_grid(N+1)

    visualize_graph(source_nodes,target_nodes,N)

    graph = tfgnn.GraphTensor.from_pieces(
          node_sets = {
              'gauss_points': tfgnn.NodeSet.from_fields(
                  sizes=tf.constant([n_gp]),
                  features={
                    "data": tf.constant(data.reshape(n_gp,nVar))
                      })},
          edge_sets = {
              'edge': tfgnn.EdgeSet.from_fields(
                  sizes=tf.constant([len(source_nodes)]),
                  features={},
                  adjacency=tfgnn.Adjacency.from_indices(
                      source=('gauss_points', source_nodes),
                      target=('gauss_points', target_nodes)))})

    print(graph.edge_sets)
    print(graph.edge_sets['edge'])
    gcnconv = gcn_conv.GCNConv(8)
    gcnconv(graph, edge_set_name='edge')   # Has shape=(4, 3).
      graph = tfgnn.keras.layers.MapFeatures(
      node_sets_fn=set_initial_node_states)(graph)
    print(graph)
