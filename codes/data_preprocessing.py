import numpy as np
import argparse
from collections import Counter
import networkx as nx
from scipy.sparse import coo_matrix
import pickle

def load_data(args):
    data_folder_path = '../data/'+args.dataset

    adjs_path = data_folder_path+'/'+args.dataset+'_A.csv'
    graph_indicators_path = data_folder_path+'/'+args.dataset+'_graph_indicator.csv'
    graph_labels_path = data_folder_path+'/'+args.dataset+'_graph_labels.csv'
    node_attributes_path = data_folder_path+'/'+args.dataset+'_node_attributes.csv'

    adjs = np.loadtxt(adjs_path, dtype=int, delimiter=',')
    graph_indicators = np.loadtxt(graph_indicators_path, dtype=int, delimiter=',')
    graph_labels = np.loadtxt(graph_labels_path, dtype=int, delimiter=',')
    node_attributes = np.loadtxt(node_attributes_path, delimiter=',')

    return adjs, graph_indicators, graph_labels, node_attributes

def graph_labels_process(args, graph_labels):
    processed_labels = [[0, 0] for i in range(len(graph_labels))]
    for i in range(len(graph_labels)):
        if args.dataset == 'PROTEINS_full':
            if graph_labels[i] == 1:
                processed_labels[i] = [1, 0]
            else:
                processed_labels[i] = [0, 1]
        elif args.dataset == 'FRANKENSTEIN':
            if graph_labels[i] == 1:
                processed_labels[i] = [0, 1]
            else:
                processed_labels[i] = [1, 0]
        elif args.dataset == 'AIDS':
            if graph_labels[i] == 1:
                processed_labels[i] = [0, 1]
            else:
                processed_labels[i] = [1, 0]
    return processed_labels

def node_attributes_process(args, node_attributes):
    if args.dataset == 'AIDS':
        chems = node_attributes[:, 0]
        charges = node_attributes[:, 1]
        x_y = node_attributes[:, 2:]

        processed_chems = [[0 for j in range(38)] for i in range(len(node_attributes))]
        processed_charges = [[0, 0, 0] for i in range(len(node_attributes))]
        for i in range(len(node_attributes)):
            if chems[i] >= 38:
                continue
            else:
                processed_chems[i][int(chems[i])] = 1
            if charges[i] == -1:
                processed_charges[i][0] = 1
            elif charges[i] == 0:
                processed_charges[i][1] = 1
            elif charges[i] == 1:
                processed_charges[i][2] = 1
        
        processed_node_attrs = np.concatenate((processed_chems, processed_charges, x_y), axis=1)
        return processed_node_attrs
    else:
        return node_attributes

def extraction(adjs, graph_indicators, graph_labels, node_attributes):
    ix = adjs[:, 0]-1
    iy = adjs[:, 1]-1
    values = [1 for i in range(len(adjs))]
    adjs_sparse = coo_matrix((values, (ix, iy)), shape=(len(node_attributes), len(node_attributes)))
    adjs_dense = adjs_sparse.todense()

    nodes_num_list = []
    cnt = Counter(graph_indicators)
    for i in range(1, len(graph_labels)+1):
        nodes_num_list.append(cnt[i])

    adjs_coordinate_offset = 0
    graphs_list = []
    for i in range(len(graph_labels)):
        ith_adj = adjs_dense[adjs_coordinate_offset: adjs_coordinate_offset+nodes_num_list[i], adjs_coordinate_offset: adjs_coordinate_offset+nodes_num_list[i]]
        ith_adj_sparse = coo_matrix(ith_adj)
        ith_adj_sparse_row = np.reshape(ith_adj_sparse.row, (-1, 1))
        ith_adj_sparse_col = np.reshape(ith_adj_sparse.col, (-1, 1))
        ith_adj_coo = np.concatenate((ith_adj_sparse_row, ith_adj_sparse_col), axis=1)

        ith_nodes_attrs = node_attributes[adjs_coordinate_offset: adjs_coordinate_offset+nodes_num_list[i]]

        G = nx.Graph(label=graph_labels[i])
        for j in range(nodes_num_list[i]):
            G.add_node(j, attrs=ith_nodes_attrs[j])
        G.add_edges_from(ith_adj_coo)
        graphs_list.append(G)

        adjs_coordinate_offset += nodes_num_list[i]

    return graphs_list

def save_graphs(args, graphs_list):
    path = '../data/'+args.dataset+'/nx_graphs.pkl'
    with open(path, 'wb') as f:
        pickle.dump(graphs_list, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='PROTEINS_full')

    args = parser.parse_args()

    adjs, graph_indicators, graph_labels, node_attributes = load_data(args)
    # processed_labels = graph_labels_process(args, graph_labels)
    processed_node_attrs = node_attributes_process(args, node_attributes)
    graphs_list = extraction(adjs, graph_indicators, graph_labels, processed_node_attrs)
    
    save_graphs(args, graphs_list)
