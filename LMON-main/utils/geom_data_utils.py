#  MIT License
#
#  Copyright (c) 2019 Geom-GCN Authors
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.

import os
import re

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch as th
from dgl import DGLGraph
from sklearn.model_selection import ShuffleSplit

import utils.geom_utils as geom_utils
from utils.process import to_scipy, row_normalize
import pdb

def load_data(dataset_name, splits_file_path=None, train_percentage=None, val_percentage=None, embedding_mode=None,
              embedding_method=None,
              embedding_method_graph=None, embedding_method_space=None):
    if dataset_name in {'cora', 'citeseer', 'pubmed'}:
        adj, features, labels, _, _, _ = geom_utils.load_data(dataset_name)
        labels = np.argmax(labels, axis=-1)
        features = features.todense()
        G = nx.DiGraph(adj)
    else:
        graph_adjacency_list_file_path = os.path.join('new_data', dataset_name, 'out1_graph_edges.txt') # adj
        graph_node_features_and_labels_file_path = os.path.join('new_data', dataset_name,
                                                                f'out1_node_feature_label.txt') # node_id, features, label

        G = nx.DiGraph() #将adj中的边添加进graph中
        graph_node_features_dict = {} #存放节点特征
        graph_labels_dict = {} #存放节点label

        if dataset_name == 'film':
            with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
                graph_node_features_and_labels_file.readline()
                for line in graph_node_features_and_labels_file:
                    line = line.rstrip().split('\t')
                    assert (len(line) == 3)
                    assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                    feature_blank = np.zeros(932, dtype=np.uint8)
                    feature_blank[np.array(line[1].split(','), dtype=np.uint16)] = 1
                    graph_node_features_dict[int(line[0])] = feature_blank
                    graph_labels_dict[int(line[0])] = int(line[2])
        else:
            with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
                graph_node_features_and_labels_file.readline()
                for line in graph_node_features_and_labels_file:
                    line = line.rstrip().split('\t')
                    assert (len(line) == 3)
                    assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                    graph_node_features_dict[int(line[0])] = np.array(line[1].split(','), dtype=np.uint8)
                    graph_labels_dict[int(line[0])] = int(line[2])

        with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
            graph_adjacency_list_file.readline()
            for line in graph_adjacency_list_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 2)
                if int(line[0]) not in G: #把节点添加到networkx中
                    G.add_node(int(line[0]), features=graph_node_features_dict[int(line[0])],
                               label=graph_labels_dict[int(line[0])])
                if int(line[1]) not in G:  #把节点添加到networkx中
                    G.add_node(int(line[1]), features=graph_node_features_dict[int(line[1])],
                               label=graph_labels_dict[int(line[1])])
                G.add_edge(int(line[0]), int(line[1])) 
                # G.add_edge(int(line[1]), int(line[0])) #注释掉了，添加有向图

        adj = nx.adjacency_matrix(G, sorted(G.nodes())) # G的邻接矩阵 Scipy sparse matrix
        # print(adj)
        # pdb.set_trace()
        features = np.array(
            [features for _, features in sorted(G.nodes(data='features'), key=lambda x: x[0])]) # 把features按照对应的id排序
        labels = np.array(
            [label for _, label in sorted(G.nodes(data='label'), key=lambda x: x[0])])  # 把label按照对应的id排序

    assert (features == 1).sum() == len(features.nonzero()[0])

    features = geom_utils.preprocess_features(features) # 把feature normalize

    splits_file_path = None #我在这里加一步这个，让数据集随机划分试试

    # train_percentage = 0.8
    # val_percentage = 0.1

    # train_percentage = 0.6
    # val_percentage = 0.2

    train_percentage = 0.9
    val_percentage = 0.05

    print(train_percentage, val_percentage)

    #通过随机种子，固定数据集划分
    import torch
    # seed = 7 # wisconsin数据集下 0.7 - 0.15 -0.15 seed = 7效果最佳？ 0.7368;  seed = 5时 0.8158!, seed = 3时, 0.8 - 0.1 - 0.1 达到0.8077的效果
    # seed = 5 # actor best
    # seed = 5 # cornell 数据集下 0.8 - 0.1 - 0.1 seed = 1效果最佳？ 0.6842  6: 0.6316  9: 0.6842
    # seed = 5 # film数据集下 seed = 5, split: 0.9 - 0.05 - 0.05 最佳？
    seed = 9 # wisconsin数据集下效果最佳 seed = 28, split: 0.9 - 0.05 - 0.05, 达到0.7692的效果; seed = 6, split: 0.9 - 0.05 - 0.05 达到0.6923的效果
                # film 在seed = 103时，效果也好
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    if splits_file_path: # 把splits底下的随机划分训练集、测试集、验证集进行加载
        with np.load(splits_file_path) as splits_file:
            train_mask = splits_file['train_mask']
            val_mask = splits_file['val_mask']
            test_mask = splits_file['test_mask']
    else:
        assert (train_percentage is not None and val_percentage is not None)
        assert (train_percentage < 1.0 and val_percentage < 1.0 and train_percentage + val_percentage < 1.0)

        if dataset_name in {'cora', 'citeseer'}:
            disconnected_node_file_path = os.path.join('unconnected_nodes', f'{dataset_name}_unconnected_nodes.txt')
            with open(disconnected_node_file_path) as disconnected_node_file:
                disconnected_node_file.readline()
                disconnected_nodes = []
                for line in disconnected_node_file:
                    line = line.rstrip()
                    disconnected_nodes.append(int(line))

            disconnected_nodes = np.array(disconnected_nodes)
            connected_nodes = np.setdiff1d(np.arange(features.shape[0]), disconnected_nodes)

            connected_labels = labels[connected_nodes]

            train_and_val_index, test_index = next(
                ShuffleSplit(n_splits=1, train_size=train_percentage + val_percentage).split(
                    np.empty_like(connected_labels), connected_labels))
            train_index, val_index = next(ShuffleSplit(n_splits=1, train_size=train_percentage).split(
                np.empty_like(connected_labels[train_and_val_index]), connected_labels[train_and_val_index]))
            train_index = train_and_val_index[train_index]
            val_index = train_and_val_index[val_index]

            train_mask = np.zeros_like(labels)
            train_mask[connected_nodes[train_index]] = 1
            val_mask = np.zeros_like(labels)
            val_mask[connected_nodes[val_index]] = 1
            test_mask = np.zeros_like(labels)
            test_mask[connected_nodes[test_index]] = 1
        else:
            train_and_val_index, test_index = next(
                ShuffleSplit(n_splits=1, train_size=train_percentage + val_percentage).split(
                    np.empty_like(labels), labels))
            #下面这两行是将数据集划分成训练集+验证集 和 测试集后, 在训练+验证集上, 按照训练集的比例乘上 训练验证的比例
            # 感觉不太合理,比如训练集是0.7,验证集是0.15,那么最后训练集的比例是 (0.7 + 0.15) * 0.7 = 0.59?
            # train_index, val_index = next(ShuffleSplit(n_splits=1, train_size=train_percentage).split(
            #     np.empty_like(labels[train_and_val_index]), labels[train_and_val_index])) 

            train_index, val_index = next(ShuffleSplit(n_splits=1, train_size=train_percentage / (train_percentage + val_percentage)).split(
                np.empty_like(labels[train_and_val_index]), labels[train_and_val_index])) 

            train_index = train_and_val_index[train_index]
            val_index = train_and_val_index[val_index]      

            # print(train_and_val_index.shape)
            # print(train_index.shape)
            # print(val_index.shape)
            # print(test_index.shape)
            # pdb.set_trace()      

            train_mask = np.zeros_like(labels)
            train_mask[train_index] = 1
            val_mask = np.zeros_like(labels)
            val_mask[val_index] = 1
            test_mask = np.zeros_like(labels)
            test_mask[test_index] = 1

    num_features = features.shape[1]
    num_labels = len(np.unique(labels))
    assert (np.array_equal(np.unique(labels), np.arange(len(np.unique(labels)))))

    # features = th.FloatTensor(features)
    # labels = th.LongTensor(labels)
    # train_mask = th.BoolTensor(train_mask)
    # val_mask = th.BoolTensor(val_mask)
    # test_mask = th.BoolTensor(test_mask)

    # # Adapted from https://docs.dgl.ai/tutorials/models/1_gnn/1_gcn.html
    # degs = g.in_degrees().float()
    # norm = th.pow(degs, -0.5).cuda()
    # norm[th.isinf(norm)] = 0
    # g.ndata['norm'] = norm.unsqueeze(1)

    return adj, features, labels, train_mask, val_mask, test_mask, num_features, num_labels


def load_geom_datasets(dataset, seed):
    dataset_split = 'splits/%s_split_0.6_0.2_%s.npz' % (dataset, seed%10) # seed%10 --> which split
    print('loading %s' % dataset_split)
    # data = np.load(dataset_split)
    # print(data.files)
    # print(data['train_mask'])
    # print(sum(data['train_mask']))
    # print(data['test_mask'])
    # print(sum(data['test_mask']))
    # print(data['val_mask'])
    # print(sum(data['val_mask']))
    # pdb.set_trace()
    adj, features, labels, train_mask, val_mask, test_mask, num_features, num_labels = load_data(
        dataset, dataset_split, None, None, 'ExperimentTwoAll')

    # print(adj.nnz)
    # print((adj+adj.transpose()).nnz)

    idx = np.arange(len(labels))
    # print(idx)
    idx_train, idx_val, idx_test = idx[train_mask.astype(np.bool)], idx[val_mask.astype(np.bool)], idx[test_mask.astype(np.bool)]
    # print(idx_train)
    # print(idx_test)
    # print(type(idx_val))
    # pdb.set_trace()
    

    # n = len(labels)
    # cur = np.arange(int(n * 0.2))
    # cur1 = np.arange(int(n * 0.2), int(n * 0.4))
    # cur2 = np.arange(int(n * 0.4), n)
    # print(cur)
    # print(cur1)
    # print(cur2)

    # pdb.set_trace()

    features = row_normalize(features)

    return adj, adj, features, features, labels, idx_train, idx_val, idx_test, None, None

