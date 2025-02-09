import pickle
import random
import sys

import networkx as nx
import numpy as np
import scipy
import scipy.sparse as sp
import torch
from dgl.data import DGLDataset

from data_loader import data_loader


class GraphDataset(DGLDataset):
    def __init__(self, graphs, labels, len_seq_detect=8):
        self.graphs = graphs
        self.labels = labels
        self.node_seqs = []
        for ii in range(len(graphs)):
            nodes = self.graphs[ii].num_nodes()
            # len_seq_detect = 8
            node_seq = torch.zeros(nodes, len_seq_detect).long()
            n = 0
            sampled_idx = []
            pre_sampled_idx = random.sample(range(nodes), int(nodes * 0.8))

            for x in pre_sampled_idx:
                cnt = 0
                scnt = 0
                node_seq[n, cnt] = x
                cnt += 1
                start = node_seq[n, scnt].item()
                while cnt < len_seq_detect:
                    sample_list = self.graphs[ii].successors(start).cpu().numpy().tolist()
                    nsampled = len(sample_list)
                    for i in range(nsampled):
                        if sample_list[i] not in node_seq[n]:
                            node_seq[n, cnt] = sample_list[i]
                            cnt += 1
                        if cnt == len_seq_detect:
                            break
                    scnt += 1
                    if scnt == len_seq_detect:
                        break
                    start = node_seq[n, scnt].item()
                if cnt == len_seq_detect:
                    def custom_similarity(list1, list2):
                        set1 = set(list1)
                        set2 = set(list2)
                        common_elements = set1 & set2
                        return len(common_elements) / max(len(set1), len(set2))

                    temp = node_seq[n]
                    flag = True
                    for i in range(len(sampled_idx)):
                        sim = custom_similarity(temp.tolist(), node_seq[sampled_idx[i]].tolist())
                        # print(temp.tolist(), end="\t")
                        # print(node_seq[sampled_idx[i]].tolist())
                        # print(f'similarity: {sim}')
                        if sim > 0.5:
                            flag = False
                            # print('similarity:')
                            # print(node_seq[sampled_idx[i]])
                            break
                    if flag:
                        sampled_idx.append(n)
                n += 1
            sampled_idx = random.sample(sampled_idx, int(len(sampled_idx) * 1))
            # print(sampled_idx)
            self.node_seqs.append(node_seq[sampled_idx])
            # print(f"len of node_seq: {len(node_seq[sampled_idx])}")

    def __repr__(self):
        return "CustomClass"

    def __getitem__(self, i):
        # 返回第 i 个图和对应的标签
        # print(node_seq.shape)
        # print(f'{i}: {self.graphs[i].num_nodes()}')
        return self.graphs[i], self.labels[i], self.node_seqs[i]

    def __len__(self):
        # 返回数据集中的图数量
        return len(self.graphs)


def load_data(pretrain, path):
    dl = data_loader(path)
    """
        dl.nodes(dict):
            total: total number of nodes
            count: a dict of int, number of nodes for each type
            attr: a dict of np.array (or None), attribute matrices for each type of nodes
            shift: node_id shift for each type. You can get the id range of a type by 
                        [ shift[node_type], shift[node_type]+count[node_type] )
        dl.links(dict):
            total: total number of links
            count: a dict of int, number of links for each type
            meta: a dict of tuple, explaining the link type is from what type of node to what type of node
            data: a dict of sparse matrices, each link type with one matrix. Shapes are all (nodes['total'], nodes['total'])
    """
    features = []
    for i in range(dl.nodes['total']):
        th = dl.nodes['attr'][i]
        if th is None:
            # features.append(sp.eye(dl.nodes['count'][i]))
            features.append(np.eye(dl.nodes['count'][i]))
        else:
            features.append(th)
    adjM = sum(dl.links['data'].values())
    '''
    adjM:
        (0, 1)	1.0
        (0, 3)	1.0
        (1, 2)	1.0
        (1, 6)	1.0
        (2, 5)	1.0
        ...
    self.g.nodes[node]['type_list'] = []
    self.g.nodes[node]['type_list'] = list(set(self.g.nodes[node]['type_list']))
    '''
    # return
    if pretrain:
        return features, adjM, dl
    else:
        labels = np.zeros((dl.nodes['count'][0], dl.labels_train['num_classes']), dtype=int)
        val_ratio = 0.2
        train_idx = np.nonzero(dl.labels_train['mask'])[0]##返回mask中值为True的索引
        np.random.shuffle(train_idx)
        split = int(train_idx.shape[0]*val_ratio)
        val_idx = train_idx[:split]
        train_idx = train_idx[split:]
        train_idx = np.sort(train_idx)
        val_idx = np.sort(val_idx)
        test_idx = np.nonzero(dl.labels_test['mask'])[0]
        labels[train_idx] = dl.labels_train['data'][train_idx]
        labels[val_idx] = dl.labels_train['data'][val_idx]
        # if prefix != 'IMDB' and prefix != 'IMDB-HGB':
        #     labels = labels.argmax(axis=1)
        train_val_test_idx = {}
        train_val_test_idx['train_idx'] = train_idx
        train_val_test_idx['val_idx'] = val_idx
        train_val_test_idx['test_idx'] = test_idx
        return features,\
               adjM, \
               labels,\
               train_val_test_idx,\
                dl


# def load_data_positive(pretrain, path):