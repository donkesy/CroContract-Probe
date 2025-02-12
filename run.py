import argparse
import gc
import os
import random
import shutil
import sys
import time

import dgl
import numpy as np
import torch
from sklearn import metrics
from datetime import datetime
from collections import Counter
from sklearn.model_selection import train_test_split
from torchmetrics import Accuracy, Precision, Recall, F1Score
import torchmetrics
from tqdm import tqdm

from model import HINormer
import torch.nn.functional as F
from data_loader import data_loader
from data import load_data, GraphDataset
from dgl.dataloading import GraphDataLoader
import scipy.sparse as sp


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def sp_to_spt(mat):
    coo = mat.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))


def mat2tensor(mat):
    if type(mat) is np.ndarray:
        return torch.from_numpy(mat).type(torch.FloatTensor)
    return sp_to_spt(mat)


ap = argparse.ArgumentParser(
        description='HINormer')
ap.add_argument('--feats-type', type=int, default=0,
                help='Type of the node features used. ' +
                     '0 - loaded features; ' +
                     '1 - only target node features (zero vec for others); ' +
                     '2 - only target node features (id vec for others); ' +
                     '3 - all id vec. Default is 2' +
                '4 - only term features (id vec for others);' +
                '5 - only term features (zero vec for others).')
ap.add_argument('--device', type=int, default=0)
ap.add_argument('--hidden-dim', type=int, default=2048,
                help='Dimension of the node hidden state. Default is 32.')
ap.add_argument('--dataset', type=str, default='DBLP', help='DBLP, IMDB, Freebase, AMiner, DBLP-HGB, IMDB-HGB')
ap.add_argument('--num-heads', type=int, default=8,
                help='Number of the attention heads. Default is 2.')
ap.add_argument('--epoch', type=int, default=1000, help='Number of epochs.')
ap.add_argument('--batch-size', type=int, default=1, help='Number of batch')
ap.add_argument('--patience', type=int, default=50, help='Patience.')
ap.add_argument('--repeat', type=int, default=5, help='Repeat the training and testing for N times. Default is 1.')
ap.add_argument('--num-layers', type=int, default=3, help='The number of layers of HINormer layer')
ap.add_argument('--num-gnns', type=int, default=2, help='The number of layers of both structural and heterogeneous encoder')
ap.add_argument('--lr', type=float, default=2e-4)
ap.add_argument('--seed', type=int, default=2024103060)
ap.add_argument('--dropout', type=float, default=0.3)
ap.add_argument('--weight-decay', type=float, default=0)
ap.add_argument('--len-seq', type=int, default=8, help='The length of node sequence.')
ap.add_argument('--len-seq-detect', type=int, default=8, help='The length of node sequence in detecting stage.')
ap.add_argument('--l2norm', type=bool, default=True, help='Use l2 norm for prediction')
ap.add_argument('--mode', type=int, default=0, help='Output mode, 0 for offline evaluation and 1 for online HGB evaluation')
ap.add_argument('--temperature', type=float, default=1.0, help='Temperature of attention score')
ap.add_argument('--beta', type=float, default=0.5, help='Weight of heterogeneity-level attention score')
ap.add_argument('--feats-masked-ratio', type=float, default=0.20)
ap.add_argument('--data-masked-ratio', type=float, default=0.55)  # 0.15
ap.add_argument('--tau', type=float, default=0.3)
ap.add_argument('--fine-tune', type=bool, default=True)
ap.add_argument('--alpha', type=float, default=0.01)

args = ap.parse_args()


def graphs_augmentation(dl, adjM, features, nodetype2id):
    node_types = []
    positive_node_types = []
    negative_node_types = []
    for i in range(dl.nodes['total']):
        types = dl.nodes['type'][i]
        select_type = random.choice(types)
        if 'Fallback-Node' in types:
            while select_type == 'Fallback-Node':
                select_type = random.choice(types)
                if len(types) == 1 and select_type == 'Fallback-Node':
                    select_type = 'Common-Node'
            nodetype = nodetype2id[select_type] + 6
        else:
            nodetype = nodetype2id[select_type]

        positive_select_type = select_type
        if len(types) >= 3:
            while positive_select_type == select_type or positive_select_type == 'Fallback-Node':
                positive_select_type = random.choice(types)
        else:
            positive_select_type = select_type
        if 'Fallback-Node' in types:
            positive_nodetype = nodetype2id[positive_select_type] + 6
        else:
            positive_nodetype = nodetype2id[positive_select_type]

        negative_select_type = select_type
        if len(types) >= 3:
            miss_types = []
            for miss_nodetype in nodetype2id.keys():
                if miss_nodetype != 'Fallback-Node' and miss_nodetype not in types:
                    miss_types.append(miss_nodetype)
                    # break
            # if cur_idx == 59:
            if len(miss_types) != 0:
                negative_select_type = random.choice(miss_types)
            else:
                negative_select_type = select_type
        else:
            negative_select_type = select_type
        if 'Fallback-Node' in types:
            negative_nodetype = nodetype2id[negative_select_type] + 6
        else:
            negative_nodetype = nodetype2id[negative_select_type]

        negative_node_types.append(negative_nodetype)
        positive_node_types.append(positive_nodetype)
        node_types.append(nodetype)

    # print(f'node type is ok!')
    positive_node_types = np.array(positive_node_types, dtype=np.int64)
    negative_node_types = np.array(negative_node_types, dtype=np.int64)
    node_types = np.array(node_types, dtype=np.int64)
    # print(node_types)
    # print(f'positive_node_types: \n{positive_node_types}')
    # print(f'negative_node_types: \n{negative_node_types}')

    node_type_tensor = torch.tensor(node_types)
    node_type.append(node_type_tensor)
    negative_node_type_tensor = torch.tensor(negative_node_types)
    positive_node_type_tensor = torch.tensor(positive_node_types)

    g = dgl.DGLGraph(adjM)  # 由于没有加入自循环可能会出现r许多为0的情况
    positive_g = g.clone()
    negative_g = g.clone()

    # g = dgl.remove_self_loop(g)
    g = g
    positive_g = positive_g
    negative_g = negative_g

    features_array = np.array(features)
    positive_features_array = features_array

    # features masking
    masked_idx = random.sample(range(features_array.shape[0]), int(features_array.shape[0] * args.data_masked_ratio))
    for idx in masked_idx:
        masked_indices = np.random.choice(features_array.shape[1],
                                          int(args.feats_masked_ratio * features_array.shape[1]), replace=False)
        positive_features_array[idx, masked_indices] = 0

    g.ndata['feats'] = torch.from_numpy(features_array).float()
    g.ndata['node_type'] = node_type_tensor

    positive_g.ndata['feats'] = torch.from_numpy(features_array).float()
    positive_g.ndata['node_type'] = positive_node_type_tensor

    negative_g.ndata['feats'] = torch.from_numpy(positive_features_array).float()
    negative_g.ndata['node_type'] = negative_node_type_tensor

    srcs, dsts = g.out_edges(g.nodes())
    edge_types = []
    negative_edge_types = []
    for src, dst in zip(srcs, dsts):
        edge_types.append(dl.links['edge_type'][(int(src), int(dst))])

    negative_edge_types = edge_types
    modify_edge_idx = random.sample(range(len(edge_types)), int(len(edge_types) * 0.3))
    for idx in modify_edge_idx:
        temp = negative_edge_types[idx]
        modify_edge = temp
        if temp == 0:
            modify_edge = 1
        elif temp == 1:
            modify_edge = 0
        elif temp == 6:
            modify_edge = random.choice([7, 8])
        elif temp == 7:
            modify_edge = random.choice([6, 8])
        elif temp == 8:
            modify_edge = random.choice([6, 7])
        negative_edge_types[idx] = modify_edge

    edge_types = np.array(edge_types, dtype=np.int64)
    edge_type_tensor = torch.tensor(edge_types)
    negative_edge_types = np.array(negative_edge_types, dtype=np.int64)
    negative_edge_types_tensor = torch.tensor(negative_edge_types)

    g.edata['edge_type'] = edge_type_tensor
    positive_g.edata['edge_type'] = edge_type_tensor
    negative_g.edata['edge_type'] = negative_edge_types_tensor

    return g, positive_g, negative_g, features_array

if __name__ == '__main__':
    device = torch.device('cuda:' + str(args.device)
                          if torch.cuda.is_available() else 'cpu')
    nodetype2id = {'AssetTransfer-Node': 1, 'Fallback-Node': 0, 'Event-Node': 2, 'Invocation-Node': 3,
                   'Compute-Node': 4, 'Information-Node': 5, 'Common-Node': 6}
    edgetype2id = {'JUMPI-True': 0, 'JUMPI-False': 1, 'JUMP': 2, 'Sequence': 3, 'CALLPRIVATE': 4, 'RETURNPRIVATE': 5,
                   'CALL': 6, 'STATICCALL': 7, 'DELEGATECALL': 8, 'RETURN': 9, 'DATAFLOW': 10}
    edge_type, node_type, features_list, train_graphs, train_positive_graphs, train_negative_graphs = [], [], [], [], [], []
    test_graphs, test_positive_graphs, test_negative_graphs = [], [], []

    path_list = ['vul_data/integeroverflow', 'vul_data/reentrancy',
                 'vul_data/timestamp', 'cleandata']
    name2label = {'cleandata': 0, 'vul_data/integeroverflow': 3,
                  'vul_data/reentrancy': 1,
                  'vul_data/timestamp': 2}
    path = './data'
    print(path_list)
    cur_idx = 1
    print(f"start time: {datetime.now()}")

    cross_path_list = ['cross-reentrancy', 'cross-timestamp', 'cross-integeroverflow']
    cross_name2label = {'cross-reentrancy': 1, 'cross-timestamp': 2, 'cross-integeroverflow': 3}
    valid_labels = []
    valid_path = os.path.join(path, 'vul_data/cross-contract')
    valid_graphs, valid_positive_graphs, valid_negative_graphs = [], [], []
    for vul in cross_path_list:
        if vul != 'cross-reentrancy':
            continue
        cur_path = os.path.join(valid_path, vul)
        for project in os.listdir(cur_path):
            sub_valid_graphs, sub_valid_positive_graphs, sub_valid_negative_graphs = [], [], []
            for file in os.listdir(os.path.join(cur_path, project)):
                features, adjM, dl = load_data(pretrain=True, path=os.path.join(cur_path, project, file))
                if type(adjM) is int:
                    continue
                sub_g, sub_positive_g, sub_negative_g, _ = graphs_augmentation(dl, adjM, features, nodetype2id)
                sub_valid_graphs.append(sub_g)
                sub_valid_positive_graphs.append(sub_positive_g)
                sub_valid_negative_graphs.append(sub_negative_g)
            if len(sub_valid_graphs) != 0:
                valid_graphs.append(dgl.merge(sub_valid_graphs))
                valid_positive_graphs.append(dgl.merge(sub_valid_positive_graphs))
                valid_negative_graphs.append(dgl.merge(sub_valid_negative_graphs))
                if vul == 'cross-reentrancy':
                    valid_labels.append(1)
                else:
                    valid_labels.append(0)

    train_labels = []
    test_labels = []
    for datapath in path_list:
        if datapath != 'vul_data/reentrancy' and datapath != 'cleandata':
            continue
        for mode in ['train', 'test']:
            cur_path = os.path.join(path, datapath, mode)
            for address in os.listdir(cur_path):
                sub_graphs, sub_positive_graphs, sub_negative_graphs = [], [], []
                for subpath in os.listdir(os.path.join(cur_path, address)):
                    cur_idx += 1
                    features, adjM, dl = load_data(pretrain=True, path=os.path.join(cur_path, address, subpath))
                    if type(adjM) is int:
                        continue
                    sub_g, sub_positive_g, sub_negative_g, features_array = graphs_augmentation(dl, adjM, features, nodetype2id)
                    features_list.append(features_array)
                    sub_graphs.append(sub_g)
                    sub_positive_graphs.append(sub_positive_g)
                    sub_negative_graphs.append(sub_negative_g)

                if len(sub_graphs) != 0:
                    if mode == 'train':
                        train_graphs.append(dgl.merge(sub_graphs))
                        train_positive_graphs.append(dgl.merge(sub_positive_graphs))
                        train_negative_graphs.append(dgl.merge(sub_negative_graphs))
                        if 'reentrancy' in datapath:
                            train_labels.append(1)
                        else:
                            train_labels.append(0)
                    else:
                        test_graphs.append(dgl.merge(sub_graphs))
                        test_positive_graphs.append(dgl.merge(sub_positive_graphs))
                        test_negative_graphs.append(dgl.merge(sub_negative_graphs))
                        if 'reentrancy' in datapath:
                            test_labels.append(1)
                        else:
                            test_labels.append(0)
                        if datapath == 'cleandata/cleandata_256':
                            valid_graphs.append(dgl.merge(sub_graphs))
                            valid_positive_graphs.append(dgl.merge(sub_positive_graphs))
                            valid_negative_graphs.append(dgl.merge(sub_negative_graphs))
                            valid_labels.append(name2label[datapath])

    train_random_idx = random.sample(range(len(train_graphs)), len(train_graphs))
    test_random_idx = random.sample(range(len(test_graphs)), len(test_graphs))
    valid_random_idx = random.sample(range(len(valid_graphs)), len(valid_graphs))

    features_list = np.concatenate(features_list, axis=0)
    features_list = [mat2tensor(features)
                     for features in features_list]
    features_list = torch.stack(features_list, dim=0)
    print(f"after processing data time: {datetime.now()}")
    train_dataset = GraphDataset([train_graphs[i] for i in train_random_idx], [train_labels[i] for i in train_random_idx])
    train_positive_dataset = GraphDataset([train_positive_graphs[i] for i in train_random_idx], [train_labels[i] for i in train_random_idx])
    train_negative_dataset = GraphDataset([train_negative_graphs[i] for i in train_random_idx], [train_labels[i] for i in train_random_idx])
    test_dataset = GraphDataset([test_graphs[i] for i in test_random_idx], [test_labels[i] for i in test_random_idx])
    test_positive_dataset = GraphDataset([test_positive_graphs[i] for i in test_random_idx], [test_labels[i] for i in test_random_idx])
    test_negative_dataset = GraphDataset([test_negative_graphs[i] for i in test_random_idx], [test_labels[i] for i in test_random_idx])
    valid_dataset = GraphDataset([valid_graphs[i] for i in valid_random_idx],
                                 [valid_labels[i] for i in valid_random_idx])
    valid_positive_dataset = GraphDataset([valid_positive_graphs[i] for i in valid_random_idx],
                                          [valid_labels[i] for i in valid_random_idx])
    valid_negative_dataset = GraphDataset([valid_negative_graphs[i] for i in valid_random_idx],
                                          [valid_labels[i] for i in valid_random_idx])

    train_dataloader = GraphDataLoader(train_dataset, batch_size=args.batch_size)
    train_positive_loader = GraphDataLoader(train_positive_dataset, batch_size=args.batch_size)
    train_negative_loader = GraphDataLoader(train_negative_dataset, batch_size=args.batch_size)
    test_dataloader = GraphDataLoader(test_dataset, batch_size=args.batch_size)
    test_positive_loader = GraphDataLoader(test_positive_dataset, batch_size=args.batch_size)
    test_negative_loader = GraphDataLoader(test_negative_dataset, batch_size=args.batch_size)
    valid_dataloader = GraphDataLoader(valid_dataset, batch_size=args.batch_size)
    valid_positive_loader = GraphDataLoader(valid_positive_dataset, batch_size=args.batch_size)
    valid_negative_loader = GraphDataLoader(valid_negative_dataset, batch_size=args.batch_size)
    print(f'length of train dataloader: {len(train_dataloader)}')
    print(f'length of test dataloader: {len(test_dataloader)}')
    print(f'length of valid dataloader: {len(valid_dataloader)}')
    print(train_labels)
    indims = 768
    num_type = 13
    type_emb = torch.eye(num_type).to(device)

    edge_num_type = 10  # 修改
    edge_type_emb = torch.eye(edge_num_type).to(device)

    node_seq = torch.zeros(features_list[0].shape[0], args.len_seq).long()

    net = HINormer(indims, args.hidden_dim, args.len_seq, args.num_layers, args.num_gnns, args.num_heads, args.dropout,
                   temper=args.temperature, node_num_type=num_type, edge_num_type=edge_num_type, beta=args.beta)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    net.to(device)

    best_loss = 100
    best_accuracy = 0

    for epoch in range(150):
        net.train()
        total_loss = []
        start_time = time.time()
        for train_batched_data, train_positive_batched_data, train_negative_batched_data in \
                zip(train_dataloader, train_positive_loader, train_negative_loader):
            in_start_time = time.time()
            optimizer.zero_grad()

            gc.collect()
            torch.cuda.empty_cache()

            train_batched_graph = train_batched_data[0].to(device)
            train_positive_batched_graph = train_positive_batched_data[0].to(device)
            train_negative_batched_graph = train_negative_batched_data[0].to(device)
            train_labels = train_batched_data[1].to(device)
            train_labels = train_labels.clone().detach().float()

            train_batched_feats = train_batched_graph.ndata['feats'].to(device)
            train_batched_node_type = train_batched_graph.ndata['node_type'].to(device)
            train_batched_edge_type = train_batched_graph.edata['edge_type'].to(device)

            train_positive_batched_feats = train_positive_batched_graph.ndata['feats'].to(device)
            train_positive_batched_node_type = train_positive_batched_graph.ndata['node_type'].to(device)
            train_positive_batched_edge_type = train_positive_batched_graph.edata['edge_type'].to(device)

            node_seq = train_batched_data[2].squeeze(0)

            subgraph = True
            if len(node_seq) == 0:
                subgraph = False

            logits, _ = net(train_batched_graph, train_batched_feats, node_seq, type_emb, edge_type_emb, train_batched_node_type,
                         train_batched_edge_type, subgraph, args.fine_tune, args.l2norm)
            positive_logits, _ = net(train_positive_batched_graph, train_positive_batched_feats, node_seq, type_emb,
                                  edge_type_emb, train_positive_batched_node_type, train_positive_batched_edge_type, subgraph, args.fine_tune, args.l2norm)

            if subgraph:
                train_labels = train_labels.expand(logits.shape[0], -1)
            else:
                train_labels = train_labels.unsqueeze(0)
            #classification loss
            # loss_fn = torch.nn.CrossEntropyLoss()
            loss_fn = torch.nn.BCEWithLogitsLoss()
            cls_train_loss = loss_fn(logits, train_labels)
            cls_train_positive_loss = loss_fn(positive_logits, train_labels)
            in_end_time = time.time()
            loss = cls_train_loss + cls_train_positive_loss * 0.01
            # print(f'batch: {len(total_loss)} loss: {loss}, consume time: {in_end_time-in_start_time}, current time:{datetime.now()}')

            loss.backward()
            total_loss.append(loss.item())
            optimizer.step()

        end_time = time.time()
        print(f'epoch = {epoch+1}, loss = {sum(total_loss)/len(total_loss)}, times = {end_time - start_time}')
        temp_loss = sum(total_loss)/len(total_loss)

        net.eval()
        with torch.no_grad():
            labels = []
            output_labels = []
            for test_batched_data, test_positive_batched_data, test_negative_batched_data in \
                    zip(test_dataloader, test_positive_loader, test_negative_loader):

                test_batched_graph = test_batched_data[0].to(device)
                test_positive_batched_graph = test_positive_batched_data[0].to(device)
                test_negative_batched_graph = test_negative_batched_data[0].to(device)
                test_labels = test_batched_data[1].to(device)
                labels.append(int(test_labels.cpu()))

                test_batched_feats = test_batched_graph.ndata['feats'].to(device)
                test_batched_node_type = test_batched_graph.ndata['node_type'].to(device)
                test_batched_edge_type = test_batched_graph.edata['edge_type'].to(device)

                test_positive_batched_feats = test_positive_batched_graph.ndata['feats'].to(device)
                test_positive_batched_node_type = test_positive_batched_graph.ndata['node_type'].to(device)
                test_positive_batched_edge_type = test_positive_batched_graph.edata['edge_type'].to(device)

                node_seq = test_batched_data[2].squeeze(0)
                subgraph = True
                if len(node_seq) == 0:
                    subgraph = False

                logits, _ = net(test_batched_graph, test_batched_feats, node_seq, type_emb, edge_type_emb,
                             test_batched_node_type, test_batched_edge_type, subgraph, args.fine_tune, args.l2norm)
                output1 = torch.sigmoid(logits)
                pred1 = (output1 > 0.5).int()
                pred1_list = pred1.tolist()
                if [1] not in pred1_list:
                    output_labels.append(0)
                else:
                    output_labels.append(1)
            print(f'output_labels: {output_labels}')
            print(f'labels: {labels}')
            accuracy = metrics.accuracy_score(output_labels, labels)
            print(metrics.classification_report(output_labels, labels))

            # cross-contract
            # '''
            net.eval()
            with torch.no_grad():
                labels = []
                pred_labels = []
                output_labels = []
                for valid_batched_data, valid_positive_batched_data, valid_negative_batched_data in \
                        zip(valid_dataloader, valid_positive_loader, valid_negative_loader):

                    valid_batched_graph = valid_batched_data[0].to(device)
                    valid_positive_batched_graph = valid_positive_batched_data[0].to(device)
                    valid_negative_batched_graph = valid_negative_batched_data[0].to(device)
                    valid_labels = valid_batched_data[1].to(device)
                    labels.append(int(valid_labels.cpu()))

                    valid_batched_feats = valid_batched_graph.ndata['feats'].to(device)
                    valid_batched_node_type = valid_batched_graph.ndata['node_type'].to(device)
                    valid_batched_edge_type = valid_batched_graph.edata['edge_type'].to(device)

                    valid_positive_batched_feats = valid_positive_batched_graph.ndata['feats'].to(device)
                    valid_positive_batched_node_type = valid_positive_batched_graph.ndata['node_type'].to(device)
                    valid_positive_batched_edge_type = valid_positive_batched_graph.edata['edge_type'].to(device)

                    node_seq = valid_batched_data[2].squeeze(0)

                    subgraph = True
                    if len(node_seq) == 0:
                        subgraph = False
                    logits, _ = net(valid_batched_graph, valid_batched_feats, node_seq, type_emb, edge_type_emb,
                                 valid_batched_node_type, valid_batched_edge_type, subgraph, args.fine_tune, args.l2norm)
                    output1 = torch.sigmoid(logits)
                    pred1 = (output1 > 0.5).int()
                    pred1_list = pred1.tolist()
                    if [1] not in pred1_list:
                        output_labels.append(0)
                    else:
                        output_labels.append(1)

                print(f'cross-contract vulnerability detection result:')
                print(metrics.classification_report(output_labels, labels))
                report_dict = metrics.classification_report(output_labels, labels, output_dict=True, digits=4)
            # '''
        if not os.path.exists('./results'):
            os.makedirs('./results/')
        if True:
            best_loss = temp_loss
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': 'SC-GCL',
                'loss': best_loss,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, is_best=False, filename=f'./results/train/checkpoint_epoch{epoch + 1}_hiddendim{args.hidden_dim}'
                                       f'_layers{args.num_layers}_gnns{args.num_gnns}_lenseq{args.len_seq}'
                                       f'_batchsize{args.batch_size}_tau{args.tau}_lr{args.lr}'
                                       f'_headnum{args.num_heads}_BERT_accuracy{report_dict["accuracy"]}.pth')
        if report_dict['accuracy'] > best_accuracy:
            best_accuracy = report_dict['accuracy']
            torch.save(net, f'./results/train/model_{best_accuracy}.pth')
