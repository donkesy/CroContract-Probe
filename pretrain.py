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
from datetime import datetime
import torch.nn.functional as F

from model import HINormer
sys.path.append('pyg/')
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
ap.add_argument('--batch-size', type=int, default=16, help='Number of batch')
ap.add_argument('--patience', type=int, default=50, help='Patience.')
ap.add_argument('--repeat', type=int, default=5, help='Repeat the training and testing for N times. Default is 1.')
ap.add_argument('--num-layers', type=int, default=2, help='The number of layers of HINormer layer')
ap.add_argument('--num-gnns', type=int, default=3, help='The number of layers of both structural and heterogeneous encoder')
ap.add_argument('--lr', type=float, default=2e-4)
ap.add_argument('--seed', type=int, default=2024)
ap.add_argument('--dropout', type=float, default=0.5)
ap.add_argument('--weight-decay', type=float, default=0)
ap.add_argument('--len-seq', type=int, default=8, help='The length of node sequence.')
ap.add_argument('--l2norm', type=bool, default=True, help='Use l2 norm for prediction')
ap.add_argument('--mode', type=int, default=0, help='Output mode, 0 for offline evaluation and 1 for online HGB evaluation')
ap.add_argument('--temperature', type=float, default=1.0, help='Temperature of attention score')
ap.add_argument('--beta', type=float, default=0.5, help='Weight of heterogeneity-level attention score')
ap.add_argument('--feats-masked-ratio', type=float, default=0.20)
ap.add_argument('--data-masked-ratio', type=float, default=0.55)  # 0.15
ap.add_argument('--tau', type=float, default=0.3)
ap.add_argument('--fine-tune', type=bool, default=False)

args = ap.parse_args()


if __name__ == '__main__':
    device = torch.device('cuda:' + str(args.device)
                          if torch.cuda.is_available() else 'cpu')
    # node_type = []
    # features_list = []
    # graphs = []
    # positive_graphs = []
    # negative_graphs = []
    nodetype2id = {'AssetTransfer-Node': 1, 'Fallback-Node': 0, 'Event-Node': 2, 'Invocation-Node': 3,
                   'Compute-Node': 4, 'Information-Node': 5, 'Common-Node': 6, 'Null-Node': 6}
    edgetype2id = {'JUMPI-True': 0, 'JUMPI-False': 1, 'JUMP': 2, 'Sequence': 3, 'CALLPRIVATE': 4, 'RETURNPRIVATE': 5,
                   'CALL': 6, 'STATICCALL': 7, 'DELEGATECALL': 8, 'RETURN': 9}
    edge_type, node_type, features_list, graphs, positive_graphs, negative_graphs = [], [], [], [], [], []

    # path = 'E:\\Py_projects\\CrossVulDec\\inputdata'
    path = 'E:\\Py_projects\\CrossVulDec\\inputdata'
    # path = 'E:\\Py_projects\\CrossVulDec\\utils/inputdata_768'
    cur_idx = 1
    print(f"start time: {datetime.now()}")
    for index, address in enumerate(os.listdir(path)):
        if index > 1200:
            break
        if address == 'test' or address == 'train':
            continue
        sub_graphs, sub_positive_graphs, sub_negative_graphs = [], [], []
        for subpath in os.listdir(os.path.join(path, address)):
            files = os.listdir(os.path.join(path, address, subpath))
            if len(files) == 0:
                continue
            # print(f"processing {cur_idx}! {os.path.join(path, address, subpath)}")
            cur_idx += 1
            # print(os.path.join(path, address, subpath))
            features, adjM, dl = load_data(pretrain=True, path=os.path.join(path, address, subpath))
            if type(adjM) is int:
                continue

            # node_types = np.array([dl.nodes['type'][i] for i in range(dl.nodes['total'])], dtype=np.int64)
            node_types = []
            positive_node_types = []
            negative_node_types = []
            for i in range(dl.nodes['total']):
                # print(i)
                types = dl.nodes['type'][i]
                # print(types)
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
            '''
                正样本：节点类型替换成另一个已有的节点类型
                负样本：节点类型替换成另一个不具有的节点类型；控制流修改，条件语句变换等；将某些特征进行掩码，使一些重要信息确实
            '''

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
            edge_type.append(edge_type_tensor)

            g.edata['edge_type'] = edge_type_tensor
            positive_g.edata['edge_type'] = edge_type_tensor
            negative_g.edata['edge_type'] = negative_edge_types_tensor

            features_list.append(features_array)
            # for ii in range(64):
            sub_graphs.append(g)
            sub_positive_graphs.append(positive_g)
            sub_negative_graphs.append(negative_g)
        graphs.append(dgl.merge(sub_graphs))
        positive_graphs.append(dgl.merge(sub_positive_graphs))
        negative_graphs.append(dgl.merge(sub_negative_graphs))

    print(f"after processing data time: {datetime.now()}")
    print(len(graphs))

    data_sample_1 = time.time()

    dataset = GraphDataset([graphs[i] for i in range(len(graphs))], [0 for i in range(len(graphs))])
    positive_dataset = GraphDataset([positive_graphs[i] for i in range(len(graphs))], [0 for i in range(len(graphs))])
    negative_dataset = GraphDataset([negative_graphs[i] for i in range(len(graphs))], [0 for i in range(len(graphs))])

    dataloader = GraphDataLoader(dataset, batch_size=1, shuffle=False)
    positive_loader = GraphDataLoader(positive_dataset, batch_size=1, shuffle=False)
    negative_loader = GraphDataLoader(negative_dataset, batch_size=1, shuffle=False)
    print(f'length of dataloader: {len(dataloader)}')
    print(f"time of sampling: {time.time()-data_sample_1}")

    indims = 768

    num_type = 13
    # print(f'num_type={num_type}')
    type_emb = torch.eye(num_type).to(device)

    edge_num_type = 10  # 修改
    edge_type_emb = torch.eye(edge_num_type).to(device)

    node_seq = torch.zeros(features_list[0].shape[0], args.len_seq).long()

    net = HINormer(indims, args.hidden_dim, args.len_seq, args.num_layers, args.num_gnns, args.num_heads, args.dropout,
                   temper=args.temperature, node_num_type=num_type, edge_num_type=edge_num_type, beta=args.beta)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    # checkpoint = torch.load('./results/results_768/pretrain/checkpoint_epoch24_hiddendim128_layers2_gnns3_lenseq15_batchsize32_tau0.3_lr0.0002_inChannels_768_headnum8_loss4.978963116631991.pth')

    net.to(device)

    # net.load_state_dict(checkpoint['state_dict'], strict=False)
    # optimizer.load_state_dict(checkpoint['optimizer'])

    best_loss = 100
    for epoch in range(300):
        net.train()
        total_loss = []
        start_time = time.time()
        for batched_data, positive_batched_data, negative_batched_data in zip(dataloader, positive_loader, negative_loader):
            in_start_time = time.time()
            # print(f"node number: {batched_graph.num_nodes}")
            # print(f'index: {len(total_loss)+1}')
            # print(f"node number: {len(batched_graph.nodes())}")
            optimizer.zero_grad()

            # gc.collect()
            # torch.cuda.empty_cache()

            batched_graph = batched_data[0].to(device)
            positive_batched_graph = positive_batched_data[0].to(device)
            negative_batched_graph = negative_batched_data[0].to(device)

            batched_feats = batched_data[0].ndata['feats'].to(device)
            batched_node_type = batched_data[0].ndata['node_type'].to(device)
            batched_edge_type = batched_data[0].edata['edge_type'].to(device)

            positive_batched_feats = positive_batched_data[0].ndata['feats'].to(device)
            positive_batched_node_type = positive_batched_data[0].ndata['node_type'].to(device)
            positive_batched_edge_type = positive_batched_data[0].edata['edge_type'].to(device)

            negative_batched_feats = negative_batched_data[0].ndata['feats'].to(device)
            negative_batched_node_type = negative_batched_data[0].ndata['node_type'].to(device)
            negative_batched_edge_type = negative_batched_data[0].edata['edge_type'].to(device)

            subgraph = True
            node_seq = batched_data[2].squeeze(0)

            if len(node_seq) == 0:
                subgraph = False

            logits = net(batched_graph, batched_feats, node_seq, type_emb, edge_type_emb, batched_node_type,
                         batched_edge_type, subgraph, args.fine_tune, args.l2norm)
            positive_logits = net(positive_batched_graph, positive_batched_feats, node_seq, type_emb,
                                  edge_type_emb, positive_batched_node_type, positive_batched_edge_type, subgraph, args.fine_tune, args.l2norm)
            negative_logits = net(negative_batched_graph, negative_batched_feats, node_seq, type_emb,
                                  edge_type_emb, negative_batched_node_type, negative_batched_edge_type, subgraph, args.fine_tune, args.l2norm)

            loss = net.InfoNCE_loss_triple(logits, positive_logits, negative_logits, args.tau)
            in_end_time = time.time()
            # print(f'batch: {len(total_loss)} loss: {loss}, consume time: {in_end_time-in_start_time}')

            loss.backward()
            total_loss.append(loss.item())
            optimizer.step()

        end_time = time.time()
        print(f'epoch = {epoch + 1}, loss = {sum(total_loss)/len(total_loss)}, times = {end_time - start_time}')
        temp_loss = sum(total_loss)/len(total_loss)
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
            }, is_best=False, filename=f'./results/pretrain/checkpoint_epoch{epoch+1}_hiddendim{args.hidden_dim}'
                                       f'_layers{args.num_layers}_gnns{args.num_gnns}_lenseq{args.len_seq}'
                                       f'_batchsize{args.batch_size}_tau{args.tau}_lr{args.lr}_inChannels_{768}'
                                       f'_headnum{args.num_heads}_loss{best_loss}.pth')
        print(f"endtime: {datetime.now()}")