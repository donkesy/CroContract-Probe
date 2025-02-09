# import argparse
# import gc
# import os
# import random
# import shutil
# import sys
# import time
#
# import dgl
# import numpy as np
# import torch
# from sklearn import metrics
# from datetime import datetime
# from sklearn.model_selection import train_test_split
# import torchmetrics
# from tqdm import tqdm
#
# from model import HINormer
# import torch.nn.functional as F
# from data_loader import data_loader
# from data import load_data, GraphDataset
# from dgl.dataloading import GraphDataLoader
# import scipy.sparse as sp
#
#
# def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
#     torch.save(state, filename)
#     if is_best:
#         shutil.copyfile(filename, 'model_best.pth.tar')
#
#
# def sp_to_spt(mat):
#     coo = mat.tocoo()
#     values = coo.data
#     indices = np.vstack((coo.row, coo.col))
#
#     i = torch.LongTensor(indices)
#     v = torch.FloatTensor(values)
#     shape = coo.shape
#
#     return torch.sparse.FloatTensor(i, v, torch.Size(shape))
#
#
# def mat2tensor(mat):
#     if type(mat) is np.ndarray:
#         return torch.from_numpy(mat).type(torch.FloatTensor)
#     return sp_to_spt(mat)
#
#
# ap = argparse.ArgumentParser(
#         description='HINormer')
# ap.add_argument('--feats-type', type=int, default=0,
#                 help='Type of the node features used. ' +
#                      '0 - loaded features; ' +
#                      '1 - only target node features (zero vec for others); ' +
#                      '2 - only target node features (id vec for others); ' +
#                      '3 - all id vec. Default is 2' +
#                 '4 - only term features (id vec for others);' +
#                 '5 - only term features (zero vec for others).')
# ap.add_argument('--device', type=int, default=0)
# ap.add_argument('--hidden-dim', type=int, default=128,
#                 help='Dimension of the node hidden state. Default is 32.')
# ap.add_argument('--dataset', type=str, default='DBLP', help='DBLP, IMDB, Freebase, AMiner, DBLP-HGB, IMDB-HGB')
# ap.add_argument('--num-heads', type=int, default=8,
#                 help='Number of the attention heads. Default is 2.')
# ap.add_argument('--epoch', type=int, default=1000, help='Number of epochs.')
# ap.add_argument('--batch-size', type=int, default=1, help='Number of batch')
# ap.add_argument('--patience', type=int, default=50, help='Patience.')
# ap.add_argument('--repeat', type=int, default=5, help='Repeat the training and testing for N times. Default is 1.')
# ap.add_argument('--num-layers', type=int, default=2, help='The number of layers of HINormer layer')
# ap.add_argument('--num-gnns', type=int, default=3, help='The number of layers of both structural and heterogeneous encoder')
# ap.add_argument('--lr', type=float, default=2e-4)
# ap.add_argument('--seed', type=int, default=2024)
# ap.add_argument('--dropout', type=float, default=0.5)
# ap.add_argument('--weight-decay', type=float, default=0)
# ap.add_argument('--len-seq', type=int, default=30, help='The length of node sequence.')
# ap.add_argument('--l2norm', type=bool, default=True, help='Use l2 norm for prediction')
# ap.add_argument('--mode', type=int, default=0, help='Output mode, 0 for offline evaluation and 1 for online HGB evaluation')
# ap.add_argument('--temperature', type=float, default=1.0, help='Temperature of attention score')
# ap.add_argument('--beta', type=float, default=0.5, help='Weight of heterogeneity-level attention score')
# ap.add_argument('--feats-masked-ratio', type=float, default=0.20)
# ap.add_argument('--data-masked-ratio', type=float, default=0.55)  # 0.15
# ap.add_argument('--tau', type=float, default=0.3)
# ap.add_argument('--fine-tune', type=bool, default=True)
# ap.add_argument('--alpha', type=float, default=0.01)
#
# args = ap.parse_args()
#
#
#
#
# if __name__ == '__main__':
#     device = torch.device('cuda:' + str(args.device)
#                           if torch.cuda.is_available() else 'cpu')
#
#     nodetype2id = {'AssetTransfer-Node': 1, 'Fallback-Node': 0, 'Event-Node': 2, 'Invocation-Node': 3,
#                    'Compute-Node': 4, 'Information-Node': 5, 'Common-Node': 6}
#     edgetype2id = {'JUMPI-True': 0, 'JUMPI-False': 1, 'JUMP': 2, 'Sequence': 3, 'CALLPRIVATE': 4, 'RETURNPRIVATE': 5,
#                    'CALL': 6, 'STATICCALL': 7, 'DELEGATECALL': 8, 'RETURN': 9}
#     edge_type, node_type, features_list, train_graphs, train_positive_graphs, train_negative_graphs = [], [], [], [], [], []
#     test_graphs, test_positive_graphs, test_negative_graphs = [], [], []
#
#     path_list = ['vul_data/hidden_768/integeroverflow', 'vul_data/hidden_768/reentrancy', 'vul_data/hidden_768/timestamp', 'cleandata/cleandata_768']
#     name2label = {'cleandata/cleandata_768': 0, 'vul_data/hidden_768/integeroverflow': 1, 'vul_data/hidden_768/reentrancy': 2, 'vul_data/hidden_768/timestamp': 3}
#     path = 'E:\\Py_projects\\CrossVulDec'
#     # path = 'E:\\Py_projects\\CrossVulDec\\one'
#     cur_idx = 1
#     print(f"start time: {datetime.now()}")
#     train_labels = []
#     test_labels = []
#     for datapath in path_list:
#         for mode in ['train', 'test']:
#             cur_path = os.path.join(path, datapath, mode)
#             for address in os.listdir(cur_path):
#                 for subpath in os.listdir(os.path.join(cur_path, address)):
#                 # print(f"processing {cur_idx}! {os.path.join(path, address, subpath)}")
#                     cur_idx += 1
#                     features, adjM, dl = load_data(pretrain=True, path=os.path.join(cur_path, address, subpath))
#                     if type(adjM) is int:
#                         continue
#                     # node_types = np.array([dl.nodes['type'][i] for i in range(dl.nodes['total'])], dtype=np.int64)
#                     node_types = []
#                     positive_node_types = []
#                     negative_node_types = []
#                     for i in range(dl.nodes['total']):
#                         # print(i)
#                         types = dl.nodes['type'][i]
#                         # print(types)
#                         select_type = random.choice(types)
#                         if 'Fallback-Node' in types:
#                             while select_type == 'Fallback-Node':
#                                 select_type = random.choice(types)
#                                 if len(types) == 1 and select_type == 'Fallback-Node':
#                                     select_type = 'Common-Node'
#                             nodetype = nodetype2id[select_type] + 6
#                         else:
#                             nodetype = nodetype2id[select_type]
#
#                         positive_select_type = select_type
#                         if len(types) >= 3:
#                             while positive_select_type == select_type or positive_select_type == 'Fallback-Node':
#                                 positive_select_type = random.choice(types)
#                         else:
#                             positive_select_type = select_type
#                         if 'Fallback-Node' in types:
#                             positive_nodetype = nodetype2id[positive_select_type] + 6
#                         else:
#                             positive_nodetype = nodetype2id[positive_select_type]
#
#                         negative_select_type = select_type
#                         if len(types) >= 3:
#                             miss_types = []
#                             for miss_nodetype in nodetype2id.keys():
#                                 if miss_nodetype != 'Fallback-Node' and miss_nodetype not in types:
#                                     miss_types.append(miss_nodetype)
#                                     # break
#                             # if cur_idx == 59:
#                             if len(miss_types) != 0:
#                                 negative_select_type = random.choice(miss_types)
#                             else:
#                                 negative_select_type = select_type
#                         else:
#                             negative_select_type = select_type
#                         if 'Fallback-Node' in types:
#                             negative_nodetype = nodetype2id[negative_select_type] + 6
#                         else:
#                             negative_nodetype = nodetype2id[negative_select_type]
#
#                         negative_node_types.append(negative_nodetype)
#                         positive_node_types.append(positive_nodetype)
#                         node_types.append(nodetype)
#
#                     # print(f'node type is ok!')
#                     positive_node_types = np.array(positive_node_types, dtype=np.int64)
#                     negative_node_types = np.array(negative_node_types, dtype=np.int64)
#                     node_types = np.array(node_types, dtype=np.int64)
#                     # print(node_types)
#                     # print(f'positive_node_types: \n{positive_node_types}')
#                     # print(f'negative_node_types: \n{negative_node_types}')
#
#                     node_type_tensor = torch.tensor(node_types)
#                     node_type.append(node_type_tensor)
#                     negative_node_type_tensor = torch.tensor(negative_node_types)
#                     positive_node_type_tensor = torch.tensor(positive_node_types)
#
#                     g = dgl.DGLGraph(adjM)  # 由于没有加入自循环可能会出现r许多为0的情况
#                     positive_g = g.clone()
#                     negative_g = g.clone()
#
#                     # g = dgl.remove_self_loop(g)
#                     g = g
#                     positive_g = positive_g
#                     negative_g = negative_g
#
#                     features_array = np.array(features)
#                     if features_array.shape[1] != 768:
#                         print(os.path.join(cur_path, address, subpath))
#                     positive_features_array = features_array
#
#                     # features masking
#                     masked_idx = random.sample(range(features_array.shape[0]), int(features_array.shape[0] * args.data_masked_ratio))
#                     for idx in masked_idx:
#                         masked_indices = np.random.choice(features_array.shape[1],
#                                                           int(args.feats_masked_ratio * features_array.shape[1]), replace=False)
#                         positive_features_array[idx, masked_indices] = 0
#
#                     g.ndata['feats'] = torch.from_numpy(features_array).float()
#                     g.ndata['node_type'] = node_type_tensor
#
#                     positive_g.ndata['feats'] = torch.from_numpy(features_array).float()
#                     positive_g.ndata['node_type'] = positive_node_type_tensor
#
#                     negative_g.ndata['feats'] = torch.from_numpy(positive_features_array).float()
#                     negative_g.ndata['node_type'] = negative_node_type_tensor
#                     '''
#                         正样本：节点类型替换成另一个已有的节点类型
#                         负样本：节点类型替换成另一个不具有的节点类型；控制流修改，条件语句变换等；将某些特征进行掩码，使一些重要信息确实
#                     '''
#
#                     srcs, dsts = g.out_edges(g.nodes())
#                     edge_types = []
#                     negative_edge_types = []
#                     for src, dst in zip(srcs, dsts):
#                         edge_types.append(dl.links['edge_type'][(int(src), int(dst))])
#
#                     negative_edge_types = edge_types
#                     modify_edge_idx = random.sample(range(len(edge_types)), int(len(edge_types) * 0.3))
#                     for idx in modify_edge_idx:
#                         temp = negative_edge_types[idx]
#                         modify_edge = temp
#                         if temp == 0:
#                             modify_edge = 1
#                         elif temp == 1:
#                             modify_edge = 0
#                         elif temp == 6:
#                             modify_edge = random.choice([7, 8])
#                         elif temp == 7:
#                             modify_edge = random.choice([6, 8])
#                         elif temp == 8:
#                             modify_edge = random.choice([6, 7])
#                         negative_edge_types[idx] = modify_edge
#
#                     edge_types = np.array(edge_types, dtype=np.int64)
#                     edge_type_tensor = torch.tensor(edge_types)
#                     negative_edge_types = np.array(negative_edge_types, dtype=np.int64)
#                     negative_edge_types_tensor = torch.tensor(negative_edge_types)
#                     edge_type.append(edge_type_tensor)
#
#                     g.edata['edge_type'] = edge_type_tensor
#                     positive_g.edata['edge_type'] = edge_type_tensor
#                     negative_g.edata['edge_type'] = negative_edge_types_tensor
#
#                     features_list.append(features_array)
#                     # for ii in range(64):
#                     if mode == 'train':
#                         train_graphs.append(g)
#                         train_positive_graphs.append(positive_g)
#                         train_negative_graphs.append(negative_g)
#                         train_labels.append(name2label[datapath])
#                     else:
#                         test_graphs.append(g)
#                         test_positive_graphs.append(positive_g)
#                         test_negative_graphs.append(negative_g)
#                         test_labels.append(name2label[datapath])
#
#     # data_indices = list(range(len(graphs)))
#     # train_indices, test_indices = train_test_split(data_indices, test_size=0.2, random_state=2024)
#     # print(len(train_indices))
#     # print(len(test_indices))
#     train_random_idx = random.sample(range(len(train_graphs)), len(train_graphs))
#     test_random_idx = random.sample(range(len(test_graphs)), len(test_graphs))
#
#     # sys.exit(0)
#     features_list = np.concatenate(features_list, axis=0)
#     features_list = [mat2tensor(features)
#                      for features in features_list]
#     features_list = torch.stack(features_list, dim=0)
#     print(f"after processing data time: {datetime.now()}")
#     # node_type = np.array(node_type).ravel().tolist()
#     # sys.exit(0)
#     train_dataset = GraphDataset([train_graphs[i] for i in train_random_idx], [train_labels[i] for i in train_random_idx])
#     train_positive_dataset = GraphDataset([train_positive_graphs[i] for i in train_random_idx], [train_labels[i] for i in train_random_idx])
#     train_negative_dataset = GraphDataset([train_negative_graphs[i] for i in train_random_idx], [train_labels[i] for i in train_random_idx])
#     test_dataset = GraphDataset([test_graphs[i] for i in test_random_idx], [test_labels[i] for i in test_random_idx])
#     test_positive_dataset = GraphDataset([test_positive_graphs[i] for i in test_random_idx], [test_labels[i] for i in test_random_idx])
#     test_negative_dataset = GraphDataset([test_negative_graphs[i] for i in test_random_idx], [test_labels[i] for i in test_random_idx])
#
#     train_dataloader = GraphDataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
#     train_positive_loader = GraphDataLoader(train_positive_dataset, batch_size=args.batch_size, shuffle=False)
#     train_negative_loader = GraphDataLoader(train_negative_dataset, batch_size=args.batch_size, shuffle=False)
#     test_dataloader = GraphDataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
#     test_positive_loader = GraphDataLoader(test_positive_dataset, batch_size=args.batch_size, shuffle=False)
#     test_negative_loader = GraphDataLoader(test_negative_dataset, batch_size=args.batch_size, shuffle=False)
#     print(f'length of train dataloader: {len(train_dataloader)}')
#     print(f'length of test dataloader: {len(test_dataloader)}')
#     # print(train_labels)
#     # node_cnt = features_list.shape[0]
#     indims = 256
#     # indims = 256
#
#     num_type = 13
#     # print(f'num_type={num_type}')
#     type_emb = torch.eye(num_type).to(device)
#
#     edge_num_type = 10  # 修改
#     edge_type_emb = torch.eye(edge_num_type).to(device)
#
#     node_seq = torch.zeros(features_list[0].shape[0], args.len_seq).long()
#
#     net = HINormer(indims, args.hidden_dim, args.len_seq, args.num_layers, args.num_gnns, args.num_heads, args.dropout,
#                    temper=args.temperature, node_num_type=num_type, edge_num_type=edge_num_type, beta=args.beta)
#
#     optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
#
#     checkpoint = torch.load('./results/checkpoint_epoch31_hiddendim128_layers2_gnns3_lenseq30_batchsize32_tau0.3_lr0.0002_headnum8_loss4.059761582794836.pth')
#     # checkpoint = torch.load('./results/train/checkpoint_epoch200_hiddendim128_layers2_gnns3_lenseq30_batchsize1_tau0.3_lr0.0002_headnum8_loss0.7725966218952183.pth')
#     net.to(device)
#
#     net.load_state_dict(checkpoint['state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer'])
#
#     best_loss = 100
#
#     for epoch in range(300):
#         net.train()
#         total_loss = []
#         start_time = time.time()
#         train_dataloader = tqdm(train_dataloader, desc=f"epoch:{epoch+1} —— Training progress")
#         for train_batched_data, train_positive_batched_data, train_negative_batched_data in \
#                 zip(train_dataloader, train_positive_loader, train_negative_loader):
#             in_start_time = time.time()
#             # print(f"node number: {batched_graph.num_nodes}")
#             # print(f'index: {len(total_loss)+1}')
#             # print(f"node number: {len(batched_graph.nodes())}")
#
#             gc.collect()
#             torch.cuda.empty_cache()
#
#             train_batched_graph = train_batched_data[0].to(device)
#             train_positive_batched_graph = train_positive_batched_data[0].to(device)
#             train_negative_batched_graph = train_negative_batched_data[0].to(device)
#             train_labels = train_batched_data[1].to(device)
#
#             train_batched_feats = train_batched_graph.ndata['feats'].to(device)
#             train_batched_node_type = train_batched_graph.ndata['node_type'].to(device)
#             train_batched_edge_type = train_batched_graph.edata['edge_type'].to(device)
#
#             train_positive_batched_feats = train_positive_batched_graph.ndata['feats'].to(device)
#             train_positive_batched_node_type = train_positive_batched_graph.ndata['node_type'].to(device)
#             train_positive_batched_edge_type = train_positive_batched_graph.edata['edge_type'].to(device)
#
#             train_negative_batched_feats = train_negative_batched_graph.ndata['feats'].to(device)
#             train_negative_batched_node_type = train_negative_batched_graph.ndata['node_type'].to(device)
#             train_negative_batched_edge_type = train_negative_batched_graph.edata['edge_type'].to(device)
#
#
#             # sample
#             all_nodes = np.arange(train_batched_feats.shape[0])
#             node_seq = torch.zeros(train_batched_feats.shape[0], args.len_seq).long()
#             n = 0
#             sampled_idx = []
#             pre_sampled_idx = random.sample(range(len(all_nodes)), int(len(all_nodes) * 0.12))
#             # print(f'pre_sampled_idx length: {len(pre_sampled_idx)}')
#
#             subgraph = False
#             for x in pre_sampled_idx:
#                 cnt = 0
#                 scnt = 0
#                 node_seq[n, cnt] = x
#                 cnt += 1
#                 start = node_seq[n, scnt].item()
#                 # print(f'start = {start}')
#                 # print(batched_graph.successors(start))
#                 while cnt < args.len_seq:
#                     sample_list = train_batched_graph.successors(start).cpu().numpy().tolist()
#                     nsampled = len(sample_list)
#                     for i in range(nsampled):
#                         node_seq[n, cnt] = sample_list[i]
#                         cnt += 1
#                         if cnt == args.len_seq:
#                             break
#                     scnt += 1
#                     if scnt == args.len_seq:
#                         break
#                     start = node_seq[n, scnt].item()
#                 if cnt == args.len_seq:
#                     sampled_idx.append(x)
#                     subgraph = True
#                 # if cnt != args.len_seq:
#                 #     subgraph = False
#                 n += 1
#             # print(f'before sampling: {len(sampled_idx)}')
#             sampled_idx = random.sample(sampled_idx, int(len(sampled_idx) * 0.9))
#             # print(sampled_idx)
#
#             # print(f'all nodes: {len(all_nodes)}, sampled nodes: {len(sampled_idx)}')
#             if len(sampled_idx) * 30 < len(all_nodes) or int(train_labels) != 0:
#                 subgraph = False
#
#             logits = net(train_batched_graph, train_batched_feats, node_seq[sampled_idx], type_emb, edge_type_emb, train_batched_node_type,
#                          train_batched_edge_type, subgraph, args.fine_tune, args.l2norm)
#             positive_logits = net(train_positive_batched_graph, train_positive_batched_feats, node_seq[sampled_idx], type_emb,
#                                   edge_type_emb, train_positive_batched_node_type, train_positive_batched_edge_type, subgraph, args.fine_tune, args.l2norm)
#             negative_logits = net(train_negative_batched_graph, train_negative_batched_feats, node_seq[sampled_idx], type_emb,
#                                   edge_type_emb, train_negative_batched_node_type, train_negative_batched_edge_type, subgraph, args.fine_tune, args.l2norm)
#
#             if subgraph:
#                 train_labels = torch.full((len(sampled_idx),), 0, device='cuda:0')
#
#             #classification loss
#             loss_fn = torch.nn.CrossEntropyLoss()
#             cls_loss = loss_fn(logits, train_labels)
#             # logp = F.log_softmax(logits, 1)
#             # cls_train_loss = F.nll_loss(logp, train_labels)
#             co_loss = net.InfoNCE_loss_triple(logits, positive_logits, negative_logits, args.tau)
#             # co_loss = net.InfoNCE_loss(logits, positive_logits, args.tau)
#             # print(f'contrasive loss: {co_loss}')
#             in_end_time = time.time()
#             loss = cls_loss  # + args.alpha * co_loss
#             # print(f'batch: {len(total_loss)} loss: {loss}, consume time: {in_end_time-in_start_time}, current time:{datetime.now()}')
#
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             total_loss.append(loss.item())
#
#         end_time = time.time()
#         print(f'epoch = {epoch+1}, loss = {sum(total_loss)/len(total_loss)}, times = {end_time - start_time}')
#         temp_loss = sum(total_loss)/len(total_loss)
#
#         net.eval()
#         with torch.no_grad():
#             labels = []
#             output_labels = []
#             test_dataloader = tqdm(test_dataloader, desc=f"Testing progress")
#             for test_batched_data, test_positive_batched_data, test_negative_batched_data in \
#                     zip(test_dataloader, test_positive_loader, test_negative_loader):
#
#                 test_batched_graph = test_batched_data[0].to(device)
#                 test_positive_batched_graph = test_positive_batched_data[0].to(device)
#                 test_negative_batched_graph = test_negative_batched_data[0].to(device)
#                 test_labels = test_batched_data[1].to(device)
#                 labels.append(int(test_labels.cpu()))
#
#                 test_batched_feats = test_batched_graph.ndata['feats'].to(device)
#                 test_batched_node_type = test_batched_graph.ndata['node_type'].to(device)
#                 test_batched_edge_type = test_batched_graph.edata['edge_type'].to(device)
#
#                 test_positive_batched_feats = test_positive_batched_graph.ndata['feats'].to(device)
#                 test_positive_batched_node_type = test_positive_batched_graph.ndata['node_type'].to(device)
#                 test_positive_batched_edge_type = test_positive_batched_graph.edata['edge_type'].to(device)
#
#                 test_negative_batched_feats = test_negative_batched_graph.ndata['feats'].to(device)
#                 test_negative_batched_node_type = test_negative_batched_graph.ndata['node_type'].to(device)
#                 test_negative_batched_edge_type = test_negative_batched_graph.edata['edge_type'].to(device)
#
#                 all_nodes = np.arange(test_batched_feats.shape[0])
#                 node_seq = torch.zeros(test_batched_feats.shape[0], args.len_seq).long()
#                 n = 0
#                 sampled_idx = []
#                 pre_sampled_idx = random.sample(range(len(all_nodes)), int(len(all_nodes) * 0.12))
#
#                 subgraph = False
#                 for x in pre_sampled_idx:
#                     cnt = 0
#                     scnt = 0
#                     node_seq[n, cnt] = x
#                     cnt += 1
#                     start = node_seq[n, scnt].item()
#                     while cnt < args.len_seq:
#                         sample_list = test_batched_graph.successors(start).cpu().numpy().tolist()
#                         nsampled = len(sample_list)
#                         for i in range(nsampled):
#                             node_seq[n, cnt] = sample_list[i]
#                             cnt += 1
#                             if cnt == args.len_seq:
#                                 break
#                         scnt += 1
#                         if scnt == args.len_seq:
#                             break
#                         start = node_seq[n, scnt].item()
#                     if cnt == args.len_seq:
#                         sampled_idx.append(x)
#                         subgraph = True
#                     n += 1
#                 sampled_idx = random.sample(sampled_idx, int(len(sampled_idx) * 0.9))
#                 if len(sampled_idx) * 30 < len(all_nodes):
#                     subgraph = False
#                 logits = net(test_batched_graph, test_batched_feats, node_seq[sampled_idx], type_emb, edge_type_emb, test_batched_node_type,
#                              test_batched_edge_type, subgraph, args.fine_tune, args.l2norm)
#                 output = F.softmax(logits, 1)
#                 pred = torch.argmax(output, dim=1)
#                 if pred.shape[0] != 1:
#                     all_zeros = torch.all(torch.eq(pred, 0))
#                     if all_zeros == 1:
#                         output_labels.append(0)
#                     else:
#                         mode_value, mode_index = torch.mode(pred)
#                         output_labels.append(int(mode_value))
#                 else:
#                     output_labels.append(int(pred.cpu()))
#
#             print(f'output_labels: {output_labels}')
#             print(f'labels: {labels}')
#             print(metrics.classification_report(output_labels, labels))
#
#         if not os.path.exists('./results'):
#             os.makedirs('./results/')
#         if temp_loss < best_loss:
#             best_loss = temp_loss
#             save_checkpoint({
#                 'epoch': epoch + 1,
#                 'arch': 'SC-GCL',
#                 'loss': best_loss,
#                 'state_dict': net.state_dict(),
#                 'optimizer': optimizer.state_dict(),
#             }, is_best=False, filename=f'./results/train_3/checkpoint_epoch{epoch+1}_hiddendim{args.hidden_dim}'
#                                        f'_layers{args.num_layers}_gnns{args.num_gnns}_lenseq{args.len_seq}'
#                                        f'_batchsize{args.batch_size}_tau{args.tau}_lr{args.lr}'
#                                        f'_headnum{args.num_heads}_loss{best_loss}.pth')
#
# '''
# 1. 实现图增强的代码，生成正样本与负样本
# 2. 构造出损失函数original-positive、original-negative、positive-negative
#
# 训练了30轮才出现明显变化
# '''
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

    # path_list = ['vul_data/hidden_768/timestamp', 'vul_data/hidden_768/reentrancy',
    #              'cleandata/cleandata_768', 'vul_data/hidden_768/access_control']
    # name2label = {'cleandata/cleandata_768': 0, 'vul_data/hidden_768/timestamp': 1,
    #               'vul_data/hidden_768/reentrancy': 2, 'vul_data/hidden_768/access_control': 3}
    # path_list = ['vul_data/hidden_768/reentrancy', #'vul_data/hidden_768/integeroverflow',
    #              'vul_data/hidden_768/timestamp', 'cleandata/cleandata_768']
    # name2label = {'cleandata/cleandata_768': 0, 'vul_data/hidden_768/reentrancy': 1, 'vul_data/hidden_768/timestamp': 2}
    # path_list = ['filtered/overflow', 'filtered/reentrancy',
    #              'filtered/timestamp', 'clean']
    # name2label = {'clean': 0, 'filtered/overflow': 3, 'filtered/reentrancy': 1,
    #               'filtered/timestamp': 2}
    # path = 'E:\\Py_projects\\DataFlow'
    path_list = ['vul_data/hidden_768/filtered/BERT/integeroverflow', 'vul_data/hidden_768/filtered/BERT/reentrancy',
                 'vul_data/hidden_768/filtered/BERT/timestamp', 'cleandata/cleandata_256']
    name2label = {'cleandata/cleandata_256': 0, 'vul_data/hidden_768/filtered/BERT/integeroverflow': 3,
                  'vul_data/hidden_768/filtered/BERT/reentrancy': 1,
                  'vul_data/hidden_768/filtered/BERT/timestamp': 2}
    path = 'E:\\Py_projects\\CrossVulDec'
    print(path_list)
    # path = 'E:\\Py_projects\\CrossVulDec\\one'
    cur_idx = 1
    print(f"start time: {datetime.now()}")

    # cross_path_list = ['cross-reentrancy', 'cross-timestamp', 'cross-access_control']
    # cross_name2label = {'cross-reentrancy': 1, 'cross-timestamp': 2, 'cross-access_control': 3}
    cross_path_list = ['cross-reentrancy', 'cross-timestamp', 'cross-integeroverflow']
    cross_name2label = {'cross-reentrancy': 1, 'cross-timestamp': 2, 'cross-integeroverflow': 3}
    valid_labels = []
    valid_path = os.path.join(path, 'vul_data/hidden_768/cross-contract/BERT')
    valid_graphs, valid_positive_graphs, valid_negative_graphs = [], [], []
    for vul in cross_path_list:
        if vul != 'cross-reentrancy':
            continue
        cur_path = os.path.join(valid_path, vul)
        for project in os.listdir(cur_path):
            sub_valid_graphs, sub_valid_positive_graphs, sub_valid_negative_graphs = [], [], []
            for file in os.listdir(os.path.join(cur_path, project)):
                # print(os.path.join(cur_path, project, file))
                features, adjM, dl = load_data(pretrain=True, path=os.path.join(cur_path, project, file))
                if type(adjM) is int:
                    # print(f'error file:{os.path.join(cur_path, project, file)}')
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
                # valid_labels.append(cross_name2label[vul])

    train_labels = []
    test_labels = []
    for datapath in path_list:
        # if datapath != 'vul_data/hidden_768/filtered/reentrancy':
        #     continue
        # if len(train_graphs) > 40:
        #     break
        if datapath != 'vul_data/hidden_768/filtered/BERT/reentrancy' and datapath != 'cleandata/cleandata_256':
            continue
        for mode in ['train', 'test']:
            cur_path = os.path.join(path, datapath, mode)
            for address in os.listdir(cur_path):
                sub_graphs, sub_positive_graphs, sub_negative_graphs = [], [], []
                for subpath in os.listdir(os.path.join(cur_path, address)):
                    cur_idx += 1
                    features, adjM, dl = load_data(pretrain=True, path=os.path.join(cur_path, address, subpath))
                    if type(adjM) is int:
                        # print(f'error file:{os.path.join(cur_path, address, subpath)}')
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
                        # train_labels.append(name2label[datapath])
                    else:
                        test_graphs.append(dgl.merge(sub_graphs))
                        test_positive_graphs.append(dgl.merge(sub_positive_graphs))
                        test_negative_graphs.append(dgl.merge(sub_negative_graphs))
                        if 'reentrancy' in datapath:
                            test_labels.append(1)
                        else:
                            test_labels.append(0)
                        # test_labels.append(name2label[datapath])
                        if datapath == 'cleandata/cleandata_256':
                            valid_graphs.append(dgl.merge(sub_graphs))
                            valid_positive_graphs.append(dgl.merge(sub_positive_graphs))
                            valid_negative_graphs.append(dgl.merge(sub_negative_graphs))
                            valid_labels.append(name2label[datapath])

    # data_indices = list(range(len(graphs)))
    # train_indices, test_indices = train_test_split(data_indices, test_size=0.2, random_state=2024)
    # print(len(train_indices))
    # print(len(test_indices))
    train_random_idx = random.sample(range(len(train_graphs)), len(train_graphs))
    test_random_idx = random.sample(range(len(test_graphs)), len(test_graphs))
    valid_random_idx = random.sample(range(len(valid_graphs)), len(valid_graphs))

    # sys.exit(0)
    features_list = np.concatenate(features_list, axis=0)
    features_list = [mat2tensor(features)
                     for features in features_list]
    features_list = torch.stack(features_list, dim=0)
    print(f"after processing data time: {datetime.now()}")
    # node_type = np.array(node_type).ravel().tolist()
    # sys.exit(0)
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
    # node_cnt = features_list.shape[0]
    indims = 768
    # indims = 256

    num_type = 13
    # print(f'num_type={num_type}')
    type_emb = torch.eye(num_type).to(device)

    edge_num_type = 10  # 修改
    edge_type_emb = torch.eye(edge_num_type).to(device)

    node_seq = torch.zeros(features_list[0].shape[0], args.len_seq).long()

    net = HINormer(indims, args.hidden_dim, args.len_seq, args.num_layers, args.num_gnns, args.num_heads, args.dropout,
                   temper=args.temperature, node_num_type=num_type, edge_num_type=edge_num_type, beta=args.beta)
    # net = torch.load('./results/model_filter_0.9368421052631579.pth')
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    net.to(device)
    # checkpoint = torch.load('./results/train_2/checkpoint_epoch53_hiddendim128_layers2_gnns3_lenseq30_batchsize1_tau0.3_lr0.0002_headnum8_loss0.8789599524754834_contrasive.pth')
    # checkpoint = torch.load('./results/pretrain/checkpoint_epoch122_hiddendim256_layers2_gnns3_lenseq8_batchsize16_tau0.3_lr0.0002_inChannels_768_headnum8_loss0.03744418395597545.pth')
    checkpoint = torch.load('./results/pretrain/checkpoint_epoch245_hiddendim2048_layers2_gnns3_lenseq8_batchsize16_tau0.3_lr0.0002_inChannels_768_headnum8_loss2.345286667873462.pth')

    model_dict = net.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() if
                       k in model_dict and v.size() == model_dict[k].size()}
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)

    # net.load_state_dict(checkpoint['state_dict'], strict=False)
    # optimizer.load_state_dict(checkpoint['optimizer'])

    best_loss = 100
    best_accuracy = 0

    for epoch in range(150):
        net.train()
        total_loss = []
        start_time = time.time()
        # train_dataloader = tqdm(train_dataloader, desc=f"epoch:{epoch+1} —— Training progress")
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
            # if train_batched_data[1] == 1:
            #     train_labels = torch.FloatTensor([0, 1])
            # else:
            #     train_labels = torch.FloatTensor([1, 0])
            # train_labels = train_labels.to(device)

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

            # if subgraph:
            #     train_labels = torch.full((len(node_seq),), int(train_labels), device='cuda:0')
            # if train_labels.dim() == 3:
            #     train_labels = train_labels.squeeze(0)
            if subgraph:
                # print(train_labels)
                train_labels = train_labels.expand(logits.shape[0], -1)
            else:
                train_labels = train_labels.unsqueeze(0)
            #classification loss
            # loss_fn = torch.nn.CrossEntropyLoss()
            loss_fn = torch.nn.BCEWithLogitsLoss()
            cls_train_loss = loss_fn(logits, train_labels)
            cls_train_positive_loss = loss_fn(positive_logits, train_labels)
            # co_loss = net.InfoNCE_loss_triple(logits, positive_logits, negative_logits, args.tau)
            # co_loss = net.InfoNCE_loss(logits, positive_logits, args.tau)
            # print(f'contrasive loss: {co_loss}')
            in_end_time = time.time()
            loss = cls_train_loss + cls_train_positive_loss * 0.01
            # loss = cls_train_loss
            # loss = net.SumLoss(cls_train_loss, cls_train_positive_loss)  #+ args.alpha * co_loss
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
                # positive_logits, _ = net(test_positive_batched_graph, test_positive_batched_feats, node_seq, type_emb, edge_type_emb,
                #              test_positive_batched_node_type, test_positive_batched_edge_type, subgraph, args.fine_tune, args.l2norm)
                # output1 = F.softmax(logits, 1)
                output1 = torch.sigmoid(logits)
                # output2 = F.softmax(positive_logits, 1)
                # pred1 = torch.argmax(output1, dim=1)
                pred1 = (output1 > 0.5).int()
                # pred2 = torch.argmax(output2, dim=1)
                pred1_list = pred1.tolist()
                # all_zeros1 = torch.all(torch.eq(pred1, 0)).bool()
                if [1] not in pred1_list:
                    output_labels.append(0)
                else:
                    output_labels.append(1)
                    # if int(test_labels.cpu()) in pred1_list:
                    #     output_labels.append(int(test_labels.cpu()))
                    # else:
                    #     count = Counter(pred1_list)
                    #     # 找出出现次数最多的值（众数）
                    #     most_common = count.most_common(1)
                    #     mode, mode_count = most_common[0]
                    #     output_labels.append(int(mode))
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
                    # positive_logits, _ = net(valid_positive_batched_graph, valid_positive_batched_feats, node_seq[sampled_idx],
                    #                       type_emb, edge_type_emb, valid_positive_batched_node_type,valid_positive_batched_edge_type,
                    #                       subgraph, args.fine_tune, args.l2norm)
                    # output1 = F.softmax(logits, 1)
                    output1 = torch.sigmoid(logits)
                    # output2 = F.softmax(positive_logits, 1)
                    # pred1 = torch.argmax(output1, dim=1)
                    pred1 = (output1 > 0.5).int()
                    # pred2 = torch.argmax(output2, dim=1)
                    pred1_list = pred1.tolist()
                    # pred2_list = pred2.tolist()

                    # temp = torch.mean(output1, dim=0)
                    # pred_labels.append(torch.argmax(temp, dim=0).tolist())

                    # all_zeros1 = torch.all(torch.eq(pred1, 0)).bool()
                    if [1] not in pred1_list:
                        output_labels.append(0)
                    else:
                        output_labels.append(1)
                        # if int(valid_labels.cpu()) in pred1_list:
                        #     output_labels.append(int(valid_labels.cpu()))
                        # else:
                        #     count = Counter(pred1_list)
                        #     # 找出出现次数最多的值（众数）
                        #     most_common = count.most_common(1)
                        #     mode, mode_count = most_common[0]
                        #     output_labels.append(int(mode))

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
            }, is_best=False, filename=f'./results/train_5/checkpoint_epoch{epoch + 1}_hiddendim{args.hidden_dim}'
                                       f'_layers{args.num_layers}_gnns{args.num_gnns}_lenseq{args.len_seq}'
                                       f'_batchsize{args.batch_size}_tau{args.tau}_lr{args.lr}'
                                       f'_headnum{args.num_heads}_BERT_accuracy{report_dict["accuracy"]}.pth')
        if report_dict['accuracy'] > best_accuracy:
            best_accuracy = report_dict['accuracy']
            torch.save(net, f'./results/model_filter_{best_accuracy}.pth')

'''
1. 实现图增强的代码，生成正样本与负样本
2. 构造出损失函数original-positive、original-negative、positive-negative
'''