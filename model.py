import math

import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
from sklearn.metrics import precision_score
from torch.nn import init

import torch as th
from dgl import function as fn
from dgl._ffi.base import DGLError
from dgl.nn.pytorch import edge_softmax
from dgl.nn.pytorch.utils import Identity
from dgl.utils import expand_as_pair
from torch.nn import init


class WSReadout(nn.Module):

    def __init__(self, embedding_dim):
        super(WSReadout, self).__init__()
        self.embedding_dim = embedding_dim

    def forward(self, seq, query):
        query = query.permute(0, 2, 1)
        sim = torch.matmul(seq, query)
        sim = F.softmax(sim, dim=1)
        sim = sim.repeat(1, 1, self.embedding_dim)
        out = torch.mul(seq, sim)
        out = torch.sum(out, 1)
        print(out.shape)
        return out


class REConv(nn.Module):
    """
    Adapted from
    https://docs.dgl.ai/_modules/dgl/nn/pytorch/conv/gatconv.html#GATConv
    """

    def __init__(self,
                 node_in_feats,
                 node_out_feats,
                 edge_in_feats,
                 edge_out_feats,
                 norm='both',
                 node_num_type=4,
                 edge_num_type=6,
                 weight=True,
                 bias=True,
                 activation=None):
        super(REConv, self).__init__()
        if norm not in ('none', 'both', 'right', 'left'):
            raise DGLError('Invalid norm value. Must be either "none", "both", "right" or "left".'
                           ' But got "{}".'.format(norm))
        self._node_in_feats = node_in_feats
        self._node_out_feats = node_out_feats
        self._edge_in_feats = edge_in_feats
        self._edge_out_feats = edge_out_feats
        self._norm = norm

        if weight:
            self.node_weight = nn.Parameter(th.Tensor(node_in_feats, node_out_feats))
            # self.edge_weight = nn.Parameter(th.Tensor(edge_in_feats, edge_out_feats))
            self.edge_weight = nn.Parameter(th.Tensor(node_in_feats, node_out_feats))
            # self.node_transform = nn.Linear(node_in_feats, node_out_feats)
            # self.edge_transform = nn.Linear(edge_in_feats, node_out_feats)
        else:
            self.register_parameter('weight', None)

        if bias:
            self.bias = nn.Parameter(th.Tensor(node_out_feats))
        else:
            self.register_parameter('bias', None)

        self.node_type_weight = nn.Parameter(th.ones(node_num_type))
        self.edge_type_weight = nn.Parameter(th.ones(edge_num_type))

        self.reset_parameters()

        self._activation = activation

    def reset_parameters(self):
        if self.node_weight is not None:
            init.xavier_uniform_(self.node_weight)
        if self.edge_weight is not None:
            init.xavier_uniform_(self.edge_weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, graph, edge_feat, node_feat, edge_type, node_type):
        with graph.local_scope():
            # aggregate_fn = fn.copy_u('h', 'm')
            aggregate_fn = fn.u_mul_e('h', 'e', 'm')
            # print(f"node shape:{node_feat.shape}, edge shape:{edge_feat.shape}")

            if self._norm in ['left', 'both']:
                degs = graph.out_degrees().float().clamp(min=1)
                if self._norm == 'both':
                    norm = th.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (node_feat.dim() - 1)
                norm = th.reshape(norm, shp)
                feat = node_feat * norm

            feat = th.matmul(feat, self.node_weight)
            graph.srcdata['h'] = feat * self.node_type_weight[node_type].reshape(-1, 1)

            edge_feat = th.matmul(edge_feat, self.edge_weight)
            graph.edata['e'] = edge_feat * self.edge_type_weight[edge_type].reshape(-1, 1)

            graph.update_all(aggregate_fn, fn.sum(msg='m', out='h'))
            rst = graph.dstdata['h']

            if self._norm in ['right', 'both']:
                degs = graph.in_degrees().float().clamp(min=1)
                if self._norm == 'both':
                    norm = th.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat.dim() - 1)
                norm = th.reshape(norm, shp)
                rst = rst * norm

            if self.bias is not None:
                rst = rst + self.bias

            if self._activation is not None:
                rst = self._activation(rst)

            return rst

class AGTLayer(nn.Module):
    def __init__(self, embeddings_dimension, nheads=2, att_dropout=0.5, emb_dropout=0.5, temper=1.0, rl=False, rl_dim=4,
                 beta=1):

        super(AGTLayer, self).__init__()

        self.nheads = nheads
        self.embeddings_dimension = embeddings_dimension

        self.head_dim = self.embeddings_dimension // self.nheads

        self.leaky = nn.LeakyReLU(0.01)

        self.temper = temper

        self.rl_dim = rl_dim

        self.beta = beta

        self.linear_l = nn.Linear(
            self.embeddings_dimension, self.head_dim * self.nheads, bias=False)
        self.linear_r = nn.Linear(
            self.embeddings_dimension, self.head_dim * self.nheads, bias=False)

        self.att_l = nn.Linear(self.head_dim, 1, bias=False)
        self.att_r = nn.Linear(self.head_dim, 1, bias=False)

        if rl:
            self.r_source = nn.Linear(rl_dim, rl_dim * self.nheads, bias=False)
            self.r_target = nn.Linear(rl_dim, rl_dim * self.nheads, bias=False)

        self.linear_final = nn.Linear(
            self.head_dim * self.nheads, self.embeddings_dimension, bias=False)
        self.dropout1 = nn.Dropout(att_dropout)
        self.dropout2 = nn.Dropout(emb_dropout)

        self.LN = nn.LayerNorm(embeddings_dimension)

    def forward(self, h, rh=None):
        batch_size = h.size()[0]
        fl = self.linear_l(h).reshape(batch_size, -1, self.nheads, self.head_dim).transpose(1, 2)
        fr = self.linear_r(h).reshape(batch_size, -1, self.nheads, self.head_dim).transpose(1, 2)

        score = self.att_l(self.leaky(fl)) + self.att_r(self.leaky(fr)).permute(0, 1, 3, 2)

        if rh is not None:
            r_k = self.r_source(rh).reshape(batch_size, -1, self.nheads, self.rl_dim).transpose(1, 2)
            r_q = self.r_target(rh).reshape(batch_size, -1, self.nheads, self.rl_dim).permute(0, 2, 3, 1)
            score_r = r_k @ r_q
            score = score + self.beta * score_r

        score = score / self.temper

        score = F.softmax(score, dim=-1)
        score = self.dropout1(score)

        context = score @ fr

        h_sa = context.transpose(1, 2).reshape(batch_size, -1, self.head_dim * self.nheads)
        fh = self.linear_final(h_sa)
        fh = self.dropout2(fh)

        h = self.LN(h + fh)

        return h


class SubgraphAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super(SubgraphAttention, self).__init__()
        self.self_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model)
        )
        self.dropout = nn.Dropout(dropout)


    def forward(self, readout_h):
        # readout_h: [n, m]
        # 增加一个 batch_size 维度（假设 batch_size=1）
        readout_h = readout_h.unsqueeze(1)  # [n, m] -> [n, 1, m]
        readout_h = readout_h.permute(0, 1, 2)  # [n, 1, m] -> [n, 1, m]
        attn_output, _ = self.self_attention(readout_h, readout_h, readout_h)  # [n, 1, m]

        readout_h = readout_h + self.dropout(attn_output)
        readout_h = self.norm(readout_h)

        ff_output = self.ffn(readout_h)  # [n, 1, m]
        readout_h = readout_h + self.dropout(ff_output)
        readout_h = self.norm(readout_h)

        readout_h = readout_h.squeeze(1)  # [n, 1, m] -> [n, m]
        return readout_h


class CrossAttention(nn.Module):
    def __init__(self, dim, attn_drop=0.1, proj_drop=0.1):
        super(CrossAttention, self).__init__()
        self.embedding_dim = dim
        self.scale = self.embedding_dim ** -0.5
        self.wq = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.wk = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.wv = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.proj = nn.Dropout(proj_drop)
        self.layer_norm = nn.LayerNorm(self.embedding_dim)

    def forward(self, x, y):
        """
            x: 表示全局图嵌入  [1, embedding_dim]
            y: 表示子图嵌入   [n_subgraphs, embedding_dim]
        """
        q = self.wq(x)  # [1, embedding_dim]
        k = self.wk(y)  # [n_subgraphs, embedding_dim]
        v = self.wv(y)  # [n_subgraphs, embedding_dim]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [1, n_subgraphs]
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        subgraph_h = v * attn.T
        subgraph_h = self.proj(subgraph_h)
        subgraph_h = self.attn_drop(subgraph_h)
        subgraph_h = self.layer_norm(q + subgraph_h)
        return subgraph_h



class HINormer(nn.Module):
    def __init__(self, input_dimensions, embeddings_dimension=64, len_seq=30, num_layers=8, num_gnns=2, nheads=2,  # embeddings_dimension是否可以更大些
                 dropout=0, temper=1.0, node_num_type=4, edge_num_type=6, beta=1, num_subgraph_attention_layers=1):

        super(HINormer, self).__init__()

        self.embeddings_dimension = embeddings_dimension
        self.num_layers = num_layers
        self.num_gnns = num_gnns
        self.nheads = nheads
        self.leaky = nn.LeakyReLU(0.01)
        self.embed_type = 128
        # self.fc = nn.Linear(input_dimensions, embeddings_dimension)
        self.fc = nn.Sequential(
            nn.Linear(input_dimensions, embeddings_dimension),
            nn.ReLU(inplace=True)
        )
        # self.node_transform = nn.Sequential(
        #     nn.Linear(node_num_type, self.embed_type),
        #     self.leaky
        # )
        # self.edge_transform = nn.Sequential(
        #     nn.Linear(edge_num_type, self.embed_type),
        #     self.leaky
        # )
        self.node_transform = nn.Linear(node_num_type, self.embed_type)
        self.edge_transform = nn.Linear(edge_num_type, self.embed_type)
        self.len_seq = len_seq
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.layer_norm_node = nn.LayerNorm(self.embeddings_dimension)
        self.layer_norm_graph = nn.LayerNorm(self.embeddings_dimension)

        self.dropout = dropout
        self.GCNLayers = torch.nn.ModuleList()
        self.RELayers = torch.nn.ModuleList()
        self.GTLayers = torch.nn.ModuleList()

        for layer in range(self.num_gnns):
            self.GCNLayers.append(GraphConv(
                self.embeddings_dimension, self.embeddings_dimension, activation=F.relu,
                allow_zero_in_degree=True))  # 由于存在入度为0的点，所以添加参数allow_zero_in_degree=0   self.embed_type
            # self.RELayers.append(REConv(node_num_type, node_num_type, edge_num_type, edge_num_type, activation=F.relu,
            #                             node_num_type=node_num_type, edge_num_type=edge_num_type))
            self.RELayers.append(REConv(self.embed_type, self.embed_type, self.embed_type, self.embed_type, activation=F.relu,
                                        node_num_type=node_num_type, edge_num_type=edge_num_type))
        for layer in range(self.num_layers):
            self.GTLayers.append(
                AGTLayer(self.embeddings_dimension, self.nheads, self.dropout, self.dropout, temper=temper, rl=True,
                         rl_dim=self.embed_type, beta=beta))
        self.Drop = nn.Dropout(self.dropout)
        self.expand = nn.Linear(self.embeddings_dimension, 2 * self.embeddings_dimension)

        # self.Node_W_q = nn.Linear(self.embeddings_dimension, self.embeddings_dimension)
        # self.Node_W_k = nn.Linear(self.embeddings_dimension, self.embeddings_dimension)
        # self.Node_W_v = nn.Linear(self.embeddings_dimension, self.embeddings_dimension)
        # self.attn_node = nn.Sequential(
        #     nn.Linear(2 * self.embeddings_dimension, self.embeddings_dimension),  # 拼接后的维度
        #     self.leaky,
        #     self.Drop,
        #     nn.Linear(self.embeddings_dimension, 1) 
        # )

        self.num_subgraph_attention_layers = num_subgraph_attention_layers
        self.subgraph_attention_layers = nn.ModuleList([
            SubgraphAttention(self.embeddings_dimension, self.nheads) for _ in range(self.num_subgraph_attention_layers)
        ])

        self.crossAttn = CrossAttention(self.embeddings_dimension, attn_drop=dropout, proj_drop=dropout)

        # self.attention_query = nn.Parameter(torch.randn(1, self.embeddings_dimension))
        # self.Graph_W_q = nn.Linear(self.embeddings_dimension, self.embeddings_dimension)
        # self.Graph_W_k = nn.Linear(self.embeddings_dimension, self.embeddings_dimension)
        # self.Graph_W_v = nn.Linear(self.embeddings_dimension, self.embeddings_dimension)
        # self.attn_graph = nn.Sequential(
        #     nn.Linear(2 * self.embeddings_dimension, self.embeddings_dimension),  # 拼接后的维度
        #     self.leaky,
        #     self.Drop,
        #     nn.Linear(self.embeddings_dimension, 1)  # 输出一个标量分数
        # )

        self.fc_fusion = nn.Sequential(
            nn.Linear(2 * self.embeddings_dimension, 4 * self.embeddings_dimension),
            self.leaky,
            self.Drop,
            nn.Linear(self.embeddings_dimension * 4, self.embeddings_dimension * 2)
        )#nn.Linear(2 * self.embeddings_dimension, self.embeddings_dimension)

        self.Prediction = nn.Sequential(
            nn.Linear(self.embeddings_dimension * 2, self.embeddings_dimension * 3),
            nn.ReLU(),
            self.Drop,
            nn.Linear(self.embeddings_dimension * 3, self.embeddings_dimension // 2),
            nn.ReLU(),
            nn.LayerNorm(self.embeddings_dimension // 2),
            nn.Linear(self.embeddings_dimension // 2, 1)
        )
        # self.Prediction_3 = nn.Linear(self.embeddings_dimension, 3)

    def forward(self, g, features_list, seqs, node_type_emb, edge_type_emb, node_type, edge_type, subgraph,
                fine_tune=False, norm=False):
        h = self.fc(features_list)
        gh = h
        node_r = node_type_emb[node_type]
        edge_r = edge_type_emb[edge_type]

        # 统一尺寸###############################
        node_r = self.node_transform(node_r)
        edge_r = self.edge_transform(edge_r)
        ########################################

        for layer in range(self.num_gnns):
            gh = self.GCNLayers[layer](g, gh)
            gh = self.Drop(gh)
            node_r = self.RELayers[layer](g, edge_r, node_r, edge_type, node_type)  # 节点类型特征
            # 加入激活函数
            # node_r = self.leaky(node_r)
            # r = self.RELayers[layer](g, node_r, node_type)

        full_h = gh.unsqueeze(0)
        full_r = node_r.unsqueeze(0)
        if subgraph:
            h = gh[seqs]
            r = node_r[seqs]
        else:
            h = gh.unsqueeze(0)
            r = node_r.unsqueeze(0)

        for layer in range(self.num_layers):
            h = self.GTLayers[layer](h, rh=r)
            if subgraph:
                full_h = self.GTLayers[layer](full_h, rh=full_r)

        readout_h = self.readout_with_attention(h)
        # print(readout_h.shape)
        if subgraph:
            readout_full_h = self.readout_with_attention(full_h)
            # 子图间注意力机制
            for layer in self.subgraph_attention_layers:
                readout_h = layer(readout_h)
            subgraph_h = self.crossAttn(readout_full_h, readout_h)
            readout_h = torch.cat([subgraph_h, readout_full_h.expand(subgraph_h.size(0), -1)], dim=-1)
            readout_h = self.fc_fusion(readout_h)
        else:
            readout_h = self.expand(readout_h)
        if fine_tune:
            # readout_h = self.leaky(readout_h)
            cls_res = self.Prediction(readout_h)
            # if norm:
            #     cls_res = cls_res / (torch.norm(cls_res, dim=1, keepdim=True) + 1e-12)
            return cls_res, readout_h
        return readout_h

    def readout_with_attention(self, h):
        return torch.mean(h, dim=1)
        # 注意力读出代码
        Q = self.Node_W_q(h)  # [batch_size, seq_len, hidden_dim]
        K = self.Node_W_k(h)  # [batch_size, seq_len, hidden_dim]
        V = self.Node_W_v(h)  # [batch_size, seq_len, hidden_dim]

        attention_scores = torch.matmul(Q, K.transpose(-2, -1))  # [batch_size, seq_len, seq_len]
        attention_scores = torch.bmm(Q, K.transpose(-2, -1))
        attention_scores = attention_scores / (self.embeddings_dimension ** 0.5)  # 缩放
        attention_weights = F.softmax(attention_scores, dim=-1)
        readout_vector = torch.matmul(attention_weights, V)  # [batch_size, seq_len_q, hidden_dim]
        readout_vector = readout_vector + Q
        readout_vector = self.layer_norm_node(readout_vector)
        readout_vector = readout_vector.mean(dim=1)
        return readout_vector

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        if z1.dim() == 1:
            z1 = z1.unsqueeze(0)  # 将 [N] 变为 [1, N]
        if z2.dim() == 1:
            z2 = z2.unsqueeze(0)  # 将 [N] 变为 [1, N]
        z1 = F.normalize(z1, dim=1, p=2)
        z2 = F.normalize(z2, dim=1, p=2)
        return torch.mm(z1, z2.t())

    # InfoNCE 损失函数
    def InfoNCE_loss(self, z1: torch.Tensor, z2: torch.Tensor, tau):  # tau——温度系数，作用就是控制模型对负样本的区分度
        def f(x):
            return torch.exp(x / tau)

        # f = lambda x: torch.exp(x / tau)
        between_sim = f(self.sim(z1, z2))
        refl_sim1 = f(self.sim(z1, z1))

        return (-torch.log(
            (between_sim.diag() + refl_sim1.diag())
            / (refl_sim1.sum(1) + between_sim.sum(1)))).mean()

    # InfoNCE 损失函数，origin, positive, negative
    def InfoNCE_loss_triple(self, z1: torch.Tensor, z2: torch.Tensor, z3: torch.Tensor,
                            tau):  # z1: 原始数据, z2: 正样本, z3: 负样本
        def f(x):
            return torch.exp(x / tau)

        # 计算相似度
        pos_sim = f(self.sim(z1, z2))  # 原始数据与正样本
        neg_sim = f(self.sim(z1, z3))  # 原始数据与负样本
        ori_sim = f(self.sim(z1, z1))  # 原始数据与原始数据

        # 计算正样本的对数概率
        # 正样本的相似度除以所有样本相似度的总和
        # temp = -torch.log(pos_sim.diag() / (pos_sim.sum(1) + neg_sim.sum(1)))
        loss = -torch.log(
            (pos_sim.diag() + ori_sim.diag()) / (ori_sim.sum(1) + pos_sim.sum(1) + neg_sim.sum(1))
        ).mean()

        return loss

    def SumLoss(self, original, positive):
        return original + self.alpha * positive
