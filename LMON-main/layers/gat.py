import torch
import torch.nn as nn
import torch_geometric as tg
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch.nn import init
import pdb

# class GAT(torch.nn.Module):
#     def __init__(self, input_dim, feature_dim, hidden_dim, output_dim,
#                  feature_pre=True, layer_num=2, dropout=True, **kwargs):
#         super(GAT, self).__init__()
#         self.feature_pre = feature_pre
#         self.layer_num = layer_num
#         self.dropout = dropout
#         if feature_pre:
#             self.linear_pre = nn.Linear(input_dim, feature_dim)
#             self.conv_first = tg.nn.GATConv(feature_dim, hidden_dim)
#         else:
#             self.conv_first = tg.nn.GATConv(input_dim, hidden_dim)
#         self.conv_hidden = nn.ModuleList([tg.nn.GATConv(hidden_dim, hidden_dim) for i in range(layer_num - 2)])
#         self.conv_out = tg.nn.GATConv(hidden_dim, output_dim)

#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index
#         if self.feature_pre:
#             x = self.linear_pre(x)
#         x = self.conv_first(x, edge_index)
#         x = F.relu(x)
#         if self.dropout:
#             x = F.dropout(x, training=self.training)
#         for i in range(self.layer_num-2):
#             x = self.conv_hidden[i](x, edge_index)
#             x = F.relu(x)
#             if self.dropout:
#                 x = F.dropout(x, training=self.training)
#         x = self.conv_out(x, edge_index)
#         x = F.normalize(x, p=2, dim=-1)
#         return x

class GAT(nn.Module):
    def __init__(self, in_ft, out_ft, act, bias=True):
        super(GAT, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act           #这样写当激活函数选择不是prelu的话还是无法创建激活函数
        self.conv = tg.nn.GATConv(out_ft, out_ft)
        # self.conv = tg.nn.GCNConv(out_ft, out_ft)
        # self.conv = tg.nn.SAGEConv(out_ft, out_ft)
        
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):           #判断m是否为linear层
            torch.nn.init.xavier_uniform_(m.weight.data)  #预防一些参数过大或过小的情况，再保证方差一样的情况下进行缩放，便于计算
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    # Shape of seq: (batch, nodes, features)
    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        # print(seq_fts.shape)
        # print(adj._indices().shape)
        # pdb.set_trace()
        out = self.conv(seq_fts, adj._indices())
        # if sparse:
        #     out = torch.spmm(adj, seq_fts)
        # else:
        #     out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        return self.act(out)
        # return out