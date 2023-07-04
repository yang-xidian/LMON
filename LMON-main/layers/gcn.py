import torch
import torch.nn as nn
import pdb

class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act
        
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    # Shape of seq: (batch, nodes, features)
    #为什么这里没有特征传播函数和特征聚合函数只是对输入做了一层非线性映射
    def forward(self, seq, adj, sparse=False):  #如果没有加sparse=False返回结果为稀疏矩阵（将非0值按位置表示出来）
        # print('seq:', seq.shape) # (2708, 1433)
        seq_fts = self.fc(seq)
        # print('seq_fts', seq_fts.shape) # (2708, 512)
        # print(sparse)
        # print(seq_fts.shape)
        # print((torch.squeeze(seq_fts, 0)).shape)
        # pdb.set_trace()
        if sparse:
            # out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
            # out = torch.spmm(adj, torch.squeeze(seq_fts, 0))
            out = torch.spmm(adj, seq_fts)    #矩阵乘法
        else:
            out = torch.bmm(adj, seq_fts)    #计算两个tensor矩阵乘法
        if self.bias is not None:
            out += self.bias
        # print('out:', out.shape) # (2708, 512)
        return self.act(out)
        # return out

