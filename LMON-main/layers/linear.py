import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg
import pdb

#包含一层线性层和一层GAT层
class LinearModel(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LinearModel, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        self.act = nn.PReLU()

        for m in self.modules():
            self.weights_init(m)

        self.gnn = tg.nn.GATConv(nb_classes, nb_classes)   
        # self.gnn = tg.nn.GCNConv(nb_classes, nb_classes)     


    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj, sparse):
        ret = self.fc(seq)
        # print(ret.shape)
        # pdb.set_trace()
        ret = self.gnn(ret, adj._indices())
        return self.act(ret)

