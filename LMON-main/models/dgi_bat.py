import torch
import torch.nn as nn
from layers import GCN, AvgReadout, Discriminator, LSTMContext, LinearModel, GAT, FNN
import pdb
import torch_geometric as tg

class DGI(nn.Module):
    def __init__(self, n_in, n_h, activation, subgraph):
        super(DGI, self).__init__()
        # self.gcn = GCN(n_in, n_h, activation)
        # self.gcn = LinearModel(n_in, n_h)
        # self.gcn = tg.nn.GATConv(n_in, n_h)
        # self.gcn = GAT(n_in, n_h, activation)

        self.subgraph = subgraph

        self.gcn = LSTMContext(obj_classes=7, hidden_dim=n_h, nhidlayer=2, nl_edge=2, dropout=0.4, in_channels=n_in)
        # self.gcn = LSTMContext(obj_classes=7, hidden_dim=n_h, nhidlayer=2, nl_edge=1, dropout=0.4, in_channels=n_in)

        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(n_h)

        self.feature_decoder = FNN(n_h, n_h, n_in, 3)

        self.feature_loss_func = nn.MSELoss()

    def forward2(self, seq1, seq2, adj, sparse, msk, samp_bias1, samp_bias2):
        h_1 = self.gcn(seq1, adj, sparse) # (1, 2708, 512) -> (2708, 512)

        # h_1, c_out = self.gcn(seq1, self.subgraph, adj)
        # h_1 = self.gcn(seq1, adj._indices())

        c = self.read(h_1, msk) # (1, 512) #(512)
        c = self.sigm(c)
        h_2 = self.gcn(seq2, adj, sparse) # (2708, 512)
        # h_2, _ = self.gcn(seq2, self.subgraph, adj)
        # h_2 = self.gcn(seq2, adj._indices())
        # print(c.shape)
        # print(c_out.shape)
        # pdb.set_trace()
        ret = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)
        # ret = self.disc(c_out, h_1, h_2, samp_bias1, samp_bias2)

        return ret
    
    def forward(self, seq1, subgraph2, adj, sparse, msk, samp_bias1, samp_bias2):
        # h_1 = self.gcn(seq1, adj, sparse) # (1, 2708, 512) -> (2708, 512)
        h_1, c_out = self.gcn(seq1, self.subgraph, adj)
        # h_1 = self.gcn(seq1, adj._indices())

        c = self.read(h_1, msk) # (1, 512) #(512)
        c = self.sigm(c)
        # h_2 = self.gcn(seq2, adj, sparse) # (2708, 512)
        h_2, _ = self.gcn(seq1, subgraph2, adj)
        # h_2 = self.gcn(seq2, adj._indices())
        # print(c.shape)
        # print(c_out.shape)
        # pdb.set_trace()
        # ret = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)
        ret = self.disc(c_out, h_1, h_2, samp_bias1, samp_bias2) #discriminator得到的输出，用于计算loss的

        # Decoder 部分
        feature_loss = self.feature_loss_func(seq1, self.feature_decoder(h_1)) # feature层面的decoder

        return ret, feature_loss

    # Detach the return variables
    def embed(self, seq, adj, sparse, msk):
        # h_1 = self.gcn(seq, adj, sparse)
        h_1, _ = self.gcn(seq, self.subgraph, adj)
        # h_1 = self.gcn(seq, adj._indices())

        c = self.read(h_1, msk)

        return h_1.detach(), c.detach()

class DGI2(nn.Module):
    def __init__(self, n_in, n_h, activation):
        super(DGI2, self).__init__()
        self.gcn = GCN(n_in, n_h, activation)
        # self.gcn = LinearModel(n_in, n_h)
        # self.gcn = tg.nn.GATConv(n_in, n_h)
        # self.gcn = GAT(n_in, n_h, activation)

        # self.gcn = LSTMContext(obj_classes=7, hidden_dim=n_h, nhidlayer=2, nl_edge=2, dropout=0.5, in_channels=n_in)

        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(n_h)

    def forward(self, seq1, seq2, adj, sparse, msk, samp_bias1, samp_bias2):
        h_1 = self.gcn(seq1, adj, sparse) # (1, 2708, 512) -> (2708, 512)
        # h_1 = self.gcn(seq1, self.subgraph, adj)
        # h_1 = self.gcn(seq1, adj._indices())

        c = self.read(h_1, msk) # (1, 512) #(512)
        c = self.sigm(c)
        h_2 = self.gcn(seq2, adj, sparse) # (2708, 512)
        # h_2 = self.gcn(seq2, self.subgraph, adj)
        # h_2 = self.gcn(seq2, adj._indices())

        ret = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)

        return ret

    # Detach the return variables
    def embed(self, seq, adj, sparse, msk):
        h_1 = self.gcn(seq, adj, sparse)
        # h_1 = self.gcn(seq, self.subgraph, adj)
        # h_1 = self.gcn(seq, adj._indices())

        c = self.read(h_1, msk)

        return h_1.detach(), c.detach()

