import torch
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence,pad_sequence, pack_padded_sequence, pack_sequence
import random
import pdb

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from layers import * 

import torch_geometric as tg

class LSTMContext(nn.Module):
    # def __init__(self,  obj_classes, hidden_dim, nhidlayer, nl_edge, dropout, in_channels, device):
    def __init__(self,  obj_classes, hidden_dim, nhidlayer, nl_edge, dropout, in_channels):
        super(LSTMContext, self).__init__()
        self.obj_classes = obj_classes
        self.obj_dim = in_channels
        self.hidden_dim = hidden_dim
        self.hidden_layer = nhidlayer
        self.dropout_rate = dropout
        self.nl_edge = nl_edge
        # self.device = device

        activation = F.relu
        withbn = False
        withloop = False

        self.obj_ctx_rnn = torch.nn.LSTM(            #如果bidirectional = True则代表是双向LSTM层
            input_size = self.obj_dim,
            # input_size = self.hidden_dim,
            hidden_size = self.hidden_dim,
            num_layers = self.hidden_layer,
            dropout = self.dropout_rate,
            bidirectional = True
        )

        self.decoder_rnn = torch.nn.LSTM(
            input_size = self.hidden_dim + self.obj_dim,
            # input_size = self.hidden_dim + self.hidden_dim,
            # input_size = self.hidden_dim,
            hidden_size = self.hidden_dim,
            num_layers = self.hidden_layer,
            dropout = self.dropout_rate,
            bidirectional = False
        )

        self.edge_ctx_rnn = torch.nn.LSTM(
            input_size = self.obj_dim + self.hidden_dim,
            hidden_size = self.hidden_dim,
            num_layers = self.nl_edge,
            dropout = self.dropout_rate,
            bidirectional = True
        )

        self.lin_obj_h = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.lin_edge_h = nn.Linear(self.hidden_dim * 2, self.hidden_dim)

        self.disc = Discriminator(self.hidden_dim)

        self.act = nn.PReLU()
        self.act2 = nn.PReLU()
        # self.gnn = tg.nn.GATConv(self.hidden_dim, self.hidden_dim)
        self.gnn = tg.nn.GCNConv(in_channels, self.hidden_dim)

        self.gnn2 = tg.nn.GCNConv(self.hidden_dim, self.hidden_dim)

        # self.out_layer = Dense(self.hidden_dim, obj_classes)
        self.lin1 = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.lin2 = nn.Linear(self.hidden_dim, self.hidden_dim)

    def obj_ctx2(self, obj_feas, subgraph):
    # def obj_ctx(self, obj_feas, original_fea, subgraph):
        #subgraph: 长度为7600的一个list: [72, 29, 36,....]
        perm = sorted(range(len(subgraph)), key=lambda i:len(subgraph[i]), reverse=True)
        subgraph.sort(key=lambda x:len(x), reverse=True)

        # new_input = torch.cat((obj_feas, original_fea), 1) #把gcn得到的32维特征和original feature拼接
        # new_input = obj_feas #这里只使用gcn得到的32维特征

        self.length = [] #记录子图的节点数
        self.arr = []
        for val in subgraph:
            self.length.append(len(val))
            self.arr.append(obj_feas[val]) #self.arr里每个元素都是一个tensor，它是由tensor组成的list
            # self.arr.append(new_input[val]) #self.arr里每个元素都是一个tensor，它是由tensor组成的list
        
        max_length = max(self.length) #所有子图中最大的节点数

        arr_pad = pad_sequence(self.arr, batch_first=True) # 三维tensor (7600, 72, D)，对tensor做padding
        
        arr_pack = pack_padded_sequence(arr_pad, self.length, batch_first=True) #把0去掉的二维tensor () * D
        output = self.decoder_rnn(arr_pack)
        output_unpack = pad_packed_sequence(output[0], batch_first=True) #把添加0后的padding tensor还原，三维(7600, 72, hidden_dim * 2)
        
        obj_new_feature = []
        for v in output_unpack[0]: #这里用的是简单做法，把每个序列中的第0个向量直接作为该目标节点的特征
            obj_new_feature.append(v[0])
        
        obj_new_feature = torch.stack(obj_new_feature, dim=0) #得到二维向量 N * (hidden * 2)

        return obj_new_feature

        # return obj_new_feature #不考虑后面两层LSTM了，只用第一层
        
        new_input = torch.cat((obj_new_feature, obj_feas), 1) #将新特征和旧特征拼接，作为第二个LSTM的输入
        
        # new_input = torch.cat((obj_new_feature, original_fea), 1)

        self.arr = []
        for val in subgraph: #把新得到的特征，根据子图subgraph进行划分
            self.arr.append(new_input[val])
        
        arr_new_pad = pad_sequence(self.arr, batch_first=True)
        arr_new_pack = pack_padded_sequence(arr_new_pad, self.length, batch_first=True)
        output = self.decoder_rnn(arr_new_pack)

        output_unpack = pad_packed_sequence(output[0], batch_first=True) #把添加0后的padding tensor还原，三维(7600, 72, hidden)
        output_lstm = []
        
        for v in output_unpack[0]: #这里用的是简单做法，把每个序列中的第0个向量直接作为该目标节点的特征
            output_lstm.append(v[0])

        output_lstm = torch.stack(output_lstm) #将lstm输出的tensor组合成list，然后用stack转成tensor

        # c_out = torch.sum(output_unpack[0], 1) #这个是把游走路径包含的所有节点取均值

        # print(output_lstm.shape)
        # c_out = torch.sum(output_lstm, 0)
        # print(c_out.shape)

        # pdb.set_trace()

        # return output_lstm, c_out #前者是节点的特征，后者是均值特征(7600, hidden)
        return output_lstm

    #该函数有多个return应该有问题
    def obj_ctx(self, obj_feas, subgraph):
    # def obj_ctx(self, obj_feas, original_fea, subgraph):
        #subgraph: 长度为7600的一个list: [72, 29, 36,....]
        #下面这两行是对序列长度做一个排序处理，因为担心各个序列长度不等，所以按从大到小的顺序先排列
        # perm = sorted(range(len(subgraph)), key=lambda i:len(subgraph[i]), reverse=True)
        # subgraph.sort(key=lambda x:len(x), reverse=True)

        # new_input = torch.cat((obj_feas, original_fea), 1) #把gcn得到的32维特征和original feature拼接
        # new_input = obj_feas #这里只使用gcn得到的32维特征

        self.length = [] #记录子图的节点数
        self.arr = []
        for val in subgraph:
            self.length.append(len(val))
            self.arr.append(obj_feas[val]) #self.arr里每个元素都是一个tensor，它是由tensor组成的list
            # self.arr.append(new_input[val]) #self.arr里每个元素都是一个tensor，它是由tensor组成的list
        
        max_length = max(self.length) #所有子图中最大的节点数

        arr_pad = pad_sequence(self.arr, batch_first=True) # 三维tensor (7600, 72, D)
        arr_pack = pack_padded_sequence(arr_pad, self.length, batch_first=True) #把0去掉的二维tensor () * D

        output = self.obj_ctx_rnn(arr_pack)  #双向LSTM
        output_unpack = pad_packed_sequence(output[0], batch_first=True) #把添加0后的padding tensor还原，三维(7600, 72, hidden_dim * 2)

        obj_new_feature = []
        for v in output_unpack[0]: #这里用的是简单做法，把每个序列中的第0个向量直接作为该目标节点的特征
            obj_new_feature.append(v[0])

        
        obj_new_feature = torch.stack(obj_new_feature, dim=0) #得到二维向量 N * (hidden * 2)
        obj_new_feature = self.lin_obj_h(obj_new_feature) #得到 N * hidden的特征维度   一层liner

        # obj_new_feature = self.lin1(obj_new_feature)
        # obj_new_feature = self.lin2(obj_new_feature)

        # c_out = torch.sum(output_unpack[0], 1) #这个是把游走路径包含的所有节点取均值
        c_out = torch.mean(output_unpack[0], 1) #这个是把游走路径包含的所有节点取均值
        c_out = self.lin_edge_h(c_out)
        # print(c_out.shape)
        # pdb.set_trace()

        # return obj_new_feature, c_out #这个返回的话，相当于只考虑了第一层LSTM的输出
        #return obj_new_feature, None, c_out

        # return obj_new_feature #不考虑后面两层LSTM了，只用第一层
        
        new_input = torch.cat((obj_new_feature, obj_feas), 1) #将新特征和旧特征拼接，作为第二个LSTM的输入
        
        # new_input = torch.cat((obj_new_feature, original_fea), 1)

        self.arr = []
        for val in subgraph: #把新得到的特征，根据子图subgraph进行划分
            self.arr.append(new_input[val])
        
        arr_new_pad = pad_sequence(self.arr, batch_first=True)
        arr_new_pack = pack_padded_sequence(arr_new_pad, self.length, batch_first=True)
        output = self.decoder_rnn(arr_new_pack)     #LSTM

        output_unpack = pad_packed_sequence(output[0], batch_first=True) #把添加0后的padding tensor还原，三维(7600, 72, hidden)
        output_lstm = []
        
        for v in output_unpack[0]: #这里用的是简单做法，把每个序列中的第0个向量直接作为该目标节点的特征
            output_lstm.append(v[0])

        output_lstm = torch.stack(output_lstm) #将lstm输出的tensor组合成list，然后用stack转成tensor

        c_out = torch.mean(output_unpack[0], 1) #这个是把游走路径包含的所有节点取均值


        return obj_new_feature, output_lstm, c_out #obj_new_feature是双向LSTM层和一层linear层后输出的节点的特征，output_lstm再经过一层LSTM输出的节点特征，c_out是取均值后的特征

    #  return output_lstm

    def edge_ctx(self, inp_feats, subgraph):
        self.arr = []
        for val in subgraph:
            self.arr.append(inp_feats[val])
        arr_pad = pad_sequence(self.arr, batch_first=True)
        arr_pack = pack_padded_sequence(arr_pad, self.length, batch_first=True) #把0去掉的二维tensor () * (hidden + D)
        output = self.edge_ctx_rnn(arr_pack)   
        output_unpack = pad_packed_sequence(output[0], batch_first=True) #返回的是个元组，元组第一个值是一个序列，把添加0后的padding tensor还原，三维(7600, 72, hidden * 2)
        
        obj_new_feature = []
        for v in output_unpack[0]: # 这里我只使用了每个序列的第一个节点特征，其他节点都不考虑，不知道这么做好不好？
            obj_new_feature.append(v[0])

        obj_new_feature = torch.stack(obj_new_feature, dim=0) # 二维tensor (7600, hidden * 2)
        output_edge = self.lin_edge_h(obj_new_feature)
        return output_edge

    def forward(self, x, subgraph, adj):

        obj_feas, obj_feas2, c_out = self.obj_ctx(x, subgraph) # (7600, hidden) #采用这个
        # obj_feas = self.gnn(obj_feas, adj._indices())
        # obj_feas = self.gnn(x, adj._indices())
        # print(obj_feas.shape)
        # pdb.set_trace()
        
        # obj_feas = self.lin2(obj_feas)
        # obj_feas = self.act(obj_feas)

        # obj_feas = self.obj_ctx(obj_feas, subgraph)
        # obj_feas = self.act(obj_feas)

        # obj_feas = self.gnn2(obj_feas, adj._indices())
        # obj_feas = self.lin2(obj_feas)

        

        # #底下这两行是有监督得到节点类别
        # out_class = self.out_layer(obj_feas, None) # (7600, num_classes)
        # out_class = F.log_softmax(out_class, dim=1) # (7600, num_classes)
        # return out_class, obj_feas

        # return obj_feas
        
        # return self.act(obj_feas), c_out
        # return self.act(obj_feas), self.act(obj_feas2), c_out
        return self.act(obj_feas), obj_feas, c_out
