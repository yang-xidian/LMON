import torch
import torch.nn as nn
import pdb

class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)   #双向线性层

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c_x, h_pl, h_mi, s_bias1=None, s_bias2=None):
        # print('c:', c.shape) # (1, 512) -> (512)
        # c_x = torch.unsqueeze(c, 1) # (1, 1, 512)

        # c_x = c_x.expand_as(h_pl) #这一行在cora数据集上得用！

        # c_x = c.expand_as(h_pl) # (2708, 512)

        # print(c_x.shape)
        # pdb.set_trace()
        # print('c_x:', c_x.shape) # (1, 2708, 512)

        #h_pl, h_mi (1, 2708, 512)
        # print(self.f_k(h_pl, c_x).shape) # (1, 2708, 1) -> (2708, 1)
        # pdb.set_trace()
        # sc_1 = torch.squeeze(self.f_k(h_pl, c_x), 2) # (1, 2708)
        # sc_2 = torch.squeeze(self.f_k(h_mi, c_x), 2) # (1, 2708)
        sc_1 = torch.squeeze(self.f_k(h_pl, c_x), 1) #(2708)            #这里squeeze维度压缩，删除掉所有大小为1的维度
        sc_2 = torch.squeeze(self.f_k(h_mi, c_x), 1) #(2708)            #当给定 dim 时，那么只在给定的维度（dimension）上进行压缩操作。
        # print(sc_1.shape)                
        # pdb.set_trace()

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        # logits = torch.cat((sc_1, sc_2), 1)
        logits = torch.cat((sc_1, sc_2)) # 5416
        # print(logits.shape)
        # pdb.set_trace()

        return logits

