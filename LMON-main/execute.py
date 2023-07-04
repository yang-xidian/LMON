import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from models import DGI, LogReg, DGI2
from utils import process, path, args
import pdb
import random

#dataset = 'cora'

#dataset = 'cornell'
#dataset = 'texas'
#dataset = 'wisconsin'
dataset = 'film'

# training params
batch_size = 1
nb_epochs = 10000
patience = 20
lr = 0.001
l2_coef = 0.0
# l2_coef = 5e-04
drop_prob = 0.0
#hid_units = 512
hid_units = 32
sparse = True
nonlinearity = 'prelu' # special name to separate parameters

args = args.make_args()


device = torch.device('cuda')
#device = torch.device('cpu')

# seed = 8
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# np.random.seed(seed)

# torch.cuda.manual_seed_all(seed)
# random.seed(seed)
# torch.backends.cudnn.deterministic = True


adj, features, labels, idx_train, idx_val, idx_test = process.load_data(dataset) #邻接矩阵，特征，标签，训练集，验证集，测试集
#features, _ = process.preprocess_features(features)

#print(adj)
#print(labels.size)
#print(idx_train)
#print(idx_val)
#print(idx_test)

# edge_index = adj._indices() # train_adj本身就是以稀疏矩阵的方式存储的，因此直接调用indices()就能得到边了

#对图中每个节点计算random walk
subgraph = path.get_target_random_walks(args, adj)
print(subgraph[0])
#print(subgraph)

# print(idx_train)
# print(idx_test)
# print(features.shape) #(2708, 1433)
# pdb.set_trace()

nb_nodes = features.shape[0]   #节点数量
ft_size = features.shape[1]    #特征大小
# print(nb_nodes)
# print(ft_size)
#nb_classes = labels.shape[1]
nb_classes = max(labels) + 1


adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))  #对称归一化矩阵


if sparse:
    sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj)
else:
    adj = (adj + sp.eye(adj.shape[0])).todense()  #adj + sp.eye(adj.shape[0])表示加入自连接
# print(features[np.newaxis].shape)

#features = torch.FloatTensor(features[np.newaxis]) #np.newaxis是给features前面加上一个维度
features = torch.FloatTensor(features)

if not sparse:
    adj = torch.FloatTensor(adj[np.newaxis])
    #adj = torch.FloatTensor(adj)
#labels = torch.FloatTensor(labels[np.newaxis])
#labels = torch.FloatTensor(labels)

idx_train = torch.LongTensor(idx_train)
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)


labels = torch.LongTensor(labels)
def one_hot(x, class_count):
    return torch.eye(class_count)[x,:]
labels = one_hot(labels, nb_classes)

# print(labels.shape) # (1, 2708, 7) 改完变成 (2708, 7)
# print(adj.shape) # (2708, 2708)
# print(sp_adj.shape) #稀疏矩阵 (2708, 2708)
# pdb.set_trace()

# model = DGI(hid_units, hid_units, nonlinearity, subgraph)
model = DGI(ft_size, hid_units, nonlinearity, subgraph)
optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)
# optimiser = torch.optim.SGD(model.parameters(), lr = lr, momentum = 1)

if torch.cuda.is_available():
    print('Using CUDA')
    # model.cuda()
    model.to(device)
    # features = features.cuda()
    features = features.to(device)
    if sparse:
        # sp_adj = sp_adj.cuda()
        sp_adj = sp_adj.to(device)
    else:
        # adj = adj.cuda()
        adj = adj.to(device)
    # labels = labels.cuda()
    # idx_train = idx_train.cuda()
    # idx_val = idx_val.cuda()
    # idx_test = idx_test.cuda()
    labels = labels.to(device)
    idx_train = idx_train.to(device)
    idx_val = idx_val.to(device)
    idx_test = idx_test.to(device)

b_xent = nn.BCEWithLogitsLoss() #二分类交叉熵损失函数，只能解决二分类问题 （前面会加上sigmoid函数）
xent = nn.CrossEntropyLoss() #交叉熵损失函数，用于解决多分类问题 （内部会自动加上softmax层）
cnt_wait = 0
best = 1e9
best_t = 0

# print(features.shape)
# print(adj.shape)
# pdb.set_trace()

lbl_1 = torch.ones(nb_nodes) #(1, 2708)
lbl_2 = torch.zeros(nb_nodes)
lbl = torch.cat((lbl_1, lbl_2)) # (1, 5416)

# lbl = lbl.cuda()
lbl = lbl.to(device)


for epoch in range(nb_epochs):
# for epoch in range(2):
    model.train()
    optimiser.zero_grad()

    idx = np.random.permutation(nb_nodes)

    #shuf_fts = features[:, idx, :]
    shuf_fts = features[idx, :]
    # print(features.shape)
    # print(shuf_fts.shape)
    # pdb.set_trace()

    # lbl_1 = torch.ones(batch_size, nb_nodes) #(1, 2708)
    # lbl_2 = torch.zeros(batch_size, nb_nodes)
    # lbl = torch.cat((lbl_1, lbl_2), 1) # (1, 5416)
    # print(idx)
    
    tmp = [[i] + [random.randint(0, nb_nodes - 1) for j in range(args.walk_length - 1)] for i in range(nb_nodes)] #随机构造节点游走序列

    # tmp = [[random.randint(0, nb_nodes - 1) for j in range(args.walk_length)] for i in range(nb_nodes)] #随机构造节点游走序列
    # print(tmp[0])
    # print(len(tmp), len(tmp[0]))
    # pdb.set_trace()

    if torch.cuda.is_available():
        # shuf_fts = shuf_fts.cuda()
        # lbl = lbl.cuda()
        shuf_fts = shuf_fts.to(device)
        lbl = lbl.to(device)
    
    # logits = model(features, shuf_fts, sp_adj if sparse else adj, sparse, None, None, None) #(1, 5416)
    logits, loss_f = model(features, tmp, sp_adj if sparse else adj, sparse, None, None, None) #(1, 5416)

    # print(logits.shape) #(5416)
    # print(lbl.shape) #(5416)
    loss_d = b_xent(logits, lbl) #前2708是原图得到的特征(positive 1)，后2708是扰动特征的图得到的特征(negative 0)
    
    loss = loss_f + loss_d
    # loss = loss_d
    #loss = loss_f #不可行 只用这一个效果不佳

    print('epoch: {} Loss: {}'.format(epoch, loss))
    # print(logits)
    # pdb.set_trace()

    if loss < best:
        best = loss
        best_t = epoch
        cnt_wait = 0
        torch.save(model.state_dict(), dataset + '_best_dgi.pkl')
    else:
        cnt_wait += 1

    if cnt_wait == patience:
        print('Early stopping!')
        break

    loss.backward()
    optimiser.step()


print('Loading {}th epoch'.format(best_t))
model.load_state_dict(torch.load(dataset + '_best_dgi.pkl'))

embeds, _ = model.embed(features, sp_adj if sparse else adj, sparse, None) # (1, 2708, 512)
torch.save(embeds,'cornell_file')
torch.save(labels,'cornell_labels')

print('embeds类型',type(embeds))
print(embeds)
print(embeds.shape) #(2708, 512)

print(embeds[0])
print(features[0])
print(sum(embeds[0]))
print('sum feature:', sum(features[0]))
# pdb.set_trace()

train_embs = embeds[idx_train]
val_embs = embeds[idx_val]
test_embs = embeds[idx_test]

# print(labels.shape)
# pdb.set_trace()

# train_lbls = torch.argmax(labels[0, idx_train], dim=1)
# val_lbls = torch.argmax(labels[0, idx_val], dim=1)
# test_lbls = torch.argmax(labels[0, idx_test], dim=1)

# print(labels[idx_test])

train_lbls = torch.argmax(labels[idx_train], dim=1)
val_lbls = torch.argmax(labels[idx_val], dim=1)
test_lbls = torch.argmax(labels[idx_test], dim=1)


# print(val_lbls)
# pdb.set_trace()

# print(test_embs.shape) # (1000, 512)
# print(test_lbls.shape) # (1000, )
# print(labels[0].shape) # (2708, 7)
# print(labels[0, idx_test].shape) # (1000, 7)
# pdb.set_trace()

tot = torch.zeros(1)
# tot = tot.cuda()
tot = tot.to(device)

accs = []

for _ in range(10): #10次实验取平均
    log = LogReg(hid_units, nb_classes) # linear model 512 -> 7
    opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)
    # log.cuda()
    log.to(device)

    pat_steps = 0
    best_acc = torch.zeros(1)
    # best_acc = best_acc.cuda()
    best_acc = best_acc.to(device)
    for _ in range(7): #texas
    #for _ in range(9):
        log.train()
        opt.zero_grad()

        logits = log(train_embs)
        loss = xent(logits, train_lbls) # 这里用到label了！！！

        # print(train_lbls)
        # print(loss)
        # pdb.set_trace()

        loss.backward()
        opt.step()

    logits = log(test_embs)
    preds = torch.argmax(logits, dim=1)
    print(preds)
    print(test_lbls)
    acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
    accs.append(acc * 100)
    print(acc)
    tot += acc

print('Average accuracy:', tot / 10)

accs = torch.stack(accs)
print(accs.mean())
print(accs.std())

