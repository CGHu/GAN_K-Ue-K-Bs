import torch
from torch_geometric.utils import to_undirected
import numpy as np
import pandas as pd
# from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from torch_geometric.data import Data
from GCNmodel import GraphModel

row_data = torch.from_numpy(np.load("../data/demo_K12_10.npy"))[0,:,:]#
row_data = row_data.real.float()
print("原始数据",row_data.shape)

K = 12
N = 10
N_T = 16

#构建节点特证 （N+K）* N_T
g_Node = torch.zeros((row_data.shape[0],K, N_T)) #
g_Node = torch.zeros((K, N_T))
N_Node = row_data[:, 0:16] #
x = torch.cat((N_Node, g_Node), dim = 0)
print("节点特征",x.shape)

#构建边特征 edge * 1
edge_attr = torch.flatten(row_data[:, 16:28], start_dim = 0) #
edge_attr = edge_attr.reshape(-1,1)
print("边特征（1（1……k），……，N（1……k）",edge_attr.shape)

#构建边索引 2 * edge
edge_index_list = []
for i in range(N):
    for j in range(K):
        edge_index_list.append([i, N + j])

edge_index = torch.tensor(edge_index_list).t()
print("边索引",edge_index.shape)

data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


# 初始化模型
model = GraphModel(in_channels=N_T, hidden_channels=32, out_channels=16)
# 定义优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# 定义损失函数，使用点特征计算损失
criterion = torch.nn.NLLLoss()

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out)
    loss.backward()
    optimizer.step()
    return loss.item()

model.train()
optimizer.zero_grad()
out = model(data)
print("模型输出",out.shape,out)

# for epoch in range(200):
#     loss = train()
#     print(f'Epoch: {epoch+1}, Loss: {loss:.4f}')