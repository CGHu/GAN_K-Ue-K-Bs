import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_undirected
import numpy as np
import pandas as pd
# from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# 加载邻接矩阵数据
adj_matrix = pd.read_csv('m_d.csv', header=None, index_col=None).values

# 将邻接矩阵转换为张量，这里要将列索引加上总节点数，使得列节点的索引不与行节点重复
num_nodes = adj_matrix.shape[0] + adj_matrix.shape[1]
edge_index_pos = np.column_stack(np.argwhere(adj_matrix != 0))
edge_index_pos[:, 1] += adj_matrix.shape[0]
edge_index_pos = torch.tensor(edge_index_pos, dtype=torch.long)

# 获取不相关连接 为了构建训练数据
edge_index_neg = np.column_stack(np.argwhere(adj_matrix == 0))
edge_index_neg[:, 1] += adj_matrix.shape[0]
edge_index_neg = torch.tensor(edge_index_neg, dtype=torch.long)

# 获取平衡样本
num_pos_edges_number = edge_index_pos.shape[1]
selected_neg_edge_indices = torch.randint(high=edge_index_neg.shape[1], size=(num_pos_edges_number,), dtype=torch.long)
edge_index_neg_selected = edge_index_neg[:, selected_neg_edge_indices]

edg_index_all = torch.cat((edge_index_pos, edge_index_neg_selected), dim=1)

# 创建数据
x = torch.ones((num_nodes, 1)) # 没有节点特征，所以设置为1
y = torch.cat((torch.ones((edge_index_pos.shape[1], 1)),
               torch.zeros((edge_index_neg_selected.shape[1], 1))), dim=0) # 将所有y值设置为1,0

# 将邻接矩阵转换为无向图（会将每条边复制成两条相反的边）
edge_index = to_undirected(edge_index_pos)

# 将数据拆分为训练、验证和测试集
idx = np.arange(y.shape[0])
np.random.shuffle(idx)
train_idx = idx[:int(0.8 * len(idx))] # 用前80%的样本作为训练集
test_idx = idx[int(0.8 * len(idx)):] # 用后20%的样本作为测试集


# 创建一个GCN模型
class GCNII(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNII, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True)
        self.conv2 = GCNConv(2 * out_channels, out_channels)
        self.fc = torch.nn.Linear(out_channels*2, 1)

    def forward(self, x, edge_index, edge_index_pos, edge_index_neg, edge_weight=None,param={}):
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.relu(self.conv2(x, edge_index, edge_weight))

        # 按照边的顺序将节点特征向量重新排列
        edg_index_all = torch.cat((edge_index_pos, edge_index_neg), dim=1)
        Em, Ed = self.pro_data(x, edg_index_all)  # 筛选数据 获得源节点特征和目标节点

        # 将源节点特征和目标节点特征拼接起来
        x = torch.cat((Em, Ed),dim=1)
        x = self.fc(x)

        return x

    def pro_data(self, x, edg_index):
        m_index = edg_index[0]
        d_index = edg_index[1]
        Em = torch.index_select(x, 0, m_index)  # 沿着x1的第0维选出m_index
        Ed = torch.index_select(x, 0, d_index)
        return Em, Ed


# 模型构建
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCNII(in_channels=x.shape[1], out_channels=16).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)

# 训练模型
model.train()
for epoch in range(1, 201):
    optimizer.zero_grad()
    out = model(x, edge_index, edge_index_pos=edge_index_pos, edge_index_neg=edge_index_neg_selected,
                param={'m_number': adj_matrix.shape[0], 'd_number': adj_matrix.shape[1]})

    # 使用train数据进行训练
    loss = F.binary_cross_entropy_with_logits(out[train_idx], y[train_idx].float())
    loss.backward()
    optimizer.step()
    loss = loss.item()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

# 模型验证
model.eval()
with torch.no_grad():
    # 获得所有数据
    out = model(x, edge_index, edge_index_pos=edge_index_pos, edge_index_neg=edge_index_neg_selected,
                param={'m_number':adj_matrix.shape[0],'d_number':adj_matrix.shape[1]})

    # 提取验证集数据
    out_pred = out[test_idx]
    y_pred = y[test_idx]

    # 计算AUC
    auc = roc_auc_score(y_pred, out_pred)
    print('AUC:', auc)
