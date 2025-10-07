import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import torch.nn.functional as F

class EdgeFeatureMessagePassing(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(EdgeFeatureMessagePassing, self).__init__(aggr='add')  # 使用求和聚合
        self.lin = torch.nn.Linear(in_channels * 2 + 1, out_channels)  # 考虑节点特征拼接和边特征

    def forward(self, x, edge_index, edge_attr):
        # 添加自环以考虑节点自身信息
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        # 复制边特征以匹配自环
        self_loop_edge_attr = torch.zeros(x.size(0), edge_attr.size(1)).to(edge_attr.device)
        edge_attr = torch.cat([edge_attr, self_loop_edge_attr], dim=0)

        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        # 拼接源节点特征、目标节点特征和边特征
        tmp = torch.cat([x_i, x_j, edge_attr], dim=1)
        print(".....",tmp.shape)
        return self.lin(tmp)

    def update(self, aggr_out):
        # 可以在这里添加额外的操作，如激活函数
        return F.relu(aggr_out)


class GraphModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphModel, self).__init__()
        self.conv1 = EdgeFeatureMessagePassing(in_channels, hidden_channels)
        self.conv2 = EdgeFeatureMessagePassing(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        return F.log_softmax(x, dim=1)


from torch_geometric.data import Data

# 模拟节点特征
x = torch.randn(10, 16)  # 10 个节点，每个节点特征维度为 16
# 模拟边索引
edge_index = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.long)
# 模拟边特征
edge_attr = torch.randn(4, 1)  # 4 条边，每条边特征维度为 1
# 模拟节点标签
y = torch.randint(0, 2, (10,))

data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

# 初始化模型
model = GraphModel(in_channels=16, hidden_channels=32, out_channels=2)
# 定义优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# 定义损失函数，使用点特征计算损失
criterion = torch.nn.NLLLoss()

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out, data.y)
    loss.backward()
    optimizer.step()
    return loss.item()

for epoch in range(200):
    loss = train()
    print(f'Epoch: {epoch+1}, Loss: {loss:.4f}')