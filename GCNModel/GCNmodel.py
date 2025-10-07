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
        self_loop_edge_attr = torch.zeros(x.size(0), edge_attr.size(1)).to(edge_attr.device) #
        edge_attr = torch.cat([edge_attr, self_loop_edge_attr], dim=0) #
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        # 拼接源节点特征、目标节点特征和边特征
        print("xi的大小",x_i.shape, x_j.shape, edge_attr.shape)
        tmp = torch.cat([x_i, x_j, edge_attr], dim=1)
        print(tmp.shape)
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