import torch
from mynodegnn import mynodeGNN
import torch.nn.functional as F
from torch import nn

# myGNN to generate node and edge embedding
class mynodeEmbedding(torch.nn.Module):
    """
    Output:
        node representations
    """

    def __init__(self, num_layers, emb_dim, drop_ratio, JK="last", residual=False):
        """myGNN Node Embedding Module"""

        super(mynodeEmbedding, self).__init__()

        # num_layers (int, optional): number of GINConv layers. Defaults to 2.
        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        self.JK = JK

        # add residual connection or not
        self.residual = residual
        self.sw1 = nn.Sequential(nn.Linear(12, 27),
                                 nn.ReLU(),
                                 nn.Linear(27, 32))
        self.sw2 = nn.Sequential(nn.Linear(3, 9),
                                 nn.ReLU(),
                                 nn.Linear(9, 32))

        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

    # emb_dim (int, optional): dimension of node embedding. Defaults to 300.

        # List of GNNs
        self.convs = torch.nn.ModuleList()#简单的说，就是把子模块存储在list中
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(num_layers):
            self.convs.append(mynodeGNN(emb_dim))#列表 append() 方法用于在列表末尾追加新的对象
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))#BatchNorm就是在深度神经网络训练过程中使得每一层神经网络的输入保持相同分布的

    def forward(self, batched_data):
        x, edge_index, edge_attr = batched_data.x, batched_data.edge_index, batched_data.edge_attr
        x = self.sw1(x)
        edge_attr = self.sw2(edge_attr)
        h_list =[x]

        # computing input node embedding
        for layer in range(self.num_layers):
            h = self.convs[layer](h_list[layer], edge_index,edge_attr )
            h = self.batch_norms[layer](h)
            if layer == self.num_layers - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

            if self.residual:
                h += h_list[layer]

            h_list.append(h)

        # Different implementations of Jk-concat
        # JK (str, optional): 可选的值为"last"和"sum"。选"last"，只取最后一层的结点的嵌入，选"sum"对各层的结点的嵌入求和。Defaults to "last".
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":#if…elif的逻辑是，程序先走if，能走就走，走完就不走elif了，走不通的情况才走elif
            node_representation = 0
            for layer in range(self.num_layers + 1):
                node_representation += h_list[layer]

        return node_representation #9*32的矩阵

