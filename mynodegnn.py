from torch_geometric.nn import MessagePassing
from myedgegnn import myedgeGNN
from torch import nn

class mynodeGNN(MessagePassing):
    def __init__(self, emb_dim):
        '''
                    emb_dim (int): nodes and edges embedding dimensionality
                '''
        super(mynodeGNN, self).__init__(aggr='add')  #  "add" aggregation.

        self.lin = nn.Linear(emb_dim, emb_dim,bias=False)#线性化层

        self.mlp1 = nn.Sequential(nn.Linear(emb_dim, 2*emb_dim),
                   nn.ReLU(),
                   nn.Linear(2*emb_dim, emb_dim))#全连接神经网络

        self.edge_attr = myedgeGNN(emb_dim)

    def forward(self, x, edge_index, edge_attr):

        Edge_attr = self.edge_attr(x=x,edge_index=edge_index,edge_attr=edge_attr)

        m = self.lin(x) + self.propagate(edge_index, x=x, edge_attr= Edge_attr)

        output =x+m

        return output

    def message(self, x_j, edge_attr):

        return x_j * self.mlp1(edge_attr)

    def update(self, aggr_out):

        return aggr_out

