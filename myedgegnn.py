import torch
from torch_geometric.nn import MessagePassing
from torch import nn

class myedgeGNN(MessagePassing):
    def __init__(self, emb_dim):
        '''
                    emb_dim (int): nodes and edges embedding dimensionality
                '''
        super(myedgeGNN, self).__init__(aggr=None)  #  "add" aggregation.

        self.lin = nn.Linear(emb_dim, emb_dim,bias=False)#线性化层

        self.mlp = nn.Sequential(nn.Linear(2*emb_dim, 4*emb_dim),
                   nn.ReLU(),
                   nn.Linear(4*emb_dim, emb_dim))#全连接神经网络


    def forward(self, x, edge_index, edge_attr):

        m = self.lin(edge_attr) + self.propagate(edge_index, x= x)

        output=m+edge_attr

        return output

    def message(self, x_i, x_j):
        tmp = torch.cat((x_i, x_j), dim=1)

        return self.mlp(tmp)

    def aggregate(self, inputs):
        return inputs
    
    def update(self, aggr_out):

        return aggr_out

