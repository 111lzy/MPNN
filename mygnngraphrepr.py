import torch
from torch import nn
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
from mynodeembedding import mynodeEmbedding
# from myedgeembedding import myedgeEmbedding


class mygnnGraphRepr(nn.Module):

    def __init__(self, num_tasks=2, num_layers=2, emb_dim=32, residual=False, drop_ratio=0.5, JK="last", graph_pooling="set2set"):
        """GNN Graph Pooling Module
        Args:
            num_tasks (int, optional): number of labels to be predicted. Defaults to 1 (控制了图表征的维度，dimension of graph representation).
            num_layers (int, optional): number of GINConv layers. Defaults to 2.
            emb_dim (int, optional): dimension of node and edge  embedding. Defaults to 32.
            residual (bool, optional): adding residual connection or not. Defaults to False.
            drop_ratio (float, optional): dropout rate. Defaults to 0.5.
            JK (str, optional): 可选的值为"last"和"sum"。选"last"，只取最后一层的结点的嵌入，选"sum"对各层的结点的嵌入求和。Defaults to "last".
            graph_pooling (str, optional): pooling method of node embedding. 可选的值为"sum"，"mean"，"max"，"attention"和"set2set"。 Defaults to "set2set".

        Out:
            graph representation
        """
        super(mygnnGraphRepr, self).__init__()

        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.softmax = nn.Softmax(dim=1)

        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn_node = mynodeEmbedding(num_layers, emb_dim, JK=JK, drop_ratio=drop_ratio, residual=residual)
        # self.gnn_edge = myedgeEmbedding(num_layers, emb_dim, JK=JK, drop_ratio=drop_ratio, residual=residual)#?

        # Pooling function to generate whole-graph embeddings
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn=nn.Sequential(
                nn.Linear(emb_dim, emb_dim), nn.BatchNorm1d(emb_dim), nn.ReLU(), nn.Linear(emb_dim, 1)))
        elif graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps=2, num_layers = 1)
        else:
            raise ValueError("Invalid graph pooling type.")

        if graph_pooling == "set2set":
            self.graph_pred_linear = nn.Sequential(nn.Linear(2*self.emb_dim, 4*self.emb_dim),
                                     nn.ReLU(),
                                     nn.Linear(4*self.emb_dim, self.num_tasks ))
        else:
            self.graph_pred_linear = nn.Sequential(nn.Linear(self.emb_dim, 2*self.emb_dim),
                                     nn.ReLU(),
                                     nn.Linear(2*self.emb_dim, self.num_tasks ))

    def forward(self, batched_data):
        h_node = self.gnn_node(batched_data)#9*32
        h_graph = self.pool( h_node,batched_data.batch)#1*64？
        output = self.graph_pred_linear(h_graph)#1*2?
        out = self.softmax(output)
        return out

    def predict(self,batched_data):
        # pred = self.softmax(self.forward(batched_data))
        pred = self.forward(batched_data)
        ans = []
        for t in pred:
            if t[0] > t[1]:
                ans.append(0)
            else:
                ans.append(1)
        predict=torch.tensor(ans)
        return predict


