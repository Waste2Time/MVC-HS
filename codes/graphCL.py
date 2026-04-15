from Generator import GNN
from torch import nn


class MLP(nn.Module):
    def __init__(self, attrs_dim, dim_list=[16, 8, 2]):
        super(MLP, self).__init__()

        attrs_dim = [attrs_dim]
        attrs_dim.extend(dim_list)
        self.layers = nn.ModuleList([nn.Linear(attrs_dim[i], attrs_dim[i+1]) for i in range(len(dim_list))])
        self.activation = nn.Tanh()

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            x = self.activation(x)
        return x


class GraphCL(nn.Module):
    def __init__(self, args, attrs_dim):
        super(GraphCL, self).__init__()
        self.attrs_dim = attrs_dim
        if attrs_dim >= 32:
            self.mlp = MLP(attrs_dim, [2*attrs_dim, attrs_dim, 32])
        self.pos_gnn = GNN(args, attrs_dim)
        self.neg_gnn = GNN(args, attrs_dim)

    def forward(self, origianl_batch, pp_batch, fm_batch):
        pos_original_embeddings = self.pos_gnn(origianl_batch)
        neg_original_embeddings = self.neg_gnn(origianl_batch)
        pp_embeddings = self.neg_gnn(pp_batch)
        fm_embeddings = self.neg_gnn(fm_batch)

        if self.attrs_dim >= 32:
            pos_original_embeddings = self.mlp(pos_original_embeddings)
            neg_original_embeddings = self.mlp(neg_original_embeddings)
            pp_embeddings = self.mlp(pp_embeddings)
            fm_embeddings = self.mlp(fm_embeddings)

        return pos_original_embeddings, neg_original_embeddings, pp_embeddings, fm_embeddings