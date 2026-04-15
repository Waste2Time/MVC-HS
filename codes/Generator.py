from torch import nn
from torch.nn import ParameterList
from torch.nn.parameter import Parameter
import torch
from torch_geometric.nn import GCNConv, GINConv
from torch_scatter import scatter_mean
import copy


class GNN(nn.Module):
    def __init__(self, args, attrs_dim):
        super(GNN, self).__init__()

        self.args = args
        if args.gnn == 'GCN':
            self.gnn_layers = nn.ModuleList([GCNConv(attrs_dim, attrs_dim) for i in range(args.gnn_layers_num)])
        if args.gnn == 'GIN':
            self.gnn_layers = nn.ModuleList([GINConv(MLP(attrs_dim, [2*attrs_dim, 2*attrs_dim, attrs_dim])) for i in range(args.gnn_layers_num)])
        self.activation = nn.Tanh()

    def forward(self, data):
        x = data.attrs.float()
        for i in range(self.args.gnn_layers_num):
            x = self.gnn_layers[i](x, data.edge_index)
            x = self.activation(x)
        x = scatter_mean(x, data.batch, dim=0)
        return x


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


class Predictor(nn.Module):
    def __init__(self, args, attrs_dim):
        super(Predictor, self).__init__()

        self.gnn = GNN(args, attrs_dim)
        self.mlp = MLP(attrs_dim)
        # self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, data):
        x = self.gnn(data)
        graph_embedding = x
        x = self.mlp(x)
        # x = self.logsoftmax(x)
        return x, graph_embedding


class Generator(nn.Module):
    def __init__(self, args, graphs_num, nodes_num_list, attrs_dim):
        super(Generator, self).__init__()

        self.args = args
        self.attrs_dim = attrs_dim

        self.predictor = Predictor(args, attrs_dim)

        self.perturbation_matrices = ParameterList([Parameter(torch.FloatTensor(nodes_num_list[i], nodes_num_list[i])) for i in range(graphs_num)])
        for each in self.perturbation_matrices:
            #print(each.data)
            torch.nn.init.xavier_uniform_(each.data, gain=5/3)
        self.perturbation_biases = ParameterList([Parameter(torch.FloatTensor(nodes_num_list[i], nodes_num_list[i])) for i in range(graphs_num)])
        for each in self.perturbation_biases:
            torch.nn.init.xavier_uniform_(each.data, gain=5/3)
        self.masking_matrices = ParameterList([Parameter(torch.FloatTensor(nodes_num_list[i], attrs_dim)) for i in range(graphs_num)])
        for each in self.masking_matrices:
            torch.nn.init.xavier_uniform_(each.data, gain=5/3)

    def forward(self, batch):
        perturbation_matrices = tuple([self.perturbation_matrices[id] for id in batch.id])
        perturbation_matrices = torch.block_diag(*perturbation_matrices)

        perturbation_biases = tuple([self.perturbation_biases[id] for id in batch.id])
        perturbation_biases = torch.block_diag(*perturbation_biases)

        masking_matrices = [self.masking_matrices[int(id)] for id in batch.id]
        masking_matrices = torch.cat(masking_matrices, dim=0)

        batch_perturbation = copy.deepcopy(batch)
        batch_masking = copy.deepcopy(batch)

        values = torch.Tensor([1 for i in range(batch.edge_index.size()[1])])
        if self.args.cuda:
            values = values.cuda()
        adjs = torch.sparse_coo_tensor(batch.edge_index, values, (batch.num_nodes, batch.num_nodes), dtype=torch.float)
        adjs_dense = adjs.to_dense()
        perturbation_adjs = torch.mm(perturbation_matrices, adjs_dense)+perturbation_biases
        perturbation_adjs = torch.sigmoid(perturbation_adjs)
        perturbation_adjs = torch.where(perturbation_adjs<=0.5, torch.zeros_like(perturbation_adjs), torch.ones_like(perturbation_adjs))
        perturbation_adjs_sparse = perturbation_adjs.to_sparse()
        batch_perturbation.edge_index = perturbation_adjs_sparse.indices()

        masking_matrices = torch.sigmoid(masking_matrices)
        masking_matrices = torch.where(masking_matrices<=0.5, torch.zeros_like(masking_matrices), torch.ones_like(masking_matrices))
        masked_attrs = torch.mul(masking_matrices, batch.attrs)
        batch_masking.attrs = masked_attrs

        predicted_results, _ = self.predictor(batch)
        perturbation_predicted_results, _ = self.predictor(batch_perturbation)
        masking_predicted_results, _ = self.predictor(batch_masking)

        return adjs_dense, perturbation_adjs, masking_matrices, predicted_results, perturbation_predicted_results, masking_predicted_results, batch_perturbation, batch_masking