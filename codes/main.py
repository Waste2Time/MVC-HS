import argparse
import torch
import numpy as np
from tqdm import tqdm
import torch_geometric
from torch_geometric.loader import DataLoader
from graphCL import GraphCL
from Generator import Generator, Predictor
import pickle
import os
from sklearn.metrics import accuracy_score, f1_score
import time
import copy
import random
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.svm import SVC
import math

def load_data(args):
    data_folder_path = '../data/'+args.dataset

    adjs_path = data_folder_path+'/'+args.dataset+'_A.csv'
    graph_indicators_path = data_folder_path+'/'+args.dataset+'_graph_indicator.csv'
    graph_labels_path = data_folder_path+'/'+args.dataset+'_graph_labels.csv'
    node_attributes_path = data_folder_path+'/'+args.dataset+'_node_attributes.csv'

    adjs = np.loadtxt(adjs_path, dtype=int, delimiter=',')
    graph_indicators = np.loadtxt(graph_indicators_path, dtype=int, delimiter=',')
    graph_labels = np.loadtxt(graph_labels_path, dtype=int, delimiter=',')
    node_attributes = np.loadtxt(node_attributes_path, delimiter=',')

    return adjs, graph_indicators, graph_labels, node_attributes

def load_nx_graphs(args):
    data_path = '../data/'+args.dataset+'/nx_graphs.pkl'
    with open(data_path, 'rb') as f:
        graphs_list = pickle.load(f)
    return graphs_list

def graph_labels_process(args, graph_labels):
    processed_labels = [[0, 0] for i in range(len(graph_labels))]
    for i in range(len(graph_labels)):
        if args.dataset == 'PROTEINS_full':
            if graph_labels[i] == 1:
                processed_labels[i] = 0
            else:
                processed_labels[i] = 1
        elif args.dataset == 'FRANKENSTEIN':
            if graph_labels[i] == 1:
                processed_labels[i] = 1
            else:
                processed_labels[i] = 0
        elif args.dataset == 'AIDS':
            if graph_labels[i] == 1:
                processed_labels[i] = 1
            else:
                processed_labels[i] = 0
    return processed_labels

def generation(args, graphs_num, nodes_num_list, attrs_dim, data):
    dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=True)
    dataloader = [batch for batch in dataloader]
    if args.cuda:
        dataloader = [batch.cuda() for batch in dataloader]
    
    generator = Generator(args, graphs_num, nodes_num_list, attrs_dim)
    if args.cuda:
        generator.cuda()

    optimizer = torch.optim.Adam(
        params=generator.parameters(),
        lr=args.generation_lr
    )

    data_perturbation_list = []
    data_masking_list = []

    pbar = tqdm(range(args.generation_epochs))
    for epoch in pbar:
        pbar.set_description('Hard Negative Samples Generation Epoch %d...' % epoch)

        for batch in dataloader:
            optimizer.zero_grad()

            adjs_dense, perturbation_adjs, masking_matrix, predicted_results, perturbation_predicted_results, masking_predicted_results, data_perturbation, data_masking = generator(batch)
            sim_loss = similarity_loss(adjs_dense, perturbation_adjs, masking_matrix)
            kl_loss = kl_div(predicted_results, perturbation_predicted_results, masking_predicted_results)
            l = sim_loss-kl_loss

            l.backward()
            optimizer.step()

            pbar.set_postfix(sim_loss=sim_loss.item(), kl_loss=kl_loss.item())

    pbar = tqdm(range(len(data)))
    pbar.set_description('Augmented Graphs Generation...')
    for i in pbar:
        each = data[i]
        if args.cuda:
            each = each.cuda()

        each_p = copy.deepcopy(each)
        each_m = copy.deepcopy(each)

        p_matrix = generator.perturbation_matrices[each.id]
        p_bias = generator.perturbation_biases[each.id]
        m_matrix = generator.masking_matrices[each.id]

        values = torch.Tensor([1 for i in range(each.edge_index.size()[1])])
        if args.cuda:
            values = values.cuda()
        adjs = torch.sparse_coo_tensor(each.edge_index, values, (each.num_nodes, each.num_nodes), dtype=torch.float)
        adjs_dense = adjs.to_dense()
        perturbation_adjs = torch.mm(p_matrix, adjs_dense)+p_bias
        perturbation_adjs = torch.sigmoid(perturbation_adjs)
        perturbation_adjs = torch.where(perturbation_adjs<=args.gamma, torch.zeros_like(perturbation_adjs), torch.ones_like(perturbation_adjs))
        # print(perturbation_adjs)
        perturbation_adjs_sparse = perturbation_adjs.to_sparse()
        # print(perturbation_adjs_sparse)
        each_p.edge_index = perturbation_adjs_sparse.indices()
        data_perturbation_list.append(each_p)

        masking_matrices = torch.sigmoid(m_matrix)
        masking_matrices = torch.where(masking_matrices<=args.gamma, torch.zeros_like(masking_matrices), torch.ones_like(masking_matrices))
        masked_attrs = torch.mul(masking_matrices, each.attrs)
        each_m.attrs = masked_attrs
        data_masking_list.append(each_m)

    return data_perturbation_list, data_masking_list
        
def similarity_loss(adjs_dense, perturbation_adjs, masking_matrix):
    diff = adjs_dense-perturbation_adjs
    diff_norm = torch.linalg.matrix_norm(diff, ord=1)/torch.ones_like(diff).sum()
    masking_matrix_norm = torch.linalg.matrix_norm(masking_matrix, ord=1)/torch.ones_like(masking_matrix).sum()
    return diff_norm-masking_matrix_norm

def kl_div(predicted_results, perturbation_predicted_results, masking_predicted_results):
    predicted_results = predicted_results.softmax(dim=1)
    perturbation_predicted_results = perturbation_predicted_results.log_softmax(dim=1)
    masking_predicted_results = masking_predicted_results.log_softmax(dim=1)
    loss_func = torch.nn.KLDivLoss(reduction='batchmean')
    kl_loss_1 = loss_func(perturbation_predicted_results, predicted_results)
    kl_loss_2 = loss_func(masking_predicted_results, predicted_results)

    return kl_loss_1+kl_loss_2

def save_generated_data(args, fn, data):
    path = '../data/'+args.dataset+'/'+fn
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def load_generated_data(args, fn):
    path = '../data/'+args.dataset+'/'+fn
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def cl_train(args, attrs_dim, original_data, pp_data, fm_data):
    random.seed(args.random_seed)
    randnum = random.randint(0,100)
    random.seed(randnum)
    random.shuffle(original_data)
    random.seed(randnum)
    random.shuffle(pp_data)
    random.seed(randnum)
    random.shuffle(fm_data)

    original_dataloader = DataLoader(original_data, batch_size=args.batch_size)
    pp_dataloader = DataLoader(pp_data, batch_size=args.batch_size)
    fm_dataloader = DataLoader(fm_data, batch_size=args.batch_size)
    if args.cuda:
        original_dataloader = [batch.cuda() for batch in original_dataloader]
        pp_dataloader = [batch.cuda() for batch in pp_dataloader]
        fm_dataloader = [batch.cuda() for batch in fm_dataloader]
    else:
        original_dataloader = [batch.cpu() for batch in original_dataloader]
        pp_dataloader = [batch.cpu() for batch in pp_dataloader]
        fm_dataloader = [batch.cpu() for batch in fm_dataloader]

    model = GraphCL(args, attrs_dim)
    if args.cuda:
        model.cuda()
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=args.pre_lr
    )

    graph_embeddings = []
    graph_labels = []

    pbar = tqdm(range(args.pre_epochs))
    for epoch in pbar:
        for i in range(len(original_dataloader)):
            pbar.set_description('Graph Contrastive Learning Epoch %d Batch %d...' % (epoch, i))

            original_batch = original_dataloader[i]
            pp_batch = pp_dataloader[i]
            fm_batch = fm_dataloader[i]

            optimizer.zero_grad()

            pos_original_embeddings, neg_original_embeddings, pp_embeddings, fm_embeddings = model(original_batch, pp_batch, fm_batch)
            loss = cl_loss(pos_original_embeddings, neg_original_embeddings, pp_embeddings, fm_embeddings)

            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=loss.item())

            if epoch == args.pre_epochs-1:
                graph_embeddings.append(pos_original_embeddings)
                graph_labels.append(original_batch.label)

    graph_embeddings = torch.cat(graph_embeddings, dim=0).cpu().detach().numpy()
    graph_labels = np.concatenate(graph_labels)

    return graph_embeddings, graph_labels

def cl_loss(pos_original_embeddings, neg_original_embeddings, pp_embeddings, fm_embeddings):
    pos_pair = torch.cosine_similarity(pos_original_embeddings, neg_original_embeddings)
    neg_pair_pp = torch.cosine_similarity(pos_original_embeddings, pp_embeddings)
    neg_pair_fm = torch.cosine_similarity(pos_original_embeddings, fm_embeddings)

    pos_pair = torch.exp(pos_pair/args.temp)
    neg_pair_pp = torch.exp(neg_pair_pp/args.temp)
    neg_pair_fm = torch.exp(neg_pair_fm/args.temp)

    loss = -torch.log(pos_pair/(pos_pair+neg_pair_pp+neg_pair_fm)).mean()

    return loss

def svc_classify(x, y, search):
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    accuracies = []
    f1mis = []
    f1mas = []
    for train_index, test_index in kf.split(x, y):

        # test
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)
        if search:
            params = {'C':[0.001, 0.01,0.1,1,10,100,1000]}
            classifier = GridSearchCV(SVC(), params, cv=5, scoring='accuracy', verbose=0)
        else:
            classifier = SVC(C=10)
        classifier.fit(x_train, y_train)
        accuracies.append(accuracy_score(y_test, classifier.predict(x_test)))
        f1mis.append(f1_score(y_test, classifier.predict(x_test), average="micro"))
        f1mas.append(f1_score(y_test, classifier.predict(x_test), average="macro"))

    return f1mis, f1mas

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--random_seed', type=int, default=12345)
    parser.add_argument('--dataset', type=str, default='FRANKENSTEIN')
    parser.add_argument('--gnn_layers_num', type=int, default=3)
    parser.add_argument('--gnn', type=str, default='GIN')
    parser.add_argument('--generation_lr', type=float, default=1e-4)
    parser.add_argument('--pre_lr', type=float, default=1e-4)
    parser.add_argument('--generation_epochs', type=int, default=80)
    parser.add_argument('--pre_epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--temp', type=float, default=1)
    parser.add_argument('--gamma', type=float, default=0.3)

    args = parser.parse_args()

    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    adjs, graph_indicators, graph_labels, node_attributes = load_data(args)
    # graph_labels = graph_labels_process(args, graph_labels)
    nx_graphs = load_nx_graphs(args)
    pyg_graphs = []

    pbar = tqdm(range(len(graph_labels)))
    for i in pbar:
        pbar.set_description('Converting the %d-th networkx graphs to PyG graphs...' % (i+1))
        pyg_G = torch_geometric.utils.from_networkx(nx_graphs[i])
        pyg_G.id = i
        pyg_G.label = graph_labels[i]
        pyg_graphs.append(pyg_G)

    graphs_num = len(pyg_graphs)
    nodes_num_list = [each.num_nodes for each in pyg_graphs]
    attrs_dim = pyg_graphs[0].attrs.size()[1]
    
    if os.path.exists('../data/'+args.dataset+'/original_graphs.pkl'):
        data_perturbation_list = load_generated_data(args, 'perturbation_graphs.pkl')
        data_masking_list = load_generated_data(args, 'masking_graphs.pkl')
    else:
        data_perturbation_list, data_masking_list = generation(args, graphs_num, nodes_num_list, attrs_dim, pyg_graphs)
        save_generated_data(args, 'original_graphs.pkl', pyg_graphs)
        save_generated_data(args, 'perturbation_graphs.pkl', data_perturbation_list)
        save_generated_data(args, 'masking_graphs.pkl', data_masking_list)

    graph_embeddings, graph_labels = cl_train(args, attrs_dim, pyg_graphs, data_perturbation_list, data_masking_list)
    #acc, acc_std, f1mi, f1mi_std, f1ma, f1ma_std = svc_classify(graph_embeddings, graph_labels, True)
    #print(acc, acc_std, f1mi, f1mi_std, f1ma, f1ma_std)
    f1mis, f1mas = svc_classify(graph_embeddings, graph_labels, True)
    print(f1mis)
    print(f1mas)