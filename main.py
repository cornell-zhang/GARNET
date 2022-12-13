import argparse
import os
import pickle

import numpy as np
from scipy.sparse import load_npz
import torch
from deeprobust.graph.data import Dataset, PrePtbDataset

from model import ada_filter, GCN
from logger import Logger, SimpleLogger
from utils import *
from garnet import garnet


def main(args):
    seed()
    device = f'cuda:{args.device}' if args.device > -1 else 'cpu'
    device = torch.device(device)

    ## load dataset
    dataset = args.dataset
    if dataset in ['chameleon', 'squirrel']:
        with open(f'data/{dataset}_data.pickle', 'rb') as handle:
            data = pickle.load(handle)
        features = data["features"]
        labels = data["labels"]
        idx_train = data["idx_train"]
        idx_val = data["idx_val"]
        idx_test = data["idx_test"]
        if args.perturbed:
            adj_mtx = load_npz(f'data/{dataset}_perturbed_{args.ptb_rate}.npz')
        else:
            adj_mtx = load_npz(f'data/{dataset}.npz')
        if args.attack == 'nettack':
            idx_test = np.load(f"data/{dataset}_idx_test.npy")

    else:
        data = Dataset(root='./data/', name=dataset, setting='prognn')
        labels = data.labels
        features = data.features.todense()
        idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
        perturbed_data = PrePtbDataset(root='./data/',
            name=dataset,
            attack_method=args.attack,
            ptb_rate=args.ptb_rate)
        if args.perturbed:
            adj_mtx = perturbed_data.adj
        else:
            adj_mtx = data.adj
        if args.attack == 'nettack':
            idx_test = perturbed_data.target_nodes

    if args.dataset == "chameleon" and args.attack == "nettack":
        embedding_symmetric = True
    else:
        embedding_symmetric = False

    ## purify input graph via GARNET
    if not args.no_garnet:
        adj_mtx = garnet(
                adj_mtx,
                features,
                r=args.r,
                k=args.k,
                gamma=args.gamma,
                use_feature=args.use_feature,
                embedding_norm=args.embedding_norm,
                embedding_symmetric=embedding_symmetric,
                full_distortion=args.full_distortion,
                adj_norm=args.adj_norm,
                weighted_knn=args.weighted_knn)

    edge_index = SparseTensor.from_scipy(adj_mtx).float().to(device)
    labels = torch.as_tensor(labels, dtype=torch.long).to(device)
    x = torch.from_numpy(features).to(device)
    d = x.shape[1]
    c = labels.max().item() + 1

    ## choose backbone GNN model
    if args.backbone == "gprgnn":
        model = ada_filter(d, args.hidden_dim, c, dropout=args.dropout, coe=args.c, P=args.p)
    elif args.backbone == "gcn":
        model = GCN(d, args.hidden_dim, c, num_layers=3, dropout=args.dropout, use_bn=False, norm=True)
    else:
        raise NotImplementedError
    model = model.to(device)
    logger = Logger(args.runs, args)

    ## GNN training
    for run in range(args.runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)
        best_val = float('-inf')
        for epoch in range(args.epochs):
            loss = train(model, labels, x, edge_index, idx_train, optimizer)
            result = test(model, labels, x, edge_index, idx_train, idx_val, idx_test)

            if args.attack == "nettack" and epoch <= int(0.7*args.epochs) \
                and args.dataset == "pubmed":
                continue
            logger.add_result(run, result[:-1])

            if result[1] >= best_val:
                best_out = result[-1]
                best_val = result[1]

            if epoch % args.display_step == 0:
                print(f'Epoch: {epoch:02d}, '
                    f'Loss: {loss:.4f}, '
                    f'Train: {100 * result[0]:.2f}%, '
                    f'Valid: {100 * result[1]:.2f}%, '
                    f'Test: {100 * result[2]:.2f}%')
        logger.print_statistics(run)

    ## print results
    best_val, best_test = logger.print_statistics()

    ## save results
    if not os.path.exists("results"):
        os.makedirs("results")
    filename = f'results/{args.dataset}.csv'
    print(f"Saving results to {filename}")
    with open(f"{filename}", 'a+') as write_obj:
        write_obj.write(f"GNN: {args.backbone}, " +
                        f" Perturbed: {args.perturbed}, " +
                        f" Attack: {args.attack}, " +
                        f" Perturbation: {args.ptb_rate}, " +
                        f" GARNET: {not args.no_garnet}, " +
                        f" Test Acc: {best_test.mean():.2f} ± {best_test.std():.2f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0,
                        help='choose GPU device id')
    parser.add_argument('--dataset', type=str, default='cora',
                        choices=['cora', 'citeseer', 'pubmed', 'chameleon', 'squirrel'],
                        help='choose graph dataset')
    parser.add_argument('--perturbed', action='store_true',
                        help='use adversarial graph as input')
    parser.add_argument('--attack', type=str, default='meta',
                        choices=['nettack', 'meta', 'grbcd'],
                        help='used to choose attack method and test nodes')
    parser.add_argument('--backbone', type=str, default='gcn',
                        choices=['gcn', 'gprgnn'],
                        help='backbone GNN model')
    parser.add_argument('--ptb_rate', type=float, default=.2,
                        help='adversarial perturbation budget:\
                        suggest to use 0.2 for meta attack, 5.0 for nettack attack, 0.5 for grbcd attack')
    parser.add_argument('--runs', type=int, default=10,
                        help='how many runs to compute accuracy mean and std')
    parser.add_argument('--display_step', type=int,
                        default=1, help='how often to print')
    parser.add_argument('--full_distortion', action='store_true',
                        help='Use the non-simplified spectral embedding distortion')
    parser.add_argument('--no_garnet', action='store_true',
                        help='No using GARNET to purify input graph')
    parser.add_argument('--k', type=int,
                        help='k for kNN graph construction')
    parser.add_argument('--weighted_knn', type=str,
                        choices=[None, 'True', 'False'],
                        help='use weighted knn graph')
    parser.add_argument('--adj_norm', type=str,
                        choices=[None, 'True', 'False'],
                        help='normalize adjacency matrix')
    parser.add_argument('--use_feature', type=str,
                        choices=[None, 'True', 'False'],
                        help='incorporate node features for kNN construction')
    parser.add_argument('--embedding_norm', type=str,
                        choices=[None, 'unit_vector', 'standardize', 'minmax'],
                        help='normalize node embeddings for kNN construction')
    parser.add_argument('--gamma', type=float,
                        help='threshold to sparsify kNN graph')
    parser.add_argument('--r', type=int, help='number of eigenpairs')
    parser.add_argument('--epochs', type=int, help='number of epochs')
    parser.add_argument('--lr', type=float, help='learning rate')
    parser.add_argument('--dropout', type=float, help='dropout rate')
    parser.add_argument('--hidden_dim', type=int, help='GNN hidden dimension')
    parser.add_argument('--weight_decay', type=float, help='weight decay for GNN training')
    parser.add_argument('--p', type=int,
                        help='adaptive filter degree in GPRGNN')
    parser.add_argument('--c', type=float,
                        help='coefficients of adaptive filter in GPRGNN')

    ## combine input arguments w/ arguments in configuration files
    args = parser.parse_args()
    args = preprocess_args(args)
    print(args)
    main(args)
