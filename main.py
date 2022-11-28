import argparse
import os
import pickle
from opt_einsum import contract

import numpy as np
from numpy import linalg as LA
from scipy.sparse import csr_matrix, load_npz, diags
from scipy.sparse.linalg import svds, eigs
import torch
from deeprobust.graph.data import Dataset, PrePtbDataset

from model import ada_filter, GCN
from logger import Logger, SimpleLogger
from utils import *
from hnsw import *


def main(args):
    seed()
    device = f'cuda:{args.device}' if args.device > -1 else 'cpu'
    device = torch.device(device)

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

    if args.no_garnet:
        edge_index = SparseTensor.from_scipy(adj_mtx).float().to(device)
    else:
        adj_mtx = adj_mtx.asfptype()
        num_nodes = adj_mtx.shape[0]
        if args.adj_norm:
            adj_mtx = normal_adj(adj_mtx)
        U, S, Vt = svds(adj_mtx, k=args.r)

        spec_embed = np.sqrt(S.reshape(1,-1))*U
        spec_embed_Vt = np.sqrt(S.reshape(1,-1))*Vt.transpose()
        spec_embed = embedding_normalize(spec_embed, args.embedding_norm)
        spec_embed_Vt = embedding_normalize(spec_embed_Vt, args.embedding_norm)
        if args.use_feature:
            feat_embed = adj_mtx @ (adj_mtx @ features)/2
            feat_embed = embedding_normalize(feat_embed, args.embedding_norm)
            spec_embed = np.concatenate((spec_embed, feat_embed), axis=1)
            spec_embed_Vt = np.concatenate((spec_embed_Vt, feat_embed), axis=1)

        adj_mtx = hnsw(spec_embed, k=args.k)
        diag_mtx = diags(adj_mtx.diagonal(), 0)
        row, col = adj_mtx.nonzero()
        lower_diag_idx = np.argwhere(row>col).reshape(-1)
        row = row[lower_diag_idx]
        col = col[lower_diag_idx]
        row_embed = spec_embed[row]
        if args.dataset == "chameleon" and args.attack == "nettack":
            col_embed = spec_embed[col]
        else:
            col_embed = spec_embed_Vt[col]
        embed_sim = contract("ik, ik -> i" , row_embed, col_embed)

        if args.full_distortion:
            ori_dist = LA.norm((row_embed-col_embed), axis=1)
            S_b, U_b = eigs(adj2laplacian(adj_mtx), k=args.r, which='SM')
            S_b, U_b = S_b[1:].real, U_b[:, 1:].real
            base_spec_embed = U_b/np.sqrt(S_b.reshape(1,-1))
            base_spec_embed = embedding_normalize(base_spec_embed, args.embedding_norm)
            base_row_embed = base_spec_embed[row]
            base_col_embed = base_spec_embed[col]
            base_dist = LA.norm((base_row_embed-base_col_embed), axis=1)
            spec_dist = base_dist/ori_dist
            idx = np.argwhere(spec_dist>args.gamma).reshape(-1,)
        else:
            idx = np.argwhere(embed_sim>args.gamma).reshape(-1,)

        new_row = row[idx]
        new_col = col[idx]
        if args.weighted_knn:
            val = embed_sim[idx]
        else:
            val = np.repeat(1, new_row.shape[0])
        adj_mtx = csr_matrix((val, (new_row, new_col)), shape=(num_nodes, num_nodes))
        adj_mtx = adj_mtx + adj_mtx.transpose() + diag_mtx
        edge_index = SparseTensor.from_scipy(adj_mtx).float().to(device)

    labels = torch.as_tensor(labels, dtype=torch.long).to(device)
    x = torch.from_numpy(features).to(device)
    d = x.shape[1]
    c = labels.max().item() + 1

    if args.backbone == "gprgnn":
        model = ada_filter(d, args.hidden_dim, c, dropout=args.dropout, coe=args.c, P=args.p)
    elif args.backbone == "gcn":
        model = GCN(d, args.hidden_dim, c, num_layers=3, dropout=args.dropout, use_bn=False, norm=True)
    model = model.to(device)

    logger = Logger(args.runs, args)
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

    ### Print results ###
    best_val, best_test = logger.print_statistics()

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
    args = parser.parse_args()
    args = preprocess_args(args)
    print(args)
    main(args)
