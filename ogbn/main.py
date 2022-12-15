import argparse
import os
import gdown

import numpy as np
from scipy.sparse import load_npz
import torch
from ogb.nodeproppred import NodePropPredDataset

import sys
sys.path.append("../")
from model import GCN
from logger import Logger, SimpleLogger
from utils import *


def main(args):
    seed()
    device = f'cuda:{args.device}' if args.device > -1 else 'cpu'
    device = torch.device(device)

    ## load dataset
    ogbn_dataset = NodePropPredDataset(name=f'ogbn-{args.dataset}', root='./ogbn_data/')
    ogbn_data = ogbn_dataset[0]
    split_idx = ogbn_dataset.get_idx_split()
    idx_train = split_idx['train']
    idx_val = split_idx['valid']
    idx_test = split_idx['test']
    features = ogbn_data[0]['node_feat']
    labels = ogbn_data[1].reshape(-1,)

    if args.no_garnet:
        if args.perturbed:
            adj_mtx = load_npz(f'data/{args.dataset}/{args.dataset}_grbcd_{args.ptb_rate}.npz')
        else:
            adj_mtx = load_npz(f'data/{args.dataset}/{args.dataset}_clean.npz')
    ## load purified graph via GARNET
    else:
        if args.perturbed:
            adj_mtx = load_npz(f'data/{args.dataset}/{args.dataset}_grbcd_{args.ptb_rate}_garnet.npz')
        else:
            adj_mtx = load_npz(f'data/{args.dataset}/{args.dataset}_clean_garnet.npz')

    if args.dataset == "products":
        adj_mtx = normal_adj(adj_mtx)
        args.epochs = 300

    edge_index = SparseTensor.from_scipy(adj_mtx).float().to(device)
    labels = torch.as_tensor(labels, dtype=torch.long).to(device)
    x = torch.from_numpy(features).to(device)
    d = x.shape[1]
    c = labels.max().item() + 1

    ## choose backbone GNN model
    if args.dataset == "arxiv":
        model = GCN(d, args.hidden_dim, c, num_layers=3, dropout=args.dropout, use_bn=True, norm=True)
    elif args.dataset == "products":
        model = GCN(d, args.hidden_dim, c, num_layers=3, dropout=args.dropout, use_bn=False, norm=False)
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
    parser.add_argument('--dataset', type=str, default='arxiv',
                        choices=['arxiv', 'products'],
                        help='choose graph dataset')
    parser.add_argument('--perturbed', action='store_true',
                        help='use adversarial graph as input')
    parser.add_argument('--attack', type=str, default='grbcd',
                        choices=['grbcd'],
                        help='used to choose attack method and test nodes')
    parser.add_argument('--backbone', type=str, default='gcn',
                        choices=['gcn'],
                        help='backbone GNN model')
    parser.add_argument('--ptb_rate', type=float, default=0.25,
                        choices=[0.25, 0.5],
                        help='adversarial perturbation budget')
    parser.add_argument('--runs', type=int, default=10,
                        help='how many runs to compute accuracy mean and std')
    parser.add_argument('--display_step', type=int,
                        default=1, help='how often to print')
    parser.add_argument('--no_garnet', action='store_true',
                        help='No using GARNET to purify input graph')
    parser.add_argument('--epochs', type=int, default=500,
                        help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='GNN hidden dimension')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='weight decay for GNN training')

    ## combine input arguments w/ arguments in configuration files
    args = parser.parse_args()
    print(args)
    if not os.path.exists(f"data"):
        os.makedirs(f"data")
    if not os.path.exists(f"data/{args.dataset}"):
        if args.dataset == "arxiv":
            drive_id = '193-QvtS6LbwNnlGAcYNJvz_z-5nybqAE'
        else:
            drive_id = '11pgwa4J4w5IhW6TsZa82Wzt6qAF96Om-'
        gdown.download(id=drive_id, output="data/")
        os.system(f"unzip data/{args.dataset}.zip -d data/{args.dataset}")
        os.system(f"rm data/{args.dataset}.zip")
    main(args)
