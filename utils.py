import torch
import torch.nn.functional as F
from torch_sparse import SparseTensor

import numpy as np
import yaml
import random
from scipy.sparse import identity, diags
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import normalize

def seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normal_adj(adj):
    adj = SparseTensor.from_scipy(adj)
    deg = adj.sum(dim=1).to(torch.float)
    D_isqrt = deg.pow(-0.5)
    D_isqrt[D_isqrt == float('inf')] = 0
    DAD = D_isqrt.view(-1,1) * adj * D_isqrt.view(1,-1)

    return DAD.to_scipy(layout='csr')


def adj2laplacian(A):
    norm_adj = normal_adj(A)
    L = identity(norm_adj.shape[0]).multiply(1+1e-6) - norm_adj

    return L


def accuracy(output, labels):
    preds = output.cpu().max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def train(model, labels, x, adj, idx_train, optimizer):
    model.train()

    optimizer.zero_grad()
    out = model(x, adj)
    loss = F.nll_loss(out[idx_train], labels[idx_train])
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, labels, x, edge_index, idx_train, idx_val, idx_test, out=None):
    if out is None:
        model.eval()
        out = model(x, edge_index)

    train_acc = accuracy(out[idx_train], labels[idx_train])
    valid_acc = accuracy(out[idx_val], labels[idx_val])
    test_acc = accuracy(out[idx_test], labels[idx_test])

    return train_acc.item(), valid_acc.item(), test_acc.item(), out


def svds_jl(adj, k):
    from julia.api import Julia
    jl = Julia(compiled_modules=False)
    from julia import Main
    Main.include("./svds.jl")
    print('Perform Truncated SVD')
    U, S, Vt = Main.main(adj, k)

    return U.real, S.real, Vt.real


def embedding_normalize(embedding, norm):
    if norm == "unit_vector":
        return normalize(embedding, axis=1)
    elif norm == "standardize":
        scaler = StandardScaler()
        return scaler.fit_transform(embedding)
    elif norm == "minmax":
        scaler = MinMaxScaler()
        return scaler.fit_transform(embedding)
    else:
        return embedding


def minmax_gamma(embedding_norm, gamma, ptb_rate, embed_sim):
    if embedding_norm == "minmax":
        if ptb_rate == 2.0:
            gamma = embed_sim.min()+2.5
        elif ptb_rate == 3.0:
            gamma = embed_sim.min()+2
        elif ptb_rate == 4.0:
            gamma = embed_sim.min()+2
    return gamma


def preprocess_args(args):
    arg_data = args.__dict__
    if args.perturbed:
        config_file = \
            f"configs/{args.dataset}/{args.backbone}/{args.attack}/perturbed_{args.ptb_rate}.yaml"
    else:
        config_file = \
            f"configs/{args.dataset}/{args.backbone}/{args.attack}/clean.yaml"
    with open(config_file) as file:
        yaml_data= yaml.safe_load(file)
    for arg, value in arg_data.items():
        if value is None:
            continue
        if value in ['True', 'False']:
            yaml_data[arg] = value=='True'
        else:
            yaml_data[arg] = value
    args.__dict__ = yaml_data
    if args.full_distortion:
        if args.dataset == "chameleon" and args.attack=="meta":
            args.gamma = 0.3
        elif args.dataset == "squirrel":
            args.gamma = 0.08
    return args
