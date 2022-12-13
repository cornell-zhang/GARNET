import numpy as np
from numpy import linalg as LA
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import svds, eigs
from opt_einsum import contract

from utils import normal_adj, embedding_normalize, adj2laplacian
from hnsw import hnsw


def garnet(
    adj_mtx,
    features,
    r=50,
    k=30,
    gamma=0.0,
    use_feature=True,
    embedding_norm=None,
    embedding_symmetric=False,
    full_distortion=False,
    adj_norm=True,
    weighted_knn=True):

    num_nodes = adj_mtx.shape[0]
    if num_nodes > 100000:
        batch_version = True
    else:
        batch_version = False

    ## weighted spectral embedding on input graph
    adj_mtx = adj_mtx.asfptype()
    if adj_norm:
        adj_mtx = normal_adj(adj_mtx)
    U, S, Vt = svds(adj_mtx, r)

    spec_embed = np.sqrt(S.reshape(1,-1))*U
    spec_embed_Vt = np.sqrt(S.reshape(1,-1))*Vt.transpose()
    spec_embed = embedding_normalize(spec_embed, embedding_norm)
    spec_embed_Vt = embedding_normalize(spec_embed_Vt, embedding_norm)
    if use_feature:
        feat_embed = adj_mtx @ (adj_mtx @ features)/2
        feat_embed = embedding_normalize(feat_embed, embedding_norm)
        spec_embed = np.concatenate((spec_embed, feat_embed), axis=1)
        spec_embed_Vt = np.concatenate((spec_embed_Vt, feat_embed), axis=1)

    ## knn base graph construction
    adj_mtx = hnsw(spec_embed, k)
    diag_mtx = diags(adj_mtx.diagonal(), 0)
    row, col = adj_mtx.nonzero()
    lower_diag_idx = np.argwhere(row>col).reshape(-1)
    row = row[lower_diag_idx]
    col = col[lower_diag_idx]

    ## use batch version of edge pruning on large graphs
    if batch_version:
        '''
        We choose simplified disortion metric for the batch version,
        as computing the full distortion metric is expensive on large graphs.
        '''
        idx = []
        embed_sim = []
        batch_size = 20000
        num_edges = row.shape[0]
        for i in range(num_edges//batch_size):
            bstart = i*batch_size
            if i < num_edges//batch_size - 1:
                bend = (i+1)*batch_size
            else:
                bend = num_edges
            batch_row_embed = spec_embed[row[bstart:bend]]
            batch_col_embed = spec_embed_Vt[col[bstart:bend]]
            batch_embed_sim = contract("ik, ik -> i" , batch_row_embed, batch_col_embed)
            batch_idx = np.argwhere(batch_embed_sim>gamma).reshape(-1,)+bstart
            idx = np.concatenate((idx, batch_idx))
            embed_sim = np.concatenate((embed_sim, batch_embed_sim))
        idx = idx.astype(int)

    ## prune all uncritical edges simultaneously on small graphs
    else:
        row_embed = spec_embed[row]
        if embedding_symmetric:
            col_embed = spec_embed[col]
        else:
            col_embed = spec_embed_Vt[col]

        '''
        We replace Euclidean distance w/ dot product to measure embedding distance,
        which has two benefits:
        (1) dot product is more efficient to compute by leveraging eisum (i.e., contract).
        (2) the results of dot product could be used as edge weights in the refined graph.
        '''
        embed_sim = contract("ik, ik -> i" , row_embed, col_embed)

        if full_distortion:
            ori_dist = LA.norm((row_embed-col_embed), axis=1)
            S_b, U_b = eigs(adj2laplacian(adj_mtx), r, which='SM')
            S_b, U_b = S_b[1:].real, U_b[:, 1:].real
            base_spec_embed = U_b/np.sqrt(S_b.reshape(1,-1))
            base_spec_embed = embedding_normalize(base_spec_embed, embedding_norm)
            base_row_embed = base_spec_embed[row]
            base_col_embed = base_spec_embed[col]
            base_dist = LA.norm((base_row_embed-base_col_embed), axis=1)
            spec_dist = base_dist/ori_dist
            idx = np.argwhere(spec_dist>gamma).reshape(-1,)
        else:
            idx = np.argwhere(embed_sim>gamma).reshape(-1,)

    new_row = row[idx]
    new_col = col[idx]
    if weighted_knn:
        val = embed_sim[idx]
    else:
        val = np.repeat(1, new_row.shape[0])
    adj_mtx = csr_matrix((val, (new_row, new_col)), shape=(num_nodes, num_nodes))
    adj_mtx = adj_mtx + adj_mtx.transpose() + diag_mtx

    return adj_mtx
