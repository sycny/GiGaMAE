import math
import os.path as osp
import torch
from torch_geometric.utils import negative_sampling
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, CitationFull, WikiCS, Coauthor, Amazon


def get_dataset(path, name):
            assert name in ['Cora', 'CiteSeer', 'WikiCS', 'Coauthor-CS', 'Coauthor-Phy','PubMed','dblp',
                            'Amazon-Computers', 'Amazon-Photo']
            name = 'dblp' if name == 'DBLP' else name
            root_path = osp.expanduser('~/datasets')

            if name == 'Coauthor-CS':
                return Coauthor(root=path, name='cs', transform=T.NormalizeFeatures())

            if name == 'Coauthor-Phy':
                return Coauthor(root=path, name='physics', transform=T.NormalizeFeatures())

            if name == 'WikiCS':
                return WikiCS(root=path)

            if name == 'Amazon-Computers':
                return Amazon(root=path, name='computers', transform=T.NormalizeFeatures())

            if name == 'Amazon-Photo':
                return Amazon(root=path, name='photo', transform=T.NormalizeFeatures())

            return (CitationFull if name == 'dblp' else Planetoid)(osp.join(root_path, 'Citation'), name, transform=T.NormalizeFeatures())



def do_edge_split_direct(dataset, fast_split=False, val_ratio=0.05, test_ratio=0.1):
    data = dataset.clone()

    num_nodes = data.num_nodes
    row, col = data.edge_index
    # Return upper triangular portion.
    mask = row < col
    row, col = row[mask], col[mask]
    n_v = int(math.floor(val_ratio * row.size(0)))
    n_t = int(math.floor(test_ratio * row.size(0)))
    # Positive edges.
    perm = torch.randperm(row.size(0))
    row, col = row[perm], col[perm]
    r, c = row[:n_v], col[:n_v]
    data.val_pos_edge_index = torch.stack([r, c], dim=0)
    r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
    data.test_pos_edge_index = torch.stack([r, c], dim=0)
    r, c = row[n_v + n_t:], col[n_v + n_t:]
    data.train_pos_edge_index = torch.stack([r, c], dim=0)
    # Negative edges (cannot guarantee (i,j) and (j,i) won't both appear)
    neg_edge_index = negative_sampling(
        data.edge_index, num_nodes=num_nodes,
        num_neg_samples=row.size(0))
    data.val_neg_edge_index = neg_edge_index[:, :n_v]
    data.test_neg_edge_index = neg_edge_index[:, n_v:n_v + n_t]
    data.train_neg_edge_index = neg_edge_index[:, n_v + n_t:]

    split_edge = {'train': {}, 'valid': {}, 'test': {}}
    split_edge['train']['edge'] = data.train_pos_edge_index.t()
    split_edge['train']['edge_neg'] = data.train_neg_edge_index.t()
    split_edge['valid']['edge'] = data.val_pos_edge_index.t()
    split_edge['valid']['edge_neg'] = data.val_neg_edge_index.t()
    split_edge['test']['edge'] = data.test_pos_edge_index.t()
    split_edge['test']['edge_neg'] = data.test_neg_edge_index.t()
    return split_edge



