#includes four kinds of data augmentation
import torch  


def mask_feature(x, node_p):
    
    #maks node_p% node
    mask = torch.empty((x.size(0), ), dtype=torch.float32, device=x.device).uniform_(0, 1) < node_p
    x = x.clone()
    x[mask,:] = 0
    
    return x, mask


def filter_adj(row, col, mask):
    return row[mask], col[mask]
      

def dropout_edge(edge_index, edge_p):
    #drop edge_p% edge

    row, col = edge_index
    p = torch.zeros(edge_index.shape[1]).to(edge_index.device) + 1 - edge_p
    stay = torch.bernoulli(p).to(torch.bool)
    mask = ~stay
    row, col = filter_adj(row, col, stay)
    edge_index = torch.stack([row, col], dim=0)

    return edge_index.long(), mask



