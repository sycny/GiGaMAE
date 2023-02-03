import torch as tc
from arg import args
from torch_geometric.nn import Node2Vec

def svd_flip(u, v):
    # columns of u, rows of v
    max_abs_cols = tc.argmax(tc.abs(u), 0)
    i = tc.arange(u.shape[1]).to(u.device)
    signs = tc.sign(u[max_abs_cols, i])
    u *= signs
    v *= signs.view(-1, 1)
    return u, v


def postprocess(embs, ratio):
    # PCA
    mu = tc.mean(embs, dim=0, keepdim=True)
    X = embs - mu
    U, S, V = tc.svd(X)
    U, Vt = svd_flip(U, V)
    accumulate, sum_S = 0.0, sum(S.detach().cpu().tolist())
    for idx, s in enumerate(S.detach().cpu().tolist(), 1):
        accumulate += s / sum_S
        if accumulate > ratio:
            break
    X = tc.mm(X, Vt[:idx].T)
    
    # whitening
    u, s, vt = tc.svd(tc.mm(X.T, X) / (X.shape[0] - 1.0))
    W = tc.mm(u, tc.diag(1.0 / tc.sqrt(s)))
    X = tc.mm(X, W)
    return X


def PCA(embs, ratio):
    
    if embs.shape[0]>embs.shape[1]:
        pca_emb = postprocess(embs, ratio)
    else:
        embs1 = embs[:,0:embs.shape[1]//2]
        embs2 = embs[:,embs.shape[1]//2:]
        
        pca_emb1 = postprocess(embs1, ratio)
        pca_emb2 = postprocess(embs2, ratio)
        
        pca_emb = tc.cat((pca_emb1,pca_emb2), 1)
        
    return  pca_emb
        
        
def node2vec(data):
    
    model = Node2Vec(data.edge_index, embedding_dim=args.embedding_dim, walk_length=args.walk_length,
                 context_size=args.context_size, walks_per_node=args.walks_per_node,
                 num_negative_samples=args.node2vec_neg_samples, p=args.p, q=args.q, sparse=True).to(data.x.device)
    #print(f'Node2vec_config: p={args.p},q={args.q} embedding_dim:{args.embedding_dim}, walk_length={args.walk_length}')
    #print(f'context_size={args.context_size}, walks_per_node={args.walks_per_node}, num_negative_samples={args.node2vec_neg_samples},lr:{args.node2vec_lr}, batchsize:{args.node2vec_batchsize} epoch {args.node2vec_epoch}')
    loader = model.loader(batch_size=args.node2vec_batchsize, shuffle=True, num_workers=0)
    optimizer = tc.optim.SparseAdam(list(model.parameters()), lr=args.node2vec_lr)
    model.train()
    for epoch in range(1, args.node2vec_epoch+1):
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(data.x.device), neg_rw.to(data.x.device))
            loss.backward()
            optimizer.step()
    model.eval()
    emb = model()
    
    return emb.detach()

