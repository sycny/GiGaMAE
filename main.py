import os
import os.path as osp
from pickle import FALSE
import datetime
import random
import numpy as np
from time import perf_counter as t

import torch
import torch.nn.functional as F
from torch_geometric.nn import GAE, GATConv

from arg import args
from data_aug import mask_feature, dropout_edge
from eval import label_classification, clustering
from teacher import PCA, node2vec
from datasets import get_dataset
from model import GAE


def seed_torch(seed):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) 
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True


def pretrain():
    
    print('Start pretrain')
    emb_node2vec = node2vec(data)
    emb_pca = PCA(data.x, args.ratio)
    print('pretrain is done!')

    return  emb_node2vec, emb_pca


def train(mask_x, mask_edge, mask_index_node,  mask_index_edge, mask_both_node_edge):

    model.train()
    optimizer.zero_grad()
    
    embs = model(mask_x, mask_edge)
    recon_emb_1,recon_emb_2, recon_12 = embs
    
    loss1_f = semi_loss(emb_1[mask_index_node], recon_emb_1[mask_index_node])
    loss1_e = semi_loss(emb_1[mask_index_edge], recon_emb_1[mask_index_edge])
    loss1_both = semi_loss(emb_1[mask_both_node_edge], recon_emb_1[mask_both_node_edge])

    loss2_f = semi_loss(emb_2[mask_index_node], recon_emb_2[mask_index_node])
    loss2_e = semi_loss(emb_2[mask_index_edge], recon_emb_2[mask_index_edge])
    loss2_both = semi_loss(emb_2[mask_both_node_edge], recon_emb_2[mask_both_node_edge])
 
    loss12_f = semi_loss(torch.cat((emb_1,emb_2), 1)[mask_index_node], recon_12[mask_index_node])
    loss12_e = semi_loss(torch.cat((emb_1,emb_2), 1)[mask_index_edge], recon_12[mask_index_edge])
    loss12_both= semi_loss(torch.cat((emb_1,emb_2), 1)[mask_both_node_edge], recon_12[mask_both_node_edge])
    
    loss_e = args.l1_e*loss1_e + args.l2_e*loss2_e + args.l12_e*loss12_e
    loss_f = args.l1_f*loss1_f + args.l2_f*loss2_f + args.l12_f*loss12_f
    loss_both = args.l1_b*loss1_both + args.l1_b*loss2_both + args.l12_b*loss12_both

    info_loss = loss_e.mean() + loss_f.mean() + loss_both.mean()

    info_loss.backward()
    optimizer.step()
    
    return float(info_loss)


@torch.no_grad()
def test(data):
    
    model.eval()
    z = model.encoder(data.x, data.edge_index)
    
    acc = label_classification(z, data.y, ratio=0.1)
    nmi, ari, _  = clustering(z, data.y, dataset.num_classes)
    
    acc_mean = acc.get('F1Mi').get('mean')
    
    return acc_mean, nmi, ari


def adjust_learning_rate(optimizer, epoch):
       
    lr = args.lr * (args.lrdec_1 ** (epoch // args.lrdec_2))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def sim(z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

def semi_loss(z1: torch.Tensor, z2: torch.Tensor):
    

    f = lambda x: torch.exp(x / args.tau)

    refl_sim = f(sim(z1, z1))
    
    between_sim = f(sim(z1, z2))
    
    loss = -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    
    return loss




if __name__ == '__main__':
    
        seed_torch(args.seed)
        device = torch.device(args.gpu_num if torch.cuda.is_available() else 'cpu')

        path = osp.join(osp.expanduser('~'), 'datasets', args.dataset)
        dataset = get_dataset(path, args.dataset)
        data = dataset[0]
        data = data.to(device)
        

        emb_node2vec, emb_pca  = pretrain()
        emb_1 = emb_node2vec
        if args.ratio == 1:
            emb_2 = data.x
        else:
            emb_2 = emb_pca

        student_start = t()
        in_channels, hidden_num, head, out_channels = dataset.num_features, args.hidden_num, args.head, args.out_channels

        emb_size_1 = emb_node2vec.shape[1]
        emb_size_2 = emb_pca.shape[1]

        activation = torch.nn.ELU()
    
        model = GAE(dataset.num_features, hidden_num, head, out_channels, emb_size_1, emb_size_2, activation, GATConv).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) 
        
        total_loss = []
        accuracy = []  
        for epoch in range(1, args.epoch + 1):
            
            adjust_learning_rate(optimizer,epoch)
            
            mask_x, mask_index_node_binary = mask_feature(data.x, args.node_p)
            mask_edge, mask_index_edge = dropout_edge(data.edge_index, args.edge_p)

            
            mask_edge_node = mask_index_edge*data.edge_index[0] 
            mask_index_edge_binary = torch.zeros(data.x.shape[0]).to(device) 
            mask_index_edge_binary[mask_edge_node] = 1
            mask_index_edge_binary = mask_index_edge_binary.to(bool)
            mask_both_node_edge = mask_index_edge_binary & mask_index_node_binary
            mask_index_node_binary_sole = mask_index_node_binary &(~mask_both_node_edge)
            mask_index_edge_binary_sole  = mask_index_edge_binary &(~mask_both_node_edge)

        
            info_loss = train(mask_x, mask_edge, mask_index_node_binary_sole, mask_index_edge_binary_sole, mask_both_node_edge)


        acc, nmi, ari = test(data)
        print(f'Final result: acc: {acc:.4f}, nmi:{nmi:.4f}, ari: {ari:.4f}' )  
        

        

                
