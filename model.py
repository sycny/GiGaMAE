import torch
import torch.nn as nn
from arg import args
import torch.nn.functional as F
from torch_geometric.nn import GAE, VGAE, GATConv, BatchNorm, GCNConv


class prediction_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim): # bottleneck structure
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        
    def forward(self, x):
        
        x = self.layer1(x)
        x = self.layer2(x)
        return x 



class GAE(torch.nn.Module):
    def __init__(self, in_channels, hidden_num, head,  out_channels, emb_size_1, emb_size_2, activation, base_model=GATConv):
        super().__init__()
        self.base_model = base_model

        self.conv1 = base_model(in_channels, hidden_num, heads=head, dropout=args.dropout)
        self.conv2 = base_model(hidden_num * head, out_channels, heads=1, concat=False,dropout=args.dropout)
        
        self.bn1 = BatchNorm(hidden_num * head, momentum=0.99)
        self.bn2 = BatchNorm(out_channels, momentum=0.99)


        self.p_1 = prediction_MLP(out_channels, out_channels*2, emb_size_1)
        self.p_2 = prediction_MLP(out_channels, out_channels*2, emb_size_2)
        self.p_12 = prediction_MLP(out_channels, out_channels*2, (emb_size_1 + emb_size_2))


        self.activation = activation
        
    def encoder(self, x, edge_index):
        
        x = F.dropout(x, p=args.dropout, training=self.training)
        x = self.activation(self.bn1(self.conv1(x, edge_index)))
        x = F.dropout(x, p=args.dropout, training=self.training)
        x = self.activation(self.conv2(x, edge_index))

       
        return x
    
    
    def decoder_all(self, z):
        
        recon_emb_1 = self.p_1(z)
        recon_emb_2 = self.p_2(z)
        
        recon_12 = self.p_12(z)

        return recon_emb_1,recon_emb_2, recon_12
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):

        z = self.encoder(x, edge_index)
        embs = self.decoder_all(z)
        
        
        return embs
