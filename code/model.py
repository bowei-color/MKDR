# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 14:52:05 2025

@author: Administrator
"""

import torch
import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader, Subset




class MultiOmicsEncoder(nn.Module):
    def __init__(self, input_dims, embed_dim):
        
        """
        input_dims: dict, 如{'gex': gex_dim, 'cnv': cnv_dim, 'mu': mu_dim}
        embed_dim: 每个组学映射后的统一向量维度
        """
        
        super(MultiOmicsEncoder, self).__init__()
        
        self.mlp_gex = nn.Sequential(nn.Linear(input_dims['gex'], embed_dim), nn.ReLU())
        self.mlp_cnv = nn.Sequential(nn.Linear(input_dims['cnv'], embed_dim), nn.ReLU())
        self.mlp_mu = nn.Sequential(nn.Linear(input_dims['mu'], embed_dim), nn.ReLU())
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        
    def forward(self, gex, cnv, mu):
        h_gex = self.mlp_gex(gex)
        h_cnv = self.mlp_cnv(cnv)
        h_mu = self.mlp_mu(mu)
        h_seq = torch.stack([h_gex, h_cnv, h_mu], dim=0)   # [seq_len=3, batch, embed_dim]
        h_multi = self.transformer(h_seq)
        h_multi = h_multi.mean(dim=0)    # [batch, embed_dim]
        return h_multi
    
    
class DrugEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=3):
        super(DrugEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.norm = nn.BatchNorm1d(hidden_dim)
        
        
    def forward(self, smiles_seq):
        x = self.embedding(smiles_seq)  # [batch, seq_len, embed_dim]
        _, (h_n, _) = self.lstm(x)      # h_n 形状: [num_layers, batch, hidden_dim]
        return self.norm(h_n[-1])     
        
        
class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super(CrossAttentionFusion, self).__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads)
        
    def forward(self, feat1, feat2):
        feat1 = feat1.unsqueeze(0)      # [1, batch, embed_dim]
        feat2 = feat2.unsqueeze(0)    # [1, batch, embed_dim]
        fused, _ = self.cross_attn(feat1, feat2, feat2)
        return fused.squeeze(0)
    
    


class Classifier(nn.Module):
    def __init__(self, embed_dim, out_dim):
        super(Classifier, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim//2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim//2, out_dim)
            )
    
    def forward(self, x):
        return self.mlp(x)
    

class TeacherModel(nn.Module):
    def __init__(self, omics_input_dims, embed_dim, vocab_size, drug_hidden_dim, out_dim):
        super(TeacherModel, self).__init__()
        self.omics_encoder = MultiOmicsEncoder(omics_input_dims, embed_dim)
        self.drug_encoder = DrugEncoder(vocab_size, embed_dim, drug_hidden_dim)
        self.fusion = CrossAttentionFusion(embed_dim)
        self.classifier = Classifier(embed_dim, out_dim)
        self.linear = nn.Linear(embed_dim*2, embed_dim)
        
        
    def forward(self, gex, cnv, mu, smiles_seq):
        h_omics = self.omics_encoder(gex, cnv, mu)
        h_drug = self.drug_encoder(smiles_seq)
        if h_drug.shape[-1] != h_omics.shape[-1]:
            mapper = nn.Linear(h_drug.shape[-1], h_omics.shape[-1]).to(h_drug.device)
            h_drug = mapper(h_drug)
        h_fused1 = self.fusion(h_omics, h_drug)
        h_fused2 = self.fusion(h_drug, h_omics)
        h_fused = torch.cat([h_fused1, h_fused2], dim=-1)    # [batch, 2*embed_dim]
        h_fused = self.linear(h_fused)
        out = self.classifier(h_fused)
        return out
    
    
class StudentModel(nn.Module):    
    def __init__(self, omics_input_dims, embed_dim, vocab_size, drug_hidden_dim, out_dim):
        super(StudentModel, self).__init__()
        self.omics_encoder = MultiOmicsEncoder(omics_input_dims, embed_dim)
        self.drug_encoder = DrugEncoder(vocab_size, embed_dim, drug_hidden_dim)
        self.fusion = CrossAttentionFusion(embed_dim)
        self.classifier = Classifier(embed_dim, out_dim)
        self.linear = nn.Linear(embed_dim*2, embed_dim)
        
        
    def forward(self, gex, cnv, mu, smiles_seq):
        h_omics = self.omics_encoder(gex, cnv, mu)
        h_drug = self.drug_encoder(smiles_seq)
        if h_drug.shape[-1] != h_omics.shape[-1]:
            mapper = nn.Linear(h_drug.shape[-1], h_omics.shape[-1]).to(h_drug.device)
            h_drug = mapper(h_drug)
        h_fused1 = self.fusion(h_omics, h_drug)
        h_fused2 = self.fusion(h_drug, h_omics)
        h_fused = torch.cat((h_fused1, h_fused2), dim=-1) # [batch, 2*embed_dim]
        h_fused = self.linear(h_fused)
        out = self.classifier(h_fused)
        return out
        
    

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, output_dims):
        super(VAE, self).__init__()
        
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim*2),
            nn.BatchNorm1d(hidden_dim*2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2)
            )
        
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.BatchNorm1d(hidden_dim*2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim*2, sum(output_dims))
            )
        
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
