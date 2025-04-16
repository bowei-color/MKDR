# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 16:08:37 2025

@author: Administrator
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm # Make sure to import tqdm
import random
from datasets import VAEDataset
from model import VAE
from utils import vae_loss



# -----------------------------
# random seed
# -----------------------------






    
def train_vae():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # seed = np.random.randint(10000)
    seed = 4211
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    
    # file path
    gex_path = "../data/source/cell_gex_features.csv"
    cnv_path = "../data/source/cell_cnv_features.csv"
    mu_path = "../data/source/cell_mu_features.csv"
    cell_data_path = "../data/source/cell_sample.csv"

    gex_data = pd.read_csv(gex_path)
    cnv_data = pd.read_csv(cnv_path)
    mu_data = pd.read_csv(mu_path)
    cell_data = pd.read_csv(cell_data_path)


    batch_size = 32
    hidden_dim = 256
    latent_dim = 64
    learning_rate = 1e-3
    num_epochs = 1000
    beta=0.5
    

    dataset = VAEDataset(cell_data, gex_data, cnv_data, mu_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    output_dims=[dataset.gex_dim, dataset.cnv_dim, dataset.mu_dim]
    # print(output_dims)

    model = VAE(
        input_dim=dataset.gex_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        output_dims=output_dims
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10)
    
    best_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            gex = batch['gex'].to(device)
            cnv = batch['cnv'].to(device)
            mu = batch['mu'].to(device)
            
            optimizer.zero_grad()
            
            reconstructed, mu_latent, logvar_latent = model(gex)
            
            loss = vae_loss(reconstructed, [gex, cnv, mu], mu_latent, logvar_latent, beta)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # 梯度裁剪
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "../data/model/vae_model.pth")
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.2e}")
    
    print("Training completed!")



if __name__ == "__main__":
    train_vae()