# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 17:32:34 2025

@author: Administrator
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from model import StudentModel, VAE
from datasets import StudentDataset
from utils import loss_funciton, split_data
from scipy.stats  import pearsonr, spearmanr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Fixed random seed
# -----------------------------





def train_student() :
    
    # seed = np.random.randint(10000)  # seed
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
    drug_smiles_path = "../data/source/GDSC_Smiles.csv"
    cell_data_path = "../data/result/teacher_all_predictions.csv"
    vae_model_path = "../data/model/vae_model.pth"
    
    
    gex_data = pd.read_csv(gex_path)
    cnv_data = pd.read_csv(cnv_path)
    mu_data = pd.read_csv(mu_path)
    cell_data = pd.read_csv(cell_data_path)
    drug_smiles = pd.read_csv(drug_smiles_path)
    
    gex_dim = gex_data.shape[1] - 1
    cnv_dim = cnv_data.shape[1] - 1
    mu_dim = mu_data.shape[1] -1 
    output_dim = [gex_dim, cnv_dim, mu_dim]
    
    
    latent_dim = 64
    hidden_dim=256
    embed_dim = 128
    drug_hidden_dim = 128
    out_dim = 1   
    batch_size = 32
    learning_rate = 1e-3
    num_epochs = 40
    
    # Loading the trained VAE model
    vae_model = VAE(input_dim=gex_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, output_dims = output_dim).to(device)
    
    vae_model.load_state_dict(torch.load(vae_model_path))
    vae_model.eval()
    
    dataset = StudentDataset(cell_data, gex_data, cnv_data, mu_data, drug_smiles)
    
    
    omics_input_dims = {'gex': gex_dim, 'cnv': cnv_dim, 'mu': mu_dim}
    
    # Building student model
    student_model = StudentModel(omics_input_dims, embed_dim, dataset.vocab_size, drug_hidden_dim, out_dim).to(device)
    
    
    
    
    train_dataset, test_dataset = split_data(dataset) 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    optimizer = optim.Adam(student_model.parameters(), lr=learning_rate)
    
    student_model.train()
    # training
    for epoch in range(num_epochs):
    
        epoch_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):  # Using tqdm to display a progress bar
            # Make sure the data is on the right device
            gex = batch['gex'].to(device)
            with torch.no_grad():
                vae_model.eval()
                generated, _, _ = vae_model(gex)
                generated_gex, generated_cnv, generated_mu = torch.split(generated, [gex_dim, cnv_dim, mu_dim], dim=1)
            
            cnv = generated_cnv.to(device)
            mu = generated_mu.to(device)
            smiles_seq = batch['smiles'].to(device)
            labels_tensor = batch["label"].to(device).unsqueeze(1)
    
            # Get the output of the student model
            student_output = student_model(gex, cnv, mu, smiles_seq) 
            
            # Get the corresponding soft label
            s_labels = batch["s_label"].to(device).unsqueeze(1)
    
            # Calculating Losses
            loss = loss_funciton(labels_tensor, student_output, s_labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            epoch_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(train_loader):.4f}")
    
    # -----------------------------
    # Conduct test evaluation
    # -----------------------------
    student_model.eval()
    y_true = []
    y_pred = []
    s_labels = []
    samples = []
    pubchem_cids = []  # save pubchem cid
    
    with torch.no_grad():
        for batch in test_loader:
            gex = batch['gex'].to(device)
            with torch.no_grad():
                vae_model.eval()
                generated, _, _ = vae_model(gex)
                generated_gex, generated_cnv, generated_mu = torch.split(generated, [gex_dim, cnv_dim, mu_dim], dim=1)
            cnv = generated_cnv.to(device)
            mu = generated_mu.to(device)
            smiles_seq = batch['smiles'].to(device)
            labels_tensor = batch["label"].to(device).unsqueeze(1)
            s_labels_tensor = batch["s_label"].to(device).unsqueeze(1)
            sample_names = batch["sample"]
            
            # get pubchem_id
            pubchem_cid_batch = batch['pubchem_cid']
            pubchem_cids.extend(pubchem_cid_batch)  # record pubchem cid
            
            # Get the prediction results of the student model
            student_output = student_model(gex, cnv, mu, smiles_seq)
    
            y_true.extend(labels_tensor.cpu().numpy())
            y_pred.extend(student_output.cpu().numpy())
            s_labels.extend(s_labels_tensor.cpu().numpy())
            samples.extend(sample_names)
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculating MSE, RMSE, and R2
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred) 
    r2 = r2_score(y_true, y_pred)
    y_true1 = np.ravel(y_true)  # 或者 y_true.flatten()
    y_pred1 = np.ravel(y_pred)
    pcc, p = pearsonr(y_true1, y_pred1)
    scc, p = spearmanr(y_true1, y_pred1)
    
    print(f"Test MSE: {mse:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    print(f"Test R²: {r2:.4f}")
    print(f'Test PCC: {pcc:.4f}')
    print(f'Test SCC: {scc:.4f}')
    
    # Save the weights of the student model
    model_save_path = "../data/model/student_model.pth"
    torch.save(student_model.state_dict(), model_save_path)
    print(f"Model weights saved to {model_save_path}")
    
    
    print(seed)


if __name__ == "__main__":
    train_student()