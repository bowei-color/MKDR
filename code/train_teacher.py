# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 17:21:55 2025

@author: Administrator
"""


import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from datasets import TeacherDataset
from model import TeacherModel
from utils import collate_fn
from scipy.stats  import pearsonr, spearmanr






# -----------------------------
# File path definition
# -----------------------------
def train_teacher():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # -----------------------------
    # Fixed random seed
    # -----------------------------
    # seed = np.random.randint(10000)
    seed = 4211
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    cell_data_path = "../data/source/cell_data.csv"
    gex_all_path = "../data/source/cell_all_gex_features.csv"
    cnv_path = "../data/source/cell_cnv_features.csv"
    mu_path = "../data/source/cell_mu_features.csv"
    drug_smiles_path = "../data/source/GDSC_Smiles.csv"
    
    df_smiles = pd.read_csv(drug_smiles_path)
    df_cell_data = pd.read_csv(cell_data_path)
    df_gex = pd.read_csv(gex_all_path)
    df_cnv = pd.read_csv(cnv_path)
    df_mu = pd.read_csv(mu_path)
    
    dataset = TeacherDataset(df_cell_data, df_gex, df_cnv, df_mu, df_smiles)
    print("Number of samples in the teacher dataset：", len(dataset))
    
    
    # Hyperparameters
    embed_dim = 128
    drug_hidden_dim = 128
    out_dim = 1   # Regression output dimensions
    batch_size = 32
    learning_rate = 1e-3
    num_epochs = 40
    
    
    indices = np.arange(len(dataset))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=seed)
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    
    # -----------------------------
    # Teacher model training
    # -----------------------------
    
    # Get the omics input dimensions and dictionary size from dataset
    gex_dim = dataset.gex_dim
    cnv_dim = dataset.cnv_dim
    mu_dim = dataset.mu_dim
    vocab_size = dataset.vocab_size
    
    omics_input_dims = {'gex': gex_dim, 'cnv': cnv_dim, 'mu': mu_dim}
    
    
    
    teacher_model = TeacherModel(omics_input_dims, embed_dim, vocab_size, drug_hidden_dim, out_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(teacher_model.parameters(), lr=learning_rate)
    
    print("========== Start model training ==========")
    teacher_model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            gex = batch["gex"].to(device)
            cnv = batch["cnv"].to(device)
            mu_ = batch["mu"].to(device)
            smiles_seq = batch["smiles"].to(device)
            # print(smiles_seq)
            labels_tensor = batch["label"].to(device).unsqueeze(1)  # [batch, 1]
            
            optimizer.zero_grad()
            outputs = teacher_model(gex, cnv, mu_, smiles_seq)
            loss = criterion(outputs, labels_tensor)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * gex.size(0)
        avg_loss = epoch_loss / len(train_dataset)
        print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")
    
    # -----------------------------
    # Get the prediction results of the validation set and calculate the MSE
    # -----------------------------
    teacher_model.eval()
    all_results = []
    all_preds = []
    all_true = []
    
    with torch.no_grad():
        for batch in val_loader:
            gex = batch["gex"].to(device)
            cnv = batch["cnv"].to(device)
            mu_ = batch["mu"].to(device)
            smiles_seq = batch["smiles"].to(device)
            labels_tensor = batch["label"].to(device).unsqueeze(1)
            outputs = teacher_model(gex, cnv, mu_, smiles_seq)
            preds = outputs.squeeze(1).cpu().numpy().tolist()
            true_vals = labels_tensor.squeeze(1).cpu().numpy().tolist()
            
            # Record the prediction results of each sample and its corresponding sample and pubchem_id
            samples = batch["sample"]
            pubchem_ids = batch["pubchem_cid"]
            
            for s, p, t, pred in zip(samples, pubchem_ids, true_vals, preds):
                all_results.append({
                    "sample": s,
                    "pubchem_cid": p,
                    "label": t,
                    "predict": pred
                })
            all_preds.extend(preds)
            all_true.extend(true_vals)
    
    # Calculate MSE, RMSE and R2
    mse = np.mean((np.array(all_preds) - np.array(all_true))**2)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_true, all_preds) 
    r2 = r2_score(all_true, all_preds)
    y_true1 = np.ravel(all_true) 
    y_pred1 = np.ravel(all_preds)
    pcc, p = pearsonr(y_true1, y_pred1)
    scc, p = spearmanr(y_true1, y_pred1)
    
    print(f"Test MSE: {mse:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    print(f"Test R²: {r2:.4f}")
    print(f'Test PCC: {pcc:.4f}')
    print(f'Test SCC: {scc:.4f}')
    
    
    
    
    
    model_save_path = "../data/model/teacher_model.pth"
    torch.save(teacher_model.state_dict(), model_save_path)
    print(f"Model weights saved to {model_save_path}")
    
    all_samples_loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    teacher_model.eval()
    all_results = []
    all_preds = []
    all_true = []
    
    with torch.no_grad():
        for batch in tqdm(all_samples_loader, desc="Predicting all samples"):
            original_indices = batch["original_indices"].numpy()  
            gex = batch["gex"].to(device)
            cnv = batch["cnv"].to(device)
            mu_ = batch["mu"].to(device)
            smiles_seq = batch["smiles"].to(device)
            outputs = teacher_model(gex, cnv, mu_, smiles_seq)
            preds = outputs.squeeze(1).cpu().numpy().tolist()
            
    
            raw_labels = dataset.sample_df['label'].iloc[original_indices].values.tolist()
            
     
            samples = batch["sample"]
            pubchem_ids = batch["pubchem_cid"]
            
            for s, p, true_label, pred in zip(samples, pubchem_ids, raw_labels, preds):
                all_results.append({
                    "pubchem_cid": p,
                    "sample": s,
                    "label": true_label,
                    "s_label": pred
                })
            all_preds.extend(preds)
            all_true.extend(raw_labels)
    
    
    all_results_df = pd.DataFrame(all_results)
    all_results_df.to_csv("../data/result/teacher_all_predictions.csv", index=False)




if __name__ == "__main__":
    train_teacher()