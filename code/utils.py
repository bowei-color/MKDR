# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 14:55:34 2025

@author: Administrator
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
    

def collate_fn(batch):
    """ 处理 DataLoader 批次 """
    original_indices = [item["original_idx"] for item in batch]  # 收集原始索引
    batch_dict = {
        key: torch.stack([item[key] for item in batch]) 
        if isinstance(batch[0][key], torch.Tensor) 
        else [item[key] for item in batch] 
        for key in batch[0] if key != "original_idx"  # 排除 original_idx，单独处理
    }
    batch_dict["original_indices"] = torch.tensor(original_indices)  # 添加原始索引
    return batch_dict


# -----------------------------
# SMILES word segmentation and dictionary construction
# -----------------------------

def build_vocab(smiles_list):
    vocab = {"<PAD>": 0}
    for s in smiles_list:
        for ch in s:
            if ch not in vocab:
                vocab[ch] = len(vocab)
    return vocab


def tokenize_smiles(smiles, vocab, max_len=50):
    tokens = [vocab.get(ch, 0) for ch in smiles]
    if len(tokens) < max_len:
        tokens += [vocab["<PAD>"]] * (max_len - len(tokens))
    else:
        tokens = tokens[:max_len]
    return tokens


def vae_loss(reconstructed, targets, mu, logvar, beta=1.0):
    # Split refactoring results and goals
    gex_rec, cnv_rec, mu_rec = torch.split(reconstructed, [t.shape[1] for t in targets], dim=1)
    
    # Calculate the reconstruction loss of each component
    mse_loss = nn.MSELoss(reduction='sum')
    recon_loss = (mse_loss(gex_rec, targets[0])+
                  mse_loss(cnv_rec, targets[1])+
                  mse_loss(mu_rec, targets[2])
                  ) / sum(t.size(0) for t in targets) # Calculate the average loss
    
    # Calculate KL loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / targets[0].size(0)
    
    
    return recon_loss + beta * kl_loss
                 

def loss_funciton(y_true, y_pred, y_soft):
    """
    Calculate the total loss of the student model
    """
    # The loss of the true label
    loss_true = nn.MSELoss()(y_pred, y_true)
    
    # Soft label loss
    loss_soft = nn.MSELoss()(y_pred, y_soft)
    
    # The total loss is the weighted sum of the two losses
    total_loss = loss_true + loss_soft
    # total_loss = loss_true
    
    return total_loss


def split_data(dataset, test_size=0.2, random_seed=42):
    """
    Divide the training set and test set
    """
    indices = np.arange(len(dataset))
    train_idx, val_idx = train_test_split(indices, test_size=test_size, random_state=random_seed)
    
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    
    return train_dataset, val_dataset

