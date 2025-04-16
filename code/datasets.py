# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 14:55:46 2025

@author: Administrator
"""

from torch.utils.data import Dataset
from transformers import RobertaTokenizerFast
import numpy as np
import torch

class TeacherDataset(Dataset):
    def __init__(self, sample_data, gex_data, cnv_data, mu_data, smiles_seq, max_length=50):
        """
        SMILES encoding using RobertaTokenizerFast
        
        """    
    
        # get sample information
        self.sample_df = sample_data
        self.sample_df['sample'] = self.sample_df['sample'].astype(str).str.strip()
        self.sample_df['pubchem_cid'] = self.sample_df['pubchem_cid'].astype(str).str.strip()
        self.sample_df['label'] = self.sample_df['label'].astype(np.float32)
        
        # save original label
        self.raw_labels = self.sample_df['label'].values
        
        # get multi-omics data
        self.gex_df = gex_data.set_index('sample')
        self.cnv_df = cnv_data.set_index('sample')
        self.mu_df = mu_data.set_index('sample')
        
        self.gex_dim = self.gex_df.shape[1]
        self.cnv_dim = self.cnv_df.shape[1]
        self.mu_dim = self.mu_df.shape[1]
        
        # get drug smiles
        self.df_smiles = smiles_seq
        self.df_smiles['pubchem_cid'] = self.df_smiles['pubchem_cid'].astype(str).str.strip()
        self.df_smiles['canonical_smiles'] = self.df_smiles['canonical_smiles'].astype(str).str.strip()
        self.smiles_dict = dict(zip(self.df_smiles['pubchem_cid'], self.df_smiles['canonical_smiles']))
        
        # Initialize the RoBERTa Tokenizer
        self.tokenizer = RobertaTokenizerFast.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
        self.max_length = max_length
        
        # add vocab_size
        self.vocab_size = self.tokenizer.vocab_size
        
        
        
    def __len__(self):
        return len(self.sample_df)
    
    def __getitem__(self, idx):
        row = self.sample_df.iloc[idx]
        sample = row['sample']
        pubchem_cid = row['pubchem_cid']
        label = row['label']
        
        # get omics features
        gex = self.gex_df.loc[sample].values.astype(np.float32)
        cnv = self.cnv_df.loc[sample].values.astype(np.float32)
        mu = self.mu_df.loc[sample].values.astype(np.float32)
        
        # get smiles and convert it into token
        smiles = self.smiles_dict.get(pubchem_cid, "")
        encoded_smiles = self.tokenizer.encode(smiles, padding="max_length", truncation=True, max_length=self.max_length)
        
        return {
            "original_idx": idx,    # original index
            "sample": sample,
            "pubchem_cid": pubchem_cid,
            "gex": torch.tensor(gex, dtype=torch.float32),
            "cnv": torch.tensor(cnv, dtype=torch.float32),
            "mu": torch.tensor(mu, dtype=torch.float32),
            "smiles": torch.tensor(encoded_smiles, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.float32)
            }
    

class StudentDataset(TeacherDataset):
    def __init__(self, sample_data, gex_data, cnv_data, mu_data, smiles_seq, max_length=50):
        """
        Student Dataset, which contains GEX data, drug data, and soft labels predicted by the teacher model.
        """
        # # Call the parent class constructor and pass only necessary parameters
        super().__init__(sample_data, gex_data, cnv_data=cnv_data, mu_data=mu_data, smiles_seq=smiles_seq, max_length=max_length)
        
    def __getitem__(self, idx):
        row = self.sample_df.loc[idx]
        sample = row['sample']
        pubchem_cid = row['pubchem_cid']
        label = row['label']
        s_label = row['s_label']
        
        # get gex features
        gex = self.gex_df.loc[sample].values.astype(np.float32)
        
        # get smiles and convert it into token
        smiles = self.smiles_dict.get(pubchem_cid, "")
        encoded_smiles = self.tokenizer.encode(smiles, padding="max_length", truncation=True, max_length=self.max_length)
        
        return {
            "original_idx": idx,
            "sample": sample,
            "pubchem_cid": pubchem_cid,
            "gex": torch.tensor(gex, dtype=torch.float32),
            "smiles": torch.tensor(encoded_smiles, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.float32),
            "s_label": torch.tensor(s_label, dtype=torch.float32),
            }

    
class VAEDataset(Dataset):
    def __init__(self, sample_data, gex_data, cnv_data, mu_data):
        self.gex_df = gex_data.set_index('sample')
        self.cnv_df = cnv_data.set_index('sample')
        self.mu_df = mu_data.set_index('sample')
        self.cell_df = sample_data
        
        self.gex_dim = self.gex_df.shape[1]
        self.cnv_dim = self.cnv_df.shape[1]
        self.mu_dim = self.mu_df.shape[1]

    def __len__(self):
        return len(self.cell_df)

    def __getitem__(self, idx):
        sample = self.cell_df.iloc[idx]
        sample_id = sample['sample']
        
        gex = self.gex_df.loc[sample_id].values.astype(np.float32)
        cnv = self.cnv_df.loc[sample_id].values.astype(np.float32)
        mu = self.mu_df.loc[sample_id].values.astype(np.float32)

        # 将 GEX 数据作为输入，CNV 和 MU 作为目标输出
        return {
            'gex': gex,
            'cnv': cnv,
            'mu': mu
        }
    
    
    
    
        