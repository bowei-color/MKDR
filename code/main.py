# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 14:12:08 2025

@author: Administrator
"""


import numpy as np
import random
import torch
from train_vae import train_vae
from train_teacher import train_teacher
from train_student import train_student



if __name__ == "__main__":
    

    
    print("==> Step 1: Training VAE model...")
    train_vae()
    
    print("==> Step 2: Training Teacher model...")
    train_teacher()
    
    print("==> Step 3: Training Student model...")
    train_student()

    print("==> All steps completed successfully.")

