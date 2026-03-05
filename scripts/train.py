import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import sys
import os
import warnings
from tqdm import tqdm

# Suppress harmless PyTorch 2.0 compiler and DataParallel warnings for clean output
warnings.filterwarnings("ignore", category=UserWarning)

# Add src to path so we can import the architecture
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.model import SLM

# Hyperparameters
batch_size = 64  # Reduced to prevent GPU 0 OOM during DataParallel gathering
block_size = 256
max_iters = 5000 
learning_rate = 3e-4
eval_interval = 500
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_batch(data):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)

def train():
    print("Loading binary data into RAM...")
    data_path = 'data/train.bin'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Cannot find {data_path}. Did you run data_prep.py?")
    data = np.fromfile(data_path, dtype=np.uint16)

    model = SLM()
    
    if torch.cuda.device_count() > 1:
        print(f"🔥 Optimization: Utilizing {torch.cuda.device_count()} GPUs via DataParallel!")
        model = nn.DataParallel(model)
        
    model = model.to(device)
    
    print("Compiling model into C++ kernels (this takes ~1 min)...")
    model = torch.compile(model)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, fused=True)
    
    # Updated to the modern PyTorch AMP syntax
    scaler = torch.amp.GradScaler('cuda')
    
    print(f"Training fully optimized on {device}...")
    
    pbar = tqdm(range(max_iters), desc="Training Progress")
    
    for iter in pbar:
        xb, yb = get_batch(data)
        
        optimizer.zero_grad(set_to_none=True)
        
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            logits, loss = model(xb, yb)
            if torch.cuda.device_count() > 1:
                loss = loss.mean()
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        if iter % 10 == 0:
            current_loss = loss.item()
            pbar.set_description(f"Loss: {current_loss:.4f}")
            
        if iter > 0 and iter % eval_interval == 0:
            checkpoint_path = f'checkpoint_step_{iter}.pt'
            
            raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
            if isinstance(raw_model, nn.DataParallel):
                raw_model = raw_model.module
                
            torch.save(raw_model.state_dict(), checkpoint_path)
            
    print("Training complete.")
    
    raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    if isinstance(raw_model, nn.DataParallel):
        raw_model = raw_model.module
    torch.save(raw_model.state_dict(), 'checkpoint_final.pt')

if __name__ == "__main__":
    train()