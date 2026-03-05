import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import sys
import os
import wandb
from tqdm import tqdm

# Add src to path so we can import the architecture
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.model import SLM

# Hyperparameters
batch_size = 128  # Maximized to feed Dual T4 GPUs
block_size = 256
max_iters = 5000 
learning_rate = 3e-4
eval_interval = 500  # Saves an intermediate checkpoint every 500 steps
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_batch(data):
    # Randomly sample starting indices for the batch
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    
    # Asynchronous data transfer (non_blocking=True) keeps the GPU fed without waiting
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)

def train():
    # 1. Initialize Weights & Biases for live visualization
    wandb.init(project="slm-tinystories", name="optimized-multi-gpu-run")

    # Load data directly into RAM
    print("Loading binary data into RAM...")
    data_path = 'data/train.bin'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Cannot find {data_path}. Did you run data_prep.py?")
    data = np.fromfile(data_path, dtype=np.uint16)

    # 2. Initialize base architecture
    model = SLM()
    
    # 3. Multi-GPU Support via DataParallel
    if torch.cuda.device_count() > 1:
        print(f"🔥 Optimization: Utilizing {torch.cuda.device_count()} GPUs via DataParallel!")
        model = nn.DataParallel(model)
        
    model = model.to(device)
    
    # 4. PyTorch 2.0 Compilation (Speeds up execution by removing Python overhead)
    print("Compiling model into C++ kernels (this takes ~1 min)...")
    model = torch.compile(model)
    
    # 5. Fused Optimizer & Mixed Precision Setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, fused=True)
    scaler = torch.cuda.amp.GradScaler()
    
    print(f"Training fully optimized on {device}...")
    
    pbar = tqdm(range(max_iters), desc="Training Progress")
    
    for iter in pbar:
        xb, yb = get_batch(data)
        
        optimizer.zero_grad(set_to_none=True)
        
        # 6. Automatic Mixed Precision (AMP) logic
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            logits, loss = model(xb, yb)
            # DataParallel returns a loss vector (one per GPU); we must average it
            if torch.cuda.device_count() > 1:
                loss = loss.mean()
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Update progress bar and WandB dashboard
        if iter % 10 == 0:
            current_loss = loss.item()
            pbar.set_description(f"Loss: {current_loss:.4f}")
            wandb.log({"step": iter, "train/loss": current_loss})
            
        # 7. Save Intermediate Checkpoints
        if iter > 0 and iter % eval_interval == 0:
            checkpoint_path = f'checkpoint_step_{iter}.pt'
            
            # Correctly unwrap the model to save the raw weights (ignoring DataParallel/Compile wrappers)
            raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
            if isinstance(raw_model, nn.DataParallel):
                raw_model = raw_model.module
                
            torch.save(raw_model.state_dict(), checkpoint_path)
            
    print("Training complete.")
    
    # Save the absolute final state just in case
    raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    if isinstance(raw_model, nn.DataParallel):
        raw_model = raw_model.module
    torch.save(raw_model.state_dict(), 'checkpoint_final.pt')
    
    wandb.finish()

if __name__ == "__main__":
    train()