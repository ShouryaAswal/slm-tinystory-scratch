import torch
import numpy as np
import sys
import os

# Add src to path so we can import our model
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.model import SLM

# Hyperparameters
batch_size = 32
block_size = 256
max_iters = 5000
learning_rate = 3e-4
eval_interval = 500
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load binary data via memory mapping (zero RAM cost)
data_path = 'data/train.bin'
data = np.memmap(data_path, dtype=np.uint16, mode='r')

def get_batch():
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)

def train():
    model = SLM().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    print(f"Training on {device}...")
    
    for iter in range(max_iters):
        xb, yb = get_batch()
        
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        if iter % eval_interval == 0:
            print(f"Step {iter}: Loss {loss.item():.4f}")
            # Save checkpoint
            torch.save(model.state_dict(), f'checkpoint_step_{iter}.pt')
            
    print("Training complete. Final checkpoint saved.")

if __name__ == "__main__":
    train()