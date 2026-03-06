import torch
import torch.nn.functional as F
from transformers import GPT2TokenizerFast
import sys
import os
import time
import argparse

# Add src to path so we can import the architecture
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.model import SLM

def generate_story(prompt, checkpoint_path, max_new_tokens, temperature, top_k):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[*] Initializing inference on: {device.upper()}")

    # 1. Load Tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    
    # 2. Initialize Architecture
    model = SLM().to(device)

    # 3. Load Weights gracefully
    if not os.path.exists(checkpoint_path):
        print(f"[!] Error: Could not find checkpoint at '{checkpoint_path}'")
        print("    Please ensure the file exists and the path is correct.")
        sys.exit(1)

    print(f"[*] Loading weights from: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    
    # Clean up DataParallel prefixes if they accidentally slipped in
    clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict)
    model.eval()

    # 4. Encode Prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    print("\n" + "="*50)
    print("📖 GENERATING STORY...")
    print("="*50 + "\n")
    
    # Print the prompt with the typewriter effect
    for char in prompt:
        print(char, end="", flush=True)
        time.sleep(0.01)

    # 5. Generation Loop
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Crop context to the block_size (256) to prevent crashes
            idx_cond = input_ids[:, -256:]
            
            # Forward pass
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] # Focus on last time step
            
            # Apply Temperature
            logits = logits / temperature
            
            # Apply Top-K Sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            # Get probabilities and sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            input_ids = torch.cat((input_ids, idx_next), dim=1)
            
            # Decode the single new token
            word = tokenizer.decode(idx_next[0].item())
            
            # Typewriter effect print
            print(word, end="", flush=True)
            time.sleep(0.03) # Adjust for faster/slower typing
            
            # Stop early if the model generates the End Of Text token
            if idx_next[0].item() == tokenizer.eos_token_id:
                break
                
    print("\n\n" + "="*50)
    print("✨ THE END")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate stories using your trained SLM.")
    parser.add_argument("--prompt", type=str, default="Once upon a time, a little girl named Lily found a magic", help="The starting sentence of the story.")
    parser.add_argument("--checkpoint", type=str, default="checkpoint_final.pt", help="Path to your trained .pt file.")
    parser.add_argument("--tokens", type=int, default=150, help="Maximum number of new tokens to generate.")
    parser.add_argument("--temp", type=float, default=0.8, help="Temperature (higher = more creative/random, lower = more safe/boring).")
    parser.add_argument("--top_k", type=int, default=40, help="Top-K sampling constraint.")
    
    args = parser.parse_args()
    
    generate_story(
        prompt=args.prompt,
        checkpoint_path=args.checkpoint,
        max_new_tokens=args.tokens,
        temperature=args.temp,
        top_k=args.top_k
    )