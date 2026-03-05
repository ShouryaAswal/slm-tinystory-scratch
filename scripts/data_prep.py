import os
import numpy as np
from datasets import load_dataset
from transformers import GPT2TokenizerFast
from tqdm import tqdm

def prepare_data():
    print("Loading TinyStories dataset...")
    dataset = load_dataset("roneneldan/TinyStories", split="train")
    
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    
    # We will store the tokens in a flat numpy array
    all_tokens = []
    
    print("Tokenizing stories...")
    # Limiting to 100,000 stories for initial rapid testing. Remove indexing to process all.
    for story in tqdm(dataset['text'][:100000]): 
        tokens = tokenizer.encode(story, add_special_tokens=False)
        all_tokens.extend(tokens)
        all_tokens.append(tokenizer.eos_token_id) # Append End of Text token
        
    print(f"Total tokens: {len(all_tokens)}")
    
    # Save to binary file
    os.makedirs('data', exist_ok=True)
    all_tokens_np = np.array(all_tokens, dtype=np.uint16)
    all_tokens_np.tofile('data/train.bin')
    print("Saved to data/train.bin successfully.")

if __name__ == "__main__":
    prepare_data()