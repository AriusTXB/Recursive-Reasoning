import sys
import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.dataset import SudokuDataset
from model import TinyRecursiveReasoningModel_ACTV1

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

def visualize():
    # 1. Config (Must match training config)
    config = {
        "batch_size": 1, 
        "seq_len": 81, "vocab_size": 11, "puzzle_emb_ndim": 0, "num_puzzle_identifiers": 1,
        "H_cycles": 1, "L_cycles": 2, "H_layers": 1, "L_layers": 2, "hidden_size": 128,
        "expansion": 4.0, "num_heads": 4, "pos_encodings": "rope", "halt_max_steps": 20,
        "halt_exploration_prob": 0.0, "forward_dtype": "float32",
        "puzzle_emb_len": 0
    }
    
    # 2. Load Model
    model = TinyRecursiveReasoningModel_ACTV1(config).to(DEVICE)
    model_path = os.path.join(OUTPUT_DIR, "model.pt")
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    # 3. Load Data
    ds = SudokuDataset("data/processed", "test")
    
    # --- FIX: Unpack tuple instead of dict access ---
    inp_sample, label_sample = ds[0] 
    
    # Prepare single batch dictionary
    batch = {
        "inputs": inp_sample.unsqueeze(0).to(DEVICE),
        "puzzle_identifiers": torch.tensor([0], dtype=torch.int32).to(DEVICE),
        "labels": label_sample.unsqueeze(0).to(DEVICE)
    }
    
    # 4. Init state
    # Pass device explicitly to ensure tensors are on GPU
    carry = model.initial_carry(batch)
    
    print("Running Inference...")
    
    # Step through
    for i in range(20):
        # Optional: Manually force continue to see what happens
        # if i < 5: carry.halted[:] = False 
        
        with torch.no_grad():
            carry, outputs = model(carry, batch)
        
        logits = outputs["logits"]
        # Access the correct key for Q-value
        q_val = outputs.get("q_halt_logits", outputs.get("q_halt")).item()
        
        preds = torch.argmax(logits, dim=-1)
        acc = (preds == batch["labels"]).float().mean().item()
        
        print(f"Step {i}: Accuracy: {acc:.2f} | Q-Halt: {q_val:.4f} | Halted: {carry.halted.item()}")
        
        if carry.halted.item():
            print("Model decided to HALT.")
            break

if __name__ == "__main__":
    visualize()