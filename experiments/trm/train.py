import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np

# Adjust path to find src folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.dataset import SudokuDataset
from model import TinyRecursiveReasoningModel_ACTV1

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def train():
    # 1. Configuration
    config = {
        "batch_size": 64,
        "seq_len": 81,
        "vocab_size": 11,
        
        # Ensure dimensions match (0 extra dims for embedding)
        "puzzle_emb_ndim": 0,
        "puzzle_emb_len": 0, 
        
        "num_puzzle_identifiers": 1,
        "H_cycles": 1, 
        "L_cycles": 2, 
        "H_layers": 1, 
        "L_layers": 2, 
        "hidden_size": 128,
        "expansion": 4.0,
        "num_heads": 4,
        "pos_encodings": "rope",
        "halt_max_steps": 10,
        "halt_exploration_prob": 0.1,
        "forward_dtype": "float32"
    }

    # 2. Data
    ds = SudokuDataset("data/processed", "train")
    loader = DataLoader(ds, batch_size=config["batch_size"], shuffle=True, drop_last=True) 

    # 3. Model
    model = TinyRecursiveReasoningModel_ACTV1(config).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    
    # 4. State Management (The "Carry")
    # --- CRITICAL FIX: Explicitly use dtype=torch.long for inputs/labels ---
    # PyTorch defaults to Float32 if dtype is not specified, causing the Loss error.
    dummy_batch = {
        "inputs": torch.zeros(config["batch_size"], 81, dtype=torch.long).to(DEVICE),
        "labels": torch.zeros(config["batch_size"], 81, dtype=torch.long).to(DEVICE),
        "puzzle_identifiers": torch.zeros(config["batch_size"], dtype=torch.int32).to(DEVICE)
    }
    carry = model.initial_carry(dummy_batch)

    criterion = nn.CrossEntropyLoss()
    losses = []

    print(f"Starting Training on {DEVICE}...")
    
    for epoch in range(100):
        model.train()
        loop = tqdm(loader, desc=f"Epoch {epoch+1}")
        
        for inputs, labels in loop:
            # Unpack tuple
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            
            # Construct dictionary batch
            batch_data = {
                "inputs": inputs,
                "labels": labels,
                "puzzle_identifiers": torch.zeros(config["batch_size"], dtype=torch.int32).to(DEVICE)
            }
            
            # Forward Pass
            carry, outputs = model(carry, batch_data)
            
            # --- Loss Calculation ---
            logits = outputs["logits"] 
            targets = carry.current_data["labels"] 
            
            # targets needs to be Long (int64) for CrossEntropy
            loss_ce = criterion(logits.view(-1, config["vocab_size"]), targets.view(-1))
            
            # Q-Learning Halting Loss
            preds = torch.argmax(logits, dim=-1)
            is_correct = (preds == targets).all(dim=1).float() 
            
            # Reward: 1.0 if correct, -0.1 if wrong/slow
            reward = is_correct - 0.1
            
            q_halt = outputs["q_halt_logits"] 
            loss_q = F.mse_loss(q_halt, reward)
            
            total_loss = loss_ce + 0.1 * loss_q

            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            
            # Detach carry
            carry.inner_carry.z_H = carry.inner_carry.z_H.detach()
            carry.inner_carry.z_L = carry.inner_carry.z_L.detach()
            
            optimizer.step()
            
            losses.append(total_loss.item())
            loop.set_postfix(loss=total_loss.item(), steps=carry.steps.float().mean().item())

    # Save
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "model.pt"))
    plt.figure()
    plt.plot(losses)
    plt.title("Training Loss")
    plt.savefig(os.path.join(OUTPUT_DIR, "loss.png"))
    print("Training Complete.")

if __name__ == "__main__":
    train()