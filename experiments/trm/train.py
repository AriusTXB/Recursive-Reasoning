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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.dataset import SudokuDataset
from model import TinyRecursiveReasoningModel_ACTV1

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def train():
    # 1. Config matches your Pydantic model
    config = {
        "batch_size": 64,
        "seq_len": 81,
        "vocab_size": 11, # 0=PAD, 1=Empty, 2-10=Digits
        "puzzle_emb_ndim": 0,
        "num_puzzle_identifiers": 1,
        "H_cycles": 1, 
        "L_cycles": 2, # Recursion depth per step
        "H_layers": 1, # Ignored in code but needed for config
        "L_layers": 2, 
        "hidden_size": 128,
        "expansion": 4.0,
        "num_heads": 4,
        "pos_encodings": "rope",
        "halt_max_steps": 10,  # Max ACT steps
        "halt_exploration_prob": 0.1,
        "forward_dtype": "float32" # Use float32 for simplicity
    }

    # 2. Data
    ds = SudokuDataset("data/processed", "train")
    # Drop last to ensure batch size match for carry
    loader = DataLoader(ds, batch_size=config["batch_size"], shuffle=True, drop_last=True) 

    # 3. Model
    model = TinyRecursiveReasoningModel_ACTV1(config).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    
    # 4. State Management (The "Carry")
    # We need an initial dummy batch to initialize the carry state
    dummy_batch = {
        "inputs": torch.zeros(config["batch_size"], 81).to(DEVICE),
        "labels": torch.zeros(config["batch_size"], 81).to(DEVICE),
        "puzzle_identifiers": torch.zeros(config["batch_size"], dtype=torch.int32).to(DEVICE)
    }
    carry = model.initial_carry(dummy_batch)

    criterion = nn.CrossEntropyLoss()
    losses = []

    print("Starting Training...")
    
    for epoch in range(5):
        model.train()
        loop = tqdm(loader)
        
        # --- FIX: Unpack tuple (inputs, labels) instead of treating batch as dict ---
        for inputs, labels in loop:
            # Move to device
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            
            # Construct the dictionary TRM expects
            batch_data = {
                "inputs": inputs,
                "labels": labels,
                # Create dummy puzzle IDs (zeros) since the simple dataset doesn't have them
                "puzzle_identifiers": torch.zeros(config["batch_size"], dtype=torch.int32).to(DEVICE)
            }
            
            # Forward Pass
            # The model updates 'carry' internally based on 'halted' status
            carry, outputs = model(carry, batch_data)
            
            # --- Loss Calculation ---
            # 1. Prediction Loss (reconstruct sudoku solution)
            logits = outputs["logits"] # [B, 81, Vocab]
            targets = carry.current_data["labels"] # Use targets from carry (handles alignment)
            
            loss_ce = criterion(logits.view(-1, config["vocab_size"]), targets.view(-1))
            
            # 2. Q-Learning Halting Loss
            preds = torch.argmax(logits, dim=-1)
            # Correctness per sample
            is_correct = (preds == targets).all(dim=1).float() 
            
            # Simple Reward: 1.0 if correct, -0.1 penalty per step
            reward = is_correct - 0.1
            
            # MSE between predicted Q-value (should I stop?) and Reward
            q_halt = outputs["q_halt_logits"] 
            loss_q = F.mse_loss(q_halt, reward)
            
            total_loss = loss_ce + 0.1 * loss_q

            optimizer.zero_grad()
            total_loss.backward()
            
            # Important: Detach carry to prevent infinite backprop graph
            carry.inner_carry.z_H = carry.inner_carry.z_H.detach()
            carry.inner_carry.z_L = carry.inner_carry.z_L.detach()
            
            optimizer.step()
            
            losses.append(total_loss.item())
            loop.set_postfix(loss=total_loss.item(), steps=carry.steps.float().mean().item())

    # Save
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "model.pt"))
    
    # Simple Plotting
    plt.figure()
    plt.plot(losses)
    plt.title("Training Loss")
    plt.savefig(os.path.join(OUTPUT_DIR, "loss.png"))
    print("Training Complete.")

if __name__ == "__main__":
    train()