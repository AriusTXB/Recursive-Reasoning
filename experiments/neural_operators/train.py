import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.dataset import SudokuDataset
from src.visualizer import plot_convergence
from model import NeuralOperatorSolver

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_trajectory_target(initial_puzzle, solution, t):
    """
    Creates a synthetic target grid at time t.
    t: [Batch, 1] ranging 0 to 1
    
    Logic:
    - Fixed Cells (from initial puzzle): Always correct.
    - Empty Cells: 
      - With probability t, they are the Correct Solution.
      - With probability (1-t), they are Random Noise (1-9).
    """
    B, L = initial_puzzle.shape
    device = initial_puzzle.device
    
    # 1. Identify which cells are fixed (given clues)
    # Recall: Preprocessing maps 0->1 (Empty), 1..9->2..10
    # So '1' is the empty token.
    is_clue = (initial_puzzle != 1)
    
    # 2. Generate Random Noise Grid (random digits 2-10)
    noise_grid = torch.randint(2, 11, (B, L), device=device)
    
    # 3. Create a mask based on time t
    # For each cell, pick a random float. If val < t, use Solution. Else use Noise.
    rand_mask = torch.rand(B, L, device=device)
    use_solution_mask = rand_mask < t
    
    # 4. Compose the Target
    # Start with noise
    target = noise_grid.clone()
    
    # Apply correct solution where time allows
    target = torch.where(use_solution_mask, solution, target)
    
    # Always enforce the original clues (they shouldn't change)
    target = torch.where(is_clue, initial_puzzle, target)
    
    return target

def train():
    print(f"--- Training Neural Operator on {DEVICE} ---")
    
    # Config
    BATCH_SIZE = 64
    EPOCHS = 15
    LR = 0.001
    
    # Data
    train_ds = SudokuDataset("data/processed", "train")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    
    # Approx 12 Million Parameters
    model = NeuralOperatorSolver(vocab_size=11, hidden_dim=384, depth=4).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    loss_history = []
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for inputs, labels in loop:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            B = inputs.shape[0]
            
            # 1. Sample Time t for this batch
            # We want the model to learn the WHOLE trajectory, so we sample t randomly
            t = torch.rand(B, 1, device=DEVICE)
            
            # 2. Generate the Expected Output at this time t
            # This is our "Ground Truth" for this specific moment in the reasoning process
            target_at_t = generate_trajectory_target(inputs, labels, t)
            
            # 3. Forward Pass
            logits = model(inputs, t) # [B, 11, 9, 9]
            
            # 4. Loss
            # Reshape for CE: [B, 11, 9, 9] -> [B, 11, 81] vs [B, 81]
            loss = criterion(logits.flatten(2), target_at_t)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f"Avg Loss: {avg_loss:.4f}")
        
        # Save Checkpoint
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "model_nop.pt"))

    plot_convergence(loss_history, os.path.join(OUTPUT_DIR, "convergence.png"))

if __name__ == "__main__":
    train()