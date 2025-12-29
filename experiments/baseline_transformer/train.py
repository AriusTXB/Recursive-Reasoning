import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.dataset import SudokuDataset
from src.visualizer import plot_convergence
from src.evaluators import evaluate_model
from model import BaselineTransformer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def train():
    print(f"--- Training LARGE Baseline Transformer on {DEVICE} ---")
    
    # 1. Config (Scaled to ~11M Params)
    BATCH_SIZE = 64
    EPOCHS = 100
    LR = 0.0005
    
    # Large Architecture
    MODEL_ARGS = {
        "vocab_size": 11,
        "d_model": 384,      # Scaled up
        "nhead": 8,          # Scaled up
        "num_layers": 6      # Scaled up
    }

    train_ds = SudokuDataset("data/processed", split="train")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    
    model = BaselineTransformer(**MODEL_ARGS).to(DEVICE)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    loss_history = []
    best_puzzle_acc = -1.0
    best_cell_acc = -1.0
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for inputs, labels in loop:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            logits = model(inputs)
            loss = criterion(logits.view(-1, 11), labels.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)
        
        # --- SMART SAVE LOGIC ---
        val_cell_acc, val_puzzle_acc = evaluate_model(model, DEVICE, split="test", model_type="standard")
        print(f"  Test Puzzle Acc: {val_puzzle_acc:.2f}% | Cell Acc: {val_cell_acc:.2f}%")
        
        # Always save latest
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "model_latest.pt"))
        
        # Save Best: Prioritize Puzzle Acc, break ties with Cell Acc
        is_best = False
        if val_puzzle_acc > best_puzzle_acc:
            is_best = True
        elif val_puzzle_acc == best_puzzle_acc and val_cell_acc > best_cell_acc:
            is_best = True
            
        if is_best:
            best_puzzle_acc = val_puzzle_acc
            best_cell_acc = val_cell_acc
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "model_best.pt"))
            print(f"  >>> New Best Model Saved! (P:{best_puzzle_acc:.2f} C:{best_cell_acc:.2f})")

    plot_convergence(loss_history, os.path.join(OUTPUT_DIR, "convergence.png"))

if __name__ == "__main__":
    train()