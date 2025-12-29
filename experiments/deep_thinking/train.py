import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from random import randrange

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.dataset import SudokuDataset
from src.visualizer import plot_convergence
from src.evaluators import evaluate_model
from model import SudokuDeepThinking2D

EPOCH = 100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_prog_loss(inputs, max_iters, net):
    n = randrange(0, max_iters)
    k = randrange(1, max_iters - n + 1)
    if n > 0:
        with torch.no_grad(): _, interim = net(inputs, iters_to_do=n)
        interim = interim.detach()
    else: interim = None
    return net(inputs, iters_to_do=k, interim_thought=interim)[0]

def train():
    print(f"--- Training LARGE Deep Thinking 2D on {DEVICE} ---")
    
    # 1. Config (Scaled to ~10M Params)
    BATCH_SIZE = 32 # Reduced batch size for larger model
    MAX_ITERS = 30
    WIDTH = 384     # Scaled up from 128
    
    train_ds = SudokuDataset("data/processed", split="train")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    model = SudokuDeepThinking2D(width=WIDTH, recall=True, max_iters=MAX_ITERS).to(DEVICE)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    best_puzzle_acc = -1.0
    best_cell_acc = -1.0

    for epoch in range(EPOCH):
        model.train()
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            targets_flat = targets.view(-1)
            optimizer.zero_grad()

            out_max, _ = model(inputs, iters_to_do=MAX_ITERS)
            out_prog = get_prog_loss(inputs, MAX_ITERS, model)
            
            loss = 0.5 * criterion(out_max.reshape(-1, 11), targets_flat) + \
                   0.5 * criterion(out_prog.reshape(-1, 11), targets_flat)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        # --- Smart Save ---
        val_cell_acc, val_puzzle_acc = evaluate_model(model, DEVICE, split="test", model_type="deep_thinking")
        print(f"  Test Puzzle Acc: {val_puzzle_acc:.2f}% | Cell Acc: {val_cell_acc:.2f}%")
        
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "model_latest.pt"))
        
        if val_puzzle_acc > best_puzzle_acc or (val_puzzle_acc == best_puzzle_acc and val_cell_acc > best_cell_acc):
            best_puzzle_acc = val_puzzle_acc
            best_cell_acc = val_cell_acc
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "model_best.pt"))
            print("  >>> New Best Model Saved!")
if __name__ == "__main__":
    train()