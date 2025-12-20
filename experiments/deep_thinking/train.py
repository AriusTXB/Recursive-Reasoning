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

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_output_for_prog_loss(inputs, max_iters, net):
    n = randrange(0, max_iters)
    k = randrange(1, max_iters - n + 1)

    if n > 0:
        with torch.no_grad():
            _, interim_thought = net(inputs, iters_to_do=n)
        interim_thought = interim_thought.detach()
    else:
        interim_thought = None

    outputs, _ = net(inputs, iters_to_do=k, interim_thought=interim_thought)
    return outputs, k

def train():
    print(f"--- Training Deep Thinking 2D on {DEVICE} ---")

    BATCH_SIZE = 64
    MAX_ITERS = 30
    ALPHA = 0.5
    LR = 0.0005
    EPOCHS = 20
    CLIP = 1.0

    train_ds = SudokuDataset("data/processed", split="train")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    model = SudokuDeepThinking2D(width=128, recall=True, max_iters=MAX_ITERS).to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(reduction="none") 
    
    loss_history = []
    best_puzzle_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for inputs, targets in loop:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE).long()
            targets_flat = targets.view(-1)

            optimizer.zero_grad()

            # 1. Max Iters Loss
            outputs_max, _ = model(inputs, iters_to_do=MAX_ITERS)
            loss_max = criterion(outputs_max.reshape(-1, 11), targets_flat).mean()

            # 2. Progressive Loss
            if ALPHA > 0:
                outputs_prog, k = get_output_for_prog_loss(inputs, MAX_ITERS, model)
                loss_prog = criterion(outputs_prog.reshape(-1, 11), targets_flat).mean()
            else:
                loss_prog = torch.tensor(0.0, device=DEVICE)

            loss = (1 - ALPHA) * loss_max + ALPHA * loss_prog
            
            loss.backward()
            if CLIP:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        
        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)
        
        # --- EVALUATION & SAVING ---
        # Note: model_type="deep_thinking" ensures the evaluator handles the tuple return correctly
        val_cell_acc, val_puzzle_acc = evaluate_model(model, DEVICE, split="test", model_type="deep_thinking")
        
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Train Loss: {avg_loss:.4f}")
        print(f"  Test Puzzle Acc: {val_puzzle_acc:.2f}% (Cell Acc: {val_cell_acc:.2f}%)")
        
        # Save Latest
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "model_latest.pt"))
        
        # Save Best
        if val_puzzle_acc > best_puzzle_acc:
            best_puzzle_acc = val_puzzle_acc
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "model_best.pt"))
            print(f"  >>> New Best Model Saved! <<<")

        with open(os.path.join(OUTPUT_DIR, "metrics.csv"), "a") as f:
            if epoch == 0: f.write("Epoch,Loss,CellAcc,PuzzleAcc\n")
            f.write(f"{epoch+1},{avg_loss},{val_cell_acc},{val_puzzle_acc}\n")

    plot_convergence(loss_history, os.path.join(OUTPUT_DIR, "convergence.png"))
    print(f"Training Complete. Best Accuracy: {best_puzzle_acc:.2f}%")

if __name__ == "__main__":
    train()