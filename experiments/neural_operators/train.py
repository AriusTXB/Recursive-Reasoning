import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.dataset import SudokuDataset
from src.evaluators import evaluate_model
from model import NeuralOperatorSolver

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

class EvalWrapper(nn.Module):
    def __init__(self, inner): super().__init__(); self.inner = inner
    def forward(self, x):
        B = x.shape[0]
        t = torch.ones(B, 1, device=x.device)
        return self.inner(x, t).flatten(2).transpose(1, 2)

def train():
    print(f"--- Training LARGE Neural Operator on {DEVICE} ---")
    
    # Scaled Config (~12M params)
    BATCH_SIZE = 64
    EPOCHS = 100
    HIDDEN_DIM = 384
    DEPTH = 4
    
    train_ds = SudokuDataset("data/processed", "train")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    
    model = NeuralOperatorSolver(vocab_size=11, hidden_dim=HIDDEN_DIM, depth=DEPTH).to(DEVICE)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    best_puzzle_acc = -1.0
    best_cell_acc = -1.0
    
    for epoch in range(EPOCHS):
        model.train()
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            B, L = inputs.shape
            
            # Random time t
            t = torch.rand(B, 1, device=DEVICE)
            
            # Generate Target (Noise <-> Solution interpolation)
            is_clue = (inputs != 1)
            noise_grid = torch.randint(2, 11, (B, L), device=DEVICE)
            use_solution = torch.rand(B, L, device=DEVICE) < t
            target = torch.where(use_solution, labels, noise_grid)
            target = torch.where(is_clue, inputs, target)
            
            logits = model(inputs, t)
            loss = criterion(logits.flatten(2), target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # Eval
        eval_wrapper = EvalWrapper(model)
        val_cell_acc, val_puzzle_acc = evaluate_model(eval_wrapper, DEVICE, split="test", model_type="standard")
        print(f"  Test Puzzle Acc: {val_puzzle_acc:.2f}% | Cell Acc: {val_cell_acc:.2f}%")
        
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "model_latest.pt"))
        if val_puzzle_acc > best_puzzle_acc or (val_puzzle_acc == best_puzzle_acc and val_cell_acc > best_cell_acc):
            best_puzzle_acc = val_puzzle_acc
            best_cell_acc = val_cell_acc
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "model_nop.pt")) # Keep 'model_nop.pt' for consistency
            print("  >>> New Best Model Saved!")

if __name__ == "__main__":
    train()