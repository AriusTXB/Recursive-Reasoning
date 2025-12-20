import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.dataset import SudokuDataset
from src.visualizer import plot_convergence
from model import BaselineTransformer

# Config
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 64
EPOCHS = 10
LR = 0.001
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def train():
    print(f"Training on {DEVICE}")
    
    # Load Data
    train_dataset = SudokuDataset("data/processed", split="train")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    
    # Init Model
    model = BaselineTransformer().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    loss_history = []
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for inputs, labels in loop:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            # Forward
            logits = model(inputs) # [Batch, 81, 11]
            
            # Reshape for loss: [Batch*81, 11] vs [Batch*81]
            loss = criterion(logits.view(-1, 11), labels.view(-1))
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)
        
        # Save checkpoint
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "model_latest.pt"))
        print(f"Epoch {epoch+1} Complete. Avg Loss: {avg_loss:.4f}")

    # Plot Convergence
    plot_convergence(loss_history, os.path.join(OUTPUT_DIR, "convergence.png"))
    print("Training Complete. Model and Graph saved.")

if __name__ == "__main__":
    train()