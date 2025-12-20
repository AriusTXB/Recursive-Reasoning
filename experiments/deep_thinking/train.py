import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from random import randrange
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.dataset import SudokuDataset
from src.visualizer import plot_convergence
from model import SudokuDeepThinking2D as SudokuDeepThinkingModel

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- The Paper's Logic Functions ---

def get_output_for_prog_loss(inputs, max_iters, net):
    """
    Stochastic Depth Logic:
    1. Pick a random start point 'n'
    2. Pick a random duration 'k'
    3. Run 'n' steps (detached)
    4. Run 'k' steps (with grad)
    """
    # get features from n iterations to use as input
    n = randrange(0, max_iters)

    # do k iterations using intermediate features as input
    k = randrange(1, max_iters - n + 1)

    if n > 0:
        # Run first n steps without gradients for the intermediate state
        with torch.no_grad():
            _, interim_thought = net(inputs, iters_to_do=n)
        interim_thought = interim_thought.detach()
    else:
        interim_thought = None

    # Run k steps from that state
    outputs, _ = net(inputs, iters_to_do=k, interim_thought=interim_thought)
    return outputs, k

def train():
    print("--- Training Deep Thinking 2D (Original Paper Implementation) ---")

    # Hyperparameters
    BATCH_SIZE = 64
    MAX_ITERS = 30
    ALPHA = 0.5        # Weighting between Max-Iter loss and Progressive loss
    LR = 0.0005
    EPOCHS = 20
    CLIP = 1.0

    # Data
    train_ds = SudokuDataset("data/processed", split="train")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # Model
    net = SudokuDeepThinkingModel(width=128, recall=True, max_iters=30).to(DEVICE)
    
    optimizer = optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
    # Using 'none' reduction because the paper's code calculates mean manually after masking
    criterion = nn.CrossEntropyLoss(reduction="none") 
    
    loss_history = []

    for epoch in range(EPOCHS):
        net.train()
        total_loss = 0
        correct = 0
        total_samples = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch_idx, (inputs, targets) in enumerate(loop):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE).long()
            
            # Flatten targets for CrossEntropy: [Batch, 81] -> [Batch*81]
            targets_flat = targets.view(-1)

            optimizer.zero_grad()

            # --- 1. Max Iters Loss (L_max) ---
            # Run the model for the full 30 steps
            outputs_max_iters, _ = net(inputs, iters_to_do=MAX_ITERS)
            
            # Flatten outputs: [Batch, 81, 11] -> [Batch*81, 11]
            outputs_max_flat = outputs_max_iters.reshape(-1, 11)
            
            if ALPHA != 1:
                loss_max_iters = criterion(outputs_max_flat, targets_flat)
            else:
                loss_max_iters = torch.zeros_like(targets_flat).float()

            # --- 2. Progressive Loss (L_prog) ---
            # Run the stochastic depth logic
            if ALPHA != 0:
                outputs_prog, k = get_output_for_prog_loss(inputs, MAX_ITERS, net)
                outputs_prog_flat = outputs_prog.reshape(-1, 11)
                loss_progressive = criterion(outputs_prog_flat, targets_flat)
            else:
                loss_progressive = torch.zeros_like(targets_flat).float()

            # Note: The original paper handles "Mazes" masking here. 
            # For Sudoku, we might want to mask the loss for the "given" digits 
            # (only calculate loss on the empty cells).
            # However, standard Sudoku training often trains on all cells to enforce consistency.
            # We will use mean() over all cells.

            loss_max_iters_mean = loss_max_iters.mean()
            loss_progressive_mean = loss_progressive.mean()

            # Combine Losses
            loss = (1 - ALPHA) * loss_max_iters_mean + ALPHA * loss_progressive_mean
            
            loss.backward()

            if CLIP is not None:
                torch.nn.utils.clip_grad_norm_(net.parameters(), CLIP)
            optimizer.step()

            # Metrics
            train_loss = loss.item()
            
            # Calculate accuracy based on Max Iters output
            preds = torch.argmax(outputs_max_iters, dim=-1) # [Batch, 81]
            # Accuracy: How many full rows are 100% correct?
            row_matches = (preds == targets).all(dim=1)
            correct += row_matches.sum().item()
            total_samples += inputs.size(0)
            
            total_loss += train_loss
            loop.set_postfix(loss=train_loss, acc=f"{100.*correct/total_samples:.2f}%")

        # End of Epoch
        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)
        
        # Save Checkpoint
        torch.save(net.state_dict(), os.path.join(OUTPUT_DIR, "model_dt_paper.pt"))
        print(f"Epoch {epoch+1} done. Avg Loss: {avg_loss:.4f}")

    plot_convergence(loss_history, os.path.join(OUTPUT_DIR, "convergence.png"))

if __name__ == "__main__":
    train()