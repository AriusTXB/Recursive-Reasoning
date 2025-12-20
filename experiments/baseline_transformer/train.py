import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import time

# --- IMPORTS ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.dataset import SudokuDataset
from src.evaluators import evaluate_model
from model import TinyRecursiveReasoningModel_ACTV1

# --- CONFIGURATION ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_CONFIG = {
    "batch_size": 64, # Matches loader batch size
    "seq_len": 81,
    "vocab_size": 11,
    "puzzle_emb_ndim": 0,
    "num_puzzle_identifiers": 1,
    "H_cycles": 1, "L_cycles": 2, "H_layers": 0, "L_layers": 2,
    "hidden_size": 128, "expansion": 4.0, "num_heads": 4,
    "pos_encodings": "rope", 
    "halt_max_steps": 20, # We will unroll up to this many steps
    "halt_exploration_prob": 0.0, # Handled manually in loop
    "forward_dtype": "float32"
}

def train():
    print(f"--- Classic Unrolled Training on {DEVICE} ---")
    
    # 1. Hyperparameters
    BATCH_SIZE = 64
    EPOCHS = 10
    LR = 0.001
    MAX_STEPS = 20  # How many thinking steps to unroll
    
    # 2. Data
    train_ds = SudokuDataset("data/processed", "train")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)
    
    # 3. Model
    model = TinyRecursiveReasoningModel_ACTV1(MODEL_CONFIG).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    
    best_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        total_epoch_loss = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for inputs, labels in loop:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            # Prepare Batch Dict
            batch_data = {
                "inputs": inputs,
                "labels": labels,
                "puzzle_identifiers": torch.zeros(BATCH_SIZE, dtype=torch.int32).to(DEVICE)
            }
            
            # Init State
            carry = model.initial_carry(batch_data)
            
            # --- UNROLLED TRAINING LOOP (BPTT) ---
            batch_loss = 0
            
            # We assume every puzzle needs at least some steps. 
            # We force it to run MAX_STEPS to ensure gradients flow for long-term reasoning.
            for step in range(MAX_STEPS):
                # 1. Force Continue: Override any internal decision to halt
                carry.halted[:] = False 
                
                # 2. Forward Step
                carry, outputs = model(carry, batch_data)
                
                # 3. Calculate Loss for THIS step
                logits = outputs["logits"]
                q_halt = outputs["q_halt"]
                
                # A. Reconstruction Loss (The reasoning)
                loss_ce = F.cross_entropy(logits.view(-1, 11), labels.view(-1))
                
                # B. Q-Learning Loss (The halting decision)
                # Calculate Reward: +1.0 if fully correct, -1.0 if ANY mistake
                preds = torch.argmax(logits, dim=-1)
                is_correct_row = (preds == labels).all(dim=1).float() # [Batch]
                
                # If correct -> Target Q is 1.0. If wrong -> Target Q is -1.0
                target_q = torch.where(is_correct_row == 1.0, 
                                     torch.tensor(1.0, device=DEVICE), 
                                     torch.tensor(-1.0, device=DEVICE))
                
                loss_q = F.mse_loss(q_halt, target_q)
                
                # Accumulate
                # We weight later steps slightly more? No, uniform is fine for "Classic".
                step_loss = loss_ce + 0.5 * loss_q
                batch_loss += step_loss
            
            # Average loss over time steps (Standard BPTT)
            final_loss = batch_loss / MAX_STEPS
            
            # 4. Backward & Update
            optimizer.zero_grad()
            final_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_epoch_loss += final_loss.item()
            loop.set_postfix(loss=final_loss.item())

        # --- Evaluation ---
        # We need a stateless wrapper for the evaluator
        class StatelessWrapper(nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner
            def forward(self, x):
                # Create batch
                B = x.shape[0]
                batch = {
                    "inputs": x,
                    "labels": torch.zeros_like(x),
                    "puzzle_identifiers": torch.zeros(B, dtype=torch.int32).to(x.device)
                }
                carry = self.inner.initial_carry(batch)
                
                # Run Inference Loop (Letting it halt naturally)
                # Or force max steps for robust benchmark
                carry.halted[:] = False
                final_logits = None
                
                for _ in range(self.inner.config.halt_max_steps):
                    carry, outputs = self.inner(carry, batch)
                    final_logits = outputs["logits"]
                
                return final_logits, None

        eval_wrapper = StatelessWrapper(model)
        cell_acc, puzzle_acc = evaluate_model(eval_wrapper, DEVICE, split="test", model_type="deep_thinking")
        
        print(f"Epoch {epoch+1} Done. Loss: {total_epoch_loss/len(train_loader):.4f} | Val Puzzle Acc: {puzzle_acc:.2f}%")
        
        # Save
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "model_trm_classic.pt"))
        if puzzle_acc > best_acc:
            best_acc = puzzle_acc
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "model_best.pt"))
            print(">>> New Best Model Saved!")

if __name__ == "__main__":
    train()