import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import copy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.dataset import SudokuDataset
from src.evaluators import evaluate_model
from model import TinyRecursiveReasoningModel_ACTV1

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- LARGE MODEL CONFIGURATION ---
CONFIG = {
    "batch_size": 32, # Safe batch size
    "seq_len": 81, "vocab_size": 11,
    "puzzle_emb_ndim": 0, "num_puzzle_identifiers": 1,
    
    # Recurrence Args
    "H_cycles": 1, "L_cycles": 2, "H_layers": 0, "L_layers": 2,
    
    # Large Dimensions (~10M Params)
    "hidden_size": 768, 
    "expansion": 4.0, 
    "num_heads": 12,
    
    "pos_encodings": "rope", "forward_dtype": "float32",
    "puzzle_emb_len": 0, # Fixes size mismatch error
    
    "halt_max_steps": 20,
    "halt_exploration_prob": 0.0 # Controlled via Curriculum
}

# Training Hyperparams
LR = 5e-5          # Critical Fix: Lower LR for large recurrent model
WEIGHT_DECAY = 1e-2
EPOCHS = 15
GRAD_CLIP = 1.0

class EMAHelper:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.register(model)
    def register(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad: self.shadow[name] = param.data.clone()
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
    def apply_shadow(self, model):
        ema = copy.deepcopy(model)
        for name, param in ema.named_parameters():
            if param.requires_grad: param.data.copy_(self.shadow[name])
        return ema

def train():
    print(f"--- Training LARGE TRM (Stabilized) on {DEVICE} ---")
    
    # 1. Data
    train_ds = SudokuDataset("data/processed", "train")
    # Drop_last=True helps with shape consistency
    loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True, drop_last=True, num_workers=2)

    # 2. Model
    model = TinyRecursiveReasoningModel_ACTV1(CONFIG).to(DEVICE)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 3. Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion_ce = nn.CrossEntropyLoss()
    ema = EMAHelper(model, decay=0.995)
    
    best_acc = -1.0
    
    for epoch in range(EPOCHS):
        model.train()
        
        # --- Curriculum: Decay Force Prob ---
        # Epoch 0 -> 100% Forced. Epoch 10 -> 0% Forced.
        force_prob = max(0.0, 1.0 - (epoch / (0.7 * EPOCHS)))
        model.config.halt_exploration_prob = force_prob
        
        total_loss = 0
        avg_steps = 0
        
        loop = tqdm(loader, desc=f"Ep {epoch+1} (Force {force_prob:.2f})")
        
        for inputs, labels in loop:
            # Unpack and Dictionary Setup
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            batch_data = {
                "inputs": inputs, 
                "labels": labels,
                "puzzle_identifiers": torch.zeros(CONFIG["batch_size"], dtype=torch.int32).to(DEVICE)
            }
            
            # Init State
            carry = model.initial_carry(batch_data)
            
            # Save state for Stagnation Check
            prev_z_L = None
            
            # Forward
            # We must handle the carry carefully. 
            # In epoch-based training, we reset carry every batch, so no need to detach 'prev' carry.
            
            # Run Forward
            carry, outputs = model(carry, batch_data)
            
            # Loss 1: CE
            logits = outputs["logits"]
            # Important: Use labels from batch, not carry (carry labels might be masked/swapped)
            loss_ce = criterion_ce(logits.view(-1, 11), labels.view(-1))
            
            # Loss 2: Q-Learning
            preds = torch.argmax(logits, dim=-1)
            is_correct = (preds == labels).all(dim=1).float()
            # Strong Reward signal
            reward = torch.where(is_correct == 1.0, torch.tensor(1.0, device=DEVICE), torch.tensor(-1.0, device=DEVICE))
            loss_q = F.mse_loss(outputs["q_halt"], reward)
            
            # Loss 3: Stagnation (Optional but helpful)
            # We can't easily calc stagnation here because we didn't keep prev_z_L from inside the loop
            # The model internal loop handles recurrence.
            # We rely on Q-loss and CE-loss here.
            
            loss = loss_ce + 0.1 * loss_q
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            ema.update(model)
            
            total_loss += loss.item()
            current_steps = carry.steps.float().mean().item()
            avg_steps += current_steps
            
            loop.set_postfix(loss=loss.item(), steps=current_steps)
            
        # --- End of Epoch Eval ---
        print(f"  Avg Loss: {total_loss/len(loader):.4f} | Avg Steps: {avg_steps/len(loader):.2f}")
        
        # Evaluate
        test_model = ema.apply_shadow(model)
        
        # Eval Wrapper
        class EvalWrapper(nn.Module):
            def __init__(self, inner): super().__init__(); self.inner = inner
            def forward(self, x):
                B = x.shape[0]
                b = {"inputs": x, "labels": torch.zeros_like(x), "puzzle_identifiers": torch.zeros(B, dtype=torch.int32).to(x.device)}
                c = self.inner.initial_carry(b); c.halted[:] = False
                f = None
                for _ in range(20): c, o = self.inner(c, b); f = o["logits"]
                return f, None

        wrapped = EvalWrapper(test_model)
        val_cell, val_puzzle = evaluate_model(wrapped, DEVICE, split="test", model_type="deep_thinking")
        
        print(f"  Test Puzzle Acc: {val_puzzle:.2f}% | Cell Acc: {val_cell:.2f}%")
        
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "model_latest.pt"))
        if val_puzzle > best_acc:
            best_acc = val_puzzle
            torch.save(test_model.state_dict(), os.path.join(OUTPUT_DIR, "model_best.pt"))
            print("  >>> New Best Model Saved!")

if __name__ == "__main__":
    train()