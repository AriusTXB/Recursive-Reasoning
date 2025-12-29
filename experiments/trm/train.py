import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
import numpy as np
import copy
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.dataset import SudokuDataset
from src.evaluators import evaluate_model
from model import TinyRecursiveReasoningModel_ACTV1

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
EPOCH = 100
# --- LARGE CONFIGURATION ---
CONFIG = {
    "batch_size": 32, # Lower batch size for large model
    "seq_len": 81, "vocab_size": 11, "puzzle_emb_ndim": 0, "num_puzzle_identifiers": 1,
    "H_cycles": 1, "L_cycles": 2, "H_layers": 0, "L_layers": 2,
    
    # SCALED PARAMS
    "hidden_size": 768, 
    "expansion": 4.0, 
    "num_heads": 12,
    
    "pos_encodings": "rope", "forward_dtype": "float32", "puzzle_emb_len": 0,
    "halt_max_steps": 20, "halt_exploration_prob": 0.0,
    
    "total_steps": 10000, "lr": 0.0005, "eval_interval": 500, "log_interval": 50, "ema_rate": 0.995
}

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

class SudokuStream(IterableDataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.indices = np.arange(len(dataset))
    def __iter__(self):
        while True:
            np.random.shuffle(self.indices)
            for idx in self.indices: yield self.dataset[idx]

def train():
    print(f"--- Training LARGE TRM on {DEVICE} ---")
    
    train_ds_raw = SudokuDataset("data/processed", "train")
    stream_ds = SudokuStream(train_ds_raw)
    loader = DataLoader(stream_ds, batch_size=CONFIG["batch_size"])
    iterator = iter(loader)

    model = TinyRecursiveReasoningModel_ACTV1(CONFIG).to(DEVICE)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=1e-4)
    criterion_ce = nn.CrossEntropyLoss()
    ema = EMAHelper(model, decay=CONFIG["ema_rate"])

    # Init State
    init_x, init_y = next(iterator)
    init_batch = {
        "inputs": init_x.to(DEVICE), "labels": init_y.to(DEVICE),
        "puzzle_identifiers": torch.zeros(CONFIG["batch_size"], dtype=torch.int32).to(DEVICE)
    }
    carry = model.initial_carry(init_batch)
    
    best_puzzle_acc = -1.0
    best_cell_acc = -1.0
    
    model.train()
    prev_z_L = None

    for step in range(1, CONFIG["total_steps"] + 1):
        force_prob = max(0.0, 1.0 - (step / (0.8 * CONFIG["total_steps"])))
        model.config.halt_exploration_prob = force_prob
        
        try: next_x, next_y = next(iterator)
        except: iterator = iter(loader); next_x, next_y = next(iterator)
        
        batch_data = {
            "inputs": next_x.to(DEVICE), "labels": next_y.to(DEVICE),
            "puzzle_identifiers": torch.zeros(CONFIG["batch_size"], dtype=torch.int32).to(DEVICE)
        }

        if not carry.halted.all(): prev_z_L = carry.inner_carry.z_L.clone().detach()
        else: prev_z_L = None

        carry, outputs = model(carry, batch_data)

        logits = outputs["logits"]
        targets = carry.current_data["labels"]
        loss_ce = criterion_ce(logits.view(-1, 11), targets.view(-1))
        
        preds = torch.argmax(logits, dim=-1)
        is_correct = (preds == targets).all(dim=1).float()
        reward = torch.where(is_correct == 1.0, torch.tensor(1.0, device=DEVICE), torch.tensor(-1.0, device=DEVICE))
        loss_q = F.mse_loss(outputs["q_halt"], reward)
        
        loss_stag = torch.tensor(0.0, device=DEVICE)
        if prev_z_L is not None:
            curr_z_L = carry.inner_carry.z_L
            valid = ~carry.halted
            if valid.any():
                loss_stag = F.cosine_similarity(prev_z_L[valid].flatten(1), curr_z_L[valid].flatten(1)).mean()

        total_loss = loss_ce + 0.1 * loss_q + 0.1 * loss_stag

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        ema.update(model)
        
        carry.inner_carry.z_L = carry.inner_carry.z_L.detach()

        if step % CONFIG["log_interval"] == 0:
            print(f"Step {step} | Loss: {total_loss.item():.4f} | Force%: {force_prob:.2f}")

        # --- EVALUATION ---
        if step % CONFIG["eval_interval"] == 0 or step == CONFIG["total_steps"]:
            print(">>> Evaluating...")
            test_model = ema.apply_shadow(model)
            
            class EvalWrapper(nn.Module):
                def __init__(self, inner): super().__init__(); self.inner = inner
                def forward(self, x):
                    B = x.shape[0]
                    b = {"inputs": x, "labels": torch.zeros_like(x), "puzzle_identifiers": torch.zeros(B, dtype=torch.int32).to(x.device)}
                    c = self.inner.initial_carry(b); c.halted[:] = False
                    f = None
                    for _ in range(EPOCH): c, o = self.inner(c, b); f = o["logits"]
                    return f, None

            wrapped = EvalWrapper(test_model)
            val_cell_acc, val_puzzle_acc = evaluate_model(wrapped, DEVICE, split="test", model_type="deep_thinking")
            print(f"  Result: P:{val_puzzle_acc:.2f}% C:{val_cell_acc:.2f}%")
            
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "model_latest.pt"))
            
            # Smart Save logic
            if val_puzzle_acc > best_puzzle_acc or (val_puzzle_acc == best_puzzle_acc and val_cell_acc > best_cell_acc):
                best_puzzle_acc = val_puzzle_acc
                best_cell_acc = val_cell_acc
                torch.save(test_model.state_dict(), os.path.join(OUTPUT_DIR, "model_best.pt"))
                print(">>> New Best Model Saved!")

if __name__ == "__main__":
    train()