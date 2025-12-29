import sys
import os
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.evaluators import evaluate_model

# --- IMPORT MODELS ---
# Adjust paths if your folder names differ slightly (e.g. deep_thinking vs deep_thinking_2d)
try:
    from experiments.baseline_transformer.model import BaselineTransformer
    from experiments.deep_thinking.model import SudokuDeepThinking2D
    from experiments.feed_forward_neural_net.model import SudokuFeedForward2D
    from experiments.trm.model import TinyRecursiveReasoningModel_ACTV1
    from experiments.neural_operators.model import NeuralOperatorSolver
except ImportError as e:
    print(f"Import Error: {e}. Check folder names (e.g., deep_thinking_2d vs deep_thinking).")
    sys.exit(1)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- WRAPPERS ---

class TRMEvalWrapper(nn.Module):
    """
    Adapts TRM (Carry/State/Halt) -> Standard (Logits)
    """
    def __init__(self, inner_model):
        super().__init__()
        self.inner = inner_model

    def forward(self, x):
        B = x.shape[0]
        batch = {
            "inputs": x,
            "labels": torch.zeros_like(x),
            "puzzle_identifiers": torch.zeros(B, dtype=torch.int32).to(x.device)
        }
        carry = self.inner.initial_carry(batch)
        carry.halted[:] = False 
        
        final_logits = None
        # Run full depth for benchmark consistency
        max_steps = self.inner.config.halt_max_steps
        
        for _ in range(max_steps):
            carry, outputs = self.inner(carry, batch)
            final_logits = outputs["logits"]
            if carry.halted.all(): break
        
        # Return tuple to match "deep_thinking" evaluator mode
        return final_logits, None

class NeuralOpEvalWrapper(nn.Module):
    """
    Adapts Neural Operator (Grid Output + Time Input) -> Standard (Flat Logits)
    """
    def __init__(self, inner_model):
        super().__init__()
        self.inner = inner_model
    
    def forward(self, x):
        # x: [Batch, 81]
        B = x.shape[0]
        
        # We want the solution, which corresponds to t=1.0 in our training formulation
        t_vec = torch.ones(B, 1, device=x.device)
        
        # Forward: [B, 11, 9, 9]
        logits_grid = self.inner(x, t_vec)
        
        # Reshape to [B, 81, 11] for the evaluator (which expects channel-last flattened)
        # 1. Flatten spatial: [B, 11, 81]
        # 2. Transpose: [B, 81, 11]
        logits_flat = logits_grid.flatten(2).transpose(1, 2)
        
        return logits_flat

# --- BENCHMARK LOGIC ---

def bench(name, model, path, model_type="standard"):
    print(f"Benchmarking {name}...")
    try:
        if not os.path.exists(path):
            print(f"  [Skip] Model file not found: {path}")
            return None

        state_dict = torch.load(path, map_location=DEVICE)
        
        # Unwrap if needed for loading state dict
        if hasattr(model, 'inner'):
            model.inner.load_state_dict(state_dict)
            model.inner.to(DEVICE)
        else:
            model.load_state_dict(state_dict)
            model.to(DEVICE)
            
        c, p = evaluate_model(model, DEVICE, split="test", model_type=model_type)
        print(f"  -> Result: Puzzle Acc: {p:.2f}% | Cell Acc: {c:.2f}%")
        return {"Model": name, "Cell Acc": c, "Puzzle Acc": p}
        
    except Exception as e:
        print(f"  [Error] Failed benchmarking {name}: {e}")
        return None

def main():
    results = []
    
    # 1. Baseline Transformer
    # Config from scaled version: d_model=384, layers=6
    # Or base version: d_model=128, layers=4. 
    # Try loading scaled first, fallback to base logic if needed or ensure this matches training.
    # Assuming "Scaled" configs from last step:
    m1 = BaselineTransformer(vocab_size=11, d_model=384, nhead=8, num_layers=6) 
    res1 = bench("Baseline", m1, "experiments/baseline_transformer/output/model_best.pt")
    if res1: results.append(res1)
    
    # 2. Deep Thinking 2D
    # Scaled: width=384
    m2 = SudokuDeepThinking2D(width=384, recall=True, max_iters=30)
    res2 = bench("Deep Thinking", m2, "experiments/deep_thinking/output/model_best.pt", "deep_thinking")
    if res2: results.append(res2)
    
    # 3. FeedForward 2D
    # Scaled: width=192, layers=12
    m3 = SudokuFeedForward2D(width=192, recall=True, max_iters=12)
    res3 = bench("FeedForward", m3, "experiments/feed_forward_neural_net/output/model_best.pt", "standard")
    if res3: results.append(res3)

    # 4. TRM (Tiny Recursive Model)
    trm_config = {
        "batch_size": 1, "seq_len": 81, "vocab_size": 11,
        "puzzle_emb_ndim": 0, "num_puzzle_identifiers": 1,
        "H_cycles": 1, "L_cycles": 2, "H_layers": 0, "L_layers": 2,
        # Scaled config:
        "hidden_size": 768, "expansion": 4.0, "num_heads": 12,
        "pos_encodings": "rope", "halt_max_steps": 20,
        "halt_exploration_prob": 0.0, "forward_dtype": "float32", "puzzle_emb_len": 0
    }
    trm_inner = TinyRecursiveReasoningModel_ACTV1(trm_config)
    m4 = TRMEvalWrapper(trm_inner)
    res4 = bench("TRM", m4, "experiments/trm/output/model_trm.pt", "deep_thinking")
    if res4: results.append(res4)

    # 5. Neural Operator
    # Scaled config: hidden_dim=384, depth=4
    nop_inner = NeuralOperatorSolver(vocab_size=11, hidden_dim=384, depth=4)
    m5 = NeuralOpEvalWrapper(nop_inner)
    res5 = bench("Neural Operator", m5, "experiments/neural_operators/output/model_nop.pt", "standard")
    if res5: results.append(res5)

    # --- Plotting ---
    if not results:
        print("No models successfully evaluated.")
        return

    df = pd.DataFrame(results)
    df.to_csv("benchmark_results.csv", index=False)
    print("\nFinal Results:\n", df)

    try:
        df_melt = df.melt(id_vars="Model", value_name="Accuracy (%)", var_name="Metric")
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df_melt, x="Model", y="Accuracy (%)", hue="Metric")
        plt.title("Reasoning Model Comparison (Scaled)")
        plt.ylim(0, 100)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig("benchmark_comparison.png")
        print("Chart saved to benchmark_comparison.png")
    except Exception as e:
        print(f"Plotting error: {e}")

if __name__ == "__main__":
    main()