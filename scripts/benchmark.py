import sys
import os
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.evaluator import evaluate_model

# Import Models
from experiments.baseline_transformer.model import BaselineTransformer
from experiments.deep_thinking_2d.model import SudokuDeepThinking2D
from experiments.feedforward_2d.model import SudokuFeedForward2D
from experiments.trm.model import TinyRecursiveReasoningModel_ACTV1

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TRMEvalWrapper(nn.Module):
    """
    Adapts the complex TinyRecursiveReasoningModel to work with the 
    standard evaluate_model() function.
    """
    def __init__(self, inner_model):
        super().__init__()
        self.inner = inner_model

    def forward(self, x):
        # x shape: [Batch, 81]
        B = x.shape[0]
        
        # 1. Prepare the dictionary batch TRM expects
        batch = {
            "inputs": x,
            "labels": torch.zeros_like(x), # Dummy labels, not needed for inference
            "puzzle_identifiers": torch.zeros(B, dtype=torch.int32).to(x.device)
        }

        # 2. Initialize State
        carry = self.inner.initial_carry(batch)
        
        # 3. Force Start (Reset halting flag to ensure we don't start halted)
        carry.halted[:] = False 

        # 4. Run the reasoning loop
        # We run for max_steps to give the model time to think
        final_logits = None
        max_steps = self.inner.config.halt_max_steps
        
        for _ in range(max_steps):
            carry, outputs = self.inner(carry, batch)
            final_logits = outputs["logits"]
            
            # Optimization: If all inputs have halted, we could break early.
            # But for benchmarking, running full depth is safer.
            if carry.halted.all():
                break
        
        # Return tuple matching "deep_thinking" signature: (logits, auxiliary)
        return final_logits, None

def bench(name, model, path, model_type="standard"):
    print(f"Benchmarking {name}...")
    try:
        # Check if path exists
        if not os.path.exists(path):
            print(f"  X Model file not found: {path}")
            return None

        # Load Weights
        state_dict = torch.load(path, map_location=DEVICE)
        
        # Handle wrapper cases (like TRM) where we load into the inner model
        if isinstance(model, TRMEvalWrapper):
            model.inner.load_state_dict(state_dict)
            model.inner.to(DEVICE)
        else:
            model.load_state_dict(state_dict)
            model.to(DEVICE)
            
        # Evaluate
        c, p = evaluate_model(model, DEVICE, split="test", model_type=model_type)
        print(f"  -> Result: Cell Acc: {c:.2f}%, Puzzle Acc: {p:.2f}%")
        return {"Model": name, "Cell Acc": c, "Puzzle Acc": p}
        
    except Exception as e:
        print(f"  X Error benchmarking {name}: {e}")
        return None

def main():
    results = []
    
    # --- 1. Baseline Transformer ---
    m1 = BaselineTransformer(vocab_size=11, d_model=128, nhead=4, num_layers=4)
    res1 = bench("Baseline Transf.", m1, "experiments/baseline_transformer/output/model_best.pt")
    if res1: results.append(res1)
    
    # --- 2. Deep Thinking 2D ---
    m2 = SudokuDeepThinking2D(width=128, recall=True, max_iters=30)
    res2 = bench("Deep Thinking 2D", m2, "experiments/deep_thinking_2d/output/model_best.pt", "deep_thinking")
    if res2: results.append(res2)
    
    # --- 3. FeedForward 2D ---
    m3 = SudokuFeedForward2D(width=128, recall=True, max_iters=20)
    res3 = bench("FeedForward 2D", m3, "experiments/feedforward_2d/output/model_best.pt", "standard")
    if res3: results.append(res3)

    # --- 4. TRM (Tiny Recursive Model) ---
    # Configuration must match your training config
    trm_config = {
        "batch_size": 1, # Placeholder, dynamic in forward
        "seq_len": 81,
        "vocab_size": 11,
        "puzzle_emb_ndim": 0,
        "num_puzzle_identifiers": 1,
        "H_cycles": 1, "L_cycles": 2, "H_layers": 0, "L_layers": 2,
        "hidden_size": 128, "expansion": 4.0, "num_heads": 4,
        "pos_encodings": "rope", 
        "halt_max_steps": 20,
        "halt_exploration_prob": 0.0,
        "forward_dtype": "float32"
    }
    trm_inner = TinyRecursiveReasoningModel_ACTV1(trm_config)
    m4 = TRMEvalWrapper(trm_inner)
    
    # Note: Use model_type="deep_thinking" because TRMEvalWrapper returns (logits, None)
    res4 = bench("TRM (Recursive)", m4, "experiments/trm/output/model_best.pt", "deep_thinking")
    if res4: results.append(res4)

    # --- Results & Plotting ---
    if not results:
        print("No models evaluated.")
        return

    # CSV
    df = pd.DataFrame(results)
    df.to_csv("benchmark_results.csv", index=False)
    print("\nFinal Results Table:\n", df)

    # Chart
    try:
        df_melt = df.melt(id_vars="Model", value_name="Accuracy (%)", var_name="Metric")
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df_melt, x="Model", y="Accuracy (%)", hue="Metric")
        plt.title("Reasoning Capability Comparison")
        plt.ylim(0, 100)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig("benchmark_comparison.png")
        print("Chart saved to benchmark_comparison.png")
    except Exception as e:
        print(f"Could not generate plot: {e}")

if __name__ == "__main__":
    main()