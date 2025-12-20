import sys
import os
import torch
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.evaluators import evaluate_model

# Import your model classes
# Note: Ensure you have __init__.py in experiment folders or use sys.path tricks
try:
    from experiments.baseline_transformer.model import BaselineTransformer
    from experiments.deep_thinking.model import SudokuDeepThinking2D
    from experiments.feed_forward_neural_net.model import SudokuFeedForward2D
    # Add others as needed
except ImportError as e:
    print(f"Import Error: {e}")
    print("Ensure you have __init__.py files in your experiment directories!")
    sys.exit(1)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_and_bench(name, model_class, path, model_kwargs, model_type="standard"):
    print(f"--- Benchmarking {name} ---")
    if not os.path.exists(path):
        print(f"Skipping {name}: File not found at {path}")
        return None

    # Init model
    model = model_class(**model_kwargs).to(DEVICE)
    
    # Load weights
    try:
        model.load_state_dict(torch.load(path, map_location=DEVICE))
    except Exception as e:
        print(f"Failed to load weights for {name}: {e}")
        return None

    # Evaluate
    cell_acc, puzzle_acc = evaluate_model(model, DEVICE, split="test", model_type=model_type)
    return {"Model": name, "Cell Accuracy": cell_acc, "Puzzle Accuracy": puzzle_acc}

def main():
    results = []

    # 1. Baseline Transformer
    results.append(load_and_bench(
        "Baseline Transformer",
        BaselineTransformer,
        "experiments/baseline_transformer/output/model_latest.pt", # Use model_best.pt if you implemented it
        {"vocab_size": 11, "d_model": 128, "nhead": 4, "num_layers": 4}
    ))

    # 2. Deep Thinking 2D
    results.append(load_and_bench(
        "Deep Thinking 2D",
        SudokuDeepThinking2D,
        "experiments/deep_thinking_2d/output/model_dt2d.pt", # Or model_best.pt
        {"width": 128, "recall": True, "max_iters": 30},
        model_type="deep_thinking"
    ))

    # 3. FeedForward 2D (Control)
    results.append(load_and_bench(
        "FeedForward 2D",
        SudokuFeedForward2D,
        "experiments/feedforward_2d/output/model_ff2d.pt",
        {"width": 128, "recall": True, "max_iters": 20},
        model_type="standard"
    ))

    # Filter out None (failed loads)
    results = [r for r in results if r is not None]

    # Create DataFrame
    df = pd.DataFrame(results)
    print("\nFinal Results:")
    print(df)
    
    # Save CSV
    df.to_csv("benchmark_results.csv", index=False)

    # Plot
    if not df.empty:
        plt.figure(figsize=(10, 6))
        
        # Melt for side-by-side bars
        df_melted = df.melt(id_vars="Model", var_name="Metric", value_name="Accuracy (%)")
        
        sns.barplot(data=df_melted, x="Model", y="Accuracy (%)", hue="Metric")
        plt.title("Model Comparison: Sudoku Solving Accuracy")
        plt.ylim(0, 100)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig("benchmark_comparison.png")
        print("Comparison chart saved to benchmark_comparison.png")

if __name__ == "__main__":
    main()