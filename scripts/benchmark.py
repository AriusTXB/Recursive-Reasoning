import sys
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.evaluators import evaluate_model

# Import Models
from experiments.baseline_transformer.model import BaselineTransformer
from experiments.deep_thinking.model import SudokuDeepThinking2D
from experiments.feed_forward_neural_net.model import SudokuFeedForward2D

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def bench(name, model, path, model_type="standard"):
    print(f"Benchmarking {name}...")
    try:
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        model.to(DEVICE)
        c, p = evaluate_model(model, DEVICE, split="test", model_type=model_type)
        return {"Model": name, "Cell Acc": c, "Puzzle Acc": p}
    except FileNotFoundError:
        print(f"  X Model file not found: {path}")
        return None

def main():
    results = []
    
    # 1. Baseline
    m1 = BaselineTransformer(vocab_size=11, d_model=128, nhead=4, num_layers=4)
    res1 = bench("Baseline Transf.", m1, "experiments/baseline_transformer/output/model_best.pt")
    if res1: results.append(res1)
    
    # 2. Deep Thinking
    m2 = SudokuDeepThinking2D(width=128, recall=True, max_iters=30)
    res2 = bench("Deep Thinking 2D", m2, "experiments/deep_thinking_2d/output/model_best.pt", "deep_thinking")
    if res2: results.append(res2)
    
    # 3. FeedForward
    m3 = SudokuFeedForward2D(width=128, recall=True, max_iters=20)
    res3 = bench("FeedForward 2D", m3, "experiments/feedforward_2d/output/model_best.pt", "standard")
    if res3: results.append(res3)

    if not results:
        print("No models evaluated.")
        return

    # CSV
    df = pd.DataFrame(results)
    df.to_csv("benchmark_results.csv", index=False)
    print("\nResults:\n", df)

    # Chart
    df_melt = df.melt(id_vars="Model", value_name="Accuracy (%)", var_name="Metric")
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df_melt, x="Model", y="Accuracy (%)", hue="Metric")
    plt.title("Reasoning Capability Comparison")
    plt.ylim(0, 100)
    plt.savefig("benchmark_comparison.png")
    print("Chart saved to benchmark_comparison.png")

if __name__ == "__main__":
    main()