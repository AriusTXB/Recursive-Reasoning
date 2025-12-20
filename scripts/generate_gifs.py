import sys
import os
import torch
import torch.nn.functional as F
import shutil

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dataset import SudokuDataset
from src.visualizer import draw_board, create_gif

# Import Models
from experiments.baseline_transformer.model import BaselineTransformer
from experiments.deep_thinking.model import SudokuDeepThinking2D
from experiments.feed_forward_neural_net.model import SudokuFeedForward2D

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GIF_OUT_DIR = "experiment_gifs"
os.makedirs(GIF_OUT_DIR, exist_ok=True)

def generate_gif_for_model(model_name, model, input_tensor, ground_truth, steps):
    print(f"Generating GIF for {model_name}...")
    temp_dir = os.path.join(GIF_OUT_DIR, f"temp_{model_name.replace(' ', '_')}")
    if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)

    model.eval()
    
    # 1. GENERATE STEPS
    # Different models have different forward signatures
    
    if "Baseline" in model_name:
        # Transformer: We must manually recurse
        curr_seq = input_tensor.clone()
        for i in range(steps):
            with torch.no_grad():
                logits = model(curr_seq)
                probs = F.softmax(logits, dim=-1)
                
                # Visualize
                conf = probs.max(dim=-1).values.mean().item()
                preds = torch.argmax(probs, dim=-1).cpu().numpy()[0]
                draw_board(preds, os.path.join(temp_dir, f"step_{i:02d}.png"), i, conf, ground_truth)
                
                # Update Board (Greedy fill)
                # Find empty cells (1s)
                mask = (curr_seq == 1)
                if not mask.any(): break
                
                digit_probs = probs[:, :, 2:] # Skip Pad/Empty
                max_probs, idxs = torch.max(digit_probs, dim=2)
                vals = idxs + 2
                max_probs[~mask] = -1 # Don't overwrite existing
                
                best_idx = torch.argmax(max_probs)
                curr_seq.view(-1)[best_idx] = vals.view(-1)[best_idx]

    elif "FeedForward" in model_name:
        # FeedForward: Call specific layers iteratively
        curr_thought = None
        for i in range(steps):
            with torch.no_grad():
                out, curr_thought = model(input_tensor, iters_to_do=1, interim_thought=curr_thought, iters_elapsed=i)
                
            probs = F.softmax(out, dim=-1)
            conf = probs.max(dim=-1).values.mean().item()
            preds = torch.argmax(probs, dim=-1).cpu().numpy()[0]
            draw_board(preds, os.path.join(temp_dir, f"step_{i:02d}.png"), i, conf, ground_truth)

    elif "Deep Thinking" in model_name:
        # Deep Thinking: Loop the single layer
        curr_thought = None
        for i in range(steps):
            with torch.no_grad():
                # Note: we set iters_to_do=1 each time, passing the thought forward
                out, curr_thought = model(input_tensor, iters_to_do=1, interim_thought=curr_thought)
            
            probs = F.softmax(out, dim=-1)
            conf = probs.max(dim=-1).values.mean().item()
            preds = torch.argmax(probs, dim=-1).cpu().numpy()[0]
            draw_board(preds, os.path.join(temp_dir, f"step_{i:02d}.png"), i, conf, ground_truth)

    # 2. CREATE GIF
    gif_path = os.path.join(GIF_OUT_DIR, f"{model_name.replace(' ', '_')}.gif")
    create_gif(temp_dir, gif_path)
    print(f"Saved: {gif_path}")
    shutil.rmtree(temp_dir)

def main():
    # Load Test Data
    ds = SudokuDataset("data/processed", "test")
    # Pick a hard one (e.g. index 5) or just 0
    inp, label = ds[0]
    input_tensor = inp.unsqueeze(0).to(DEVICE)
    ground_truth = label.numpy()

    # --- MODEL 1: Baseline Transformer ---
    try:
        m1 = BaselineTransformer(vocab_size=11, d_model=128, nhead=4, num_layers=4).to(DEVICE)
        m1.load_state_dict(torch.load("experiments/baseline_transformer/output/model_best.pt", map_location=DEVICE))
        generate_gif_for_model("Baseline_Transformer", m1, input_tensor, ground_truth, steps=60)
    except Exception as e:
        print(f"Skipping Baseline: {e}")

    # --- MODEL 2: Deep Thinking 2D ---
    try:
        m2 = SudokuDeepThinking2D(width=128, recall=True, max_iters=30).to(DEVICE)
        m2.load_state_dict(torch.load("experiments/deep_thinking_2d/output/model_best.pt", map_location=DEVICE))
        generate_gif_for_model("Deep_Thinking_2D", m2, input_tensor, ground_truth, steps=30)
    except Exception as e:
        print(f"Skipping Deep Thinking: {e}")

    # --- MODEL 3: FeedForward 2D ---
    try:
        m3 = SudokuFeedForward2D(width=128, recall=True, max_iters=20).to(DEVICE)
        m3.load_state_dict(torch.load("experiments/feedforward_2d/output/model_best.pt", map_location=DEVICE))
        generate_gif_for_model("FeedForward_2D", m3, input_tensor, ground_truth, steps=20)
    except Exception as e:
        print(f"Skipping FeedForward: {e}")

if __name__ == "__main__":
    main()