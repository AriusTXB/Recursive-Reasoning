import sys
import os
import torch
import torch.nn.functional as F
import shutil
import numpy as np

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dataset import SudokuDataset
from src.visualizer import draw_board, create_gif

# Import Models
try:
    from experiments.baseline_transformer.model import BaselineTransformer
    from experiments.deep_thinking.model import SudokuDeepThinking2D
    from experiments.feed_forward_neural_net.model import SudokuFeedForward2D
    from experiments.trm.model import TinyRecursiveReasoningModel_ACTV1
    from experiments.neural_operators.model import NeuralOperatorSolver
except ImportError as e:
    print(f"Import Error: {e}. Check folder structure.")
    sys.exit(1)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GIF_OUT_DIR = "experiment_gifs"
os.makedirs(GIF_OUT_DIR, exist_ok=True)

def generate_gif(model_name, model_class, model_path, input_tensor, ground_truth, model_kwargs={}, model_type="standard"):
    print(f"--- Generating GIF for {model_name} ---")
    
    if not os.path.exists(model_path):
        print(f"  [Skip] Model not found at {model_path}")
        return

    # Init & Load
    try:
        model = model_class(**model_kwargs).to(DEVICE)
        state_dict = torch.load(model_path, map_location=DEVICE)
        
        # Handle Wrappers if necessary
        if "trm" in model_name.lower() or "neural" in model_name.lower():
             model.load_state_dict(state_dict)
        else:
             model.load_state_dict(state_dict)
             
        model.eval()
    except Exception as e:
        print(f"  [Error] Loading model: {e}")
        return

    # Prep Temp Dir
    temp_dir = os.path.join(GIF_OUT_DIR, f"temp_{model_name.replace(' ', '_')}")
    if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)

    # --- INFERENCE LOOPS ---
    
    # 1. Baseline (Greedy Filling)
    if "Baseline" in model_name:
        curr_seq = input_tensor.clone()
        for i in range(50):
            with torch.no_grad():
                logits = model(curr_seq)
                probs = F.softmax(logits, dim=-1)
                
                conf = probs.max(dim=-1).values.mean().item()
                preds = torch.argmax(probs, dim=-1).cpu().numpy()[0]
                draw_board(preds, f"{temp_dir}/step_{i:02d}.png", i, conf, ground_truth)
                
                # Fill one cell
                mask = (curr_seq == 1) # 1 is empty
                if not mask.any(): break
                digit_probs = probs[:, :, 2:]
                max_probs, idxs = torch.max(digit_probs, dim=2)
                vals = idxs + 2
                max_probs[~mask] = -1
                best_idx = torch.argmax(max_probs)
                curr_seq.view(-1)[best_idx] = vals.view(-1)[best_idx]

    # 2. Deep Thinking (Recurrent)
    elif "Deep Thinking" in model_name:
        curr_thought = None
        for i in range(30):
            with torch.no_grad():
                out, curr_thought = model(input_tensor, iters_to_do=1, interim_thought=curr_thought)
            probs = F.softmax(out, dim=-1)
            conf = probs.max(dim=-1).values.mean().item()
            preds = torch.argmax(probs, dim=-1).cpu().numpy()[0]
            draw_board(preds, f"{temp_dir}/step_{i:02d}.png", i, conf, ground_truth)

    # 3. FeedForward (Sequential Layers)
    elif "FeedForward" in model_name:
        curr_thought = None
        max_layers = model_kwargs.get('max_iters', 12)
        for i in range(max_layers):
            with torch.no_grad():
                out, curr_thought = model(input_tensor, iters_to_do=1, interim_thought=curr_thought, iters_elapsed=i)
            probs = F.softmax(out, dim=-1)
            conf = probs.max(dim=-1).values.mean().item()
            preds = torch.argmax(probs, dim=-1).cpu().numpy()[0]
            draw_board(preds, f"{temp_dir}/step_{i:02d}.png", i, conf, ground_truth)

    # 4. TRM (Stateful Recursion)
    elif "TRM" in model_name:
        batch = {
            "inputs": input_tensor,
            "labels": torch.zeros_like(input_tensor),
            "puzzle_identifiers": torch.zeros(1, dtype=torch.int32).to(DEVICE)
        }
        carry = model.initial_carry(batch)
        for i in range(30):
            if i > 0: carry.halted[:] = False # Force continue for viz
            with torch.no_grad():
                carry, outputs = model(carry, batch)
            
            probs = F.softmax(outputs["logits"], dim=-1)
            conf = probs.max(dim=-1).values.mean().item()
            q_val = outputs.get("q_halt", outputs.get("q_halt_logits")).item()
            preds = torch.argmax(probs, dim=-1).cpu().numpy()[0]
            
            draw_board(preds, f"{temp_dir}/step_{i:02d}.png", i, f"{conf:.2f} (Q:{q_val:.2f})", ground_truth)

    # 5. Neural Operator (Continuous Time)
    elif "Neural Operator" in model_name:
        steps = 30
        times = np.linspace(0, 1, steps)
        for i, t_val in enumerate(times):
            t_tensor = torch.tensor([[t_val]], dtype=torch.float32).to(DEVICE)
            with torch.no_grad():
                logits = model(input_tensor, t_tensor) # [1, 11, 9, 9]
            
            # Reshape [1, 11, 9, 9] -> [1, 9, 9, 11] -> [1, 81, 11]
            probs = F.softmax(logits, dim=1)
            probs_flat = probs.permute(0, 2, 3, 1).flatten(1, 2) # [1, 81, 11]
            
            conf = probs_flat.max(dim=-1).values.mean().item()
            preds = torch.argmax(probs_flat, dim=-1).cpu().numpy()[0]
            
            draw_board(preds, f"{temp_dir}/step_{i:02d}.png", i, f"Time:{t_val:.2f}", ground_truth)

    # Create GIF
    gif_path = os.path.join(GIF_OUT_DIR, f"{model_name.replace(' ', '_')}.gif")
    create_gif(temp_dir, gif_path)
    print(f"  -> Saved: {gif_path}")
    shutil.rmtree(temp_dir)

def main():
    # Load Test Sample
    try:
        ds = SudokuDataset("data/processed", "test")
        inp, lbl = ds[0]
        input_tensor = inp.unsqueeze(0).to(DEVICE)
        ground_truth = lbl.numpy()
    except:
        print("Data not found. Run preprocess.py first.")
        return

    # 1. Baseline
    generate_gif("Baseline", BaselineTransformer, 
                 "experiments/baseline_transformer/output/model_best.pt", 
                 input_tensor, ground_truth, 
                 {"vocab_size": 11, "d_model": 384, "nhead": 8, "num_layers": 6})

    # 2. Deep Thinking
    generate_gif("Deep Thinking", SudokuDeepThinking2D, 
                 "experiments/deep_thinking_2d/output/model_best.pt", 
                 input_tensor, ground_truth, 
                 {"width": 384, "recall": True, "max_iters": 30})

    # 3. FeedForward
    generate_gif("FeedForward", SudokuFeedForward2D, 
                 "experiments/feedforward_2d/output/model_best.pt", 
                 input_tensor, ground_truth, 
                 {"width": 192, "recall": True, "max_iters": 12})

    # 4. TRM
    trm_cfg = {
        "batch_size": 1, "seq_len": 81, "vocab_size": 11,
        "puzzle_emb_ndim": 0, "num_puzzle_identifiers": 1,
        "H_cycles": 1, "L_cycles": 2, "H_layers": 0, "L_layers": 2,
        "hidden_size": 768, "expansion": 4.0, "num_heads": 12, # Scaled
        "pos_encodings": "rope", "halt_max_steps": 20, "halt_exploration_prob": 0.0,
        "forward_dtype": "float32", "puzzle_emb_len": 0
    }
    generate_gif("TRM", TinyRecursiveReasoningModel_ACTV1, 
                 "experiments/trm/output/model_trm.pt", 
                 input_tensor, ground_truth, 
                 trm_cfg)

    # 5. Neural Operator
    generate_gif("Neural Operator", NeuralOperatorSolver, 
                 "experiments/neural_operator/output/model_nop.pt", 
                 input_tensor, ground_truth, 
                 {"vocab_size": 11, "hidden_dim": 384, "depth": 4})

if __name__ == "__main__":
    main()