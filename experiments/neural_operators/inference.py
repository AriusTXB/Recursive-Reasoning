import sys
import os
import torch
import torch.nn.functional as F
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.dataset import SudokuDataset
from src.visualizer import draw_board, create_gif
from model import NeuralOperatorSolver

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
SAMPLE_DIR = os.path.join(OUTPUT_DIR, "thought_process")

def run_visual_inference():
    # Load Model
    model = NeuralOperatorSolver(hidden_dim=256).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "model_nop.pt"), map_location=DEVICE))
    model.eval()
    
    # Get Data
    ds = SudokuDataset("data/processed", "test")
    inp, label = ds[0]
    
    input_tensor = inp.unsqueeze(0).to(DEVICE) # [1, 81]
    ground_truth = label.numpy()
    
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    
    print("Generating Neural Operator Trajectory...")
    
    # We query the model at 30 time steps from 0 to 1
    steps = 30
    time_points = np.linspace(0, 1, steps)
    
    for i, t_val in enumerate(time_points):
        # Create time tensor [1, 1]
        t_tensor = torch.tensor([[t_val]], dtype=torch.float32).to(DEVICE)
        
        with torch.no_grad():
            # Query Operator G(u)(t)
            logits = model(input_tensor, t_tensor) # [1, 11, 9, 9]
            
        probs = F.softmax(logits, dim=1)
        
        # Get predictions
        # Shape: [1, 11, 9, 9] -> max over dim 1 -> [1, 9, 9]
        conf_map, preds_grid = torch.max(probs, dim=1)
        
        preds_flat = preds_grid.flatten().cpu().numpy()
        avg_conf = conf_map.mean().item()
        
        print(f"Time {t_val:.2f}: Avg Conf {avg_conf:.2f}")
        
        # Draw
        draw_board(
            preds_flat, 
            os.path.join(SAMPLE_DIR, f"step_{i:02d}.png"), 
            i, 
            f"Time: {t_val:.2f}", 
            ground_truth
        )

    create_gif(SAMPLE_DIR, os.path.join(OUTPUT_DIR, "neural_operator.gif"))
    print("GIF saved.")

if __name__ == "__main__":
    run_visual_inference()