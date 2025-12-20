import sys
import os
import torch
import torch.nn.functional as F
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.dataset import SudokuDataset
from src.visualizer import draw_board, create_gif
from model import SudokuFeedForward2D

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
SAMPLE_DIR = os.path.join(OUTPUT_DIR, "thought_process")

def run_visual_inference():
    MAX_ITERS = 20
    model = SudokuFeedForward2D(width=128, recall=True, max_iters=MAX_ITERS).to(DEVICE)
    
    model_path = os.path.join(OUTPUT_DIR, "model_ff2d.pt")
    if not os.path.exists(model_path):
        print("Model not found. Run train.py first.")
        return

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    # Get Data
    ds = SudokuDataset("data/processed", "test")
    inp, label = ds[0]
    
    input_tensor = inp.unsqueeze(0).to(DEVICE) # [1, 81]
    ground_truth = label.numpy()
    
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    
    print("Running FeedForward Layers...")
    
    # Initialize thought state
    current_thought = None
    
    # Loop through the physical layers one by one
    for i in range(MAX_ITERS):
        with torch.no_grad():
            # Run EXACTLY 1 layer (layer index 'i')
            # Input: The output from previous step (current_thought)
            out, current_thought = model(
                input_tensor, 
                iters_to_do=1, 
                interim_thought=current_thought, 
                iters_elapsed=i # Tells model to use layer[i]
            )
        
        # Visualization Logic
        probs = F.softmax(out, dim=-1)
        conf = probs.max(dim=-1).values.mean().item()
        preds = torch.argmax(probs, dim=-1).cpu().numpy()[0]
        
        draw_board(
            board_1d=preds, 
            save_path=os.path.join(SAMPLE_DIR, f"step_{i:02d}.png"), 
            step_num=i, 
            prediction_conf=conf, 
            ground_truth_1d=ground_truth
        )

    create_gif(SAMPLE_DIR, os.path.join(OUTPUT_DIR, "feedforward_2d.gif"))
    print(f"GIF saved to {os.path.join(OUTPUT_DIR, 'feedforward_2d.gif')}")

if __name__ == "__main__":
    run_visual_inference()