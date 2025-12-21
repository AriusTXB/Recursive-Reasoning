import sys
import os
import torch
import torch.nn.functional as F
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.dataset import SudokuDataset
from src.visualizer import draw_board, create_gif
from model import SudokuDeepThinking2D

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
SAMPLE_DIR = os.path.join(OUTPUT_DIR, "thought_process")

def run_visual_inference():
    # Load Model
    model = SudokuDeepThinking2D(width=196, recall=True, max_iters=30).to(DEVICE)
    model_path = os.path.join(OUTPUT_DIR, "model_dt1d.pt")
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please run train.py first.")
        return

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    # Get Data (Input AND Label)
    ds = SudokuDataset("data/processed", "test")
    
    # Pick a specific puzzle (e.g., index 0)
    inp, label = ds[0] 
    
    input_tensor = inp.unsqueeze(0).to(DEVICE) # [1, 81]
    
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    
    print("Thinking...")
    with torch.no_grad():
        # Get all 30 steps of thought
        all_logits = model(input_tensor, iters=30) # [30, 1, 81, 11]
        
    print("Generating frames...")
    # Convert label to numpy for comparison
    ground_truth = label.numpy()
    
    for i, logits in enumerate(all_logits):
        # logits: [1, 81, 11]
        probs = F.softmax(logits, dim=-1)
        
        # Confidence score
        conf = probs.max(dim=-1).values.mean().item()
        
        # Prediction (0-10)
        preds = torch.argmax(probs, dim=-1).cpu().numpy()[0]
        
        # Draw with Ground Truth Comparison
        draw_board(
            board_1d=preds, 
            save_path=os.path.join(SAMPLE_DIR, f"step_{i:02d}.png"), 
            step_num=i, 
            prediction_conf=conf,
            ground_truth_1d=ground_truth # <--- Passed here
        )
        
    create_gif(SAMPLE_DIR, os.path.join(OUTPUT_DIR, "deep_thinking_1d_validated.gif"))
    print(f"GIF saved to {os.path.join(OUTPUT_DIR, 'deep_thinking_1d_validated.gif')}")

if __name__ == "__main__":
    run_visual_inference()