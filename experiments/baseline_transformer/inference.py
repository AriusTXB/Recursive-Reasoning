import sys
import os
import torch
import torch.nn.functional as F
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.dataset import SudokuDataset
from src.visualizer import draw_board, create_gif
from model import BaselineTransformer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
TEST_SAMPLE_DIR = os.path.join(OUTPUT_DIR, "test_sample")

def recursive_inference(model, input_seq, target_seq):
    """
    Simulates 'Reasoning' by filling the board one step at a time.
    """
    model.eval()
    current_seq = input_seq.clone().to(DEVICE).unsqueeze(0) # [1, 81]
    
    # Convert target to numpy for visualization comparison
    ground_truth_np = target_seq.numpy()
    
    # Directory for steps
    os.makedirs(TEST_SAMPLE_DIR, exist_ok=True)
    
    # Step 0: Initial state
    print("Saving Step 0...")
    draw_board(
        board_1d=current_seq.cpu().numpy()[0], 
        save_path=os.path.join(TEST_SAMPLE_DIR, "step_00.png"), 
        step_num=0,
        ground_truth_1d=ground_truth_np  # <--- Pass Truth
    )
    
    step_count = 0
    max_steps = 81 # Safety break
    
    with torch.no_grad():
        while True:
            # 1. Check if done (no empty cells). 
            # Note: In preprocessing we shifted data +1, so 1 is "Empty" (originally 0)
            if not (current_seq == 1).any():
                print("Puzzle filled.")
                break
                
            if step_count >= max_steps:
                break

            # 2. Forward pass
            logits = model(current_seq) # [1, 81, 11]
            probs = F.softmax(logits, dim=-1) # [1, 81, 11]
            
            # 3. Mask out non-empty positions
            filled_mask = (current_seq != 1) 
            
            # Get max probability for digits (classes 2..10)
            digit_probs = probs[:, :, 2:] 
            max_probs, predicted_classes = torch.max(digit_probs, dim=2) 
            predicted_vals = predicted_classes + 2
            
            # Don't pick already filled cells
            max_probs[filled_mask] = -1.0
            
            # 4. Find the single cell with highest confidence
            best_cell_idx = torch.argmax(max_probs) 
            confidence = max_probs.view(-1)[best_cell_idx].item()
            chosen_val = predicted_vals.view(-1)[best_cell_idx]
            
            # 5. Update board
            current_seq.view(-1)[best_cell_idx] = chosen_val
            
            step_count += 1
            
            # Check if this specific move was correct (for print log)
            # Map 1D index to GT value
            correct_val_at_idx = ground_truth_np[best_cell_idx.item()]
            is_correct = (chosen_val.item() == correct_val_at_idx)
            status_str = "CORRECT" if is_correct else "WRONG"
            
            print(f"Step {step_count}: Cell {best_cell_idx.item()} -> {chosen_val.item()-1} | Conf: {confidence:.2f} | {status_str}")
            
            # 6. Visualize
            draw_board(
                board_1d=current_seq.cpu().numpy()[0], 
                save_path=os.path.join(TEST_SAMPLE_DIR, f"step_{step_count:02d}.png"), 
                step_num=step_count,
                prediction_conf=confidence,
                ground_truth_1d=ground_truth_np # <--- Pass Truth to color code Red/Green
            )

    # Generate GIF
    create_gif(TEST_SAMPLE_DIR, os.path.join(OUTPUT_DIR, "solution_process.gif"))
    print(f"GIF saved to {os.path.join(OUTPUT_DIR, 'solution_process.gif')}")

def run_test():
    # Load Model
    model = BaselineTransformer().to(DEVICE)
    model_path = os.path.join(OUTPUT_DIR, "model_latest.pt")
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Run train.py first.")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    
    # Load 1 example from test set (Input AND Label)
    test_ds = SudokuDataset("data/processed", split="test")
    input_sample, label_sample = test_ds[0] 
    
    recursive_inference(model, input_sample, label_sample)

if __name__ == "__main__":
    run_test()