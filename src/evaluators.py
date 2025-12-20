import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import os

# Ensure we can import dataset
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.dataset import SudokuDataset

def evaluate_model(model, device, split="test", batch_size=64, model_type="standard"):
    """
    Evaluates a model on the specified split.
    Returns:
        cell_acc: % of individual cells correct
        puzzle_acc: % of entire puzzles correct
    """
    dataset = SudokuDataset("data/processed", split=split)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=2)
    
    model.eval()
    total_cells = 0
    correct_cells = 0
    total_puzzles = 0
    correct_puzzles = 0
    
    print(f"Evaluating {model_type} on {split} set...")
    
    with torch.no_grad():
        for inputs, targets in tqdm(loader, leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 1. Forward Pass logic based on model type
            if model_type == "deep_thinking":
                # For DT, we want the FINAL output
                out, _ = model(inputs) 
            else:
                out = model(inputs)
                # Handle cases where model returns tuple (e.g. wrapper returning thought)
                if isinstance(out, tuple):
                    out = out[0]

            # 2. Get Predictions
            # out shape: [Batch, 81, 11]
            preds = torch.argmax(out, dim=-1) # [Batch, 81]
            
            # 3. Compare with Targets
            # Mask: Only check non-padding indices? 
            # In your preprocessing, targets are 2..10. 0 is Pad, 1 is Empty Input.
            # We compare everything since output should be full board.
            
            # Cell Accuracy
            mask = (targets != 0) # Should be all True for valid sudoku solutions
            correct_mask = (preds == targets) & mask
            correct_cells += correct_mask.sum().item()
            total_cells += mask.sum().item()
            
            # Puzzle Accuracy (All cells in a row must be true)
            # Check if all True along dim 1
            row_correct = (preds == targets).all(dim=1)
            correct_puzzles += row_correct.sum().item()
            total_puzzles += inputs.size(0)

    cell_acc = 100.0 * correct_cells / total_cells
    puzzle_acc = 100.0 * correct_puzzles / total_puzzles
    
    return cell_acc, puzzle_acc