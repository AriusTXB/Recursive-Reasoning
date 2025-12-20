import sys
import os
# Add root to path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import Optional
import csv
import json
import numpy as np
from argdantic import ArgParser
from pydantic import BaseModel
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from src.common import PuzzleDatasetMetadata

cli = ArgParser()

class DataProcessConfig(BaseModel):
    source_repo: str = "sapientinc/sudoku-extreme"
    # Adjusted output path relative to root
    output_dir: str = "data/processed"
    subsample_size: Optional[int] = 50000 # Limit to 50k for faster baseline training
    min_difficulty: Optional[int] = None
    num_aug: int = 0

def shuffle_sudoku(board: np.ndarray, solution: np.ndarray):
    digit_map = np.pad(np.random.permutation(np.arange(1, 10)), (1, 0))
    transpose_flag = np.random.rand() < 0.5
    bands = np.random.permutation(3)
    row_perm = np.concatenate([b * 3 + np.random.permutation(3) for b in bands])
    stacks = np.random.permutation(3)
    col_perm = np.concatenate([s * 3 + np.random.permutation(3) for s in stacks])
    mapping = np.array([row_perm[i // 9] * 9 + col_perm[i % 9] for i in range(81)])

    def apply_transformation(x: np.ndarray) -> np.ndarray:
        if transpose_flag: x = x.T
        new_board = x.flatten()[mapping].reshape(9, 9).copy()
        return digit_map[new_board]

    return apply_transformation(board), apply_transformation(solution)

def convert_subset(set_name: str, config: DataProcessConfig):
    print(f"Processing {set_name} set...")
    inputs, labels = [], []
    
    try:
        file_path = hf_hub_download(config.source_repo, f"{set_name}.csv", repo_type="dataset")
    except Exception as e:
        print(f"Could not download {set_name}: {e}")
        return

    with open(file_path, newline="") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for source, q, a, rating in reader:
            if (config.min_difficulty is None) or (int(rating) >= config.min_difficulty):
                inputs.append(np.frombuffer(q.replace('.', '0').encode(), dtype=np.uint8).reshape(9, 9) - ord('0'))
                labels.append(np.frombuffer(a.encode(), dtype=np.uint8).reshape(9, 9) - ord('0'))

    if set_name == "train" and config.subsample_size is not None and config.subsample_size < len(inputs):
        indices = np.random.choice(len(inputs), size=config.subsample_size, replace=False)
        inputs = [inputs[i] for i in indices]
        labels = [labels[i] for i in indices]

    num_augments = config.num_aug if set_name == "train" else 0
    results = {k: [] for k in ["inputs", "labels", "group_indices"]}
    
    puzzle_id = 0
    results["group_indices"].append(0)
    
    for orig_inp, orig_out in zip(tqdm(inputs), labels):
        for aug_idx in range(1 + num_augments):
            if aug_idx == 0: inp, out = orig_inp, orig_out
            else: inp, out = shuffle_sudoku(orig_inp, orig_out)
            results["inputs"].append(inp)
            results["labels"].append(out)
            puzzle_id += 1
        results["group_indices"].append(puzzle_id)
        
    def _seq_to_numpy(seq):
        arr = np.concatenate(seq).reshape(len(seq), -1)
        return arr + 1 # 0->1, 1->2... 

    final_results = {
        "inputs": _seq_to_numpy(results["inputs"]),
        "labels": _seq_to_numpy(results["labels"]),
    }

    metadata = PuzzleDatasetMetadata(
        seq_len=81, vocab_size=11, pad_id=0, ignore_label_id=0,
        blank_identifier_id=0, num_puzzle_identifiers=1,
        total_groups=len(results["group_indices"]) - 1,
        mean_puzzle_examples=1, total_puzzles=len(results["group_indices"]) - 1,
        sets=["all"]
    )

    save_dir = os.path.join(config.output_dir, set_name)
    os.makedirs(save_dir, exist_ok=True)
    
    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f)
        
    for k, v in final_results.items():
        np.save(os.path.join(save_dir, f"all__{k}.npy"), v)

@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig):
    convert_subset("train", config)
    convert_subset("test", config)

if __name__ == "__main__":
    cli()