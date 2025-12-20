from pydantic import BaseModel
from typing import List

class PuzzleDatasetMetadata(BaseModel):
    seq_len: int
    vocab_size: int
    pad_id: int
    ignore_label_id: int
    blank_identifier_id: int
    num_puzzle_identifiers: int
    total_groups: int
    mean_puzzle_examples: float
    total_puzzles: int
    sets: List[str]