import sys
import os
import torch
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.blocks import BasicBlock2D

class DTNet2D(nn.Module):
    def __init__(self, block, num_blocks, width, in_channels, vocab_size=11, recall=True):
        super().__init__()
        self.width = int(width)
        self.recall = recall
        self.proj_conv = nn.Conv2d(in_channels, width, 3, 1, 1, bias=False)
        
        recur_in = width + in_channels if recall else width
        self.conv_recall = nn.Conv2d(recur_in, width, 3, 1, 1, bias=False)
        
        layers = []
        if recall:
            layers.append(nn.Sequential(self.conv_recall, nn.ReLU()))
        
        # Deep Thinking Core
        for _ in range(sum(num_blocks)):
            layers.append(block(width, width, 1, group_norm=True))
            
        self.recur_block = nn.Sequential(*layers)

        self.projection = nn.Sequential(self.proj_conv, nn.ReLU())
        self.head = nn.Sequential(
            nn.Conv2d(width, width, 3, 1, 1, bias=False), nn.ReLU(),
            nn.Conv2d(width, width//2, 3, 1, 1, bias=False), nn.ReLU(),
            nn.Conv2d(width//2, vocab_size, 3, 1, 1, bias=False)
        )

    def forward(self, x, iters_to_do, interim_thought=None):
        if interim_thought is None:
            interim_thought = self.projection(x)

        for i in range(iters_to_do):
            if self.recall:
                combined = torch.cat([interim_thought, x], dim=1)
                interim_thought = self.recur_block(combined)
            else:
                interim_thought = self.recur_block(interim_thought)
        
        out = self.head(interim_thought)
        return out, interim_thought

class SudokuDeepThinking2D(nn.Module):
    def __init__(self, vocab_size=11, width=384, recall=True, max_iters=30):
        """
        Scaled Deep Thinking 2D (~10M Params)
        Width: 384
        """
        super().__init__()
        self.max_iters = max_iters
        self.embedding = nn.Embedding(vocab_size, width)
        
        # num_blocks=[2, 2] means 4 ResBlocks per recurrence step
        self.dt_net = DTNet2D(
            BasicBlock2D, 
            num_blocks=[2, 2], 
            width=width, 
            in_channels=width, 
            vocab_size=vocab_size, 
            recall=recall
        )

    def forward(self, x, iters_to_do=None, interim_thought=None, **kwargs):
        iters = iters_to_do if iters_to_do is not None else self.max_iters
        emb = self.embedding(x).permute(0, 2, 1).reshape(x.shape[0], -1, 9, 9)
        out_grid, final_thought = self.dt_net(emb, iters, interim_thought)
        out_flat = out_grid.flatten(2, 3).permute(0, 2, 1)
        return out_flat, final_thought