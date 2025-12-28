import sys
import os
import torch
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.blocks import BasicBlock2D

class FeedForwardNet2D(nn.Module):
    def __init__(self, block, num_blocks, width, in_channels, vocab_size=11, recall=True, max_iters=12):
        super().__init__()
        self.width = int(width)
        self.recall = recall
        self.iters = max_iters

        self.proj_conv = nn.Conv2d(in_channels, width, 3, 1, 1, bias=False)
        self.projection = nn.Sequential(self.proj_conv, nn.ReLU())
        self.recall_layer = nn.Conv2d(width + in_channels, width, 3, 1, 1, bias=False) if recall else nn.Identity()
        
        self.feedforward_layers = nn.ModuleList()
        # 12 Steps * 4 Blocks per step
        for _ in range(max_iters):
            layers = []
            for _ in range(sum(num_blocks)):
                layers.append(block(width, width, 1, group_norm=True))
            self.feedforward_layers.append(nn.Sequential(*layers))

        self.head = nn.Sequential(
            nn.Conv2d(width, width, 3, 1, 1, bias=False), nn.ReLU(),
            nn.Conv2d(width, width//2, 3, 1, 1, bias=False), nn.ReLU(),
            nn.Conv2d(width//2, vocab_size, 3, 1, 1, bias=False)
        )

    def forward(self, x, iters_to_do, interim_thought=None, iters_elapsed=0):
        if iters_elapsed + iters_to_do > self.iters: iters_to_do = self.iters - iters_elapsed
        if interim_thought is None: interim_thought = self.projection(x)
        
        current_layers = self.feedforward_layers[iters_elapsed : iters_elapsed + iters_to_do]
        
        for layer in current_layers:
            if self.recall:
                interim_thought = torch.cat([interim_thought, x], dim=1)
                interim_thought = self.recall_layer(interim_thought)
            interim_thought = layer(interim_thought)
            
        return self.head(interim_thought), interim_thought

class SudokuFeedForward2D(nn.Module):
    def __init__(self, vocab_size=11, width=192, recall=True, max_iters=12):
        """
        Scaled FeedForward (~12M Params)
        Width: 192
        Layers: 12 physical steps (unrolled)
        """
        super().__init__()
        self.max_iters = max_iters
        self.embedding = nn.Embedding(vocab_size, width)
        self.ff_net = FeedForwardNet2D(
            BasicBlock2D, 
            num_blocks=[2, 2], 
            width=width, 
            in_channels=width, 
            vocab_size=vocab_size, 
            recall=recall, 
            max_iters=max_iters
        )

    def forward(self, x, iters_to_do=None, interim_thought=None, iters_elapsed=0, **kwargs):
        iters = iters_to_do if iters_to_do is not None else self.max_iters
        emb = self.embedding(x).permute(0, 2, 1).reshape(x.shape[0], -1, 9, 9)
        out_grid, final_thought = self.ff_net(emb, iters, interim_thought, iters_elapsed)
        return out_grid.flatten(2, 3).permute(0, 2, 1), final_thought