import sys
import os
import torch
import torch.nn as nn

# Add root to path to import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from blocks import BasicBlock2D

class FeedForwardNet2D(nn.Module):
    """
    Feed-Forward 2D Network.
    This mimics the DeepThinking process but uses DISTINCT weights for every step.
    It effectively unrolls the recurrence into a very deep ResNet.
    """

    def __init__(self, block, num_blocks, width, in_channels, vocab_size=11, recall=True, max_iters=20, group_norm=True):
        super().__init__()

        self.width = int(width)
        self.recall = recall
        self.group_norm = group_norm
        self.iters = max_iters

        # 1. Initial Projection
        self.proj_conv = nn.Conv2d(in_channels, width, kernel_size=3, stride=1, padding=1, bias=False)
        self.projection = nn.Sequential(self.proj_conv, nn.ReLU())

        # 2. Recall Layer (Maps [Thought + Input] -> [Hidden])
        if self.recall:
            self.recall_layer = nn.Conv2d(width + in_channels, width, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.recall_layer = nn.Identity()

        # 3. The Layers (Unique parameters for every step 0..max_iters)
        # We create a ModuleList where index 'i' corresponds to step 'i'
        self.feedforward_layers = nn.ModuleList()
        for _ in range(max_iters):
            self.feedforward_layers.append(self._make_layer_seq(block, width, num_blocks))

        # 4. Heads (Output Projection)
        head_conv1 = nn.Conv2d(width, width, kernel_size=3, stride=1, padding=1, bias=False)
        head_conv2 = nn.Conv2d(width, width // 2, kernel_size=3, stride=1, padding=1, bias=False)
        head_conv3 = nn.Conv2d(width // 2, vocab_size, kernel_size=3, stride=1, padding=1, bias=False)

        self.head = nn.Sequential(
            head_conv1, nn.ReLU(), 
            head_conv2, nn.ReLU(), 
            head_conv3
        )

    def _make_layer_seq(self, block, planes, num_blocks_list):
        # Helper to build one "Step" (which might contain multiple ResNet blocks)
        layers = []
        total_blocks = sum(num_blocks_list) # e.g. [2, 2] -> 4 blocks per step
        for _ in range(total_blocks):
            layers.append(block(self.width, planes, stride=1, group_norm=self.group_norm))
        return nn.Sequential(*layers)

    def forward(self, x, iters_to_do, interim_thought=None, iters_elapsed=0):
        """
        x: [Batch, Channels, 9, 9]
        iters_elapsed: The starting layer index (e.g., if we already ran 5 layers, start at index 5)
        """
        # Safety check
        if iters_elapsed + iters_to_do > self.iters:
            iters_to_do = self.iters - iters_elapsed

        # Initial Projection (only if starting from scratch)
        if interim_thought is None:
            interim_thought = self.projection(x)

        # We don't store all_outputs here to save memory during training, 
        # unless we specifically wanted to. We just return the final result of this sequence.
        
        # Select the specific physical layers we need to run
        current_layers = self.feedforward_layers[iters_elapsed : iters_elapsed + iters_to_do]

        for i, layer in enumerate(current_layers):
            if self.recall:
                # Recall: Concat input 'x' again to refresh memory
                interim_thought = torch.cat([interim_thought, x], dim=1)
                interim_thought = self.recall_layer(interim_thought)
            
            # Apply the UNIQUE layer
            interim_thought = layer(interim_thought)
            
        # Predict after the last requested layer
        out = self.head(interim_thought)

        return out, interim_thought


class SudokuFeedForward2D(nn.Module):
    """
    Wrapper to handle Embedding -> 2D Grid -> Embedding
    """
    def __init__(self, vocab_size=11, width=128, recall=True, max_iters=20):
        super().__init__()
        self.max_iters = max_iters
        self.embedding = nn.Embedding(vocab_size, width)
        
        # Input channels is 'width' because we embed first
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
        batch_size = x.shape[0]

        # 1. Embed & Reshape to 2D
        emb = self.embedding(x) # [B, 81, W]
        # Permute to [B, W, 81] then reshape to [B, W, 9, 9]
        emb_grid = emb.permute(0, 2, 1).reshape(batch_size, -1, 9, 9)
        
        # 2. Forward pass through the FeedForward Net
        out_grid, final_thought = self.ff_net(
            emb_grid, 
            iters_to_do=iters, 
            interim_thought=interim_thought, 
            iters_elapsed=iters_elapsed
        )
        
        # 3. Flatten Output: [B, Vocab, 9, 9] -> [B, 81, Vocab]
        out_flat = out_grid.flatten(2, 3).permute(0, 2, 1)
        
        return out_flat, final_thought