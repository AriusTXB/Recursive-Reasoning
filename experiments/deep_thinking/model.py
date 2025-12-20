import sys
import os
import torch
import torch.nn as nn

# Allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from blocks import BasicBlock2D

class DTNet2D(nn.Module):
    """
    Core DeepThinking 2D Network.
    """
    def __init__(self, block, num_blocks, width, in_channels, vocab_size=11, recall=True, group_norm=True):
        super().__init__()
        self.width = int(width)
        self.recall = recall
        self.group_norm = group_norm

        # Project Input Channels -> Hidden Width
        self.proj_conv = nn.Conv2d(in_channels, width, kernel_size=3, stride=1, padding=1, bias=False)

        # Recurrent Layer Input: 
        # If recall=True: Hidden Width + Original Input Channels
        recur_in_channels = width + in_channels if self.recall else width
        
        self.conv_recall = nn.Conv2d(recur_in_channels, width, kernel_size=3, stride=1, padding=1, bias=False)

        if self.recall:
            recur_layers = [self.conv_recall, nn.ReLU()]
        else:
            recur_layers = []

        for i in range(len(num_blocks)):
            recur_layers.append(self._make_layer(block, width, num_blocks[i], stride=1))

        self.recur_block = nn.Sequential(*recur_layers)

        # Heads
        head_conv1 = nn.Conv2d(width, width, kernel_size=3, stride=1, padding=1, bias=False)
        head_conv2 = nn.Conv2d(width, width // 2, kernel_size=3, stride=1, padding=1, bias=False)
        head_conv3 = nn.Conv2d(width // 2, vocab_size, kernel_size=3, stride=1, padding=1, bias=False)

        self.projection = nn.Sequential(self.proj_conv, nn.ReLU())
        
        self.head = nn.Sequential(
            head_conv1, nn.ReLU(),
            head_conv2, nn.ReLU(),
            head_conv3
        )

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for strd in strides:
            layers.append(block(self.width, planes, strd, self.group_norm))
            self.width = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, iters_to_do, interim_thought=None):
        """
        x: [Batch, Channels, 9, 9]
        """
        if interim_thought is None:
            interim_thought = self.projection(x)

        # Iterate
        for i in range(iters_to_do):
            if self.recall:
                # Concatenate along channel dimension (dim 1)
                combined = torch.cat([interim_thought, x], dim=1)
                interim_thought = self.recur_block(combined)
            else:
                interim_thought = self.recur_block(interim_thought)
        
        out = self.head(interim_thought)
        return out, interim_thought


class SudokuDeepThinking2D(nn.Module):
    def __init__(self, vocab_size=11, width=128, recall=True, max_iters=30):
        super().__init__()
        self.max_iters = max_iters
        self.embedding = nn.Embedding(vocab_size, width)
        
        # Note: in_channels is 'width' because we embed the tokens first.
        # But wait, usually CNNs embed dimensions as channels.
        # Yes: Embedding(Batch, 81) -> (Batch, 81, Width)
        # Reshape -> (Batch, Width, 9, 9). So in_channels = Width.
        
        self.dt_net = DTNet2D(
            BasicBlock2D, 
            num_blocks=[2, 2], 
            width=width, 
            in_channels=width, 
            vocab_size=vocab_size, 
            recall=recall
        )

    def forward(self, x, iters_to_do=None, interim_thought=None, **kwargs):
        """
        x: [Batch, 81] indices
        """
        iters = iters_to_do if iters_to_do is not None else self.max_iters
        batch_size = x.shape[0]

        # 1. Embed: [Batch, 81, Width]
        emb = self.embedding(x)
        
        # 2. Reshape to 2D Grid: [Batch, Width, 9, 9]
        # We need to ensure the 81 length maps correctly to 9x9 row-major
        emb_grid = emb.permute(0, 2, 1).reshape(batch_size, -1, 9, 9)
        
        # 3. Pass to internal network
        out_grid, final_thought = self.dt_net(emb_grid, iters_to_do=iters, interim_thought=interim_thought)
        # out_grid: [Batch, Vocab, 9, 9]
        
        # 4. Flatten back to [Batch, 81, Vocab]
        # Reshape [Batch, Vocab, 9, 9] -> [Batch, Vocab, 81] -> [Batch, 81, Vocab]
        out_flat = out_grid.flatten(2, 3).permute(0, 2, 1)
        
        return out_flat, final_thought