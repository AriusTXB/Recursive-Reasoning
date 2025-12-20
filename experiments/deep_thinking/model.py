import sys
import os
import torch
import torch.nn as nn

# Allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from blocks import BasicBlock1D as BasicBlock

class DTNet1D(nn.Module):
    def __init__(self, block, num_blocks, width, vocab_size=11, recall=True, group_norm=True):
        super().__init__()
        self.width = int(width)
        self.recall = recall
        self.group_norm = group_norm

        self.proj_conv = nn.Conv1d(width, width, kernel_size=3, stride=1, padding=1, bias=False)
        
        # If recall, input to recurrent block is width*2
        in_channels = width * 2 if self.recall else width
        self.conv_recall = nn.Conv1d(in_channels, width, kernel_size=3, stride=1, padding=1, bias=False)

        if self.recall:
            recur_layers = [self.conv_recall, nn.ReLU()]
        else:
            recur_layers = []

        for i in range(len(num_blocks)):
            recur_layers.append(self._make_layer(block, width, num_blocks[i], stride=1))

        self.recur_block = nn.Sequential(*recur_layers)

        head_conv1 = nn.Conv1d(width, width, kernel_size=3, stride=1, padding=1, bias=False)
        head_conv2 = nn.Conv1d(width, width // 2, kernel_size=3, stride=1, padding=1, bias=False)
        head_conv3 = nn.Conv1d(width // 2, vocab_size, kernel_size=3, stride=1, padding=1, bias=False)

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
        x: [Batch, Hidden_Dim, Seq_Len] - The Embedded Input
        """
        # 1. Initialize Thought
        if interim_thought is None:
            interim_thought = self.projection(x)

        # 2. Iterate
        for i in range(iters_to_do):
            if self.recall:
                # Recall mechanism: Concat thought with original input features
                combined = torch.cat([interim_thought, x], dim=1)
                interim_thought = self.recur_block(combined)
            else:
                interim_thought = self.recur_block(interim_thought)
        
        # 3. Predict (using the state at the last step)
        out = self.head(interim_thought)
        
        # Return Tuple to match paper's training logic
        return out, interim_thought


class SudokuDeepThinking1D(nn.Module):
    def __init__(self, vocab_size=11, width=196, recall=True, max_iters=30):
        super().__init__()
        self.max_iters = max_iters
        self.embedding = nn.Embedding(vocab_size, width)
        # Using [2, 2] blocks as per standard DT-1D config
        self.dt_net = DTNet1D(BasicBlock, num_blocks=[2, 2], width=width, vocab_size=vocab_size, recall=recall)

    def forward(self, x, iters_to_do=None, interim_thought=None, **kwargs):
        """
        Wrapper to handle embeddings.
        x: [Batch, 81] indices
        """
        iters = iters_to_do if iters_to_do is not None else self.max_iters
        
        # 1. Embed and Transpose to [Batch, Width, 81]
        emb = self.embedding(x)
        emb = emb.transpose(1, 2)
        
        # 2. Pass to internal network
        # Note: We pass 'emb' as 'x' because 'Recall' needs the original features
        out, final_thought = self.dt_net(emb, iters_to_do=iters, interim_thought=interim_thought)
        
        # 3. Transpose output back to [Batch, 81, Vocab] for Loss calculation
        out = out.permute(0, 2, 1)
        
        return out, final_thought