import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalTimeEmbeddings(nn.Module):
    """
    Encodes scalar time t into a high-dimensional vector.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        # time: [Batch, 1]
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ResBlock2d(nn.Module):
    """
    Standard ResNet Block for stable deep training.
    """
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(dim)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(dim)
        self.activation = nn.GELU()

    def forward(self, x):
        residual = x
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.activation(out + residual)

class NeuralOperatorSolver(nn.Module):
    def __init__(self, vocab_size=11, hidden_dim=384, depth=4):
        """
        Scaled Up Neural Operator.
        Args:
            hidden_dim: 384 (Results in ~12M params with depth=4)
            depth: Number of ResBlocks in the branch net.
        """
        super().__init__()
        
        # --- BRANCH NET (Input Processor) ---
        # Processes the static 9x9 puzzle
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        
        # Input projection
        self.branch_entry = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU()
        )
        
        # Deep Residual Backbone
        self.branch_blocks = nn.ModuleList([
            ResBlock2d(hidden_dim) for _ in range(depth)
        ])

        # --- TRUNK NET (Time Processor) ---
        # Processes the time scalar t
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), # Added extra layer for depth
        )

        # --- HEAD (Combination & Output) ---
        # Combines Space (Branch) and Time (Trunk)
        self.head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, vocab_size, kernel_size=1) # Output 11 classes
        )

    def forward(self, x, t):
        """
        x: [Batch, 81] (The initial puzzle)
        t: [Batch, 1]  (Time from 0.0 to 1.0)
        """
        batch_size = x.shape[0]
        
        # 1. Branch: Process Grid
        # Reshape [B, 81] -> [B, 9, 9] -> Embed -> [B, H, 9, 9]
        grid = x.view(batch_size, 9, 9)
        emb_grid = self.embedding(grid).permute(0, 3, 1, 2) 
        
        branch_feat = self.branch_entry(emb_grid)
        for block in self.branch_blocks:
            branch_feat = block(branch_feat)
        
        # 2. Trunk: Process Time
        time_feat = self.time_mlp(t) # [B, H]
        
        # 3. Combine
        # Broadcast Time features across the 9x9 grid
        # [B, H] -> [B, H, 1, 1] -> add to [B, H, 9, 9]
        combined_feat = branch_feat + time_feat.unsqueeze(-1).unsqueeze(-1)
        
        # 4. Decode
        logits = self.head(combined_feat) # [B, 11, 9, 9]
        
        return logits