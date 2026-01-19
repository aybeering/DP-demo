import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool


class GNNEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        out_dim: int = 16,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        convs = [GCNConv(in_dim, hidden_dim)]
        for _ in range(num_layers - 2):
            convs.append(GCNConv(hidden_dim, hidden_dim))
        if num_layers > 1:
            convs.append(GCNConv(hidden_dim, out_dim))

        self.convs = nn.ModuleList(convs)
        self.act = nn.ReLU()

    def forward(self, data: Data) -> torch.Tensor:
        """返回分子级嵌入 z (batch_size, out_dim)。"""
        x, edge_index = data.x, data.edge_index
        batch = getattr(data, "batch", torch.zeros(x.size(0), dtype=torch.long, device=x.device))

        for idx, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if idx < len(self.convs) - 1:
                x = self.act(x)
        return global_mean_pool(x, batch)
