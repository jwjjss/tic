"""Neural network for policy/value prediction."""

import torch
from torch import nn


class GameNet(nn.Module):
    def __init__(
        self,
        board_size,
        action_size,
        hidden_size=64,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
    ):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.board_size = board_size
        self.action_size = action_size
        self.input_proj = nn.Linear(1, hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, board_size, hidden_size))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.policy_head = nn.Linear(board_size * hidden_size, action_size)
        self.value_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 1),
        )
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)

    def forward(self, x):
        tokens = x.unsqueeze(-1)
        features = self.input_proj(tokens) + self.pos_embed
        encoded = self.encoder(features)
        flat = encoded.reshape(encoded.size(0), -1)
        policy_logits = self.policy_head(flat)
        pooled = encoded.mean(dim=1)
        value = torch.tanh(self.value_head(pooled))
        return policy_logits, value.squeeze(-1)


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(
    path,
    device=None,
    board_size=9,
    action_size=9,
    hidden_size=64,
    num_heads=4,
    num_layers=2,
    dropout=0.1,
):
    device = device or torch.device("cpu")
    model = GameNet(
        board_size=board_size,
        action_size=action_size,
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model
