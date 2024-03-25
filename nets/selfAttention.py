import torch
import torch.nn as nn
import torch.nn.functional as F


# 自注意力块
class SelfAttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=8):
        super(SelfAttentionBlock, self).__init__()

        self.query_transform = nn.Linear(in_channels, out_channels)
        self.key_transform = nn.Linear(in_channels, out_channels)
        self.value_transform = nn.Linear(in_channels, out_channels)

        self.num_heads = num_heads

        self.out_transform = nn.Linear(out_channels * num_heads, in_channels)
        self.layer_norm = nn.LayerNorm(in_channels)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        query = self.query_transform(x)
        key = self.key_transform(x)
        value = self.value_transform(x)

        query = query.view(batch_size, seq_len, self.num_heads, -1)
        key = key.view(batch_size, seq_len, self.num_heads, -1)
        value = value.view(batch_size, seq_len, self.num_heads, -1)

        scores = torch.matmul(query, key.transpose(-2, -1)) / (key.size(-1) ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)

        attended_values = torch.matmul(attention_weights, value)
        attended_values = attended_values.view(batch_size, seq_len, -1)

        output = self.out_transform(attended_values)
        output = self.layer_norm(x + output)

        return output


# Residual Dense Block with Self-Attention
class ResidualDenseBlockAttention(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, num_heads=8):
        super(ResidualDenseBlockAttention, self).__init__()

        self.blocks = nn.ModuleList([
            SelfAttentionBlock(in_channels, out_channels, num_heads) for _ in range(num_blocks)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x