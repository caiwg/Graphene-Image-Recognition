import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        self.d = d
        self.depth = d // num_heads

        self.wq = nn.Linear(d, d, bias=False)
        self.wk = nn.Linear(d, d, bias=False)
        self.wv = nn.Linear(d, d, bias=False)

        self.fc_out = nn.Linear(d, d)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.transpose(2, 1)

    def forward(self, q, k, v):
        batch_size = q.shape[0]

        q = self.wq(q)  # (batch_size, seq_len, d)
        k = self.wk(k)  # (batch_size, seq_len, d)
        v = self.wv(v)  # (batch_size, seq_len, d)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v)

        scaled_attention = scaled_attention.transpose(2, 1).contiguous().view(batch_size, -1, self.d)

        output = self.fc_out(scaled_attention)

        return output, attention_weights

    def scaled_dot_product_attention(self, q, k, v):
        matmul_qk = torch.matmul(q, k.transpose(-1, -2))

        dk = torch.tensor(self.depth).float()
        scaled_attention_logits = matmul_qk / dk.sqrt()

        attention_weights = nn.Softmax(dim=-1)(scaled_attention_logits)

        output = torch.matmul(attention_weights, v)

        return output, attention_weights


class CustomModule(nn.Module):
    def __init__(self, d, num_heads):
        super(CustomModule, self).__init__()

        self.LN = nn.ModuleList([nn.LayerNorm(d) for _ in range(4)])
        self.MHA = MultiHeadAttention(d, num_heads)
        self.FC = nn.ModuleList([nn.Sequential(nn.Linear(d, d * 4),
                                               nn.ReLU(),
                                               nn.Linear(d * 4, d)) for _ in range(4)])
        self.LN2 = nn.ModuleList([nn.LayerNorm(d) for _ in range(4)])

    def forward(self, x1, x2, x3, x4):
        inputs = [x1, x2, x3, x4]
        queries = []

        # 首先对输入进行layer norm
        for i in range(4):
            inputs[i] = self.LN[i](inputs[i])
            queries.append(inputs[i])

        # 合并所有输入作为k, v
        kv = torch.cat(inputs, dim=-1)

        outputs = []
        # 用每个q进行多头注意力模块
        for i in range(4):
            out, _ = self.MHA(queries[i], kv, kv)
            # 经过LN和多层感知机进行残差计算
            out = self.LN2[i](self.FC[i](out) + out)
            outputs.append(out)

        return outputs[0], outputs[1], outputs[2], outputs[3]