import torch
from torch import nn
import torch.nn.functional as F


class BasicGPT(nn.Module):
    def __init__(self, embed_size, vocab_size, num_layers):
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embed_size)

        # Initialize all decoder layers
        self.decoder_layers = nn.ModuleList(
            [TransformerDecoderLayer(embed_size) for _ in range(num_layers)]
        )

    def forward(self, x):
        x = self.embedding_layer(x)

        for i in range(self.num_layers):
            x = self.decoder_layers[i](x)

        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.attentionLayer = MultiHeadLatentAttention(embed_size)
        self.norm1 = nn.LayerNorm(embed_size)
        self.feedForwardLayer = FeedForwardLayer(output_size=embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

    def forward(self, x):
        batch_size, seq_length, embed_size = x.shape
        attention_output = self.attentionLayer(x)

        x = self.norm1(x + attention_output)
        x_flat = x.view(batch_size * seq_length, embed_size)

        ff_output = self.feedForwardLayer(x_flat)
        ff_output = ff_output.view(batch_size, seq_length, embed_size)

        x = self.norm2(x + ff_output)
        return x


class FeedForwardLayer(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.fc1 = nn.Linear(output_size, output_size)
        self.fc2 = nn.Linear(output_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x


class MultiHeadLatentAttention(nn.Module):
    def __init__(self, d_model, num_heads, q_latent_dim, kv_latent_dim):
        super().__init__()
        # Low-rank compression for KV
        self.Wkv_d = nn.Linear(d_model, kv_latent_dim)  # Down-projection
        self.Wv_u = nn.Linear(
            kv_latent_dim, num_heads * (d_model // num_heads)
        )  # Up-projection
        # Similar projections are used for Queries (Q)
