import torch
from torch import nn
import torch.nn.functional as F


# Generic GPT
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
        # runs on decoder layer
        batch_size, seq_length, embed_size = x.shape

        # run attentin layer for contextualizing tokens
        attention_output = self.attentionLayer(x)

        # Residaul connection 1
        x = self.norm1(x + attention_output)

        # Flatten the tensor from [B, S, D] to [B * S, D]
        x_flat = x.view(batch_size * seq_length, embed_size)

        # Run feedforward layer for point-wise feedforward network
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


# actually i do not know what it is exactly. just write
class MultiHeadLatentAttention(nn.Module):
    def __init__(self, d_model, num_heads, q_latent_dim, kv_latent_dim):
        super().__init__()
        # Low-rank compression for KV
        self.Wkv_d = nn.Linear(d_model, kv_latent_dim)  # Down-projection
        self.Wv_u = nn.Linear(
            kv_latent_dim, num_heads * (d_model // num_heads)
        )  # Up-projection
        # Similar projections are used for Queries (Q)


""" ------------------------Dense MoE-----------------------------"""


class BasicGPT(nn.Module):
    def __init__(self, embed_size, vocab_size, num_layers):
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embed_size)

        # Initialize all decoder layers
        self.decoder_layers = nn.ModuleList(
            [TransformerDecoderLayer(embed_size) for _ in range(num_layers)]
        )

    def forward(self, x):
        # Input: [B, S], dtype: int64
        x = self.embedding_layer(x)
        # Output: [B, S, D], dtype:float32

        # Loop through decoder layers
        for i in range(self.num_layers):
            x = self.decoder_layers[i](x)
        # Output: [B, S, D], dtype: float32

        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.attentionLayer = MultiHeadLatentAttention(embed_size)
        self.norm1 = nn.LayerNorm(embed_size)
        self.feedForwardLayer = FeedForwardLayer(output_size=embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

    def forward(self, x):
        # runs on decoder layer
        batch_size, seq_length, embed_size = x.shape

        # run attentin layer for contextualizing tokens
        attention_output = self.attentionLayer(x)

        # Residaul connection 1
        x = self.norm1(x + attention_output)

        # Flatten the tensor from [B, S, D] to [B * S, D]
        x_flat = x.view(batch_size * seq_length, embed_size)

        # Run dense MoE layer to replace feedforward network
        ff_output = self.dense_moe(x_flat)

        ff_output = ff_output.view(batch_size, seq_length, embed_size)

        x = self.norm2(x + ff_output)
        return x


class DenseMixturOfExperts(nn.Module):
    def __init__(self, embed_size, num_experts):
        super(DenseMixturOfExperts, self).__init__()
        self.embed_size = embed_size
        self.num_experts = num_experts

        # Initialize the experts
        self.experts = nn.ModuleList(
            [FeedForwardLayer(embed_size) for _ in range(num_experts)]
        )

        # Initialize the router layer
        self.router = Router(embed_size=embed_size, num_experts=num_experts)

    def forward(self, x):
        # X shape: [T, D]
        expert_probabilities = self.router(x)
        # Expert probability Shape: [T, NUM_EXPERT]

        # Pass each token through all experts and stack th results.
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        # Expert Output Shape: [T, NUM_EXPERTS, D]

        # Scale the expert outputs by the expert probabilities
        expert_outputs = expert_outputs * expert_probabilities.unsqueeze(-1)
        # Scaled outputs Shape: [T, NUM_EXPERTS, D]

        # Sum the expert outputs over the NUM_EXPERT dimension
        expert_outputs = expert_outputs.sum(dim=1)
        # Out Shape: [T, D]

        return expert_outputs


class Router(nn.Module):
    def __init__(self, embed_size, num_experts):
        super().__init__()
        self.fc = nn.Linear(embed_size, embed_size)
        self.fc2 = nn.Linear(embed_size, num_experts)

    def forward(self, x):
        # X shape: [T, D]
        x = self.fc(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)

        # Output shape: [T, NUM_EXPERTS], Probability distribution
        return x


""" ------------------------Sparse MoE-----------------------------"""
# 15:55
