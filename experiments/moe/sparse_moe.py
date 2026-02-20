import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicExpert(nn.Module):
    # expert can be a Linear layer or MLP layer or more complicated (activation function = swiglu)
    def __init__(self, feature_in, feature_out):
        super().__init__()
        self.linear = nn.Linear(feature_in, feature_out)

    def forward(self, x):
        return self.linear(x)


class MOERouter(nn.Module):
    def __init__(self, hidden_dim, expert_number, top_k):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, expert_number)
        self.expert_number = expert_number
        self.top_k = top_k

    def forward(self, hidden_states):
        router_logits = self.gate(hidden_states)  # shape (b*s, expert_number)
        routing_probs = F.softmax(router_logits, dim=-1, dtype=torch.float)

        # shapes are (b*s, top_k)
        router_weights, selected_experts = torch.topk(routing_probs, self.top_k, dim=-1)

        # normalization on expert weights
        router_weights = router_weights / router_weights.sum(dim=-1, keepdim=True)
        router_weights = router_weights.to(hidden_states.device.dtype)
        expert_mask = F.one_hot(
            selected_experts, num_classes=self.expert_number
        )  # shape (b*s, top_k, expert_number)
        expert_mask = expert_mask.permute(2, 1, 0)  # (expert_number, top_k, b*s)
        return router_logits, router_weights, selected_experts, expert_mask, expert_mask


class MOEConfig:
    def __init__(self, hidden_dim, expert_number, top_k, shared_experts_number=2):
        self.hidden_dim = hidden_dim
        self.expert_number = expert_number
        self.top_k = top_k
        self.shared_experts_number = shared_experts_number


class SparseMOE(nn.Module):
    # each token goes to topk experts, each token gets hidden_embeddings
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.expert_number = config.expert_number
        self.top_k = config.top_k
        self.experts = nn.ModuleList(
            [
                BasicExpert(self.hidden_dim, self.hidden_dim)
                for _ in range(config.expert_number)
                for _ in range(self.expert_number)
            ]
        )
        self.router = MOERouter(self.hidden_dim, self.expert_number, self.top_k)

    def forward(self, x):
        # x shape (b, s, hidden_dim)
        batch_size, seq_len, hidden_dim = x.size()
        hidden_states = x.view(-1, hidden_dim)  # shape (b*s, hidden_dim)
        router_logits, router_weights, selected_experts_indices, expert_mask = (
            self.router(hidden_states)
        )
        # selected_experts_indices shape (b*s, top_k), expert_mask shape (expert_number, top_k, b*s)
        final_hidden_states = torch.zeros(
            (batch_size * seq_len, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        for expert_idx in range(self.expert_number):
            expert_layer = self.experts[expert_idx]
            # expert_mask[expert_idx] shape (top_k, b*s)
            idx, top_x = torch.where(expert_mask[expert_idx])
            # idx, top_x both 1-dim tensor, idx = 0/1 (expert top1 or top2)
            # top_x = index of token in batch*seq_len
            # e.g. input: batch_size = 2, seq_len = 4, top_x in [0, 7], meaning 8 tokens, idx in [0, 1], meaning this token views current expert as its top1/top2 expert
            # hidden_states shape (b*s, hidden_dim)
            # top_x's hidden_states, (selected_token_number, hidden_dim)
            current_state = hidden_states.unsqueeze(0)[:, top_x, :].reshape(
                -1, hidden_dim
            )
            # router_weight shape (b*s, top_k)
            current_hidden_states = expert_layer(current_state) * router_weights[
                top_x, idx
            ].unsqueeze(-1)  # (selected_token_number, 1), broadcast here

            # add current expert output to final_hidden_state
            final_hidden_states.index_add_(
                0, top_x, current_hidden_states.to(hidden_states.dtype)
            )

        # final_hidden_states back to original shape
        final_hidden_states = final_hidden_states.reshape(
            batch_size, seq_len, hidden_dim
        )
        return final_hidden_states, router_logits  # shape (b*s, expert_number)


def test_token_level_moe():
    x = torch.rand(2, 4, 16)
    config = MOEConfig(16, 2, 2)
    token_level_moe = SparseMOE(config)
    out = token_level_moe(x)
    print(out[0].shape, out[1].shape)


test_token_level_moe()
