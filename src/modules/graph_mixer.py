import math
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .graph.gnn import GNN


class GraphDec(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        self.rnn_hidden_dim = args.rnn_hidden_dim
        self.embed_dim = args.mixing_embed_dim
        self.state_norm = nn.LayerNorm(self.state_dim)
        self.mixer_weight_scale = getattr(args, "mixer_weight_scale", 0.1)
        self.register_buffer(
            "communication_adjacency",
            th.tensor(
                [[1.0, 1.0, 1.0], [1.0, 1.0, 0.0], [1.0, 0.0, 1.0]]
            ),
        )

        self.mixing_GNN = GNN(
            num_input_features=1,
            hidden_layers=[self.embed_dim],
            state_dim=self.state_dim,
            hypernet_embed=args.hypernet_embed,
            weights_operation="sigmoid",
            weight_scale=self.mixer_weight_scale,
        )
        self.obs_enc_dim = 16
        self.obs_encoder = nn.Sequential(
            nn.Linear(self.rnn_hidden_dim, self.obs_enc_dim), nn.ReLU()
        )
        self.W_attn_query = nn.Linear(self.obs_enc_dim, self.obs_enc_dim, bias=False)
        self.W_attn_key = nn.Linear(self.obs_enc_dim, self.obs_enc_dim, bias=False)
        self.V = nn.Sequential(
            nn.Linear(self.state_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1),
        )

    def forward(
        self, agent_qs, states, agent_obs=None, team_rewards=None, hidden_states=None
    ):
        batch_size = states.size(0)
        states = self.state_norm(states.reshape(-1, self.state_dim))
        agent_qs = agent_qs.view(-1, self.n_agents, 1)
        communication_mask = self.communication_adjacency.expand(
            agent_qs.size(0), -1, -1
        )
        active_agents = th.ones(
            agent_qs.size(0), self.n_agents, dtype=th.bool, device=agent_qs.device
        )

        encoded_states = self.obs_encoder(hidden_states).view(
            -1, self.n_agents, self.obs_enc_dim
        )
        attention = th.matmul(
            self.W_attn_query(encoded_states),
            th.transpose(self.W_attn_key(encoded_states), 1, 2),
        ) / math.sqrt(self.obs_enc_dim)
        adjacency = F.softmax(attention - 1e10 * (1 - communication_mask), dim=2)
        local_reward_fractions, mixed_q = self.mixing_GNN(
            agent_qs, adjacency * communication_mask, states, self.n_agents
        )
        total_q = (mixed_q + self.V(states).view(-1, 1, 1)).view(batch_size, -1, 1)

        local_rewards = None
        if team_rewards is not None:
            local_rewards = local_reward_fractions.view(
                batch_size, -1, self.n_agents
            ) * team_rewards.repeat(1, 1, self.n_agents)
        return total_q, local_rewards, active_agents.view(
            batch_size, -1, self.n_agents
        )
