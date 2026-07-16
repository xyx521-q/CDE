import torch
import torch.nn as nn

from .gin_layer import GINGraphConvolution


class GNN(nn.Module):
    def __init__(
        self,
        num_input_features,
        hidden_layers,
        state_dim,
        hypernet_embed,
        weights_operation=None,
        weight_scale=1.0,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.weights_operation = weights_operation
        self.weight_scale = weight_scale
        self.nonlinearity = nn.ELU()
        self.layers = nn.ModuleList(
            [
                GINGraphConvolution(
                    num_input_features if index == 0 else hidden_layers[index - 1],
                    hidden_layers[index],
                    state_dim,
                    hypernet_embed,
                    weights_operation=weights_operation,
                    weight_scale=weight_scale,
                )
                for index in range(len(hidden_layers))
            ]
        )
        self.wout = nn.Sequential(
            nn.Linear(state_dim, hypernet_embed),
            nn.ReLU(),
            nn.Linear(hypernet_embed, hidden_layers[-1]),
        )
        self.wout_per_node = nn.Sequential(
            nn.Linear(state_dim, hypernet_embed),
            nn.ReLU(),
            nn.Linear(hypernet_embed, hidden_layers[-1]),
        )
        self.bout_per_node = nn.Sequential(
            nn.Linear(state_dim, hidden_layers[-1]),
            nn.ReLU(),
            nn.Linear(hidden_layers[-1], 1),
        )

    def forward(self, input_features, adjacency, states, num_agents):
        batch_size = input_features.size(0)
        features = input_features
        for layer in self.layers:
            features = self.nonlinearity(layer(features, adjacency, states))

        alive = (torch.max(adjacency, dim=2)[0] > 0).unsqueeze(2).float()
        readout = torch.sum(features * alive, dim=1) / (
            torch.sum(alive, dim=1) + 1e-10
        )
        wout = self.wout(states).view(batch_size, -1, 1)
        if self.weights_operation == "abs":
            wout = torch.abs(wout)
        elif self.weights_operation == "clamp":
            wout = nn.ReLU()(wout)
        elif self.weights_operation == "sigmoid":
            wout = torch.sigmoid(wout) * self.weight_scale
        scalar_out = torch.matmul(readout.view(batch_size, 1, -1), wout)

        per_node_scalars = torch.matmul(
            features.view(batch_size, num_agents, -1),
            self.wout_per_node(states).view(batch_size, -1, 1),
        ) + self.bout_per_node(states).view(batch_size, -1, 1)
        per_node_scalars = torch.softmax(
            per_node_scalars - 1e10 * (1 - alive), dim=1
        )
        return per_node_scalars, scalar_out
