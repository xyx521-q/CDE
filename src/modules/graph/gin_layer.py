import torch
import torch.nn as nn


class GINGraphConvolution(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        state_dim,
        hypernet_embed,
        weights_operation=None,
        weight_scale=1.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights_operation = weights_operation
        self.weight_scale = weight_scale
        self.hidden_features = int((in_features + out_features) / 2)
        self.w1 = nn.Sequential(
            nn.Linear(state_dim, hypernet_embed),
            nn.ReLU(),
            nn.Linear(hypernet_embed, in_features * self.hidden_features),
        )
        self.b1 = nn.Linear(state_dim, self.hidden_features)
        self.w2 = nn.Sequential(
            nn.Linear(state_dim, hypernet_embed),
            nn.ReLU(),
            nn.Linear(hypernet_embed, self.hidden_features * out_features),
        )
        self.b2 = nn.Linear(state_dim, out_features)

    def forward(self, input_features, adjacency, states):
        aggregated = torch.matmul(adjacency, input_features)
        batch_size = aggregated.size(0)
        weight1 = self.w1(states).view(-1, self.in_features, self.hidden_features)
        weight2 = self.w2(states).view(
            -1, self.hidden_features, self.out_features
        )
        if self.weights_operation == "abs":
            weight1 = torch.abs(weight1)
            weight2 = torch.abs(weight2)
        elif self.weights_operation == "clamp":
            weight1 = nn.ReLU()(weight1)
            weight2 = nn.ReLU()(weight2)
        elif self.weights_operation == "sigmoid":
            weight1 = torch.sigmoid(weight1) * self.weight_scale
            weight2 = torch.sigmoid(weight2) * self.weight_scale
        hidden = torch.nn.functional.leaky_relu(
            torch.matmul(aggregated, weight1)
            + self.b1(states).view(batch_size, 1, -1)
        )
        return torch.matmul(hidden, weight2) + self.b2(states).view(
            batch_size, 1, -1
        )
