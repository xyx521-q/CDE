import torch as th
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.parametrizations import spectral_norm

class FACMACCritic(nn.Module):
    def __init__(self, scheme, args):
        super(FACMACCritic, self).__init__()
        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.input_shape = self._get_input_shape(scheme) + self.n_actions
        self.output_type = "q"
        self.hidden_states = None

        # Set up network layers
        self.fc1 = nn.Linear(self.input_shape, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc3 = nn.Linear(args.rnn_hidden_dim, 1)

    def init_hidden(self, batch_size):
        # make hidden states on same device as model
        self.hidden_states = None

    def forward(self, inputs, actions, hidden_state=None):
        if actions is not None:
            inputs = th.cat([inputs.view(-1, self.input_shape - self.n_actions),
                             actions.contiguous().view(-1, self.n_actions)], dim=-1)
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q, hidden_state

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        return input_shape

class FACMACDiscreteCritic(nn.Module):
    def __init__(self, scheme, args):
        super(FACMACDiscreteCritic, self).__init__()
        self.args = args
        self.n_actions = scheme["actions_onehot"]["vshape"][0]
        self.n_agents = args.n_agents
        self.input_shape = self._get_input_shape(scheme) + self.n_actions
        self.output_type = "q"
        self.hidden_states = None

        # Set up network layers
        # self.fc1 = nn.Linear(self.input_shape, args.rnn_hidden_dim)
        # self.fc2 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        # self.fc3 = nn.Linear(args.rnn_hidden_dim, 1)

        spectral_func = spectral_norm
        if args.spectral_regularization:
            try:
                from pytorch_spectral_utils import spectral_regularize
                spectral_func = spectral_regularize
            except ImportError:
                print("pytorch_spectral_utils not found, using spectral_norm instead")

        # Set up network layers
        self.fc1 = nn.Linear(self.input_shape, args.rnn_hidden_dim)
        if args.critic_spectral[0] == "y":
            self.fc1 = spectral_func(self.fc1)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        if args.critic_spectral[1] == "y":
            self.fc2 = spectral_func(self.fc2)
        self.fc3 = nn.Linear(args.rnn_hidden_dim, 1)
        if args.critic_spectral[2] == "y":
            self.fc3 = spectral_func(self.fc3)

    def init_hidden(self, batch_size):
        # make hidden states on same device as model
        self.hidden_states = None

    def forward(self, inputs, actions, hidden_state=None):
        if actions is not None:
            inputs = th.cat([inputs.reshape(-1, self.input_shape - self.n_actions),
                             actions.contiguous().view(-1, self.n_actions)], dim=-1)
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q, hidden_state

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        return input_shape

from collections import OrderedDict
class PeVFA_FACMACDiscreteCritic(nn.Module):
    def __init__(self, scheme, args):
        super(PeVFA_FACMACDiscreteCritic, self).__init__()
        self.args = args
        self.n_actions = scheme["actions_onehot"]["vshape"][0]
        self.n_agents = args.n_agents
        self.input_shape = self._get_input_shape(scheme) + self.n_actions
        self.output_type = "q"
        self.hidden_states = None


        # Set up network layers
        self.fc1 = nn.Linear(self.input_shape, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc3 = nn.Linear(args.rnn_hidden_dim, 1)

        self.params = OrderedDict(self.named_parameters())
        self.nonlinearity = F.leaky_relu

        # spectral_func = spectral_norm
        # if args.spectral_regularization:
        #     try:
        #         from pytorch_spectral_utils import spectral_regularize
        #         spectral_func = spectral_regularize
        #     except ImportError:
        #         print("pytorch_spectral_utils not found, using spectral_norm instead")
        #
        # # Set up network layers
        # self.fc1 = nn.Linear(self.input_shape, args.rnn_hidden_dim)
        # if args.critic_spectral[0] == "y":
        #     self.fc1 = spectral_func(self.fc1)
        # self.fc2 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        # if args.critic_spectral[1] == "y":
        #     self.fc2 = spectral_func(self.fc2)
        # self.fc3 = nn.Linear(args.rnn_hidden_dim, 1)
        # if args.critic_spectral[2] == "y":
        #     self.fc3 = spectral_func(self.fc3)

    def init_hidden(self, batch_size):
        # make hidden states on same device as model
        self.hidden_states = None

    def forward(self, inputs, actions, hidden_state=None):

        #print("2",out_p.shape)
        if actions is not None:
            inputs = th.cat([inputs.reshape(-1, self.input_shape - self.n_actions),
                             actions.contiguous().view(-1, self.n_actions)], dim=-1)
        #print("3",inputs.shape)

        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q, hidden_state

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        return input_shape