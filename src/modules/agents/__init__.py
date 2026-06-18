REGISTRY = {}

from .mlp_agent import MLPAgent
from .rnn_agent import RNNAgent
from .qmix_agent import QMIXRNNAgent, FFAgent

REGISTRY["mlp"] = MLPAgent
REGISTRY["rnn"] = RNNAgent
REGISTRY["qmixrnn"] = QMIXRNNAgent
REGISTRY["ff"] = FFAgent
