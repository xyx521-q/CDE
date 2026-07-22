import os
import sys
import unittest
from os.path import abspath, dirname
from types import SimpleNamespace

import torch as th

PROJECT_ROOT = dirname(dirname(abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from modules.graph_mixer import GraphDec


class GraphMixerTest(unittest.TestCase):
    def test_transport_communication_is_fixed_and_ignores_signed_observations(self):
        args = SimpleNamespace(
            n_agents=3,
            state_shape=33,
            rnn_hidden_dim=64,
            mixing_embed_dim=64,
            hypernet_embed=64,
            mixer_weight_scale=0.1,
        )
        mixer = GraphDec(args)
        agent_qs = th.randn(1, 1, 3, 1, requires_grad=True)
        states = th.randn(1, 1, 33)
        hidden_states = th.randn(1, 1, 3, 64)
        negative_observations = -th.ones(1, 1, 3, 11)

        q_tot, _, active_agents = mixer(
            agent_qs,
            states,
            agent_obs=negative_observations,
            hidden_states=hidden_states,
        )
        gradients = th.autograd.grad(q_tot.sum(), agent_qs)[0]

        self.assertTrue(active_agents.all())
        self.assertTrue(
            th.equal(
                mixer.communication_adjacency,
                th.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0], [1.0, 0.0, 1.0]]),
            )
        )
        self.assertTrue(th.all(gradients.abs() > 0))


if __name__ == "__main__":
    unittest.main()
