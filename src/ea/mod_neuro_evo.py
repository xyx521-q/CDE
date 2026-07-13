import random

import fastrand
import numpy as np
import torch as th

from ea.mod_utils import is_lnorm_key
from ea.ssne_base import AgentSSNEBase, unsqueeze


class SSNE(AgentSSNEBase):
    use_agent_level_clone_crossover = True

    def crossover_inplace(self, gene1, gene2, agent_index):
        bias1 = None
        bias2 = None
        for param1, param2 in zip(gene1.agent.parameters(), gene2.agent.parameters()):
            if param1.ndim == 1:
                bias1 = param1.data
                bias2 = param2.data

        for param1, param2 in zip(gene1.agent.parameters(), gene2.agent.parameters()):
            weight1 = param1.data
            weight2 = param2.data
            if weight1.ndim == 2:
                for _ in range(fastrand.pcg32bounded(weight1.shape[0] * 2)):
                    index = fastrand.pcg32bounded(weight1.shape[0])
                    if random.random() < 0.5:
                        weight1[index, :] = weight2[index, :]
                        bias1[index] = bias2[index]
                    else:
                        weight2[index, :] = weight1[index, :]
                        bias2[index] = bias1[index]

    def mutate_inplace(self, gene, agent_index, agent_level=False):
        model_params = gene.agent.state_dict()
        if agent_level:
            with th.no_grad():
                for key, weight in model_params.items():
                    if is_lnorm_key(key) or weight.ndim != 2:
                        continue
                    draws = th.rand_like(weight)
                    noise = th.randn_like(weight)
                    if self.frac < 1.0:
                        mutation_count = int(weight.shape[1] * self.frac)
                        mutation_mask = th.zeros_like(weight, dtype=th.bool)
                        if mutation_count:
                            selected = th.rand_like(weight).topk(
                                mutation_count, dim=1
                            ).indices
                            mutation_mask.scatter_(1, selected, True)
                    else:
                        mutation_mask = th.ones_like(weight, dtype=th.bool)
                    reset_mask = draws < self.prob_reset_and_sup
                    super_mutation_mask = (
                        draws >= self.prob_reset_and_sup
                    ) & (draws < 2 * self.prob_reset_and_sup)
                    mutated = weight + noise * 0.1 * weight
                    mutated = th.where(
                        reset_mask,
                        weight + noise * 10.0 * weight,
                        mutated,
                    )
                    mutated = th.where(super_mutation_mask, noise, mutated)
                    mutated = th.where(mutation_mask, mutated, weight)
                    weight.copy_(mutated.clamp(-1000000, 1000000))
            return

        probabilities = np.random.uniform(0, 1, len(model_params)) * 2
        for index, key in enumerate(model_params):
            if is_lnorm_key(key):
                continue
            weight = model_params[key]
            if weight.ndim != 2:
                continue
            action_probability = 1.0 if agent_level else probabilities[index]
            for row in range(weight.shape[0]):
                if random.random() > action_probability:
                    continue
                columns = random.sample(
                    range(weight.shape[1]), int(weight.shape[1] * self.frac)
                )
                for column in columns:
                    draw = random.random()
                    if draw < self.prob_reset_and_sup:
                        weight[row, column] += random.gauss(
                            0, 10 * weight[row, column].item()
                        )
                    elif draw < 2 * self.prob_reset_and_sup:
                        weight[row, column] = random.gauss(0, 1)
                    else:
                        weight[row, column] += random.gauss(
                            0, 0.1 * weight[row, column].item()
                        )
                weight[row, :].clamp_(-1000000, 1000000)
