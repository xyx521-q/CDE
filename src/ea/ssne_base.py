import math
import random

import fastrand
import numpy as np

from ea.mod_utils import is_lnorm_key


class SSNEBase:
    use_agent_level_clone_crossover = False
    mutation_probability_gate = True

    def __init__(self, args):
        self.current_gen = 0
        self.args = args
        self.prob_reset_and_sup = args.prob_reset_and_sup
        self.frac = args.frac
        self.population_size = args.pop_size
        self.num_elitists = max(1, int(args.elite_fraction * args.pop_size))
        self.rl_policy = None
        self.selection_stats = {
            "elite": 0,
            "selected": 0,
            "discarded": 0,
            "total": 0.0000001,
        }

    def model(self, gene, agent_index):
        raise NotImplementedError

    def selection_tournament(self, index_rank, num_offsprings, tournament_size):
        offsprings = []
        for _ in range(num_offsprings):
            winner = np.min(np.random.randint(len(index_rank), size=tournament_size))
            offsprings.append(index_rank[winner])

        offsprings = list(set(offsprings))
        if len(offsprings) % 2 != 0:
            offsprings.append(offsprings[fastrand.pcg32bounded(len(offsprings))])
        return offsprings

    def list_argsort(self, seq):
        return sorted(range(len(seq)), key=seq.__getitem__)

    def regularize_weight(self, weight, mag):
        return max(-mag, min(weight, mag))

    def crossover_inplace(self, gene1, gene2, agent_index):
        for param1, param2 in zip(
            self.model(gene1, agent_index).parameters(),
            self.model(gene2, agent_index).parameters(),
        ):
            weight1 = param1.data
            weight2 = param2.data
            swaps = fastrand.pcg32bounded(weight1.shape[0] * (2 if weight1.ndim == 2 else 1))
            for _ in range(swaps):
                index = fastrand.pcg32bounded(weight1.shape[0])
                if random.random() < 0.5:
                    weight1[index] = weight2[index]
                else:
                    weight2[index] = weight1[index]

    def clone(self, master, replacee, agent_index):
        for target_param, source_param in zip(
            self.model(replacee, agent_index).parameters(),
            self.model(master, agent_index).parameters(),
        ):
            target_param.data.copy_(source_param.data)

    def reset_genome(self, gene, agent_index):
        for param in self.model(gene, agent_index).parameters():
            param.data.copy_(param.data)

    def mutate_inplace(self, gene, agent_index, agent_level=False):
        raise NotImplementedError

    def epoch(self, pop, fitness_evals, agent_index, agent_level=False):
        index_rank = np.argsort(fitness_evals)[::-1]
        elitist_index = index_rank[: self.num_elitists]
        offsprings = self.selection_tournament(
            index_rank,
            num_offsprings=len(index_rank) - self.num_elitists,
            tournament_size=3,
        )
        unselects = [
            index
            for index in range(self.population_size)
            if index not in offsprings and index not in elitist_index
        ]
        random.shuffle(unselects)

        if self.rl_policy is not None:
            self.selection_stats["total"] += 1.0
            if self.rl_policy in elitist_index:
                self.selection_stats["elite"] += 1.0
            elif self.rl_policy in offsprings:
                self.selection_stats["selected"] += 1.0
            elif self.rl_policy in unselects:
                self.selection_stats["discarded"] += 1.0
            self.rl_policy = None

        new_elitists = []
        for index in elitist_index:
            replacee = unselects.pop(0) if unselects else offsprings.pop(0)
            new_elitists.append(replacee)
            self.clone(pop[index], pop[replacee], agent_index)

        if len(unselects) % 2 != 0:
            unselects.append(unselects[fastrand.pcg32bounded(len(unselects))])
        for index1, index2 in zip(unselects[0::2], unselects[1::2]):
            self.clone(pop[random.choice(new_elitists)], pop[index1], agent_index)
            self.clone(pop[random.choice(offsprings)], pop[index2], agent_index)
            if self.use_agent_level_clone_crossover and agent_level:
                if random.random() < 0.5:
                    self.clone(pop[index1], pop[index2], agent_index)
                else:
                    self.clone(pop[index2], pop[index1], agent_index)
            elif random.random() < self.args.crossover_prob:
                self.crossover_inplace(pop[index1], pop[index2], agent_index)

        for index in range(self.population_size):
            if index not in new_elitists and (
                not self.mutation_probability_gate
                or random.random() < self.args.mutation_prob
            ):
                self.mutate_inplace(pop[index], agent_index, agent_level)

        return new_elitists[0]


class AgentSSNEBase(SSNEBase):
    def model(self, gene, agent_index):
        return gene.agent


class AgentWeightSSNEBase(SSNEBase):
    def model(self, gene, agent_index):
        return gene.agent_W[agent_index]


class AgentWeightRowMutationSSNEBase(AgentWeightSSNEBase):
    def mutate_inplace(self, gene, agent_index, agent_level=False):
        model_params = self.model(gene, agent_index).state_dict()
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


class AgentWeightSparseMutationSSNEBase(AgentWeightSSNEBase):
    def mutate_inplace(self, gene, agent_index, agent_level=False):
        model_params = self.model(gene, agent_index).state_dict()
        probabilities = np.random.uniform(0, 1, len(model_params)) * 2
        for index, key in enumerate(model_params):
            if is_lnorm_key(key):
                continue
            weight = model_params[key]
            if weight.ndim != 2:
                continue
            mutation_count = fastrand.pcg32bounded(
                int(math.ceil(self.frac * weight.shape[0] * weight.shape[1]))
            )
            if random.random() >= probabilities[index] * self.args.mutation_prob:
                continue
            for _ in range(mutation_count):
                row = fastrand.pcg32bounded(weight.shape[0])
                column = fastrand.pcg32bounded(weight.shape[1])
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
                weight[row, column].clamp_(-1000000, 1000000)


def unsqueeze(array, axis=1):
    if axis == 0:
        return np.reshape(array, (1, len(array)))
    if axis == 1:
        return np.reshape(array, (len(array), 1))
