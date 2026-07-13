from ea.ssne_base import AgentWeightSparseMutationSSNEBase, unsqueeze


class SSNE(AgentWeightSparseMutationSSNEBase):
    mutation_probability_gate = False
