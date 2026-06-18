from .facmac_learner import FACMACLearner


class EA_FACMACLearner(FACMACLearner):
    # Continuous EA relies on rollout/evolution/sync in `run.py`, not population-aware learner updates.
    uses_population_training = False

    def train(self, batch, t_env, episode_num):
        return super().train(batch, t_env, episode_num)
