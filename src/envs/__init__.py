from functools import partial
from .maenv import MGEnv

try:
    from smac.env import StarCraft2Env
except Exception:
    StarCraft2Env = None


def env_fn(env, env_args, args):
    if env == MGEnv:
        config = env_args
        algo_name = args.name if hasattr(args, 'name') else "default"
        return env(config=config, algo_name=algo_name)
    else:
        return env(**env_args)


REGISTRY = {}
REGISTRY["microgrid"] = partial(env_fn, env=MGEnv)
if StarCraft2Env is not None:
    REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
