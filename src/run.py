import datetime
import os
import pprint
import threading
import time
from os.path import abspath, dirname
from types import SimpleNamespace as SN

import numpy as np
import torch as th

from components.episode_buffer import ReplayBuffer
from controllers.cqmix_controller import CQMixMAC
from ea import mod_neuro_evo as utils_ne
from learners.EA_facmac_learner import EA_FACMACLearner
from learners.facmac_learner import FACMACLearner
from runners.episode_runner import EpisodeRunner
from utils.logging import Logger
from utils.timehelper import time_left, time_str

def rl_to_evo(rl_agent, evo_net):
    for target_param, param in zip(
        evo_net.agent.parameters(), rl_agent.agent.parameters(), strict=False
    ):
        target_param.data.copy_(param.data)


def run(_config, _log):
    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"
    if args.use_cuda:
        th.set_float32_matmul_precision(
            getattr(args, "float32_matmul_precision", "highest")
        )

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config, indent=4, width=1)
    _log.info("\n\n" + experiment_params + "\n")

    unique_token = getattr(args, "unique_token", None) or "{}__{}".format(
        args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    args.unique_token = unique_token
    logger.setup_wandb(
        config=_config,
        name=unique_token,
        project=args.wandb_project,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
        directory=args.wandb_dir,
    )

    # Run and train
    run_sequential(args=args, logger=logger)
    logger.finish()

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print(f"Thread {t.name} is alive! Is daemon: {t.daemon}")
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

def run_sequential(args, logger):

    # Init runner so we can get env info
    runner = EpisodeRunner(args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    args.obs_shape = env_info["obs_shape"]
    args.action_spaces = env_info["action_spaces"]
    args.actions_dtype = env_info["actions_dtype"]
    args.normalise_actions = env_info.get("normalise_actions", False)

    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {
            "vshape": (max(space.shape[0] for space in args.action_spaces),),
            "group": "agents",
            "dtype": th.float,
        },
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {"agents": args.n_agents}
    preprocess = {}

    buffer = ReplayBuffer(
        scheme,
        groups,
        args.buffer_size,
        env_info["episode_limit"] + 1,
        preprocess=preprocess,
        # Keep environment rollouts and replay storage on CPU. Training samples
        # are moved to the model device immediately before learner updates.
        device="cpu",
    )
    evolver = utils_ne.SSNE(args)
    # Setup multiagent controller here
    if args.EA:
        pop = []
        mac = CQMixMAC(buffer.scheme, groups, args)
        for i in range(args.pop_size):
            pop.append(CQMixMAC(buffer.scheme, groups, args))
    else:
        mac = CQMixMAC(buffer.scheme, groups, args)
        pop = []
    fitness = []
    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    if args.EA:
        learner = EA_FACMACLearner(mac, buffer.scheme, logger, args)
    else:
        learner = FACMACLearner(mac, buffer.scheme, logger, args)

    if args.use_cuda:
        learner.cuda(device=args.device)
        for population_mac in pop:
            population_mac.cuda(device=args.device)

    if args.checkpoint_path != "":
        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info(
                f"Checkpoint directiory {args.checkpoint_path} doesn't exist"
            )
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info(f"Loading model from {model_path}")
        learner.load_models(model_path)
        runner.t_env = timestep_to_load

        if args.evaluate:
            for _ in range(args.test_nepisode):
                runner.run(mac, test_mode=True)
            runner.close_env()
            return

    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0
    ea_cycle = 0
    ea_elite_index = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info(f"Beginning training for {args.t_max} timesteps")

    while runner.t_env <= args.t_max:
        # Run for a whole episode at a time
        is_ea_cycle = (
            args.EA
            and runner.t_env > args.start_timesteps
            and episode % args.EA_freq == 0
        )
        if is_ea_cycle:
            ea_cycle += 1
            fitness = []
            for ea_mac in pop:
                episode_batch, episode_return = runner.run(ea_mac, test_mode=False)
                fitness.append(episode_return)
                buffer.insert_episode_batch(episode_batch)
            elite_index = np.argmax(fitness)
            episode_batch, _ = runner.run(mac, test_mode=False)
            buffer.insert_episode_batch(episode_batch)

            # Each generation updates the complete shared policy once.
            for _ in range(args.ea_generations_per_cycle):
                elite_index = evolver.epoch(pop, fitness, 0, agent_level=True)
            ea_elite_index = elite_index
        else:
            fitness = [0.0 for _ in range(args.pop_size)]
            elite_index = 0
            episode_batch, _ = runner.run(mac, test_mode=False)
            buffer.insert_episode_batch(episode_batch)
        if is_ea_cycle:
            learner_update_count = args.ea_learner_updates
        elif episode % args.learner_update_interval == 0:
            learner_update_count = args.learner_updates_per_episode
        else:
            learner_update_count = 0

        for _ in range(learner_update_count):
            if buffer.can_sample(args.batch_size) and (
                buffer.episodes_in_buffer >= getattr(args, "buffer_warmup", 0)
            ):
                episode_sample = buffer.sample(args.batch_size)

                # Truncate batch to only filled timesteps
                max_ep_t = episode_sample.max_t_filled()
                episode_sample = episode_sample[:, :max_ep_t]

                if episode_sample.device != args.device:
                    episode_sample.to(args.device)

                if args.EA:
                    if getattr(learner, "uses_population_training", False):
                        all_teams = []
                        for index_n in range(args.pop_size):
                            all_teams.append(pop[index_n])
                        all_teams.append(mac)
                        learner.train(episode_sample, all_teams, runner.t_env, episode)
                    else:
                        learner.train(episode_sample, runner.t_env, episode)
                else:
                    learner.train(episode_sample, runner.t_env, episode)

        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:
            logger.console_logger.info(f"t_env: {runner.t_env} / {args.t_max}")
            logger.console_logger.info(
                f"Estimated time left: {time_left(last_time, last_test_T, runner.t_env, args.t_max)}. Time passed: {time_str(time.time() - start_time)}"
            )
            last_time = time.time()

            last_test_T = runner.t_env

            rl_eval_reward = 0
            for _ in range(n_test_runs):
                _, rl_tp_reward = runner.run(mac, test_mode=True)
                rl_eval_reward += rl_tp_reward
            rl_eval_reward = rl_eval_reward / n_test_runs
            print("RL eval ", rl_eval_reward)
            logger.log_stat(
                "eval/rl_reward", rl_eval_reward, runner.t_env
            )
            if args.EA:
                ea_eval_reward = 0
                last_test_T = runner.t_env
                for _ in range(n_test_runs):
                    _, ea_tp_reward = runner.run(
                        pop[ea_elite_index], test_mode=True
                    )
                    ea_eval_reward += ea_tp_reward
                ea_eval_reward = ea_eval_reward / n_test_runs
            else:
                ea_eval_reward = 0
            print("EA eval ", ea_eval_reward)
            logger.log_stat(
                "eval/ea_reward", ea_eval_reward, runner.t_env
            )
            logger.log_stat(
                "eval/best_reward",
                max(rl_eval_reward, ea_eval_reward),
                runner.t_env,
            )
        if is_ea_cycle:
            # Replace any index different from the new elite
            replace_index = np.argmin(fitness)
            if replace_index == elite_index:
                replace_index = (replace_index + 1) % args.pop_size
            rl_to_evo(mac, pop[replace_index])
            evolver.rl_policy = replace_index
            if ea_cycle % args.ea_log_interval == 0:
                logger.console_logger.info(
                    "EA cycle %s at t_env=%s: elite=%s, synced_rl_to_population=%s",
                    ea_cycle,
                    runner.t_env,
                    elite_index,
                    replace_index,
                )

        if args.save_model and (
            runner.t_env - model_save_time >= args.save_model_interval
            or model_save_time == 0
        ):
            model_save_time = runner.t_env
            save_path = os.path.join(
                args.local_results_path, "models", args.unique_token, str(runner.t_env)
            )
            # "results/models/{}".format(unique_token)
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info(f"Saving models to {save_path}")

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            # learner.save_models(save_path, args.unique_token, model_save_time)

            learner.save_models(save_path)

        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("train/t_env", runner.t_env, runner.t_env)
            logger.flush()
            logger.print_recent_stats(runner.t_env)
            last_log_T = runner.t_env

    runner.close_env()
    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):
    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning(
            "CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!"
        )

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (
            config["test_nepisode"] // config["batch_size_run"]
        ) * config["batch_size_run"]

    return config
