import datetime
import pickle
import os
import pprint
import time
import math as mth
import threading
import torch as th
import numpy as np
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath
import random

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer, Best_experience_Buffer
from components.transforms import OneHot


def run(_run, _config, _log):

    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    # setup loggers
    logger = Logger(_log)

    # _log.info("Experiment Parameters:")
    # experiment_params = pprint.pformat(_config,
    #                                    indent=4,
    #                                    width=1)
    # _log.info("\n\n" + experiment_params + "\n")

    # # configure tensorboard logger
    # unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    # args.unique_token = unique_token
    # if args.use_tensorboard:
    #     tb_logs_direc = os.path.join(dirname(dirname(abspath(__file__))), "results", "tb_logs")
    #     tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
    #     logger.setup_tb(tb_exp_direc)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)


def evaluate_sequential(args, runner):

    for _ in range(args.test_nepisode):
        runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()

def run_sequential(args, logger):

    # create the dirs to save results
    os.makedirs("./performance/" + args.save_dir + "/won", exist_ok=True)
    os.makedirs("./performance/" + args.save_dir + "/test", exist_ok=True)
    os.makedirs("./performance/" + args.save_dir + "/ckpt", exist_ok=True)

    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    # args.unit_type_bits = env_info["unit_type_bits"]
    # args.shield_bits_ally = env_info["shield_bits_ally"]
    # args.shield_bits_enemy = env_info["shield_bits_enemy"]
    # args.n_enemies = env_info["n_enemies"]

    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
        #"policy": {"vshape": (env_info["n_agents"],)}
    }
    groups = {
        "agents": args.n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }

    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)
    off_buffer = ReplayBuffer(scheme, groups, args.off_buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)

    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    if args.use_cuda:
        learner.cuda()

    if args.checkpoint_path != "":

        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(args.checkpoint_path))
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

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)
        runner.t_env = timestep_to_load

        if args.evaluate or args.save_replay:
            evaluate_sequential(args, runner)
            return

    # start training
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0
    episode = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    if args.resume:
        won_stats, eval_returns, last_test_T, episode = load_ckpt(args.run_id, runner, learner, mac, off_buffer, args.save_dir)
    else:
        eval_returns = []
        won_stats = []

    while runner.t_env <= args.t_max:

        # critic running log
        running_log = {
            "critic_loss": [],
            "critic_grad_norm": [],
            "td_error_abs": [],
            "target_mean": [],
            "q_taken_mean": [],
            "q_max_mean": [],
            "q_min_mean": [],
            "q_max_var": [],
            "q_min_var": []
        }

        # Run for a whole episode at a time
        episode_batch = runner.run(test_mode=False)
        buffer.insert_episode_batch(episode_batch)
        off_buffer.insert_episode_batch(episode_batch)



        if buffer.can_sample(args.batch_size) and off_buffer.can_sample(args.off_batch_size):
            #train critic normall
            uni_episode_sample = buffer.uni_sample(args.batch_size)
            off_episode_sample = off_buffer.uni_sample(args.off_batch_size)
            max_ep_t = max(uni_episode_sample.max_t_filled(), off_episode_sample.max_t_filled())
            uni_episode_sample = process_batch(uni_episode_sample[:, :max_ep_t], args)
            off_episode_sample = process_batch(off_episode_sample[:, :max_ep_t], args)
            learner.train_critic(uni_episode_sample, best_batch=off_episode_sample, log=running_log)

            #train actor
            episode_sample = buffer.sample_latest(args.batch_size)
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = process_batch(episode_sample[:, :max_ep_t], args)
            learner.train(episode_sample, runner.t_env, running_log)


        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:

            # logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
            # logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
            #     time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
            last_time = time.time()

            last_test_T = runner.t_env
            for _ in range(n_test_runs):
                runner.run(test_mode=True)
            won_stats.append(np.mean(runner.won_count[0:args.test_nepisode]))
            eval_returns.append(np.mean(runner.test_returns[0:args.test_nepisode]))
            print(f"{[args.run_id]} Finished: {episode} episodes, {runner.t_env}/{args.t_max} won_rate: {won_stats[-1]} latested averaged eval returns {eval_returns[-1]} ...", flush=True)
            runner.won_count = []
            runner.test_returns = []

        if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
            model_save_time = runner.t_env
            save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(runner.t_env))
            #"results/models/{}".format(unique_token)
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            learner.save_models(save_path)

        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            # logger.log_stat("episode", episode, runner.t_env)
            # logger.print_recent_stats()
            last_log_T = runner.t_env

        if (time.time() - start_time) / 3600 >= 23:
            save_ckpt(args.run_id, runner, learner, off_buffer, mac, won_stats, eval_returns, last_test_T, episode, args.save_dir)
            break

    save_test_data(args.run_id, eval_returns, args.save_dir)
    save_won_data(args.run_id, won_stats, args.save_dir)
    runner.close_env()
    logger.console_logger.info("Finished Training")

def save_test_data(run_id, data, save_dir):
    with open("./performance/" + save_dir + "/test/test_perform" + str(run_id) + ".pickle", "wb") as handle:
        pickle.dump(data, handle)

def save_won_data(run_id, data, save_dir):
    with open("./performance/" + save_dir + "/won/won_perform" + str(run_id) + ".pickle", "wb") as handle:
        pickle.dump(data, handle)


def save_ckpt(run_idx, runner, learner, off_buffer, mac, won_stats, eval_returns, last_test_T, episode, save_dir, max_save=2):

    PATH = "./performance/" + save_dir + "/ckpt/" + str(run_idx) + "_genric_" + "{}.tar"
    for n in list(range(max_save-1, 0, -1)):
        os.system('cp -rf ' + PATH.format(n) + ' ' + PATH.format(n+1) )
    PATH = PATH.format(1)

    th.save({'runner_t': runner.t,
             'runner_t_env': runner.t_env,
             'last_test_T': last_test_T,
             'episode': episode,
             'won_stats': won_stats,
             'eval_returns': eval_returns,
             'random_state': random.getstate(),
             'np_random_state': np.random.get_state(),
             'torch_random_state': th.random.get_rng_state(),
             'critic_net_state_dict': learner.critic.state_dict(),
             'mixer_state_dict': learner.mixer.state_dict(),
             'target_critic_net_state_dict': learner.target_critic.state_dict(),
             'target_mixer_state_dict': learner.target_mixer.state_dict(),
             'critic_optimiser_state_dict': learner.critic_optimiser.state_dict(),
             'agent_optimiser_state_dict': learner.agent_optimiser.state_dict(),
             'mixer_optimiser_state_dict': learner.mixer_optimiser.state_dict(),
             'agent_net_state_dict': mac.agent.state_dict(),
             'learner_critic_training_steps': learner.critic_training_steps,
             'learner_last_target_update_step': learner.last_target_update_step,
             'off_buffer_data': off_buffer.data,
             'off_buffer_index': off_buffer.buffer_index,
             'off_buffer_episodes_in_buffer': off_buffer.episodes_in_buffer, 
             }, PATH)

def load_ckpt(run_idx, runner, learner, mac, off_buffer, save_dir):
    PATH = "./performance/" + save_dir + "/ckpt/" + str(run_idx) + "_genric_" + "1.tar"
    ckpt = th.load(PATH)
    runner.t = ckpt['runner_t']
    runner.t_env = ckpt['runner_t_env']
    won_stats = ckpt['won_stats']
    eval_returns = ckpt['eval_returns']
    random.setstate(ckpt['random_state'])
    np.random.set_state(ckpt['np_random_state'])
    th.set_rng_state(ckpt['torch_random_state'])
    learner.critic.load_state_dict(ckpt['critic_net_state_dict'])
    learner.target_critic.load_state_dict(ckpt['target_critic_net_state_dict'])
    learner.mixer.load_state_dict(ckpt['mixer_state_dict'])
    learner.target_mixer.load_state_dict(ckpt['target_mixer_state_dict'])
    learner.critic_optimiser.load_state_dict(ckpt['critic_optimiser_state_dict'])
    learner.agent_optimiser.load_state_dict(ckpt['agent_optimiser_state_dict'])
    learner.mixer_optimiser.load_state_dict(ckpt['mixer_optimiser_state_dict'])
    learner.critic_training_steps = ckpt['learner_critic_training_steps']
    learner.last_target_update_step = ckpt['learner_last_target_update_step']
    mac.agent.load_state_dict(ckpt['agent_net_state_dict'])
    off_buffer.data = ckpt['off_buffer_data']
    off_buffer.buffer_index = ckpt['off_buffer_index']
    off_buffer.episodes_in_buffer = ckpt['off_buffer_episodes_in_buffer']
    return won_stats, eval_returns, ckpt['last_test_T'], ckpt['episode']


def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"]//config["batch_size_run"]) * config["batch_size_run"]

    return config


def process_batch(batch, args):

    if batch.device != args.device:
        batch.to(args.device)
    return batch


