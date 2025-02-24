import os
import datetime
import logging
import pandas as pd

import torch as th

from config import Config

import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil
import dataclasses

# Import envs #
from simulator_vec_env import SimulatorVecEnv
from envs.iiwa_sample_joint_vel_env import IiwaJointVelEnv
from envs.iiwa_sample_env import IiwaSampleEnv
from expert import PushingPolicy

# Monitor envs 
from stable_baselines3.common.vec_env import VecMonitor

# Models #
from stable_baselines3 import PPO

from utils import rollout

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy

from imitation.policies import serialize as policies_serialize
from imitation.data import serialize
from imitation.algorithms.adversarial import common
from imitation.algorithms.adversarial.gail import GAIL
from imitation.algorithms import bc
from imitation.algorithms.dagger import SimpleDAggerTrainer, LinearBetaSchedule
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from imitation.util import logger as imit_logger

from utils import utils
import TD3_BC

SEED = 0
START_SIDES = []

rng = np.random.default_rng(SEED)

def generate_switching_numbers(length):
    switching_numbers = []
    for i in range(length):
        if i % 2 == 0:
            switching_numbers.append(1)
        else:
            switching_numbers.append(-1)
    return switching_numbers

def get_env(config_dict, localization_config_dict, dart_env_dict, reward_dict, goal_dict, log_dir, rng=None, num_envs=None):
    """
        Set-up the env according to the input dictionary settings
    """

    env_key = config_dict['env_key']
    def create_env(id=0):

        #################################################################################################################################
        # Important: 'dart' substring should always be included in the 'env_key' for dart-based envs. E.g. 'iiwa_sample_dart_unity_env' #
        # If 'dart' is not included, IK behaviour can not be used                                                                       #
        #################################################################################################################################

        # joints control without dart
        if env_key == 'iiwa_joint_vel':
            env = IiwaJointVelEnv(max_ts=dart_env_dict['max_time_step'], id=id, config=config_dict)

        # Reaching the red target sample env
        # task-space with dart or joint space control
        # model-based control with P-controller available
        elif env_key == 'iiwa_sample_dart_unity_env':
            env = IiwaSampleEnv(localization_config_dict=localization_config_dict, max_ts=dart_env_dict['max_time_step'], orientation_control=dart_env_dict['orientation_control'],
                                use_ik=dart_env_dict['use_inverse_kinematics'], ik_by_sns=dart_env_dict['linear_motion_conservation'],
                                state_type=config_dict['state'], enable_render=dart_env_dict['enable_dart_viewer'],
                                task_monitor=dart_env_dict['task_monitor'], with_objects=dart_env_dict['with_objects'],
                                target_mode=dart_env_dict['target_mode'], target_path=dart_env_dict['target_path'],
                                goal_type="target", joints_safety_limit=config_dict['joints_safety_limit'],
                                max_joint_vel=config_dict['max_joint_vel'], max_ee_cart_vel=config_dict['max_ee_cart_vel'],
                                max_ee_cart_acc=config_dict['max_ee_cart_acc'], max_ee_rot_vel=config_dict['max_ee_rot_vel'],
                                max_ee_rot_acc=config_dict['max_ee_rot_acc'], random_initial_joint_positions=config_dict['random_initial_joint_positions'],
                                initial_positions=config_dict['initial_positions'], robotic_tool=config_dict["robotic_tool"], box_dim=goal_dict["box_dim"],
                                start_side=START_SIDES[id], allow_variable_horizon=config_dict['allow_variable_horizon'], allow_dead_zone=config_dict['allow_dead_zone'],
                                min_pos_distance=config_dict['min_pos_distance'], 
                                camera_position=config_dict['camera_position'],
                                camera_rotation=config_dict['camera_rotation'],
                                rng=rng, env_id=id)

        # Set env seed #
        env.seed((id * 150) + (id + 11))

        return env

    if num_envs is None:
        num_envs = config_dict['num_envs']
    env = [create_env for i in range(num_envs)]
    env = SimulatorVecEnv(env, config_dict, manual_actions_dict=None, reward_dict=reward_dict) # Set vectorized env
    env = VecMonitor(env, log_dir, info_keywords=("success",))                                 # Monitor envs

    return env

def save(trainer: common.AdversarialTrainer, save_path: Path):
    """Save discriminator and generator."""
    # We implement this here and not in Trainer since we do not want to actually
    # serialize the whole Trainer (including e.g. expert demonstrations).
    save_path.mkdir(parents=True, exist_ok=True)
    th.save(trainer.reward_train, save_path / "reward_train.pt")
    th.save(trainer.reward_test, save_path / "reward_test.pt")
    policies_serialize.save_stable_model(
        save_path / "gen_policy",
        trainer.gen_algo,
    )

def eval_policy_td3_bc(policy, eval_env, mean, std, eval_episodes=10, render=False):

    states = eval_env.reset()
    episode_rewards = []
    for _ in range(eval_episodes):
        cum_reward = 0

        while True:
            states = (np.array(states).reshape(1,-1) - mean)/std
            action = policy.select_action(states)

            states, rewards, dones, info = eval_env.step([action])
            cum_reward += rewards

            # Render #
            if render:
                env.render()

            if dones.any():
                episode_rewards.append(cum_reward)
                break

    avg_reward = np.mean(episode_rewards)
    # d4rl_score = eval_env.get_normalized_score(avg_reward) * 100

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    
    return avg_reward

if __name__ == "__main__":

    main_config = Config()

    # Parse configs #
    config_dict = main_config.get_config_dict()
    dart_env_dict = main_config.get_dart_env_dict()
    reward_dict = main_config.get_reward_dict()
    goal_dict = main_config.get_goal_dict()
    model_config_dict = main_config.get_model_config_dict()
    localization_config_dict = main_config.get_localization_config()

    # Create new folder if not exists for logging #
    Path(config_dict["log_dir"]).mkdir(parents=True, exist_ok=True)

    # Build env #
    # START_SIDES = generate_switching_numbers(config_dict['num_envs'])
    START_SIDES = [1, 1, 1, 1, 1]
    
    # Train the agent #
    if('train' in config_dict['simulation_mode']):
        
        data_dir = config_dict["data_dir"]
        print(f"Get trajectories data from: {data_dir}")
        trajectories = serialize.load(data_dir)
        trajectories = [dataclasses.replace(traj, infos=None) for traj in trajectories]

        print(
               f"""The `rollout` function generated a list of {len(trajectories)} {type(trajectories[0])}.
            """
            )
        
        if config_dict['simulation_mode'] == "train_bc":
            print("===================================================")
            print("Train Behaviour Clonning")
            print("===================================================")
            env = get_env(config_dict, localization_config_dict, dart_env_dict, reward_dict, goal_dict, config_dict["log_dir"], rng=rng)

            # Prepocessing transitions:
            transitions = rollout.flatten_trajectories(trajectories)
            # transitions = rollout.shuffle_transitions(transitions, rng)

            # Uncomment if you want to
            print(
               f"""After flattening, this list is turned into a {type(transitions)} object containing {len(transitions)} transitions.
            The transitions object contains arrays for: {', '.join(transitions.__dict__.keys())}."
            """
            )

            #set logging dir
            now = datetime.datetime.now()
            timestamp = now.strftime("bc-%Y-%m-%d-%H-%M-%S-%f")
            folder = os.path.join(config_dict["log_dir"], timestamp)
            custom_logger = imit_logger.configure(folder=folder, 
                                                  format_strs= ["stdout", "log", "csv", "tensorboard"])

            batch_size = 32
            l2_weight = 3e-5  # L2 regularization weight
            optimizer_cls = th.optim.Adam
            optimizer_kwargs = dict(
                lr=4e-4,
            )
            
            # Initialize BC:
            policy = None
            if config_dict.get("from_pretrained") != "":
                print("Load from pretrained: ", config_dict.get("from_pretrained"))
                policy = bc.reconstruct_policy(os.path.join(config_dict.get("from_pretrained"), "saved_policy"), device='cuda')

            bc_trainer = bc.BC(
                observation_space=env.observation_space,
                action_space=env.action_space,
                demonstrations=transitions,
                # batch_size=batch_size,
                # minibatch_size=minibatch_size,
                optimizer_cls=optimizer_cls,
                optimizer_kwargs=optimizer_kwargs,
                # l2_weight=l2_weight,
                policy=policy,
                rng=rng,
                custom_logger=custom_logger
            )

            policy_folder = os.path.join(folder, "bc_policy")
            if os.path.isdir(policy_folder):
                shutil.rmtree(policy_folder)

            save_interval = 100

            def callback() -> None:
                global epoch_
                if os.path.isdir(policy_folder):
                    epoch_ = epoch_ + 1
                    checkpoints_path = os.path.join(policy_folder, "checkpoints", f"epoch_{epoch_}")
                else:
                    epoch_ = 0
                    Path(os.path.join(policy_folder, "checkpoints")).mkdir(parents=True, exist_ok=True)
                    checkpoints_path = os.path.join(policy_folder, "checkpoints", "epoch_0")
                
                if epoch_ % save_interval == 0:
                    bc_trainer.save_policy(checkpoints_path)
            
            # Train BC model for epochs_learning
            bc_trainer.train(n_epochs=model_config_dict["num_iters"],
                                on_epoch_end=callback,
                                reset_tensorboard=True,
                                progress_bar=True,
                                log_rollouts_venv=env,
                                log_rollouts_n_episodes=5,
                                log_interval=100000)

            # Stores the trained BC policy in path specified in settings
            policy_path = os.path.join(folder, "bc_policy", "saved_policy")
            bc_trainer.save_policy(policy_path)
            
        elif config_dict['simulation_mode'] == "train_ppo":
            env = get_env(config_dict, localization_config_dict, dart_env_dict, reward_dict, goal_dict, config_dict["log_dir"], rng=rng)

            timesteps_RL = 4000
            
            # Loads the pre-trained policy from the path specified
            policy_path = os.path.join(config_dict["log_dir"], "bc_policy", "saved_policy")

            bc_trainer = bc.BC(
                observation_space=env.observation_space,
                action_space=env.action_space,
                device='cuda',
                policy=bc.reconstruct_policy(policy_path, device='cuda'),
                rng=rng,
            )

            # Structure of PPO RL and initializing PPO
            policy_kwargs = {"net_arch": dict(pi=[32, 32], vf=[32, 32])}
            ppo_model = PPO(policy="MlpPolicy", env=env, policy_kwargs=policy_kwargs, verbose=1, device = 'cuda', n_steps=4001)
            
            # Loading the BC policy into PPO model
            ppo_model.policy.load_state_dict(bc_trainer.policy.state_dict())
            ppo_model.learn(total_timesteps=timesteps_RL) 
            # print("trained") # Uncomment for milestone check
            ppo_model.save(os.path.join(config_dict["log_dir"], "ppo", "ppo_trained"))
        
        elif config_dict['simulation_mode'] == "train_td3_bc":
            env = get_env(config_dict, localization_config_dict, dart_env_dict, reward_dict, goal_dict, config_dict["log_dir"], rng=rng)


            #set logging dir
            now = datetime.datetime.now()
            timestamp = now.strftime("td3_bc-%Y-%m-%d-%H-%M-%S-%f")
            folder = os.path.join(config_dict["log_dir"], timestamp)

            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.shape[0] 
            max_action = float(env.action_space.high[0])
            load_model = False
            normalize = True
            batch_size = 256
            eval_freq = 10000
            save_model = True
            max_timesteps = 1e6

            kwargs = {
                "state_dim": state_dim,
                "action_dim": action_dim,
                "max_action": max_action,
                "discount": 0.99,
                "tau": 0.005,
                # TD3
                "policy_noise": 0.2* max_action,
                "noise_clip": 0.5 * max_action,
                "policy_freq": 2,
                # TD3 + BC
                "alpha": 2.5
            }

            # Initialize policy
            policy = TD3_BC.TD3_BC(**kwargs)

            if load_model:
                policy_file = os.path.join(config_dict["log_dir"], "td3_bc", "saved_policy")
                policy.load(policy_file)

            #load data
            transitions = rollout.flatten_trajectories_with_rew(trajectories)

            replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
            replay_buffer.convert_datasets(transitions)
            if normalize:
                mean,std = replay_buffer.normalize_states() 
            else:
                mean,std = 0,1

            evaluations = []
            pbar = tqdm(total=int(max_timesteps))
            for t in range(int(max_timesteps)):
                policy.train(replay_buffer, batch_size)
                # Evaluate episode
                if (t + 1) % eval_freq == 0:
                    # evaluations.append(eval_policy_td3_bc(policy, env, mean, std))
                    # np.save(os.path.join(folder, "results"), evaluations)

                    if save_model: 
                        if not os.path.isdir(folder):
                            Path(folder).mkdir(parents=True, exist_ok=True)
                        policy.save(os.path.join(folder, "saved_policy"))

                pbar.update(1)
            pbar.close()

            np.save(os.path.join(folder, "mean"), mean)
            np.save(os.path.join(folder, "std"), std)

        elif config_dict['simulation_mode'] == "train_gail":
            print("===================================================")
            print("Train GAIL")
            print("===================================================")
            env = get_env(config_dict, localization_config_dict, dart_env_dict, reward_dict, goal_dict, config_dict["log_dir"], rng=rng)

            checkpoint_interval = 50
            max_timesteps = 800000

            gen_algo = PPO(
                env=env,
                policy=MlpPolicy,
                batch_size=64,
                ent_coef=0.0,
                learning_rate=0.0004,
                gamma=0.95,
                n_epochs=5,
                seed=SEED,
                verbose=1
            )

            reward_net = BasicRewardNet(
                observation_space=env.observation_space,
                action_space=env.action_space,
                normalize_input_layer=RunningNorm,
            )

            #set logging dir
            now = datetime.datetime.now()
            timestamp = now.strftime("gail-%Y-%m-%d-%H-%M-%S-%f")
            log_dir = os.path.join(config_dict["log_dir"], timestamp)

            custom_logger = imit_logger.configure(folder=log_dir,
                                                  format_strs= ["stdout", "log", "csv", "tensorboard"])

            gail_trainer = GAIL(
                demonstrations=trajectories,
                demo_batch_size=1024,
                gen_replay_buffer_capacity=512,
                n_disc_updates_per_round=8,
                venv=env,
                gen_algo=gen_algo,
                log_dir=log_dir,
                custom_logger=custom_logger,
                reward_net=reward_net,
                # allow_variable_horizon=True,
                init_tensorboard=True,
                init_tensorboard_graph=True
            )

            def callback(round_num: int, /) -> None:
                if checkpoint_interval > 0 and round_num % checkpoint_interval == 0:
                    save(gail_trainer, Path(os.path.join(log_dir, "gail_learner", "checkpoints",  f"round_num_{round_num:05d}")) )

            print("Training model...")
            # train the learner and evaluate again
            gail_trainer.train(max_timesteps, callback)

            save(gail_trainer, Path(os.path.join(log_dir, "gail_learner", "finals")) )
        
        elif config_dict['simulation_mode'] == "train_dagger":
            print("===================================================")
            print("Train DAgger")
            print("===================================================")
            env = get_env(config_dict, localization_config_dict, dart_env_dict, reward_dict, goal_dict, config_dict["log_dir"])

            #set logging dir
            now = datetime.datetime.now()
            timestamp = now.strftime("%Y-%m-%d-%H-%M-%S-%f")
            dagger_folder = os.path.join(config_dict["log_dir"], f"bc-dagger-{timestamp}")
            
            custom_logger_dagger = imit_logger.configure(folder=dagger_folder, 
                                                  format_strs= ["stdout", "log", "csv", "tensorboard"])
            
            expert_policy = PushingPolicy(env)
            
            batch_size = 32
            l2_weight = 3e-5  # L2 regularization weight
            optimizer_cls = th.optim.Adam
            optimizer_kwargs = dict(
                lr=4e-4,
            )

            bc_trainer = bc.BC(
                observation_space=env.observation_space,
                action_space=env.action_space,
                # batch_size=batch_size,
                # minibatch_size=minibatch_size,
                optimizer_cls=optimizer_cls,
                optimizer_kwargs=optimizer_kwargs,
                # l2_weight=l2_weight,
                rng=rng,
                custom_logger=custom_logger_dagger
            )
            
            scratch_dir = os.path.join(dagger_folder, "dagger_learner")
            if os.path.isdir(scratch_dir):
                shutil.rmtree(scratch_dir)
            Path(scratch_dir).mkdir(parents=True, exist_ok=True)
            
            dagger_trainer = SimpleDAggerTrainer(
                venv=env,
                scratch_dir=scratch_dir,
                # expert_trajs=trajectories,
                expert_policy=expert_policy,
                bc_trainer=bc_trainer,
                custom_logger=custom_logger_dagger,
                rng=rng,
                dagger_trainer_kwargs = {"beta_schedule" : LinearBetaSchedule(100)}
            )

            n_epochs = 50
            epoch_ = 0

            def callback() -> None:
                global epoch_
                epoch_ = epoch_ + 1
                
                if epoch_ % n_epochs == 0:
                    dagger_trainer.save_trainer()

            bc_train_kwargs = {"n_epochs" : n_epochs,
                               "on_epoch_end" : callback,
                               "log_interval" : 100000,
                               "progress_bar" : True}
            logging.getLogger().setLevel(logging.INFO)

            dagger_trainer.train(total_timesteps=32e+4, 
                                 rollout_round_min_timesteps=dart_env_dict["max_time_step"], 
                                 rollout_round_min_episodes=20,
                                 bc_train_kwargs=bc_train_kwargs)

            save_locations = dagger_trainer.save_trainer()
            print(f"Model saved to {save_locations}")
        else:
            raise NotImplementedError

    elif('evaluate' in config_dict['simulation_mode']):
        env = get_env(config_dict, localization_config_dict, dart_env_dict, reward_dict, goal_dict, config_dict["log_dir"], rng=rng)

        logging.getLogger().setLevel(logging.INFO)
        if config_dict['simulation_mode'] == "evaluate_expert":
            print("===================================================")
            print("Expert-Policy evaluation")
            print("===================================================")
            
            policy = PushingPolicy(env)

            traj, stats = rollout.rollout(policy, 
                                    env,
                                    rollout.make_sample_until(min_timesteps=None, min_episodes=config_dict["num_episode"]), 
                                    rng=rng,
                                    unwrap=False,
                                    exclude_infos=True,
                                    verbose=True,
                                    success_episode_only=False,
                                    return_stats=True)
            
            pd.DataFrame([stats]).T.to_csv(os.path.join(config_dict["log_dir"], "expert_stats.csv"))
        elif config_dict['simulation_mode'] == "evaluate_bc":
            print("===================================================")
            print("BC-Policy evaluation")
            print("===================================================")
            
            # Loads the pre-trained policy from the path specified
            policy_path = os.path.join(config_dict["log_bc_dir"], "saved_policy")
            if not os.path.exists(policy_path):
                raise Exception(f"""Can't find BC policy in path: {policy_path}, 
                                please download the model from the link provided in {os.path.join(config_dict["log_dir"], "README.md")}""")
            model = bc.reconstruct_policy(policy_path)

            logging.getLogger().setLevel(logging.INFO)
            traj, stats = rollout.rollout(model, 
                                    env,
                                    rollout.make_sample_until(min_timesteps=None, min_episodes=config_dict["num_episode"]), 
                                    rng=rng,
                                    unwrap=False,
                                    exclude_infos=True,
                                    verbose=True,
                                    success_episode_only=False,
                                    return_stats=True)
            
            pd.DataFrame([stats]).T.to_csv(os.path.join(config_dict["log_bc_dir"], "stats.csv"))

        elif config_dict['simulation_mode'] == "evaluate_td3_bc":
            print("===================================================")
            print("TD3-BC evaluation")
            print("===================================================")
            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.shape[0] 
            max_action = float(env.action_space.high[0])
            load_model = False
            normalize = True
            batch_size = 256
            eval_freq = 5000
            save_model = True
            max_timesteps = 1e6
            
            kwargs = {
                "state_dim": state_dim,
                "action_dim": action_dim,
                "max_action": max_action,
                "discount": 0.99,
                "tau": 0.005,
                # TD3
                "policy_noise": 0.2* max_action,
                "noise_clip": 0.5 * max_action,
                "policy_freq": 2,
                # TD3 + BC
                "alpha": 2.5
            }
            # Initialize policy
            policy = TD3_BC.TD3_BC(**kwargs)

            mean = np.load(os.path.join(config_dict["log_td3_bc"], "mean.npy"))
            std = np.load(os.path.join(config_dict["log_td3_bc"], "std.npy"))

            policy_file = os.path.join(config_dict["log_td3_bc"], "saved_policy")
            policy.load(policy_file)

            eval_policy_td3_bc(policy, env, mean, std, render=True)

        elif config_dict['simulation_mode'] == "evaluate_ppo":
            print("===================================================")
            print("PPO evaluation")
            print("===================================================")
            model = PPO.load(os.path.join(config_dict["log_dir"], "ppo", "ppo_trained"))

            traj, stats = rollout.rollout(model, 
                                    env,
                                    rollout.make_sample_until(min_timesteps=None, min_episodes=100), 
                                    rng=rng,
                                    unwrap=False,
                                    exclude_infos=True,
                                    verbose=True,
                                    success_episode_only=False,
                                    return_stats=True)
            
        elif config_dict['simulation_mode'] == "evaluate_gail":
            print("===================================================")
            print("GAIL-based evaluation")
            print("===================================================")
            # Load trained agent #
            path = os.path.join(config_dict["log_dir"], "gail_learner", "finals", "gen_policy")
            model = policies_serialize.load_stable_baselines_model(PPO, path, env)

            traj, stats = rollout.rollout(model, 
                                    env,
                                    rollout.make_sample_until(min_timesteps=None, min_episodes=100), 
                                    rng=rng,
                                    unwrap=False,
                                    exclude_infos=True,
                                    verbose=True,
                                    success_episode_only=False,
                                    return_stats=True)

        elif config_dict['simulation_mode'] == "evaluate_dagger":
            print("===================================================")
            print("DAgger evaluation")
            print("===================================================")

            # Loads the pre-trained policy from the path specified
            policy_path = os.path.join(config_dict["log_dagger_bc_dir"], "dagger_learner", "policy-latest.pt")
            model = bc.reconstruct_policy(policy_path)

            logging.getLogger().setLevel(logging.INFO)
            traj, stats = rollout.rollout(model, 
                                    env,
                                    rollout.make_sample_until(min_timesteps=None, min_episodes=100), 
                                    rng=rng,
                                    unwrap=False,
                                    exclude_infos=True,
                                    verbose=True,
                                    success_episode_only=False,
                                    return_stats=True)
            
            pd.DataFrame([stats]).T.to_csv(os.path.join(config_dict["log_bc_dir"], "stats.csv"))

    elif(config_dict['simulation_mode'] == 'expert'):
        print("===================================================")
        print("Expert demonstration")
        print("===================================================")
        env = get_env(config_dict, localization_config_dict, dart_env_dict, reward_dict, goal_dict, config_dict["log_dir"], rng=rng)

        expert_policy = PushingPolicy(env)

        logging.getLogger().setLevel(logging.INFO)
        trajectories = rollout.rollout(
            expert_policy,
            env,
            rollout.make_sample_until(min_timesteps=None, min_episodes=config_dict["num_episode"]),
            rng=rng,
            unwrap=False,
            exclude_infos=True,
            verbose=True,
            success_episode_only=config_dict["success_episode_only"]
        )
        
        print(
               f"""The `rollout` function generated a list of {len(trajectories)}.
            """
            )
        
        serialize.save(config_dict["data_dir"], trajectories)
    else:
        print("You have set an invalid simulation_mode or some other settings in the config.py are wrong - aborting")
