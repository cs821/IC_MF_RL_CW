import argparse
import torch
import numpy as np
import random
import pandas as pd
from envs import TradingEnv
from ddpg import DDPG
from q_learning import QLearning
from config import Config
from experimentmanager import ExperimentManager
import wandb
import time
# load hyperparameters
config = Config()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False 

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=config.ddpg_seed, type=int)
    args = parser.parse_args()
    return args

time1 = time.time()
args = get_args()
config.ddpg_seed = args.seed

set_seed(config.ddpg_seed)

unique_name = f'sabr_{config.sabr_flag}_freq_{config.trade_freq}_test' 
wandb.init(project="deep_hedging", config=config, name = unique_name)
# initialize environment
env = TradingEnv(continuous_action_flag=True, sabr_flag=config.sabr_flag, spread=0.01, num_contract=1, init_ttm=20, trade_freq=config.trade_freq, num_sim=500000)
env.seed(config.ddpg_seed)

# check algorithm 
if config.algo == "ddpg":
    agent = DDPG(env.observation_space.shape[0], env.action_space.shape[0], env.action_space.high[0], 
                 config.ddpg_buffer_size, config.ddpg_tau, config.ddpg_alpha,
                 max_t=config.ddpg_epsilon_decay_steps,device=config.device)
else:
    raise ValueError("Invalid algorithm. Choose 'ddpg' or 'sac'.")

# train parameters
global_step = 0
exp_manager = ExperimentManager(config)
history = {"episode": [], "episode_w_T": []}
w_T_store = []
print("\n\n*** Training Begin! ***")

episode = 0  
while episode < config.ddpg_num_episodes:
    exit_flag = False

    obs = np.array(env.reset(), dtype=np.float32)
    done = False
    episode_reward = 0
    reward_store = []
    action_store = []
    y_action = []
    while not done:
        action = agent.act(obs)  
        next_obs, reward, done, info = env.step(action)
        agent.replay_buffer.add(obs, action, reward, next_obs, done)
        if len(agent.replay_buffer) > config.ddpg_batch_size:
            agent_info  = agent.update(config.ddpg_batch_size, global_step)
            if agent_info is None:
                exit_flag = True
                break
        else:
            agent_info = {}
        obs = next_obs
        episode_reward += reward
        reward_store.append(reward)
        action_store.append(action)
        y_action.append(action) 
        global_step += 1

    if exit_flag == True:
        continue

    episode += 1

    # calculate wealth (negative reward)
    w_T_store.append(episode_reward)

    # track training history
    if episode % 30 == 0:
        if len(agent_info.keys()) > 0:
            history["episode"].append(episode)
            history["episode_w_T"].append(episode_reward)
            wandb.log({"Episode": episode,"env_steps": global_step,})
            wandb.log({"Episode_Return": episode_reward,"env_steps": global_step,})
            for k, v in agent_info.items():
                if k in history.keys():
                    history[k].append(v)
                else:
                    history[k] = [v]
                wandb.log({k: v,"env_steps": global_step,})
        else:   
            wandb.log({"Episode": episode,"env_steps": global_step,})
            wandb.log({"Episode_Return": episode_reward,"env_steps": global_step,})

    # save model every 1000 episodes
    if episode % 1000 == 0:
        exp_manager.save_checkpoint(agent.actor.state_dict(), f"{config.algo}_actor", episode)
        exp_manager.save_checkpoint(agent.critic.q_ex_net.state_dict(), f"{config.algo}_critic_q_ex", episode)
        exp_manager.save_checkpoint(agent.critic.q_ex2_net.state_dict(), f"{config.algo}_critic_q_ex2", episode)
        print(f"Checkpoint saved at episode {episode}")

    if episode % 50 == 0:
        path_row = info["path_row"]
        print(info)
        if len(agent_info.keys()) == 0:
            eps = getattr(agent,"epsilon", 0)
            print(
                f"episode: {episode} | episode final wealth: {episode_reward:.3f} | epsilon:{eps:.2f}"
            )
        else:
            print(
                "episode: {} | episode final wealth: {:.3f} | critic_loss: {:.3f} |actor_grad_norm: {:.3f} | critic_grad_norm: {:.3f} | actor_loss: {:.3f} |  epsilon:{:.2f}".format(
                    episode, episode_reward, agent_info["critic_loss"], agent_info["total_actor_grad_norm"], agent_info["total_critic_grad_norm"], agent_info["actor_loss"], getattr(agent, "epsilon", 0)
                )
            )

        with np.printoptions(precision=2, suppress=True):
            print(f"episode: {episode} | rewards {np.array(reward_store)}")
            print(f"episode: {episode} | actions taken {np.array(y_action)}")
            print(f"episode: {episode} | deltas {env.delta_path[path_row] * 100}")
            print(f"episode: {episode} | stock price {env.path[path_row]}")
            print(f"episode: {episode} | option price {env.option_price_path[path_row] * 100}\n")

    # update epsilon
    if isinstance(agent, DDPG):
        agent.update_epsilon()

# save training history
exp_manager.save_history(history)
exp_manager.save_image(history)

# save model
exp_manager.save_model(agent.actor.state_dict(), f"{config.algo}_actor.pth")
exp_manager.save_model(agent.critic.q_ex_net.state_dict(), f"{config.algo}_critic_q_ex.pth")
exp_manager.save_model(agent.critic.q_ex2_net.state_dict(), f"{config.algo}_critic_q_ex2.pth")
print("DDPG saved!")
    

print("\n*** Traning Completed! ***")
time2 = time.time()
elapsed_seconds = int(time2 - time1)
formatted_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_seconds))
print(f"Total Time cost: {formatted_time}")
wandb.finish()