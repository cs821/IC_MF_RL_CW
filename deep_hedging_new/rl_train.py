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
# 读取配置
config = Config()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False 

def get_args():
    # you can use `python rl_train.py --seed <seed> to set seed as <seed>`
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int)
    args=parser.parse_args()
    return args

args = get_args()
config.ddpg_seed = args.seed

set_seed(config.ddpg_seed)

unique_name = 'test' # this is the name for this run in wandb, can be set to whatever you want 
wandb.init(project="deep_hedging", config=config, name = unique_name)
# 创建环境
env = TradingEnv(continuous_action_flag=True, sabr_flag=config.sabr_flag, spread=0.01, num_contract=1, init_ttm=20, trade_freq=1, num_sim=500000)
env.seed(config.ddpg_seed)

# 选择算法
if config.algo == "ddpg":
    agent = DDPG(env.observation_space.shape[0], env.action_space.shape[0], env.action_space.high[0], 
                 config.ddpg_buffer_size, config.ddpg_tau, config.ddpg_alpha,
                 max_t=config.ddpg_epsilon_decay_steps,device=config.device)
elif config.algo == "qlearning":
    agent = QLearning(env, config)
else:
    raise ValueError("Invalid algorithm. Choose 'ddpg', 'sac' or 'qlearning'.")

# 训练超参数
global_step = 0
exp_manager = ExperimentManager(config)
history = {"episode": [], "episode_w_T": []}
w_T_store = []
print("\n\n*** 开始训练 ***")
for episode in range(config.ddpg_num_episodes):
    obs = np.array(env.reset(), dtype=np.float32)
    done = False
    episode_reward = 0
    reward_store = []
    action_store = []
    y_action = []

    while not done:
        action = agent.act(obs)  
        next_obs, reward, done, info = env.step(action)
        if config.algo == "qlearning":
            agent.learn(obs, action, reward, next_obs, done)
        else:
            agent.replay_buffer.add(obs, action, reward, next_obs, done)
            if len(agent.replay_buffer) > config.ddpg_batch_size:
                agent_info  = agent.update(config.ddpg_batch_size, global_step)
            else:
                agent_info = {}

        obs = next_obs
        episode_reward += reward
        reward_store.append(reward)
        action_store.append(action)
        y_action.append(action)  # 记录智能体对冲的动作
        global_step += 1

    # 计算财富
    w_T_store.append(episode_reward)

    # 记录训练历史
    if episode % 1 == 0:
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

    # **每 1000 轮保存一次***每 1000 轮保存一次**
    if episode % 1000 == 0:
        exp_manager.save_checkpoint(agent.actor.state_dict(), f"{config.algo}_actor", episode)
        exp_manager.save_checkpoint(agent.critic.q_ex_net.state_dict(), f"{config.algo}_critic_q_ex", episode)
        exp_manager.save_checkpoint(agent.critic.q_ex2_net.state_dict(), f"{config.algo}_critic_q_ex2", episode)
        print(f"Checkpoint saved at episode {episode}")
    # **源码输出部分**
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

    # Episode结束后更新epsilon
    if isinstance(agent, DDPG):
        agent.update_epsilon()

# 保存训练数据
exp_manager.save_history(history)
exp_manager.save_image(history)

# 保存模型
if config.algo == "ddpg":
    exp_manager.save_model(agent.actor.state_dict(), f"{config.algo}_actor.pth")
    exp_manager.save_model(agent.critic.q_ex_net.state_dict(), f"{config.algo}_critic_q_ex.pth")
    exp_manager.save_model(agent.critic.q_ex2_net.state_dict(), f"{config.algo}_critic_q_ex2.pth")
elif config.algo == "qlearning":
    agent.save("qlearning_table.pkl")
    print("Q-learning 模型已保存！")

print("\n*** 训练完成 ***")
wandb.finish()