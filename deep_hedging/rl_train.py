import torch
import numpy as np
import random
import pandas as pd
from envs import TradingEnv
from ddpg import DDPG
from q_learning import QLearning
from config import Config
from experimentmanager import ExperimentManager

# 读取配置
config = Config()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False 

set_seed(config.ddpg_seed)

# 创建环境
env = TradingEnv(continuous_action_flag=True, sabr_flag=config.sabr_flag, spread=0.01, num_contract=1, init_ttm=20, trade_freq=1, num_sim=500000)
env.seed(config.ddpg_seed)

# 选择算法
if config.algo == "ddpg":
    agent = DDPG(env.observation_space.shape[0], env.action_space.shape[0], env.action_space.high[0], config.ddpg_buffer_size, config.ddpg_gamma, config.ddpg_tau, config.ddpg_alpha)
elif config.algo == "qlearning":
    agent = QLearning(env, config)
else:
    raise ValueError("Invalid algorithm. Choose 'ddpg', 'sac' or 'qlearning'.")

# 训练超参数
global_step = 0
exp_manager = ExperimentManager(config)
history = {"episode": [], "episode_w_T": [], "loss_ex": [], "loss_ex2": []}
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
                loss_ex, loss_ex2 = agent.update(config.ddpg_batch_size, global_step)
            else:
                loss_ex, loss_ex2 = 0.0, 0.0  # 经验池不足时，loss 设为 0

        obs = next_obs
        episode_reward += reward
        reward_store.append(reward)
        action_store.append(action)
        y_action.append(action)  # 记录智能体对冲的动作
        global_step += 1

    # 计算财富
    w_T = sum(reward_store).item()
    w_T_store.append(w_T)

    # 记录训练历史
    history["episode"].append(episode)
    history["episode_w_T"].append(w_T)
    history["loss_ex"].append(loss_ex)
    history["loss_ex2"].append(loss_ex2)

    # **每 1000 轮保存一次***每 1000 轮保存一次**
    if episode % 1000 == 0:
        exp_manager.save_checkpoint(agent.actor.state_dict(), f"{config.algo}_actor", episode)
        exp_manager.save_checkpoint(agent.critic.q_ex_net.state_dict(), f"{config.algo}_critic_q_ex", episode)
        exp_manager.save_checkpoint(agent.critic.q_ex2_net.state_dict(), f"{config.algo}_critic_q_ex2", episode)
        print(f"Checkpoint saved at episode {episode}")
    # **源码输出部分**
    path_row = info["path_row"]
    print(info)
    print(
        "episode: {} | episode final wealth: {:.3f} | loss_ex: {:.3f} | loss_ex2: {:.3f} | epsilon:{:.2f}".format(
            episode, w_T, loss_ex, loss_ex2, getattr(agent, "epsilon", 0)
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

# 保存模型
if config.algo == "ddpg":
    exp_manager.save_model(agent.actor.state_dict(), f"{config.algo}_actor.pth")
    exp_manager.save_model(agent.critic.q_ex_net.state_dict(), f"{config.algo}_critic_q_ex.pth")
    exp_manager.save_model(agent.critic.q_ex2_net.state_dict(), f"{config.algo}_critic_q_ex2.pth")
elif config.algo == "qlearning":
    agent.save("qlearning_table.pkl")
    print("Q-learning 模型已保存！")

print("\n*** 训练完成 ***")