import os
import torch
import random
import numpy as np
from envs import TradingEnv
from ddpg import DDPG
from q_learning import QLearning
from config import Config

# 读取配置
config = Config()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False  

set_seed(config.sac_seed)
# 创建测试环境
env = TradingEnv(continuous_action_flag=True, sabr_flag=config.sabr_flag,spread=0.01, num_contract=1, init_ttm=20, trade_freq=1, num_sim=10000)
env.seed(config.sac_seed)

# 选择算法
if config.algo == "ddpg":
    agent = DDPG(env.observation_space.shape[0], env.action_space.shape[0], env.action_space.high[0], config.ddpg_buffer_size, config.ddpg_gamma, config.ddpg_tau, config.ddpg_alpha)
elif config.algo == "qlearning":
    agent = QLearning(env, config)
else:
    raise ValueError("Invalid algorithm. Choose 'ddpg' or 'qlearning'.")


# ⬇这里手动修改路径，填入你要测试的模型 
model_path = "./ddpg_experiments/sabr/ddpg_2025-03-07_16-50-00/ddpg_actor.pth" 

# 检查并加载模型
if not os.path.exists(model_path):
    raise FileNotFoundError(f"模型文件 '{model_path}' 未找到，请先训练模型！")

if config.algo in ["ddpg", "sac"]:
    agent.actor.load_state_dict(torch.load(model_path))
elif config.algo == "qlearning":
    agent.load(model_path)

# 测试选项
delta_action_test = False  # 是否使用 Delta 对冲策略
bartlett_action_test = True  # 是否使用 Bartlett 对冲策略
num_test_episodes = 10  # 设定测试 episode 数

# 统计收益
w_T_store = []

print("\n\n*** 开始测试 ***")
if delta_action_test:
    print("正在测试 Delta 对冲策略...")
elif bartlett_action_test:
    print("正在测试 Bartlett 对冲策略...")
else:
    print(f"正在测试 {config.algo.upper()} 训练的智能体策略...")

for episode in range(num_test_episodes):
    obs = env.reset()
    done = False
    reward_store = []
    action_store = []
    y_action = []  # 存储对冲动作

    while not done:
        if delta_action_test:
            action = env.delta_path[episode % env.num_path, env.t] * env.num_contract * 100
        elif bartlett_action_test:
            action = env.bartlett_delta_path[episode % env.num_path, env.t] * env.num_contract * 100
        else:
            action = agent.act(obs, eval_mode=True)  # 采用模型策略
        obs, reward, done, info = env.step(action)
        reward_store.append(reward)
        action_store.append(action)
        y_action.append(action)  # 记录对冲动作

    # 计算最终收益
    w_T = sum(reward_store)
    w_T_store.append(w_T)

    if episode % 1000 == 0:
        w_T_mean = np.mean(w_T_store)
        w_T_var = np.var(w_T_store)
        path_row = info["path_row"]

        print(info)
        print(
            "episode: {} | episode final wealth: {:.3f} | mean wealth: {:.3f} | variance: {:.3f}".format(
                episode, w_T, w_T_mean, w_T_var
            )
        )

        with np.printoptions(precision=2, suppress=True):
            print(f"episode: {episode} | Y(0) {-w_T_mean + config.ra_c * np.sqrt(w_T_var)}")
            print(f"episode: {episode} | rewards {np.array(reward_store)}")
            print(f"episode: {episode} | actions taken {np.array(y_action)}")
            print(f"episode: {episode} | deltas {env.delta_path[path_row] * 100}")
            print(f"episode: {episode} | stock price {env.path[path_row]}")
            print(f"episode: {episode} | option price {env.option_price_path[path_row] * 100}\n")


# 计算最终统计值
w_T_mean = np.mean(w_T_store)
w_T_std = np.std(w_T_store) 
y_0 = -w_T_mean + config.ra_c * w_T_std  

print(f"\n*** 测试完成 ***")
print(f"最终平均收益: {w_T_mean:.2f}, 标准差: {w_T_std:.2f}")
print(f"优化目标 Y(0): {y_0:.2f}")
