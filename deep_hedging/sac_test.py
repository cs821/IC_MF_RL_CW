from config import Config
from envs import TradingEnv
from sac_train import SAC
import random
import torch
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False  

if __name__ == "__main__":
    sacconfig = Config()
    if sacconfig.algo != "sac":
        print("Please change the algorithm in Config!")
        exit()
    set_seed(sacconfig.sac_seed)
    delta_action = False
    bartlett_action = True
    num_test_episodes = 10

    # 测试
    if delta_action:
        print("\n开始测试 Delta 对冲策略...")
    elif bartlett_action:
        print("\n开始测试 Bartlett 对冲策略...")
    else:
        print(f"\n开始测试 {sacconfig.algo.upper()} 训练的智能体策略...")
    test_env = TradingEnv(
        continuous_action_flag=True, sabr_flag=sacconfig.sabr_flag, spread=0.01, num_contract=1, init_ttm=20, trade_freq=1, num_sim=500000  # 使用更少路径进行测试
    )
    test_env.seed(sacconfig.sac_seed)
    test_agent = SAC(sacconfig, test_env)
    #输入你想用的模型路径 ↓
    test_agent.load("./sac_experiments/sabr/sac_2025-03-07_03-20-18/sac_final.pth")
    test_agent.test(num_episodes=num_test_episodes,delta_action_test = delta_action,bartlett_action_test = bartlett_action)
