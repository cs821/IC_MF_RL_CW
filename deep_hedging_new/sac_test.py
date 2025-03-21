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
    bartlett_action = False
    num_test_episodes = 1000

    if delta_action:
        print("\nTesting the Delta hedging strategy...")
    elif bartlett_action:
        print("\nTesting the Bartlett hedging strategy...")
    else:
        print(f"\nTesting {sacconfig.algo.upper()} hedging strategy...")
    test_env = TradingEnv(
        continuous_action_flag=True, sabr_flag=sacconfig.sabr_flag, spread=0.01, num_contract=1, init_ttm=20, trade_freq=1, num_sim=500000 
    )
    test_env.seed(sacconfig.sac_seed)
    test_agent = SAC(sacconfig, test_env)
    # change path to what you want
    # test_agent.load("./sac_experiments/br/sac_2025-03-16_21-09-48/checkpoints/sac_checkpoint.pth_ep26000.pth")
    test_agent.load("./sac_experiments/br/sac_2025-03-17_02-17-17/sac_final.pth")
    test_agent.test(num_episodes=num_test_episodes,delta_action_test = delta_action,bartlett_action_test = bartlett_action)
