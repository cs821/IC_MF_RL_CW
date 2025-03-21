import os
import torch
import random
import numpy as np
from envs import TradingEnv
from ddpg import DDPG
from q_learning import QLearning
from config import Config

# load hyperparameters
config = Config()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False  

set_seed(config.sac_seed)
# Initialize testing environment (the same as the training environment)
env = TradingEnv(continuous_action_flag=True, sabr_flag=False,spread=0.01, num_contract=1, init_ttm=20, trade_freq=1, num_sim=10000)
env.seed(config.sac_seed)

if config.algo == "ddpg":
    agent = DDPG(env.observation_space.shape[0], env.action_space.shape[0], env.action_space.high[0], config.ddpg_buffer_size,config.ddpg_tau, config.ddpg_alpha)
else:
    raise ValueError("Invalid algorithm. Choose 'ddpg'.")


# change path to what you want
#model_path = "./ddpg_experiments/br/ddpg_freq_1_2025-03-18_16-16-34/ddpg_actor.pth" 
model_path = "./ddpg_experiments/sabr/ddpg_2025-03-17_16-10-02/checkpoints/ddpg_actor_ep49000.pth" #49
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# check and load model
if not os.path.exists(model_path):
    raise FileNotFoundError(f"No model file '{model_path}', please train model first!")

agent.actor.load_state_dict(torch.load(model_path, map_location=device))

# How to hedge?
delta_action_test = False  # Whether to use Delta hedging strategy
bartlett_action_test = False  # Whether to use Bartlett hedging strategy
num_test_episodes = 1000  

# results
w_T_store = []
cost_ratio = []

print("\n\n*** Testing Begin! ***")
if delta_action_test:
    print("Testing the Delta hedging strategy...")
elif bartlett_action_test:
    print("Testing the Bartlett hedging strategy...")
else:
    print(f"Testing {config.algo.upper()} hedging strategy...")

for episode in range(num_test_episodes):
    obs = env.reset()
    done = False
    reward_store = []
    action_store = []
    y_action = [] 
    

    while not done:
        if delta_action_test:
            action = env.delta_path[episode % env.num_path, env.t] * env.num_contract * 100
        elif bartlett_action_test:
            action = env.bartlett_delta_path[episode % env.num_path, env.t] * env.num_contract * 100
        else:
            action = agent.act(obs, eval_mode=True) 
        obs, reward, done, info = env.step(action)
        reward_store.append(reward)
        action_store.append(action)
        y_action.append(action) 

    # print results
    path_row = info["path_row"]
    w_T = sum(reward_store)
    w_T_store.append(w_T)
    option_price = env.option_price_path[path_row, 0] * 100
    cost_ratio.append(-w_T/option_price)

    if episode % 1000 == 0:
        w_T_mean = np.mean(w_T_store)
        w_T_var = np.var(w_T_store)

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


# calculate final results
w_T_mean = np.mean(w_T_store)
w_T_std = np.std(w_T_store) 
y_0 = -w_T_mean + config.ra_c * w_T_std  
cost_ratio_mean = np.mean(cost_ratio)
cost_ratio_std = np.std(cost_ratio)

print(f"\n*** Testing Completed! ***")
print(f"Final average cost: {-w_T_mean:.2f}, std: {w_T_std:.2f}")
print(f"Final average cost/option price: {cost_ratio_mean:.2f}, std: {cost_ratio_std:.2f}")
print(f"Y(0): {y_0:.2f}")
