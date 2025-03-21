# sac_trading.py

import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.buffers import ReplayBuffer
import torch.optim as optim
from config import Config
from envs import TradingEnv
from experimentmanager import ExperimentManager
import wandb 

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False  

set_seed(42)

class SoftQNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

LOG_STD_MAX = 5
LOG_STD_MIN = -2

class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, action_scale: float):
        super().__init__()
        self.action_scale = action_scale
        self.fc1 = nn.Linear(state_dim, 256)
        self.ln = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, action_dim)
        self.fc_logstd = nn.Linear(256, action_dim)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0.0)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.fc1(x))
        x = self.ln(x)
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        #print(f"fc_mean:{mean.mean()}")
        return mean, log_std

    def get_action(self, x: torch.Tensor):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.sigmoid(x_t)  # [0, 1]
        #print(f"mean:{mean}")
        #print(f"y_t:{y_t}")
        # Zoom to the trading environment action space [0, action_scale]
        action = y_t * self.action_scale
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) / 2 + 1e-6)
        return action, log_prob.sum(-1, keepdim=True), mean

class SAC:
    def __init__(self, config: Config, env: TradingEnv):
        self.cfg = config
        self.env = env
        
        # initialization
        state_dim = 3  # price, position, ttm
        action_dim = self.env.action_space.shape[0]
        action_scale = self.env.action_space.high[0]
        
        self.actor = Actor(state_dim, action_dim, action_scale).to(config.device)
        
        self.qf1 = SoftQNetwork(state_dim, action_dim).to(config.device)
        self.qf2 = SoftQNetwork(state_dim, action_dim).to(config.device)
        self.qf1_target = SoftQNetwork(state_dim, action_dim).to(config.device)
        self.qf2_target = SoftQNetwork(state_dim, action_dim).to(config.device)
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())
        
        # Optimizer: Updates all critic network parameters
        self.q_optimizer = optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=config.sac_learning_rate)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.sac_learning_rate)

        # Automatic entropy regulation
        if config.sac_autotune:
            self.target_entropy = -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=config.device)
            self.alpha = self.log_alpha.exp().item()
            self.a_optimizer = optim.Adam([self.log_alpha], lr=config.sac_learning_rate)
        else:
            self.alpha = config.sac_alpha
        
        self.rb = ReplayBuffer(
            config.sac_buffer_size,
            env.observation_space,
            env.action_space,
            config.device,
            handle_timeout_termination=False,
        )
        
    def train(self,exp_manager):
        set_seed(self.cfg.sac_seed)
        history = {"episode": [], "episode_w_T": [], "q_loss": []}
        w_T_store = []

        wandb.init(project="SAC_hedging", config=self.cfg.__dict__,name = "test")
        ## wandb.watch(self.actor) 

        writer = SummaryWriter(f"runs/{self.env.__class__.__name__}_{time.time()}")
        obs = self.env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        reward_store = []
        action_store = []
        y_action = []
        loss = 0.0  
        episode = 0
    
        for global_step in range(self.cfg.sac_total_timesteps):
            set_seed(self.cfg.sac_seed)
            # choose action
            if global_step < self.cfg.sac_batch_size:
                action = self.env.action_space.sample().item()
            else:
                with torch.no_grad():
                    action_tensor, _, _ = self.actor.get_action(torch.Tensor(obs).to(self.cfg.device))
                    action = action_tensor.cpu().numpy().item()
            
            # Environment interaction
            next_obs, reward, done, info = self.env.step(action)
            self.rb.add(obs, next_obs, action, reward, done, info)
            
            # record reward & action
            reward_store.append(reward)
            action_store.append(action)
            y_action.append(action)

            # update observations
            obs = next_obs
            episode_reward += reward
            episode_length += 1
            
            # training logic
            if global_step >= self.cfg.sac_batch_size:
                set_seed(self.cfg.sac_seed)
                data = self.rb.sample(self.cfg.sac_batch_size)
                
                # ----------------------- Critic update -----------------------
                with torch.no_grad():
                    next_actions, next_log_pi, _ = self.actor.get_action(data.next_observations)
                    qf1_next_target = self.qf1_target(data.next_observations, next_actions)
                    qf2_next_target = self.qf2_target(data.next_observations, next_actions)
                    min_qf1_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_log_pi
                    next_q_ex = data.rewards.flatten() + (1 - data.dones.flatten()) * self.cfg.sac_gamma * min_qf1_next_target.view(-1)

                qf1_a_values = self.qf1(data.observations, data.actions).view(-1)
                qf2_a_values = self.qf2(data.observations, data.actions).view(-1)
                qf_loss = F.mse_loss(qf1_a_values, next_q_ex) + F.mse_loss(qf2_a_values, next_q_ex)

                loss = qf_loss.item()  # record loss
                wandb.log({"q_loss": loss,"env_steps": global_step,})

                self.q_optimizer.zero_grad()
                qf_loss.backward()
                self.q_optimizer.step()

                # ----------------------- Actor update -----------------------
                if global_step % self.cfg.sac_policy_frequency == 0:
                    set_seed(self.cfg.sac_seed)
                    for _ in range(self.cfg.sac_policy_frequency):
                        set_seed(self.cfg.sac_seed)
                        pi, log_pi, _ = self.actor.get_action(data.observations)
                        qf1_pi = self.qf1(data.observations, pi)
                        qf2_pi = self.qf2(data.observations, pi)
                        min_qf_pi = torch.min(qf1_pi, qf2_pi)
                        actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()
                        wandb.log({"actor_loss": actor_loss,"env_steps": global_step,})
                        
                        self.actor_optimizer.zero_grad()
                        actor_loss.backward()
                        self.actor_optimizer.step()
                        
                        if self.cfg.sac_autotune:
                            set_seed(self.cfg.sac_seed)
                            with torch.no_grad():
                                _, log_pi, _ = self.actor.get_action(data.observations)
                            alpha_loss = (-self.log_alpha.exp() * (log_pi + self.target_entropy)).mean()
                            self.a_optimizer.zero_grad()
                            alpha_loss.backward()
                            self.a_optimizer.step()
                            self.alpha = self.log_alpha.exp().item()
                
                # ----------------------- Target network update -----------------------
                if global_step % self.cfg.sac_target_network_frequency == 0:
                    set_seed(self.cfg.sac_seed)
                    for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
                        set_seed(self.cfg.sac_seed)
                        target_param.data.copy_(self.cfg.sac_tau * param.data + (1 - self.cfg.sac_tau) * target_param.data)
                    for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
                        set_seed(self.cfg.sac_seed)
                        target_param.data.copy_(self.cfg.sac_tau * param.data + (1 - self.cfg.sac_tau) * target_param.data)

            if done:
                set_seed(self.cfg.sac_seed)
                writer.add_scalar("charts/episodic_return", episode_reward, global_step)
                writer.add_scalar("charts/episodic_length", episode_length, global_step)

                # calculate final wealth w_T (negative reward)
                w_T_store.append(episode_reward)

                # record training history
                history["episode"].append(global_step)
                history["episode_w_T"].append(episode_reward)
                history["q_loss"].append(loss)
                #history["actor_loss"].append(actor_loss)

                if global_step % 1 == 0:
                    wandb.log({"Episode_Return": episode_reward,"env_steps": global_step,})
                    path_row = info["path_row"]
                    print(info)
                    print(f"global_step: {global_step} | episode final wealth: {episode_reward:.3f} | q_loss: {loss:.3f}")
                    with np.printoptions(precision=2, suppress=True):
                        print(f"global_step: {global_step} | rewards {np.array(reward_store)}")
                        print(f"global_step: {global_step} | actions taken {np.array(y_action)}")
                        print(f"global_step: {global_step} | deltas {self.env.delta_path[path_row] * 100}")
                        print(f"global_step: {global_step} | stock price {self.env.path[path_row]}")
                        print(f"global_step: {global_step} | option price {self.env.option_price_path[path_row] * 100}\n")

                obs = self.env.reset()
                episode_reward = 0
                episode_length = 0
                reward_store = []
                action_store = []
                y_action = []
                done = False
                episode += 1
            
            # save model every 1000 global_step
            if global_step % 1000 == 0:
                exp_manager.save_checkpoint({"actor": self.actor.state_dict(),
                                            "qf1": self.qf1.state_dict(),
                                            "qf2": self.qf2.state_dict(),
                                            "log_alpha": self.log_alpha if self.cfg.sac_autotune else None,
                                            }, f"sac_checkpoint.pth", global_step)
                                                    
        # save training history
        exp_manager.save_history(history)
        writer.close()
        wandb.finish()
    
    def test(self, num_episodes=1000,delta_action_test = False,bartlett_action_test = False):
        set_seed(self.cfg.sac_seed)
        """Test the trained strategy"""
        total_rewards = []
        w_T_store = []
        cost_ratio = []

        for global_step in range(num_episodes):
            obs = self.env.reset()
            episode_reward = 0
            done = False
            reward_store = []  
            action_store = []
            y_action = []

            while not done:
                with torch.no_grad():
                    if delta_action_test:
                        action = self.env.delta_path[global_step % self.env.num_path, self.env.t] * self.env.num_contract * 100
                    elif bartlett_action_test:
                        action = self.env.bartlett_delta_path[global_step % self.env.num_path, self.env.t] * self.env.num_contract * 100
                    else:
                        action_tensor, _, _ = self.actor.get_action(torch.Tensor(obs).to(self.cfg.device))
                        action = action_tensor.cpu().numpy()
                obs, reward, done, info = self.env.step(action)
                episode_reward += reward
                reward_store.append(reward)
                action_store.append(action)
                y_action.append(action)

            path_row = info["path_row"]
            w_T = sum(reward_store).item()
            w_T_store.append(w_T)
            option_price = self.env.option_price_path[path_row, 0] * 100
            cost_ratio.append(-w_T/option_price)

            if global_step% 100 == 0:# can adjust to what you want
                w_T_mean = np.mean(w_T_store)
                w_T_var = np.var(w_T_store)
                print(info)
                print(f"episode: {global_step} | episode final wealth: {w_T:.3f} | welath_mean: {w_T_mean:.3f} | wealth_var: {w_T_var:.3f}")
                with np.printoptions(precision=2, suppress=True):
                    print(f"episode: {global_step} | Y(0) {-w_T_mean + self.cfg.ra_c * np.sqrt(w_T_var)}")
                    print(f"episode: {global_step} | rewards {np.array(reward_store).ravel()}")
                    print(f"episode: {global_step} | actions taken {np.array(y_action).ravel()}")
                    print(f"episode: {global_step} | deltas {self.env.delta_path[path_row] * 100}")
                    print(f"episode: {global_step} | stock price {self.env.path[path_row]}")
                    print(f"episode: {global_step} | option price {self.env.option_price_path[path_row] * 100}\n")

            total_rewards.append(w_T)
        
        mean_reward = np.mean(w_T_store)
        std_reward = np.std(w_T_store)
        y_0 = -mean_reward + self.cfg.ra_c * std_reward  
        cost_ratio_mean = np.mean(cost_ratio)
        cost_ratio_std = np.std(cost_ratio)\
        
        print(f"\n*** Testing Completed! ***")
        print(f"Final average cost: {-mean_reward:.2f}, std: {std_reward:.2f}")
        print(f"Final average cost/option price: {cost_ratio_mean:.2f}, std: {cost_ratio_std:.2f}")
        print(f"Y(0): {y_0:.2f}")

    def load(self, path: str):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint["actor"])
        self.qf1.load_state_dict(checkpoint["qf1"])
        self.qf2.load_state_dict(checkpoint["qf2"])
        if self.cfg.sac_autotune and checkpoint["log_alpha"] is not None:
            self.log_alpha.data.copy_(checkpoint["log_alpha"])


if __name__ == "__main__":
    # initialize config and environment
    sacconfig = Config()
    if sacconfig.algo != "sac":
        print("Please change the algorithm in Config!")
        exit()
    set_seed(sacconfig.sac_seed)
    exp_manager = ExperimentManager(sacconfig)
    env = TradingEnv(
        continuous_action_flag=True, sabr_flag=sacconfig.sabr_flag, spread=0.01, num_contract=1, init_ttm=20, trade_freq=1, num_sim=500000
    )
    env.seed(sacconfig.sac_seed)
    # train
    agent = SAC(sacconfig, env)
    print("Training Begin...")
    time1 = time.time()
    agent.train(exp_manager)
    exp_manager.save_model({
        "actor": agent.actor.state_dict(),
        "qf1": agent.qf1.state_dict(),
        "qf2": agent.qf2.state_dict(),
        "log_alpha": agent.log_alpha if agent.cfg.sac_autotune else None,
    }, f"{sacconfig.algo}_final.pth")
    time2 = time.time()
    elapsed_seconds = int(time2 - time1) 
    formatted_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_seconds))
    print(f"Total time cost: {formatted_time}")