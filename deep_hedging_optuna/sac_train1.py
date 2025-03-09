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

import optuna
import json

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

LOG_STD_MAX = 2
LOG_STD_MIN = -5

class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, action_scale: float):
        super().__init__()
        self.action_scale = action_scale
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, action_dim)
        self.fc_logstd = nn.Linear(256, action_dim)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def get_action(self, x: torch.Tensor):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)  # [-1, 1]
        # 缩放到交易环境动作空间 [0, action_scale]
        action = (y_t + 1) / 2 * self.action_scale
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) / 2 + 1e-6)
        return action, log_prob.sum(-1, keepdim=True), mean

class SAC:
    def __init__(self, config: Config, env: TradingEnv, exp_manager: ExperimentManager = None):
        self.cfg = config
        self.env = env
        # 若外部未传入 exp_manager，则自行创建
        if exp_manager is None:
            self.exp_manager = ExperimentManager(config)
        else:
            self.exp_manager = exp_manager
        
        # 网络初始化
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
        
        # 优化器
        self.q_optimizer = optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=config.sac_learning_rate)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.sac_learning_rate)
        
        # 自动熵调节
        if config.sac_autotune:
            self.target_entropy = -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=config.device)
            self.alpha = self.log_alpha.exp().item()
            self.a_optimizer = optim.Adam([self.log_alpha], lr=config.sac_learning_rate)
        else:
            self.alpha = config.sac_alpha
        
        # 经验回放
        self.rb = ReplayBuffer(
            config.sac_buffer_size,
            env.observation_space,
            env.action_space,
            config.device,
            handle_timeout_termination=False,
        )
        
    def train(self):
        set_seed(self.cfg.sac_seed)
        history = {"episode": [], "episode_w_T": [], "loss": []}
        w_T_store = []

        writer = SummaryWriter(f"runs/{self.env.__class__.__name__}_{time.time()}")
        obs = self.env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        reward_store = []
        action_store = []
        y_action = []
        loss = 0.0  # 初始化 loss
    
        for global_step in range(self.cfg.sac_total_timesteps):
            # 动作选择
            if global_step < self.cfg.sac_batch_size:
                action = self.env.action_space.sample().item()
            else:
                with torch.no_grad():
                    action_tensor, _, _ = self.actor.get_action(torch.Tensor(obs).to(self.cfg.device))
                    action = action_tensor.cpu().numpy().item()
            
            # 环境交互
            next_obs, reward, done, info = self.env.step(action)
            self.rb.add(obs, next_obs, action, reward, done, info)
            
            # 记录 reward 与 action
            reward_store.append(reward)
            action_store.append(action)
            y_action.append(action)

            # 更新观察
            obs = next_obs
            episode_reward += reward
            episode_length += 1
            
            # 训练逻辑
            if global_step >= self.cfg.sac_batch_size:
                data = self.rb.sample(self.cfg.sac_batch_size)
                
                # Critic 更新
                with torch.no_grad():
                    next_actions, next_log_pi, _ = self.actor.get_action(data.next_observations)
                    qf1_next_target = self.qf1_target(data.next_observations, next_actions)
                    qf2_next_target = self.qf2_target(data.next_observations, next_actions)
                    min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_log_pi
                    next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * self.cfg.sac_gamma * min_qf_next_target.view(-1)
                
                qf1_a_values = self.qf1(data.observations, data.actions).view(-1)
                qf2_a_values = self.qf2(data.observations, data.actions).view(-1)
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                qf_loss = qf1_loss + qf2_loss
                loss = qf_loss.item()  # 记录 loss

                self.q_optimizer.zero_grad()
                qf_loss.backward()
                self.q_optimizer.step()

                # Actor 更新
                if global_step % self.cfg.sac_policy_frequency == 0:
                    for _ in range(self.cfg.sac_policy_frequency):
                        pi, log_pi, _ = self.actor.get_action(data.observations)
                        qf1_pi = self.qf1(data.observations, pi)
                        qf2_pi = self.qf2(data.observations, pi)
                        min_qf_pi = torch.min(qf1_pi, qf2_pi)
                        actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()
                        
                        self.actor_optimizer.zero_grad()
                        actor_loss.backward()
                        self.actor_optimizer.step()
                        
                        if self.cfg.sac_autotune:
                            with torch.no_grad():
                                _, log_pi, _ = self.actor.get_action(data.observations)
                            alpha_loss = (-self.log_alpha.exp() * (log_pi + self.target_entropy)).mean()
                            
                            self.a_optimizer.zero_grad()
                            alpha_loss.backward()
                            self.a_optimizer.step()
                            self.alpha = self.log_alpha.exp().item()
                
                # 目标网络更新
                if global_step % self.cfg.sac_target_network_frequency == 0:
                    for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
                        target_param.data.copy_(self.cfg.sac_tau * param.data + (1 - self.cfg.sac_tau) * target_param.data)
                    for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
                        target_param.data.copy_(self.cfg.sac_tau * param.data + (1 - self.cfg.sac_tau) * target_param.data)
            
            # 日志记录
            writer.add_scalar("charts/episodic_return", episode_reward, global_step)
            writer.add_scalar("charts/episodic_length", episode_length, global_step)

            # 计算最终财富 w_T
            w_T = sum(reward_store).item()
            w_T_store.append(w_T)

            # 记录训练历史
            history["episode"].append(global_step)
            history["episode_w_T"].append(w_T)
            history["loss"].append(loss)

            # 打印输出
            path_row = info["path_row"]
            print(info)
            print(f"episode: {global_step} | episode final wealth: {w_T:.3f} | loss: {loss:.3f} | alpha: {self.alpha:.3f}")
            with np.printoptions(precision=2, suppress=True):
                print(f"episode: {global_step} | rewards {np.array(reward_store)}")
                print(f"episode: {global_step} | actions taken {np.array(y_action)}")
                print(f"episode: {global_step} | deltas {self.env.delta_path[path_row] * 100}")
                print(f"episode: {global_step} | stock price {self.env.path[path_row]}")
                print(f"episode: {global_step} | option price {self.env.option_price_path[path_row] * 100}\n")

            # 重置环境及记录变量
            obs = self.env.reset()
            episode_reward = 0
            episode_length = 0
            reward_store = []
            action_store = []
            y_action = []
            
            # 定期保存模型（这里保存的是当前训练过程中的 checkpoint，仅供调试参考）
            if global_step % 1000 == 0:
                self.exp_manager.save_checkpoint(self, f"sac_checkpoint.pth", global_step)
        
        # 保存训练历史
        self.exp_manager.save_history(history)
        writer.close()
    
    def test(self, num_episodes=10, delta_action_test=False, bartlett_action_test=False):
        set_seed(self.cfg.sac_seed)
        total_rewards = []
        w_T_store = []

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

            w_T = sum(reward_store).item()
            w_T_store.append(w_T)

            if global_step % 1 == 0:
                w_T_mean = np.mean(w_T_store)
                w_T_var = np.var(w_T_store)
                path_row = info["path_row"]
                print(info)
                print(f"episode: {global_step} | episode final wealth: {w_T:.3f} | wealth_mean: {w_T_mean:.3f} | wealth_var: {w_T_var:.3f}")
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
        print(f"测试结果 ({num_episodes} episodes):")
        print(f"最终平均收益: {mean_reward:.2f}, 标准差: {std_reward:.2f}")
        print(f"优化目标 Y(0): {y_0:.2f}")
        return mean_reward, std_reward

    def load(self, path: str):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint["actor"])
        self.qf1.load_state_dict(checkpoint["qf1"])
        self.qf2.load_state_dict(checkpoint["qf2"])
        if self.cfg.sac_autotune and checkpoint["log_alpha"] is not None:
            self.log_alpha.data.copy_(checkpoint["log_alpha"])

if __name__ == "__main__":
    # 定义 objective，不保存每个 trial 的模型
    def objective(trial):
        sacconfig = Config()
        sacconfig.sac_learning_rate = trial.suggest_loguniform("sac_learning_rate", 1e-5, 1e-2)
        sacconfig.sac_tau = trial.suggest_uniform("sac_tau", 0.005, 0.05)
        sacconfig.sac_gamma = trial.suggest_uniform("sac_gamma", 0.8, 0.99)
        sacconfig.sac_alpha = trial.suggest_loguniform("sac_alpha", 1e-3, 1.0)
        sacconfig.sac_batch_size = trial.suggest_int("sac_batch_size", 64, 256, step=64)

        set_seed(sacconfig.sac_seed)
        exp_manager = ExperimentManager(sacconfig)
        env = TradingEnv(
            continuous_action_flag=True, sabr_flag=sacconfig.sabr_flag, spread=0.01,
            num_contract=1, init_ttm=20, trade_freq=1, num_sim=500000
        )
        env.seed(sacconfig.sac_seed)
        agent = SAC(sacconfig, env, exp_manager)
        print("开始训练 (调参模式)...")
        agent.train()
        mean_reward, std_reward = agent.test(num_episodes=10)
        return -mean_reward

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)
    best_params = study.best_trial.params
    print("最佳 trial 参数：")
    print(best_params)

    # 用最佳参数重新训练一次以保存最终模型和参数文件
    best_config = Config()
    best_config.sac_learning_rate = best_params["sac_learning_rate"]
    best_config.sac_tau = best_params["sac_tau"]
    best_config.sac_gamma = best_params["sac_gamma"]
    best_config.sac_alpha = best_params["sac_alpha"]
    best_config.sac_batch_size = best_params["sac_batch_size"]

    set_seed(best_config.sac_seed)
    exp_manager = ExperimentManager(best_config)
    env = TradingEnv(
        continuous_action_flag=True, sabr_flag=best_config.sabr_flag, spread=0.01,
        num_contract=1, init_ttm=20, trade_freq=1, num_sim=500000
    )
    env.seed(best_config.sac_seed)
    agent = SAC(best_config, env, exp_manager)
    print("使用最佳参数重新训练模型...")
    agent.train()
    mean_reward, std_reward = agent.test(num_episodes=10)
    print("最佳模型的测试结果:", mean_reward, std_reward)

    # 保存最终最佳模型及其参数 JSON 文件
    exp_manager.save_model({
        "actor": agent.actor.state_dict(),
        "qf1": agent.qf1.state_dict(),
        "qf2": agent.qf2.state_dict(),
        "log_alpha": agent.log_alpha if agent.cfg.sac_autotune else None,
    }, "best_model.pth")

    with open("best_params.json", "w") as f:
        json.dump(best_params, f, indent=4)
