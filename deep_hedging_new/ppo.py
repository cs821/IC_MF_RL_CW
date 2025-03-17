#!/usr/bin/env python
"""
PPO for Deep Hedging using TradingEnv
Adapted from CleanRL's PPO continuous implementation,
with modifications:
  - 使用你的 TradingEnv(envs.py 中的 TradingEnv)
  - 策略网络输出通过 Beta 分布控制动作在 [0, action_scale] 内
  - 在 rollout 后加入风险惩罚，对 finished episode 的累计回报进行调整
"""

import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions import Beta
from torch.utils.tensorboard import SummaryWriter

# 导入你的 TradingEnv（确保 envs.py 在 PYTHONPATH 中）
from envs import TradingEnv
from config import Config  # 如果有配置文件的话

@dataclass
class Args:
    exp_name: str = "ppo_trading"
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "deep_hedging_ppo"
    wandb_entity: str = None
    capture_video: bool = False
    save_model: bool = False
    upload_model: bool = False
    hf_entity: str = ""
    # 此处不再使用 env_id 字符串，而是直接实例化 TradingEnv
    num_envs: int = 1
    num_steps: int = 2048
    total_timesteps: int = 1000000
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 32
    update_epochs: int = 10
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = None
    # 风险惩罚系数（论文要求对收益波动进行控制）
    risk_coef: float = 0.1

    # 运行时计算
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False  

# 我们的环境创建函数，返回 TradingEnv 实例
def make_env():
    def thunk():
        env = TradingEnv(
            continuous_action_flag=True,
            sabr_flag=False,        # 根据需要设置
            spread=0.01,
            num_contract=1,
            init_ttm=20,
            trade_freq=1,
            num_sim=500000
        )
        env.seed(args.seed)
        return env
    return thunk

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# 这里我们设计一个简单的 Actor-Critic 网络
# Actor部分采用 Beta 分布，输出参数经过 softplus 保证正值，再加1确保大于1，
# 从而采样的 action 在 (0,1) 内，再乘以 action_scale 映射到 [0, action_scale]
class Agent(nn.Module):
    def __init__(self, obs_dim, act_dim, action_scale):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_hidden = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
        )
        self.actor_alpha = layer_init(nn.Linear(64, act_dim), std=0.01)
        self.actor_beta = layer_init(nn.Linear(64, act_dim), std=0.01)
        self.action_scale = action_scale

    def get_value(self, x):
        return self.critic(x).flatten()

    def get_action_and_value(self, x, action=None):
        hidden = self.actor_hidden(x)
        alpha = torch.nn.functional.softplus(self.actor_alpha(hidden)) + 1.0
        beta = torch.nn.functional.softplus(self.actor_beta(hidden)) + 1.0
        dist = Beta(alpha, beta)
        if action is None:
            action = dist.rsample()  # 采样结果在 (0,1)
        log_prob = dist.log_prob(action).sum(1)
        entropy = dist.entropy().sum(1)
        action_scaled = action * self.action_scale
        value = self.get_value(x)
        return action_scaled, log_prob, entropy, value

if __name__ == "__main__":
    args = tyro.cli(Args)
    set_seed(args.seed)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    
    writer = SummaryWriter(f"runs/{run_name}")
    
    # 创建向量化环境，使用 TradingEnv
    envs = gym.vector.SyncVectorEnv([make_env() for _ in range(args.num_envs)])
    
    # TradingEnv 的 observation_space 为 Box([-inf, ...], [inf,...])，state维度为3
    obs_dim = int(np.prod(envs.single_observation_space.shape))  # 3
    act_dim = int(np.prod(envs.single_action_space.shape))         # 1
    action_scale = envs.single_action_space.high[0]  # 应为100
    
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    
    agent = Agent(obs_dim, act_dim, action_scale).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    
    # 存储 rollout 数据
    obs_storage = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions_storage = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs_storage = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards_storage = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones_storage = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values_storage = torch.zeros((args.num_steps, args.num_envs)).to(device)
    
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    
    # PPO rollout采样 + 风险调整
    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
        
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs_storage[step] = next_obs
            dones_storage[step] = next_done
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values_storage[step] = value
            actions_storage[step] = action  # 记录已经 scaled 的 action
            logprobs_storage[step] = logprob
            next_obs_np, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = torch.tensor(np.logical_or(terminations, truncations), dtype=torch.float32).to(device)
            rewards_storage[step] = torch.tensor(reward, dtype=torch.float32).to(device)
            next_obs = torch.Tensor(next_obs_np).to(device)
        
        # —— 风险调整 —— 
        # 针对 rollout 内每个 finished episode (done==1)：
        # 计算累计回报 G，求所有 finished episode 的均值，然后对每个 finished episode计算 penalty = risk_coef * (G - mean_G)^2，
        # 并将该 penalty 均摊到该 episode 的每个 timestep上，从而调整 rewards_storage
        finished_returns = []
        for i in range(args.num_envs):
            done_indices = (dones_storage[:, i] == 1).nonzero(as_tuple=False)
            if len(done_indices) > 0:
                t_end = done_indices[-1].item()
                G = rewards_storage[:t_end+1, i].sum().item()
                finished_returns.append(G)
        if finished_returns:
            mean_return = np.mean(finished_returns)
        else:
            mean_return = 0.0
        for i in range(args.num_envs):
            done_indices = (dones_storage[:, i] == 1).nonzero(as_tuple=False)
            if len(done_indices) > 0:
                t_end = done_indices[-1].item()
                G = rewards_storage[:t_end+1, i].sum().item()
                penalty = args.risk_coef * (G - mean_return) ** 2
                adjustment = penalty / (t_end + 1)
                rewards_storage[:t_end+1, i] -= adjustment
        
        # 计算优势和 return（使用 GAE）
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards_storage).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones_storage[t+1]
                    nextvalues = values_storage[t+1]
                delta = rewards_storage[t] + args.gamma * nextvalues * nextnonterminal - values_storage[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values_storage
        
        b_obs = obs_storage.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs_storage.reshape(-1)
        b_actions = actions_storage.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_storage.reshape(-1)
        
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]
                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef)
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()
            if args.target_kl is not None and approx_kl > args.target_kl:
                break
        
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        
        # 如果 episode 结束，则记录日志并重置环境
        if done:
            writer.add_scalar("charts/episodic_return", episode_reward, global_step)
            writer.add_scalar("charts/episodic_length", episode_length, global_step)
            w_T_store.append(episode_reward)
            history["episode"].append(global_step)
            history["episode_w_T"].append(episode_reward)
            history["q_loss"].append(loss)
            wandb.log({"Episode_Return": episode_reward, "env_steps": global_step})
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
    
    exp_manager.save_history(history)
    writer.close()
    wandb.finish()
    
    # 测试函数
    def test_agent(num_episodes=1000):
        set_seed(sacconfig.sac_seed)
        total_rewards = []
        w_T_store = []
        cost_ratio = []
        for ep in range(num_episodes):
            obs = self.env.reset()
            episode_reward = 0
            done = False
            reward_store = []
            action_store = []
            y_action = []
            while not done:
                with torch.no_grad():
                    action_tensor, _, _ = self.actor.get_action(torch.Tensor(obs).to(sacconfig.device))
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
            cost_ratio.append(-w_T / option_price)
            if ep % 100 == 0:
                w_T_mean = np.mean(w_T_store)
                w_T_var = np.var(w_T_store)
                print(info)
                print(f"episode: {ep} | episode final wealth: {w_T:.3f} | wealth_mean: {w_T_mean:.3f} | wealth_var: {w_T_var:.3f}")
                print(f"episode: {ep} | rewards {np.array(reward_store).ravel()}")
                print(f"episode: {ep} | actions taken {np.array(y_action).ravel()}")
                print(f"episode: {ep} | deltas {self.env.delta_path[path_row] * 100}")
                print(f"episode: {ep} | stock price {self.env.path[path_row]}")
                print(f"episode: {ep} | option price {self.env.option_price_path[path_row] * 100}\n")
            total_rewards.append(w_T)
        mean_reward = np.mean(w_T_store)
        std_reward = np.std(w_T_store)
        y_0 = -mean_reward + sacconfig.ra_c * std_reward  
        cost_ratio_mean = np.mean(cost_ratio)
        cost_ratio_std = np.std(cost_ratio)
        print(f"测试结果 ({num_episodes} episodes):")
        print(f"最终平均成本: {-mean_reward:.2f}, 标准差: {std_reward:.2f}")
        print(f"最终平均成本/期权价格: {cost_ratio_mean:.2f}, 标准差: {cost_ratio_std:.2f}")
        print(f"优化目标 Y(0): {y_0:.2f}")
    
    test_agent(num_episodes=1000)
