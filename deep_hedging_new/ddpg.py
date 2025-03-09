import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from replay_buffer import PrioritizedReplayBuffer
from schedules import LinearSchedule
from config import Config

# 读取配置
config = Config()

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, act_limit):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, act_dim), nn.Sigmoid()
        )
        self.act_limit = act_limit

    def forward(self, obs):
        return self.act_limit * self.net(obs)

class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.q_ex_net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.q_ex2_net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, obs, act):
        q_ex = self.q_ex_net(torch.cat([obs, act], dim=-1))
        q_ex2 = self.q_ex2_net(torch.cat([obs, act], dim=-1))
        return q_ex, q_ex2

class DDPG:
    def __init__(self, obs_dim, act_dim, act_limit, buffer_size=int(2e6), gamma=0.99, tau=0.00001, alpha=0.6):
        self.actor = Actor(obs_dim, act_dim, act_limit)
        self.critic = Critic(obs_dim, act_dim)
        self.target_actor = Actor(obs_dim, act_dim, act_limit)
        self.target_critic = Critic(obs_dim, act_dim)

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-4)

        self.replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha)
        self.beta_schedule = LinearSchedule(config.prioritized_replay_beta_iters,
                                       initial_p=config.prioritized_replay_beta0,
                                       final_p=1.0)
        self.gamma = gamma
        self.tau = tau
        # 初始化参数
        self.epsilon = 1.0  # 初始探索率
        self.epsilon_decay = config.ddpg_epsilon_decay  # 衰减率
        self.epsilon_min = config.ddpg_epsilon_min     # 最小探索率

    def update_epsilon(self):
        """每Episode调用一次,衰减epsilon"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def act(self, obs, eval_mode=False):
        """
        obs is a list
        Return: an integer action 
        """
        """修改后的动作选择逻辑"""
        if not isinstance(obs, torch.Tensor):
            obs = torch.FloatTensor(obs).unsqueeze(0)  
        if not eval_mode and np.random.rand() < self.epsilon:
            # 探索：在动作空间内随机采样
            action = np.random.uniform(
                low=self.actor.act_limit*0.1, 
                high=self.actor.act_limit*0.9
            )
            return action
        else:
            # 利用：使用Actor网络输出
            action = self.actor(obs)
            return action.detach().cpu().numpy().item()

    def update(self, batch_size, global_step):
        beta = self.beta_schedule.value(global_step)
        obs, action, reward, next_obs, done, weights, idxes = self.replay_buffer.sample(batch_size, beta)
        #print(obs.shape, reward.shape, action.shape, next_obs.shape, weights.shape)
        obs = torch.FloatTensor(obs)
        action = torch.FloatTensor(action).unsqueeze(1)
        reward = torch.FloatTensor(reward).unsqueeze(1)  # 变成 (batch_size, 1)
        next_obs = torch.FloatTensor(next_obs).view(batch_size, -1)
        done = torch.FloatTensor(done).unsqueeze(1)
        weights = torch.FloatTensor(weights).unsqueeze(1)

        with torch.no_grad():
            next_action = self.target_actor(next_obs)
            target_q_ex, target_q_ex2 = self.target_critic(next_obs, next_action)
            target_critic1_mean = target_q_ex.mean()
            target_critic2_mean = target_q_ex2.mean()
            target_q_ex = reward + self.gamma * (1 - done) * target_q_ex
            target_q_ex2 = reward + self.gamma * (1 - done) * target_q_ex2

        current_q_ex, current_q_ex2 = self.critic(obs, action)
        critic1_mean = current_q_ex.mean()
        critic2_mean = current_q_ex2.mean()
        td_error_ex = target_q_ex - current_q_ex
        td_error_ex2 = target_q_ex2 - current_q_ex2

        
        loss_ex = (weights * td_error_ex.pow(2)).mean()
        loss_ex2 = (weights * td_error_ex2.pow(2)).mean()
        critic_loss = loss_ex + loss_ex2  # 合并损失

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        total_critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=10.0)
        #print(f"Total Critic Grad Norm: {total_critic_grad_norm:.4f}")
        self.critic_optimizer.step()

        priorities = (td_error_ex.abs() + td_error_ex2.abs()).cpu().detach().numpy().flatten()
        self.replay_buffer.update_priorities(idxes, priorities)

        actor_loss = -self.critic(obs, self.actor(obs))[0].mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        total_actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10.0)
        #print(f"Total Actor Grad Norm: {total_actor_grad_norm:.4f}")

        #for name, param in self.actor.named_parameters():
        #    print(f"Actor {name} grad norm: {param.grad.norm().item():.4f}")#检查梯度

        self.actor_optimizer.step()

        
        for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic.q_ex_net.parameters(), self.target_critic.q_ex_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic.q_ex2_net.parameters(), self.target_critic.q_ex2_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        info = {
            "critic1_mean": critic1_mean.item(), 
            "critic2_mean": critic2_mean.item(), 
            "target_critic1_mean": target_critic1_mean.item(), 
            "target_critic2_mean": target_critic2_mean.item(), 
            "critic1_loss": loss_ex.item(), 
            "critic2_loss": loss_ex2.item(),
            "critic_loss": critic_loss.item(), 
            "abs_td_loss1": td_error_ex.abs().mean().item(),
            "abs_td_loss2": td_error_ex2.abs().mean().item(),
            "total_actor_grad_norm": total_actor_grad_norm.item(), 
            "total_critic_grad_norm": total_critic_grad_norm.item(), 
            "actor_loss": actor_loss.item(),
            "epsilon": self.epsilon,
        }

        return info
