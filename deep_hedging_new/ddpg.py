import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from replay_buffer import PrioritizedReplayBuffer
from schedules import LinearSchedule
from config import Config

# Load hyperparameter configuration
config = Config()

# Define Actor Network
class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, act_limit):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.LayerNorm(256),# Normalize activations for stability
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, act_dim), nn.Sigmoid() # Output normalized to (0, 1)
        )
        self.act_limit = act_limit

    def forward(self, obs):
        return self.act_limit * self.net(obs) # Max action value (e.g. stock holdings)


# Define Critic Network with risk-averse utility
class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.ra_c = config.ra_c  # risk aversion coefficient
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
        
        # calculate final Q (F in paper)
        q = q_ex - self.ra_c * torch.sqrt(torch.clamp(q_ex2 - q_ex ** 2, min=0.0))
        
        return q_ex, q_ex2, q


# Define DDPG Agent
class DDPG:
    def __init__(self, obs_dim, act_dim, act_limit, buffer_size=int(2e6), tau=0.00001, alpha=0.6, max_t=500, device='cpu'):
        
        self.device = device
        self.actor = Actor(obs_dim, act_dim, act_limit).to(device)
        self.critic = Critic(obs_dim, act_dim).to(device)
        self.target_actor = Actor(obs_dim, act_dim, act_limit).to(device)
        self.target_critic = Critic(obs_dim, act_dim).to(device)

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-4)

        self.replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha)
        self.beta_schedule = LinearSchedule(config.prioritized_replay_beta_iters,
                                       initial_p=config.prioritized_replay_beta0,
                                       final_p=1.0)
        self.tau = tau
        self.gamma = config.ddpg_gamma
        self.epsilon = 1.0 #initial epsilon
        self.epsilon_decay = config.ddpg_epsilon_decay  
        self.epsilon_min = config.ddpg_epsilon_min   
        self.epsilon_decay_delta = (self.epsilon - self.epsilon_min) / max_t

    def update_epsilon(self):
        """called once per Episode to attenuate epsilon"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        # self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay_delta)
    
    def act(self, obs, eval_mode=False):
        """
        obs is a list
        Return: an integer action 
        """
        if not isinstance(obs, torch.Tensor):
            obs = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
        if not eval_mode and np.random.rand() < self.epsilon:
            # Exploration: Random sampling in the action space
            action = np.random.uniform(
                low=0, 
                high=self.actor.act_limit
            )
            return action
        else:
            # exploitation: Use Actor network output
            action = self.actor(obs)
            return action.detach().cpu().numpy().item()

    def update(self, batch_size, global_step):
        beta = self.beta_schedule.value(global_step)
        obs, action, reward, next_obs, done, weights, idxes = self.replay_buffer.sample(batch_size, beta)
        #print(obs.shape, reward.shape, action.shape, next_obs.shape, weights.shape)
        obs = torch.FloatTensor(obs).to(self.device)
        action = torch.FloatTensor(action).to(self.device).unsqueeze(1)
        reward = torch.FloatTensor(reward).to(self.device).unsqueeze(1)  
        next_obs = torch.FloatTensor(next_obs).to(self.device).view(batch_size, -1)
        done = torch.FloatTensor(done).to(self.device).unsqueeze(1)
        weights = torch.FloatTensor(weights).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_action = self.target_actor(next_obs)
            #target_q = self.target_critic(next_obs, next_action)
            #target_critic_mean = target_q.mean()
            #target = reward + self.gamma * (1 - done) * target_q
            target_q_ex, target_q_ex2, _ = self.target_critic(next_obs, next_action)
            target_q_ex = reward + self.gamma*(1 - done) * target_q_ex
            #target_q_ex2 = reward ** 2 + self.gamma* (1 - done) * (2 * reward * target_q_ex + target_q_ex2) #1
            target_q_ex2 = reward ** 2 + self.gamma* (1-done)*(2*reward*target_q_ex) + self.gamma**2 *(1-done)* target_q_ex2 #2
            #target_q_ex2 = torch.clamp(target_q_ex2, min=-1e6, max=1e6)

        current_q_ex, current_q_ex2, _ = self.critic(obs, action)
        td_error_ex = target_q_ex - current_q_ex
        td_error_ex2 = target_q_ex2 - current_q_ex2
        if torch.isnan(td_error_ex2).any():
            print("NaN detected in td_error_ex2!")
            return None
            #print(f"target_q_ex2: {target_q_ex2}")
            #print(f"current_q_ex2: {current_q_ex2}")

            #print("current_q_ex2 mean:", current_q_ex2.mean().item())
            #print("target_q_ex2 mean:", target_q_ex2.mean().item())

        loss_ex = (weights * td_error_ex.pow(2)).mean()
        loss_ex2 = (weights * td_error_ex2.pow(2)).mean()
        critic_loss = loss_ex + loss_ex2

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        total_critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=10.0)
        self.critic_optimizer.step()

        #priorities = (td_error.abs()).cpu().detach().numpy().flatten()
        #self.replay_buffer.update_priorities(idxes, priorities)
        priorities = np.abs(td_error_ex2.cpu().detach().numpy().flatten())+1e-6
        if np.min(priorities)<=0:
            print("Priorities before updating:", priorities)
        self.replay_buffer.update_priorities(idxes, priorities)

        actor_loss = -self.critic(obs, self.actor(obs))[2].mean()#用q

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        total_actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10.0)
        #print(f"Total Actor Grad Norm: {total_actor_grad_norm:.4f}")

        #for name, param in self.actor.named_parameters():
        #    print(f"Actor {name} grad norm: {param.grad.norm().item():.4f}")

        self.actor_optimizer.step()

        
        for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        info = {
            "critic1_mean": current_q_ex.mean().item(),
            "critic2_mean": current_q_ex2.mean().item(),
            "target_critic1_mean": target_q_ex.mean().item(),
            "target_critic2_mean": target_q_ex2.mean().item(),
            "critic_loss": critic_loss.item(), 
            "abs_td_loss1": td_error_ex.abs().mean().item(),
            "abs_td_loss2": td_error_ex2.abs().mean().item(),
            "total_actor_grad_norm": total_actor_grad_norm.item(), 
            "total_critic_grad_norm": total_critic_grad_norm.item(), 
            "actor_loss": actor_loss.item(),
            "epsilon": self.epsilon,
        }
        return info
