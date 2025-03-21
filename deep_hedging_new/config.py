import torch

class Config:
    def __init__(self):
        # choose algorithm: "ddpg","sac"
        self.algo = "sac"  # "ddpg" /"sac"
        self.sabr_flag = True #simulation method. False:gbm True:sabr
        #Risk aversion coefficient
        self.ra_c = 1.5
        # training hyperparameters

        # DDPG 
        self.trade_freq = 1
        self.ddpg_num_episodes = 50001
        self.ddpg_epsilon_decay_steps = 2500
        self.ddpg_batch_size = 128  
        self.ddpg_seed = 64  
        self.ddpg_gamma = 0.99
        self.ddpg_buffer_size = int(5000)  
        self.ddpg_tau = 0.005  
        self.ddpg_actor_lr = 1e-4 
        self.ddpg_critic_lr = 3e-4  
        self.ddpg_epsilon_decay = 0.99994 
        self.ddpg_epsilon_min = 0.1    
        self.prioritized_replay_beta_iters = 100000
        self.prioritized_replay_beta0 = 0.4
        self.ddpg_alpha=0.6

        # SAC  
        self.sac_seed = 20250307
        self.sac_total_timesteps = 600000 
        self.sac_buffer_size = int(1e6)
        self.sac_batch_size = 256
        self.sac_gamma = 0.99
        self.sac_tau = 0.0001
        self.sac_learning_rate = 5e-4
        self.sac_alpha = 0.5
        self.sac_autotune = True
        self.sac_policy_frequency = 5
        self.sac_target_network_frequency = 1
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.train_history = True
