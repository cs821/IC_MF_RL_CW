import torch

class Config:

    # config.py 新增方法
    def update_from_dict(self, params: dict):
        """根据参数字典更新配置"""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: Config has no attribute {key}")
    def __init__(self):
        # 选择算法: "ddpg","sac" 或 "qlearning" 【qlearning没改，只调ddpg和sac的参】
        self.algo = "ddpg"  # "ddpg" /"sac" /"qlearning"
        self.sabr_flag = True #要不要用sabr，请每次调参务必记得修改！
        #厌恶系数
        self.ra_c = 1.5
        # 训练参数
        

        # Q-Learning 相关参数
        self.q_lr = 0.1  
        self.q_gamma = 0.99  
        self.q_epsilon = 1.0  
        self.q_epsilon_decay = 0.995  
        self.q_epsilon_min = 0.01  

        # DDPG 参数
        self.ddpg_num_episodes = 10000
        self.ddpg_batch_size = 128  
        self.ddpg_seed = 20021114  
        self.ddpg_gamma = 0.99
        self.ddpg_buffer_size = int(2e6)  # 增加缓冲区大小
        self.ddpg_tau = 0.001  # 更小的 tau 让目标网络更新更平稳
        self.ddpg_actor_lr = 3e-4  
        self.ddpg_critic_lr = 3e-4  
        self.ddpg_epsilon_decay = 0.995  # 衰减率
        self.ddpg_epsilon_min = 0.1     # 最小探索率
        self.prioritized_replay_beta_iters = 100000
        self.prioritized_replay_beta0 = 0.4
        self.ddpg_alpha=0.6

        # SAC 参数 
        self.sac_seed = 20250307
        self.sac_total_timesteps = 130
        self.sac_buffer_size = int(1e6)
        self.sac_batch_size = 128
        self.sac_gamma = 0.99
        self.sac_tau = 0.005
        self.sac_learning_rate = 3e-4
        self.sac_alpha = 0.2
        self.sac_autotune = True
        self.sac_policy_frequency = 2
        self.sac_target_network_frequency = 1
        
        # 设备设置
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 是否存储训练历史
        self.train_history = True
