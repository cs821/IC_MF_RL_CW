import os
import torch
import numpy as np
import random
import pandas as pd
import optuna
import time
import json
from datetime import datetime
from envs import TradingEnv
from ddpg import DDPG
from config import Config
from experimentmanager import ExperimentManager
import shutil


def set_seed(seed):
    """设置所有随机种子保证可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def objective(trial: optuna.Trial) -> float:
    """Optuna优化目标函数（兼容4.x版本）"""
    # ==================== 立即设置路径属性 ====================
    trial_num = trial.number
    trial_dir = os.path.abspath(f"ddpg_trials/trial_{trial_num:03d}")
    model_path = os.path.join(trial_dir, "best_model.pth")
    config_path = os.path.join(trial_dir, "config.json")
    
    # 原子性设置用户属性（Optuna 4.x推荐方式）
    trial.set_user_attr("model_path", model_path)
    trial.set_user_attr("config_path", config_path)
    trial.set_user_attr("trial_dir", trial_dir)
    
    # ==================== 创建目录 ====================
    os.makedirs(trial_dir, exist_ok=True)
    
    # ==================== 初始化配置 ====================
    config = Config()
    config.algo = "ddpg"
    
    # ==================== 参数建议 ====================
    params = {
        'ddpg_gamma': trial.suggest_float('ddpg_gamma', 0.9, 0.999),
        'ddpg_tau': trial.suggest_float('ddpg_tau', 1e-5, 1e-3, log=True),
        'ddpg_actor_lr': trial.suggest_float('ddpg_actor_lr', 1e-5, 1e-3, log=True),
        'ddpg_critic_lr': trial.suggest_float('ddpg_critic_lr', 1e-5, 1e-3, log=True),
        'ddpg_alpha': trial.suggest_float('ddpg_alpha', 0.4, 0.8),
        'ddpg_batch_size': trial.suggest_categorical('ddpg_batch_size', [64, 128, 256]),
        'ddpg_buffer_size': trial.suggest_categorical('ddpg_buffer_size', [int(1e5), int(5e5), int(1e6)]),
        #'ddpg_num_episodes': trial.suggest_categorical('ddpg_num_episodes', [50, 100, 150])
    }
    config.update_from_dict(params)
    
    # ==================== 保存初始配置 ====================
    with open(config_path, "w") as f:
        json.dump(vars(config), f, indent=4)
    
    # ==================== 初始化环境 ====================
    set_seed(config.ddpg_seed)
    train_env = TradingEnv(
        continuous_action_flag=True,
        sabr_flag=config.sabr_flag,
        spread=0.01,
        num_contract=1,
        init_ttm=20,
        trade_freq=1,
        num_sim=50000
    )
    train_env.seed(config.ddpg_seed)
    
    # ==================== 初始化智能体 ====================
    agent = DDPG(
        obs_dim=train_env.observation_space.shape[0],
        act_dim=train_env.action_space.shape[0],
        act_limit=train_env.action_space.high[0],
        buffer_size=config.ddpg_buffer_size,
        gamma=config.ddpg_gamma,
        tau=config.ddpg_tau,
        alpha=config.ddpg_alpha
    )
    
    # ==================== 训练循环 ====================
    best_y0 = float('inf')
    
    for episode in range(config.ddpg_num_episodes):
        # 训练逻辑
        obs = np.array(train_env.reset(), dtype=np.float32)
        done = False
        reward_store = []
        
        while not done:
            action = agent.act(obs)
            next_obs, reward, done, info = train_env.step(action)
            
            agent.replay_buffer.add(obs, action, reward, next_obs, done)
            
            if len(agent.replay_buffer) > config.ddpg_batch_size:
                loss_ex, loss_ex2 = agent.update(config.ddpg_batch_size, episode)
            
            obs = next_obs
            reward_store.append(reward)
        if episode % 100 == 0:
            episode_reward = sum(reward_store)
            print(f"Trial {trial.number}: Episode {episode}, cumulative reward: {episode_reward:.2f}")
        
        # ==================== 中期评估 ====================
        if episode % 10 == 0:
            # 评估与保存逻辑
            torch.save(agent.actor.state_dict(), model_path)
            trial.set_user_attr("last_save_episode", episode)
            
    # ==================== 最终评估 ====================
    final_test_rewards = []
    for _ in range(20):
        final_obs = train_env.reset()
        final_done = False
        final_rewards = []
        while not final_done:
            final_action = agent.act(final_obs, eval_mode=True)
            final_obs, final_reward, final_done, _ = train_env.step(final_action)
            final_rewards.append(final_reward)
        final_test_rewards.append(sum(final_rewards))
    
    final_y0 = -np.mean(final_test_rewards) + config.ra_c * np.std(final_test_rewards)
    
    # 覆盖最佳模型（如果最终结果更好）
    if final_y0 < best_y0:
        torch.save(agent.actor.state_dict(), model_path)
        with open(config_path, "w") as f:
            json.dump(vars(config), f)
    
    return final_y0


def main():
    start_time = time.time()  # 记录开始时间

    # ==================== 使用 SQLite 存储（支持并行调参） ====================
    storage = "sqlite:///ddpg_optuna.db"
    study = optuna.create_study(
        direction="minimize",
        study_name="ddpg_hedging_v4",
        storage=storage,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10
        )
    )
    
    # ==================== 运行优化（例如并行4个试验） ====================
    try:
        study.optimize(
            objective, 
            n_trials=10,
            n_jobs=4,  # 并行运行4个 trial
            gc_after_trial=True  # 确保及时释放资源
        )
    except KeyboardInterrupt:
        print("用户中断优化过程")
    
    # ==================== 处理最佳结果 ====================
    if not study.best_trial:
        print("没有成功完成的试验")
        return
    
    # 验证最佳试验数据完整性
    best_trial = study.best_trial
    required_attrs = ["model_path", "config_path"]
    missing = [attr for attr in required_attrs if attr not in best_trial.user_attrs]
    
    if missing:
        print(f"自动修复缺失属性: {missing}")
        trial_num = best_trial.number
        repaired = {
            "model_path": f"ddpg_trials/trial_{trial_num:03d}/best_model.pth",
            "config_path": f"ddpg_trials/trial_{trial_num:03d}/config.json"
        }
        for k, v in repaired.items():
            best_trial.set_user_attr(k, v)
    
    # 保存全局最佳
    best_dir = "best_ddpg_v4"
    os.makedirs(best_dir, exist_ok=True)
    
    try:
        shutil.copy(best_trial.user_attrs["model_path"], f"{best_dir}/best_model.pth")
        shutil.copy(best_trial.user_attrs["config_path"], f"{best_dir}/config.json")
        
        # 添加版本元数据
        metadata = {
            "optuna_version": optuna.__version__,
            "saved_at": datetime.now().isoformat(),
            "best_value": best_trial.value
        }
        with open(f"{best_dir}/metadata.json", "w") as f:
            json.dump(metadata, f)
            
    except Exception as e:
        print(f"保存失败: {str(e)}")
        print("请手动复制以下文件:")
        print(f"模型: {best_trial.user_attrs['model_path']}")
        print(f"配置: {best_trial.user_attrs['config_path']}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    # 格式化成时:分:秒
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"程序总运行时长: {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}")



if __name__ == "__main__":
    main()
