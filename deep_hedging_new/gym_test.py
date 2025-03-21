#a environment for code testing, irrelevant to the paper
import torch
import numpy as np
import random
import pandas as pd
import gym
from ddpg import DDPG
from q_learning import QLearning
from config import Config
from experimentmanager import ExperimentManager
import wandb


config = Config()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False 

set_seed(config.ddpg_seed)

unique_name = 'test_gym' # this is the name for this run in wandb, can be set to whatever you want 
wandb.init(project="deep_hedging", config=config, name = unique_name)

#env = gym.make("CartPole-v1", render_mode='human')
env = gym.make("CartPole-v1")
if hasattr(env, "seed"):
    env.seed(config.ddpg_seed)
#
if config.algo == "ddpg":
    agent = DDPG(env.observation_space.shape[0], 1, 1, config.ddpg_buffer_size, config.ddpg_gamma, config.ddpg_tau, config.ddpg_alpha)
else:
    raise ValueError("Invalid algorithm. Choose 'ddpg', 'sac' or 'qlearning'.")

global_step = 0
exp_manager = ExperimentManager(config)
history = {"episode": [], "episode_w_T": []}
w_T_store = []

print("\n\n*** Training Begin! ***")
for episode in range(config.ddpg_num_episodes):
    obs, _ = env.reset()
    #env.render()
    done = False
    episode_reward = 0
    reward_store = []
    action_store = []
    y_action = []
    y_obs = [obs]
    while not done:
        action = agent.act(obs)  
        action = int(action > 0.5)
        info = env.step(action)
        next_obs, reward, done, _, info = env.step(action)
        if config.algo == "qlearning":
            agent.learn(obs, action, reward, next_obs, done)
        else:
            agent.replay_buffer.add(obs, action, reward, next_obs, done)
            if len(agent.replay_buffer) > config.ddpg_batch_size:
                agent_info  = agent.update(config.ddpg_batch_size, global_step)
            else:
                agent_info = {}

        obs = next_obs
        episode_reward += reward
        reward_store.append(reward)
        action_store.append(action)
        y_action.append(action) 
        y_obs.append(obs)
        global_step += 1
        #env.render()

    w_T_store.append(episode_reward)

    history["episode"].append(episode)
    history["episode_w_T"].append(episode_reward)
    wandb.log({"Episode": episode,"env_steps": global_step,})
    wandb.log({"Episode_Return": episode_reward,"env_steps": global_step,})
    for k, v in agent_info.items():
        if k in history.keys():
            history[k].append(v)
        else:
            history[k] = [v]
        wandb.log({k: v,"env_steps": global_step,})

    if episode % 1000 == 0:
        exp_manager.save_checkpoint(agent.actor.state_dict(), f"{config.algo}_actor", episode)
        exp_manager.save_checkpoint(agent.critic.q_ex_net.state_dict(), f"{config.algo}_critic_q_ex", episode)
        exp_manager.save_checkpoint(agent.critic.q_ex2_net.state_dict(), f"{config.algo}_critic_q_ex2", episode)
        print(f"Checkpoint saved at episode {episode}")

    print(info)
    if len(agent_info.keys()) == 0:
        eps = getattr(agent,"epsilon", 0)
        print(
            f"episode: {episode} | episode final wealth: {episode_reward:.3f} | epsilon:{eps:.2f}"
        )
    else:
        print(
            "episode: {} | episode final wealth: {:.3f} | loss_ex: {:.3f} | loss_ex2: {:.3f} |critic_loss: {:.3f} |actor_grad_norm: {:.3f} | critic_grad_norm: {:.3f} | actor_loss: {:.3f} |  epsilon:{:.2f}".format(
                episode, episode_reward, agent_info["critic1_loss"], agent_info["critic2_loss"],agent_info["critic_loss"], agent_info["total_actor_grad_norm"], agent_info["total_critic_grad_norm"], agent_info["actor_loss"], getattr(agent, "epsilon", 0)
            )
        )

    with np.printoptions(precision=2, suppress=True):
        print(f"episode: {episode} | rewards {np.array(reward_store)}")
        print(f"episode: {episode} | obs {np.array(y_obs)}")
        print(f"episode: {episode} | actions taken {np.array(y_action)}")

    if isinstance(agent, DDPG):
        agent.update_epsilon()

# save training history
exp_manager.save_history(history)
exp_manager.save_image(history)

# save model
exp_manager.save_model(agent.actor.state_dict(), f"{config.algo}_actor.pth")
exp_manager.save_model(agent.critic.q_ex_net.state_dict(), f"{config.algo}_critic_q_ex.pth")
exp_manager.save_model(agent.critic.q_ex2_net.state_dict(), f"{config.algo}_critic_q_ex2.pth")
print("DDPG Saved!")

print("\n*** Training Completed! ***")
wandb.finish()