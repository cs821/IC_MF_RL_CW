# IC_MF_RL_CW: Deep Hedging with Reinforcement Learning

This repository contains the code implementation for the **Advanced Machine Learning (MATH70120)** coursework, focusing on **Deep Hedging** using **Deep Deterministic Policy Gradient (DDPG)** and **Soft Actor-Critic (SAC)**.  

The goal of this coursework is to explore reinforcement learning approaches for hedging financial derivatives. The implementation follows standard deep reinforcement learning frameworks and utilizes **DDPG** for deterministic policy optimization and **SAC** for entropy-regularized exploration. The code is designed to replicate and analyze the performance of these algorithms in hedging tasks, considering different market dynamics and transaction costs.  

Certain components, such as the **environment setup and replay buffer**, take inspiration from [tdmdal/rl-hedge-2019](https://github.com/tdmdal/rl-hedge-2019), with necessary modifications for this coursework. The repository includes scripts for training, testing, and evaluating the models under different financial market conditions.  

## 📂 Project Structure

```bash
deep_hedging_new/
│── config.py               # Configuration file with hyperparameters
│── ddpg.py                 # DDPG algorithm implementation
│── sac_train_origin.py     # SAC algorithm training
│── sac_test.py             # SAC model evaluation
│── rl_train.py             # DDPG training script
│── rl_test.py              # DDPG evaluation script
│── envs.py                 # Hedging environment (based on rl-hedge-2019)
│── schedules.py            # Scheduling utilities (based on rl-hedge-2019)
│── replay_buffer.py        # Prioritized Experience Replay (based on rl-hedge-2019)
│── segment_tree.py         # Data structure for replay buffer (based on rl-hedge-2019)
│── utils.py                # Utility functions (based on rl-hedge-2019)
│── experimentmanager.py    # Manages experiment data and logging
│── gym_test.py             # Algorithm validation script
│── requirements.txt        # Dependency list for setting up the environment
└── .gitignore              # Git ignore file
```
