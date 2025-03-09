import numpy as np
import pickle
import os

class QLearning:
    def __init__(self, env, args):
        self.env = env
        self.lr = args.lr
        self.gamma = args.gamma
        self.epsilon = args.q_epsilon
        self.epsilon_decay = args.q_epsilon_decay
        self.epsilon_min = args.q_epsilon_min
        self.q_table = {}

    def discretize_state(self, state):
        """ 对连续状态离散化，提高精度减少冲突 """
        return tuple(np.round(state, decimals=3))  # 提高精度，减少离散化误差

    def act(self, state, eval_mode=False):
        state_tuple = self.discretize_state(state)

        if state_tuple not in self.q_table:
            self.q_table[state_tuple] = np.zeros(self.env.action_space.n)

        if eval_mode or np.random.rand() > self.epsilon:
            max_q = np.max(self.q_table[state_tuple])
            best_actions = np.where(self.q_table[state_tuple] == max_q)[0]
            action = np.random.choice(best_actions)  # 处理多个最优动作
        else:
            action = np.random.randint(self.env.action_space.n)

        return action

    def learn(self, state, action, reward, next_state, done):
        state_tuple = self.discretize_state(state)
        next_state_tuple = self.discretize_state(next_state)

        if state_tuple not in self.q_table:
            self.q_table[state_tuple] = np.zeros(self.env.action_space.n)
        if next_state_tuple not in self.q_table:
            self.q_table[next_state_tuple] = np.zeros(self.env.action_space.n)

        best_next_action = np.argmax(self.q_table[next_state_tuple])
        td_target = reward if done else reward + self.gamma * self.q_table[next_state_tuple][best_next_action]
        td_error = td_target - self.q_table[state_tuple][action]
        self.q_table[state_tuple][action] += self.lr * td_error

        self.decay_epsilon()  # 让 epsilon 逐渐衰减

    def decay_epsilon(self):
        """ ϵ 只在 episode 结束时衰减 """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.q_table, f)
        print(f"Q-table saved to {path}, size: {len(self.q_table)} states")

    def load(self, path):
        if os.path.exists(path):
            with open(path, "rb") as f:
                self.q_table = pickle.load(f)
            print(f"Q-table loaded from {path}, size: {len(self.q_table)} states")
        else:
            print(f"File {path} does not exist!")