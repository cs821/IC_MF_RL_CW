import os
import time
import json
import torch
import pandas as pd

class ExperimentManager:
    def __init__(self, config):
        self.config = config
        self.exp_time = time.strftime("%Y-%m-%d_%H-%M-%S")

        # 根据 sabr_flag 选择存储路径
        self.sub_dir = "sabr" if config.sabr_flag else "br"
        self.exp_dir = f"{config.algo}_experiments/{self.sub_dir}/{config.algo}_{self.exp_time}"
        self.checkpoint_dir = f"{self.exp_dir}/checkpoints"

        # 创建文件夹
        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # 保存 config
        with open(f"{self.exp_dir}/config.json", "w") as f:
            json.dump(vars(config), f, indent=4)

    def save_history(self, history):
        df = pd.DataFrame(history)
        df.to_csv(f"{self.exp_dir}/training_history.csv", index=False)
        print("训练历史已保存！")

    def save_model(self, model_state, filename):
        path = f"{self.exp_dir}/{filename}"
        torch.save(model_state, path)
        print(f"模型已保存：{path}")

    def save_checkpoint(self, model, filename, episode):
        """每 xx 轮保存一次 checkpoint"""
        path = f"{self.checkpoint_dir}/{filename}_ep{episode}.pth"
        torch.save(model, path)
        print(f"检查点已保存：{path}")

