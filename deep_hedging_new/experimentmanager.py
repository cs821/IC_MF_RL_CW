import os
import time
import json
import torch
import pandas as pd
from matplotlib import pyplot as plt


class ExperimentManager:
    def __init__(self, config):
        self.config = config
        self.exp_time = time.strftime("%Y-%m-%d_%H-%M-%S")

        # Select the storage path according to sabr_flag
        self.sub_dir = "sabr" if config.sabr_flag else "br"
        self.exp_dir = f"{config.algo}_experiments/{self.sub_dir}/{config.algo}_freq_{config.trade_freq}_{self.exp_time}"
        self.checkpoint_dir = f"{self.exp_dir}/checkpoints"

        # Create folder
        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # save config
        with open(f"{self.exp_dir}/config.json", "w") as f:
            json.dump(vars(config), f, indent=4)

    def save_history(self, history):
        df = pd.DataFrame(history)
        df.to_csv(f"{self.exp_dir}/training_history.csv", index=False)
        print("Traning History has been saved!")

    def save_model(self, model_state, filename):
        path = f"{self.exp_dir}/{filename}"
        torch.save(model_state, path)
        print(f"Model has been saved to :{path}")

    def save_checkpoint(self, model, filename, episode):
        """Save a checkpoint every xx rounds"""
        path = f"{self.checkpoint_dir}/{filename}_ep{episode}.pth"
        torch.save(model, path)
        print(f"Checkpoint saved to:{path}")

    def save_image(self,history):
        train_data = pd.DataFrame(history)
        for column in train_data.columns[1:]:
            save_path = f"{self.exp_dir}/plot_{column}.pdf"
            plt.figure(figsize=(8, 5))
            plt.plot(train_data['episode'], train_data[column], marker='o', linestyle='-')
            plt.xlabel('Episode')
            plt.ylabel(column)
            plt.title(f'Episode vs {column}')
            plt.grid(True)
            plt.savefig(save_path)
            plt.close() 