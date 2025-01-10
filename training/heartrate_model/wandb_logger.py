'''
Generic Weights&Biases logging file
'''

import wandb
import torch


class WandBLogger:
    def __init__(self, project_name, config):
        """
        Initialize a W&B run.
        :param project_name: Name of the W&B project.
        :param config: Dictionary of hyperparameters.
        """
        self.run = wandb.init(project=project_name, config=config)

    def log_metrics(self, metrics, step=None):
        """
        Log metrics to W&B.
        :param metrics: Dictionary of metrics.
        :param step: Training step or epoch.
        """
        wandb.log(metrics, step=step)

    def save_model(self, model, epoch):
        """
        Save model weights to W&B.
        :param model: PyTorch model.
        :param epoch: Current epoch.
        """
        model_path = f"model_epoch_{epoch}.pth"
        torch.save(model.state_dict(), model_path)
        wandb.save(model_path)

    def finish(self):
        """Finish the W&B run."""
        wandb.finish()
