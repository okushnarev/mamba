import wandb
from torch.utils.tensorboard import SummaryWriter
import os


class UnifiedLogger:
    def __init__(self, config, project_name="dreamer-project"):
        """
        Initializes a logger that writes to both wandb and TensorBoard.

        Args:
            config: A configuration object/dict with logging parameters.
            project_name (str): The name of the project for wandb.
        """
        self.config = config

        # Initialize wandb
        if self.config.USE_WANDB:
            wandb.init(
                project=project_name,
                dir=os.path.join(config.LOG_FOLDER, 'wandb'),
            )

        # Initialize TensorBoard SummaryWriter
        tb_log_dir = os.path.join(config.LOG_FOLDER, 'tb', self.config.EXP_NAME)
        self.writer = SummaryWriter(log_dir=tb_log_dir)
        self.current_step = 0
        print(f"TensorBoard logs will be saved to: {tb_log_dir}")

    def log(self, data_dict, step=None):
        """
        Logs a dictionary of metrics to both wandb and TensorBoard.

        Args:
            data_dict (dict): A dictionary of {'metric_name': value}.
            step (int): The current step or epoch to log against.
        """

        step = step or self.current_step
        # Log to wandb
        if self.config.USE_WANDB:
            wandb.log(data_dict)

        # Log to TensorBoard
        for key, value in data_dict.items():
            if hasattr(value, 'item'):  # Check if it's a tensor
                value = value.item()
            self.writer.add_scalar(key, value, global_step=step)

    def define_metric(self, metric_name, step_metric=None):
        """
        Mirrors wandb.define_metric. TensorBoard doesn't have an exact equivalent,
        but wandb uses it to set the x-axis. We'll just call the wandb function.
        """
        if self.config.USE_WANDB:
            wandb.define_metric(metric_name, step_metric=step_metric)

    def close(self):
        """
        Close the writers.
        """
        if self.config.USE_WANDB:
            wandb.finish()
        self.writer.close()
