"""
training script
"""
import os
import click
from pytorch_lightning.loggers import WandbLogger
from config_parser import (
    TrainingPipelineParams,
    read_training_pipeline_params,
)
from transforms import init_tranforms


def train_pipeline(config: TrainingPipelineParams):
    os.makedirs("runs", exist_ok=True)
    train_transforms, test_transfroms = init_tranforms(config)


@click.command(name="train_pipeline")
@click.argument("config_path")
def parse_config(config_path: str):
    params = read_training_pipeline_params(config_path)
    train_pipeline(params)


wandb_logger = WandbLogger(project="my-test-project")

if __name__ == "__main__":
    wandb_logger = WandbLogger(project="made_cv_facial_landmarks")
    parse_config()
