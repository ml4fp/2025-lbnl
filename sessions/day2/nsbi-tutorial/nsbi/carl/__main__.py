from lightning.pytorch.cli import LightningCLI

from models.carl import CARL
from datasets.balanced import BalancedDataModule

import torch
torch.set_float32_matmul_precision('high')

class CarlCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        # Link the other data arguments
        parser.link_arguments("data.features", "model.n_features", compute_fn=lambda x: len(x))
        parser.link_arguments("seed_everything", "data.random_state")

def main():
    logger_cfg = {
        "class_path": "lightning.pytorch.loggers.CSVLogger",
        "init_args": {"save_dir": "./"}
    }

    cli = CarlCLI(
        model_class=CARL,
        datamodule_class=BalancedDataModule,
        trainer_defaults={"logger": logger_cfg},
    )

if __name__ == "__main__":
    main()