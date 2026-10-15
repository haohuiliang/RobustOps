"""Shared SFT trainer setup, complete validation batches, and sampler control."""

from __future__ import annotations

from pathlib import Path
import sys

from hydra import compose, initialize_config_dir

import verl.trainer.sft_trainer as verl_sft
from verl.utils.device import auto_set_device
from verl.utils.distributed import destroy_global_process_group, initialize_global_process_group


class LengthPairedSFTTrainer(verl_sft.SFTTrainer):
    """Use the reviewed Parquet order instead of reshuffling length pairs."""

    def _build_dataloader(self) -> None:
        from robustops.training.full_validation import FullValidationDataset

        if self.val_dataset is not None:
            self.val_dataset = FullValidationDataset(
                self.val_dataset, int(self.config.data.train_batch_size)
            )
        super()._build_dataloader()
        if not bool(self.config.data.get("sampler_shuffle", True)):
            self.train_sampler.shuffle = False


def compose_config(cli_args: list[str]):
    config_dir = Path(verl_sft.__file__).resolve().parent / "config"
    overrides = [value for value in cli_args if not value.startswith("hydra.run.dir=")]
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        return compose(config_name="sft_trainer_engine", overrides=overrides)


def main() -> None:
    config = compose_config(sys.argv[1:])
    auto_set_device(config)
    initialize_global_process_group()
    try:
        trainer = LengthPairedSFTTrainer(config=config)
        trainer.fit()
    finally:
        destroy_global_process_group()


if __name__ == "__main__":
    main()
