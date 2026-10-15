"""SFT trainer with a fixed batch denominator for source-weighted weights."""

from __future__ import annotations

from functools import partial

import torch
from tensordict import TensorDict

from robustops.training.sft_base_trainer import (
    LengthPairedSFTTrainer,
    compose_config,
)
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.device import auto_set_device
from verl.utils.distributed import destroy_global_process_group, initialize_global_process_group
from verl.utils.torch_functional import masked_sum
from verl.workers.engine_workers import TrainingWorker, TrainingWorkerConfig


def decision_sft_loss(
    config: object,
    model_output: dict[str, torch.Tensor],
    data: TensorDict,
    dp_group: object | None = None,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Average weighted event losses by the fixed global batch size.

    veRL's stock SFT loss divides by ``loss_mask.sum()`` for each batch.  That
    would cancel or distort EXP-0018's per-source row masses as batch source
    composition changes.  This loss instead treats the mask as true token
    weights and divides by the configured global record batch size.
    """
    del config, dp_group
    pad_mode = tu.get_non_tensor_data(
        data=data,
        key="pad_mode",
        default=DatasetPadMode.NO_PADDING,
    )
    if pad_mode != DatasetPadMode.NO_PADDING:
        raise ValueError("decision-weighted SFT loss requires no_padding")
    global_batch_size = tu.get_non_tensor_data(
        data=data, key="global_batch_size", default=None
    )
    if not isinstance(global_batch_size, int) or global_batch_size <= 0:
        raise ValueError("global_batch_size must be a positive integer")

    log_prob_flatten = model_output["log_probs"].values()
    loss_mask_flatten = data["loss_mask"].values()
    loss_mask_flatten = torch.roll(loss_mask_flatten, shifts=-1, dims=0)
    loss = (
        -masked_sum(log_prob_flatten, loss_mask_flatten)
        / global_batch_size
        * data["dp_size"]
    )
    return loss, {}


class DecisionSFTTrainer(LengthPairedSFTTrainer):
    """Preserve reviewed ordering and source/event weighting semantics."""

    def _build_engine(self) -> None:
        self.loss_fn = partial(decision_sft_loss, config=None)
        worker_config = TrainingWorkerConfig(
            model_type="language_model",
            model_config=self.model_config,
            engine_config=self.engine_config,
            optimizer_config=self.optimizer_config,
            checkpoint_config=self.checkpoint_config,
            profiler_config=self.profiler_config,
        )
        self.training_client = TrainingWorker(config=worker_config)
        self.training_client.set_loss_fn(loss_fn=self.loss_fn)
        self.engine = self.training_client.engine


def main() -> None:
    import sys

    config = compose_config(sys.argv[1:])
    auto_set_device(config)
    initialize_global_process_group()
    try:
        trainer = DecisionSFTTrainer(config=config)
        trainer.fit()
    finally:
        destroy_global_process_group()


if __name__ == "__main__":
    main()
