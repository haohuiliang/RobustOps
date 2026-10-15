"""Event- and source-weighted SFT dataset for executed tool trajectories."""

from __future__ import annotations

import math

import torch
from verl.utils.dataset.dataset_utils import DatasetPadMode

from robustops.training.selective_sft_dataset import (
    SelectiveDatasetError,
    SelectiveTrajectoryNormalizedSFTDataset,
)
from robustops.training.selective_supervision import (
    DECISION_BALANCED_LOSS_SCOPE,
    build_selective_event_token_weights,
)
from robustops.training.sequence_length_contract import check_sequence_length


class DecisionSFTDataset(SelectiveTrajectoryNormalizedSFTDataset):
    """Normalize events within rows and Base/Teacher mass across one epoch."""

    expected_loss_scope = DECISION_BALANCED_LOSS_SCOPE
    token_weight_builder = staticmethod(build_selective_event_token_weights)

    def _read_files_and_process(self) -> None:
        super()._read_files_and_process()
        self._sequence_token_status = (
            self.dataframe['sequence_tokens_status'].tolist()
            if 'sequence_tokens_status' in self.dataframe.columns
            else ['verified'] * len(self.dataframe)
        )
        required = {"trajectory_loss_mass", "positive_decision_events"}
        missing = sorted(required - set(self.dataframe.columns))
        if missing:
            raise SelectiveDatasetError(
                f"decision-weighted SFT Parquet is missing columns: {missing}"
            )
        self._trajectory_loss_masses = [
            float(value) for value in self.dataframe["trajectory_loss_mass"].tolist()
        ]
        if any(
            not math.isfinite(value) or value <= 0.0
            for value in self._trajectory_loss_masses
        ):
            raise SelectiveDatasetError(
                "trajectory_loss_mass values must be finite and positive"
            )
        self._expected_positive_events = [
            int(value) for value in self.dataframe["positive_decision_events"].tolist()
        ]
        if any(value <= 0 for value in self._expected_positive_events):
            raise SelectiveDatasetError(
                "positive_decision_events values must be positive"
            )

    def _build_exact_training_item(self, item: int) -> dict[str, torch.Tensor]:
        if self.pad_mode != DatasetPadMode.NO_PADDING:
            raise SelectiveDatasetError(
                "decision-weighted SFT requires no_padding"
            )
        messages = self.messages[item]
        tools = self.tools[item]
        enable_thinking = (
            self.enable_thinking[item]
            if self.enable_thinking is not None
            else self.enable_thinking_default
        )
        template_kwargs = {**self.apply_chat_template_kwargs}
        if enable_thinking is not None:
            template_kwargs["enable_thinking"] = bool(enable_thinking)
        renderer = self.processor if self.processor is not None else self.tokenizer
        input_id_values, weight_values = self.token_weight_builder(
            renderer=renderer,
            tokenizer=self.tokenizer,
            messages=messages,
            tools=tools,
            policies=self._policy_rows[item],
            trajectory_mass=self._trajectory_loss_masses[item],
            template_kwargs=template_kwargs,
        )
        input_ids = torch.tensor(input_id_values, dtype=torch.long)
        loss_mask = torch.tensor(weight_values, dtype=torch.float32)
        sequence_tokens = int(input_ids.numel())
        positive_events = sum(
            policy["target"] != "masked" for policy in self._policy_rows[item]
        )
        if positive_events != self._expected_positive_events[item]:
            raise SelectiveDatasetError(
                f"positive event drift for row {item}: expected "
                f"{self._expected_positive_events[item]}, found {positive_events}"
            )
        try:
            check_sequence_length(sequence_tokens, self._expected_sequence_tokens[item],
                                  self.max_length, self._sequence_token_status[item])
        except ValueError as exc:
            raise SelectiveDatasetError(f'row {item}: {exc}') from exc
        if not torch.isclose(
            loss_mask.sum(),
            torch.tensor(self._trajectory_loss_masses[item], dtype=torch.float32),
            rtol=1e-5,
            atol=1e-6,
        ):
            raise SelectiveDatasetError(
                f"trajectory loss mass drift for row {item}"
            )
        return {
            "input_ids": input_ids,
            "position_ids": torch.arange(sequence_tokens, dtype=torch.long),
            "loss_mask": loss_mask,
        }
