"""Apply selective token supervision with per-trajectory normalization."""

from __future__ import annotations

import torch

from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset

from robustops.training.assistant_target_dataset import (
    SelectedAssistantTargetsSFTDataset,
    _decode_json_column,
)
from robustops.training.selective_supervision import (
    LOSS_SCOPE,
    SelectiveSupervisionError,
    build_selective_token_mask,
    validate_supervision_policy,
)


class SelectiveDatasetError(SelectiveSupervisionError):
    """A selective clean-alignment row violates its training contract."""


class SelectiveTrajectoryNormalizedSFTDataset(SelectedAssistantTargetsSFTDataset):
    """Keep full context while supervising only reviewed assistant spans."""

    expected_loss_scope = LOSS_SCOPE

    def _read_files_and_process(self) -> None:
        MultiTurnSFTDataset._read_files_and_process(self)
        required = {
            "loss_scope",
            self.messages_key,
            self.tools_key,
            "assistant_supervision_json",
            "sequence_tokens",
        }
        missing = sorted(required - set(self.dataframe.columns))
        if missing:
            raise SelectiveDatasetError(
                f"selective clean SFT Parquet is missing columns: {missing}"
            )
        scopes = set(self.dataframe["loss_scope"].tolist())
        if scopes != {self.expected_loss_scope}:
            raise SelectiveDatasetError(
                f"unexpected selective clean SFT loss_scope values: {sorted(scopes)}"
            )

        self.messages = [
            _decode_json_column(value, column=self.messages_key)
            for value in self.dataframe[self.messages_key].tolist()
        ]
        self.tools = [
            _decode_json_column(value, column=self.tools_key)
            for value in self.dataframe[self.tools_key].tolist()
        ]
        policies = [
            _decode_json_column(value, column="assistant_supervision_json")
            for value in self.dataframe["assistant_supervision_json"].tolist()
        ]
        if not all(isinstance(tools, list) and tools for tools in self.tools):
            raise SelectiveDatasetError(
                "each selective clean SFT row must expose at least one tool"
            )
        self._policy_rows = []
        for messages, policy in zip(self.messages, policies, strict=True):
            validate_supervision_policy(messages, policy)
            self._policy_rows.append(list(policy))
        self._expected_sequence_tokens = [
            int(value) for value in self.dataframe["sequence_tokens"].tolist()
        ]

    def _build_exact_training_item(self, item: int) -> dict[str, torch.Tensor]:
        if self.pad_mode != DatasetPadMode.NO_PADDING:
            raise SelectiveDatasetError(
                "selective trajectory-normalized SFT requires no_padding"
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
        input_id_values, mask_values = build_selective_token_mask(
            renderer=renderer,
            tokenizer=self.tokenizer,
            messages=messages,
            tools=tools,
            policies=self._policy_rows[item],
            template_kwargs=template_kwargs,
        )
        input_ids = torch.tensor(input_id_values, dtype=torch.long)
        raw_loss_mask = torch.tensor(mask_values, dtype=torch.float32)
        sequence_tokens = int(input_ids.numel())
        target_tokens = int(raw_loss_mask.sum().item())
        if sequence_tokens != self._expected_sequence_tokens[item]:
            raise SelectiveDatasetError(
                f"sequence token drift for row {item}: expected "
                f"{self._expected_sequence_tokens[item]}, found {sequence_tokens}"
            )
        if sequence_tokens > self.max_length:
            raise SelectiveDatasetError(
                f"sequence_length={sequence_tokens} exceeds max_length={self.max_length}"
            )
        if target_tokens <= 0:
            raise SelectiveDatasetError(
                f"row {item} contains no supervised target tokens"
            )
        position_ids = torch.arange(sequence_tokens, dtype=torch.long)
        # Unit mass per real row: no Base/Teacher or A/B source weighting.
        loss_mask = raw_loss_mask / target_tokens
        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }
