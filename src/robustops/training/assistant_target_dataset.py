"""Select assistant targets while preserving full multi-turn SFT context."""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Any

import torch
import torch.nn.functional as F

from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
from verl.utils.dataset.dataset_utils import DatasetPadMode


EXPECTED_LOSS_SCOPE = "selected_assistant_tool_call_turns"


class AssistantTargetDatasetError(ValueError):
    """A generated assistant-target SFT row cannot satisfy the training loss contract."""


def _decode_json_column(value: Any, *, column: str) -> Any:
    if not isinstance(value, str):
        raise AssistantTargetDatasetError(f"{column} must contain a JSON string")
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise AssistantTargetDatasetError(f"{column} contains invalid JSON") from exc


def validate_assistant_target_indices(
    messages: list[dict[str, Any]],
    supervised: list[int],
    masked: list[int],
) -> frozenset[int]:
    """Validate the assistant-target SFT message-level supervision partition."""
    if not messages or not all(isinstance(message, dict) for message in messages):
        raise AssistantTargetDatasetError("messages must be a non-empty list of objects")
    for name, indices in (("supervised", supervised), ("masked", masked)):
        if (
            not isinstance(indices, list)
            or any(not isinstance(index, int) or isinstance(index, bool) for index in indices)
            or indices != sorted(set(indices))
        ):
            raise AssistantTargetDatasetError(f"{name} assistant indices must be sorted unique integers")
    if not supervised:
        raise AssistantTargetDatasetError("each assistant-target SFT record must supervise at least one assistant turn")
    if set(supervised) & set(masked):
        raise AssistantTargetDatasetError("supervised and masked assistant indices overlap")
    if supervised[-1] != len(messages) - 1:
        raise AssistantTargetDatasetError("the final assistant-target SFT message must be a supervised target")

    indexed = set(supervised) | set(masked)
    tool_assistant_indices: set[int] = set()
    for index, message in enumerate(messages):
        tool_calls = message.get("tool_calls")
        if message.get("role") == "assistant" and isinstance(tool_calls, list) and tool_calls:
            tool_assistant_indices.add(index)
        if index in indexed and (
            message.get("role") != "assistant"
            or not isinstance(tool_calls, list)
            or not tool_calls
        ):
            raise AssistantTargetDatasetError(
                f"assistant target index {index} does not contain a tool call"
            )
    if indexed != tool_assistant_indices:
        raise AssistantTargetDatasetError(
            "every assistant tool-call turn must be explicitly supervised or masked"
        )
    return frozenset(supervised)


def _select_padding_source_item(
    dataframe: Any,
    *,
    messages_key: str,
    tools_key: str,
    real_length: int,
) -> int:
    """Choose a deterministic low-cost row for zero-loss sampler padding."""
    return min(
        range(real_length),
        key=lambda item: (
            len(dataframe.iloc[item][messages_key])
            + len(dataframe.iloc[item][tools_key]),
            item,
        ),
    )


class SelectedAssistantTargetsSFTDataset(MultiTurnSFTDataset):
    """Decode assistant-target SFT rows and supervise only the reviewed assistant targets."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        config = kwargs.get("config")
        if config is None and len(args) >= 3:
            config = args[2]
        pad_to_multiple = config.get("pad_to_multiple", 1) if config is not None else 1
        if (
            not isinstance(pad_to_multiple, int)
            or isinstance(pad_to_multiple, bool)
            or pad_to_multiple <= 0
        ):
            raise AssistantTargetDatasetError("pad_to_multiple must be a positive integer")
        self._pad_to_multiple = pad_to_multiple
        super().__init__(*args, **kwargs)
        self._real_length = len(self.messages)
        if self._real_length == 0:
            raise AssistantTargetDatasetError("assistant-target SFT dataset must contain at least one record")
        self._padded_length = (
            (self._real_length + self._pad_to_multiple - 1)
            // self._pad_to_multiple
            * self._pad_to_multiple
        )
        # Tool catalogs dominate token count for EnterpriseOps records, so the
        # number of conversation messages is a poor proxy for padding cost.
        # The generated columns are canonical JSON strings; their combined
        # serialized length is a cheap, deterministic proxy that avoids an
        # eager tokenizer pass over the whole dataset during construction.
        self._padding_source_item = _select_padding_source_item(
            self.dataframe,
            messages_key=self.messages_key,
            tools_key=self.tools_key,
            real_length=self._real_length,
        )

    def __len__(self) -> int:
        if hasattr(self, "_padded_length"):
            return self._padded_length
        return super().__len__()

    def _read_files_and_process(self) -> None:
        super()._read_files_and_process()
        required = {
            "loss_scope",
            self.messages_key,
            self.tools_key,
            "supervised_assistant_message_indices_json",
            "masked_assistant_message_indices_json",
        }
        missing = sorted(required - set(self.dataframe.columns))
        if missing:
            raise AssistantTargetDatasetError(f"assistant-target SFT Parquet is missing columns: {missing}")
        scopes = set(self.dataframe["loss_scope"].tolist())
        if scopes != {EXPECTED_LOSS_SCOPE}:
            raise AssistantTargetDatasetError(
                f"unexpected assistant-target SFT loss_scope values: {sorted(map(str, scopes))}"
            )

        self.messages = [
            _decode_json_column(value, column=self.messages_key)
            for value in self.dataframe[self.messages_key].tolist()
        ]
        self.tools = [
            _decode_json_column(value, column=self.tools_key)
            for value in self.dataframe[self.tools_key].tolist()
        ]
        supervised_rows = [
            _decode_json_column(
                value,
                column="supervised_assistant_message_indices_json",
            )
            for value in self.dataframe[
                "supervised_assistant_message_indices_json"
            ].tolist()
        ]
        masked_rows = [
            _decode_json_column(
                value,
                column="masked_assistant_message_indices_json",
            )
            for value in self.dataframe[
                "masked_assistant_message_indices_json"
            ].tolist()
        ]
        if not all(isinstance(tools, list) and tools for tools in self.tools):
            raise AssistantTargetDatasetError("each assistant-target SFT row must expose at least one tool")
        self._supervised_rows = [
            validate_assistant_target_indices(messages, supervised, masked)
            for messages, supervised, masked in zip(
                self.messages,
                supervised_rows,
                masked_rows,
                strict=True,
            )
        ]

    def _build_messages(self, example: dict[str, Any]) -> list[dict[str, Any]]:
        messages = _decode_json_column(example[self.messages_key], column=self.messages_key)
        return deepcopy(messages)

    def _apply_full_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
        enable_thinking: bool | None,
        add_generation_prompt: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize one exact Qwen conversation prefix.

        Qwen groups consecutive tool observations inside one user wrapper.  The
        upstream per-message SFT tokenizer cannot preserve that grouping, so
        assistant-target SFT must render the whole prefix at each assistant boundary.
        """
        processor = self.processor if self.processor is not None else self.tokenizer
        kwargs = {**self.apply_chat_template_kwargs}
        if enable_thinking is not None:
            kwargs["enable_thinking"] = enable_thinking
        inputs = processor.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            **kwargs,
        )
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        if input_ids.ndim != 1 or input_ids.shape != attention_mask.shape:
            raise AssistantTargetDatasetError("chat template returned invalid text tensor shapes")
        return input_ids, attention_mask

    def _build_exact_training_item(
        self,
        item: int,
    ) -> dict[str, torch.Tensor]:
        row = self.dataframe.iloc[item].to_dict()
        messages = self._build_messages(row)
        tools = self.tools[item]
        enable_thinking = (
            self.enable_thinking[item]
            if self.enable_thinking is not None
            else self.enable_thinking_default
        )
        if enable_thinking is not None:
            enable_thinking = bool(enable_thinking)

        input_ids, attention_mask = self._apply_full_template(
            messages,
            tools=tools,
            enable_thinking=enable_thinking,
            add_generation_prompt=False,
        )
        loss_mask = torch.zeros_like(attention_mask)
        for assistant_index in self._supervised_rows[item]:
            generation_prefix, _ = self._apply_full_template(
                messages[:assistant_index],
                tools=tools,
                enable_thinking=enable_thinking,
                add_generation_prompt=True,
            )
            assistant_prefix, _ = self._apply_full_template(
                messages[: assistant_index + 1],
                tools=tools,
                enable_thinking=enable_thinking,
                add_generation_prompt=False,
            )
            generation_end = generation_prefix.numel()
            assistant_end = assistant_prefix.numel()
            if generation_end >= assistant_end:
                raise AssistantTargetDatasetError(
                    f"assistant target {assistant_index} has no completion tokens"
                )
            if not torch.equal(input_ids[:generation_end], generation_prefix):
                raise AssistantTargetDatasetError(
                    f"assistant target {assistant_index} generation prefix is unstable"
                )
            if not torch.equal(input_ids[:assistant_end], assistant_prefix):
                raise AssistantTargetDatasetError(
                    f"assistant target {assistant_index} completion prefix is unstable"
                )
            loss_mask[generation_end:assistant_end] = 1

        sequence_length = input_ids.numel()
        position_ids = torch.arange(sequence_length, dtype=torch.long)
        if self.pad_mode == DatasetPadMode.RIGHT:
            if sequence_length < self.max_length:
                padding = self.max_length - sequence_length
                pad_token_id = self.tokenizer.pad_token_id or 0
                input_ids = F.pad(input_ids, (0, padding), value=pad_token_id)
                attention_mask = F.pad(attention_mask, (0, padding), value=0)
                loss_mask = F.pad(loss_mask, (0, padding), value=0)
                position_ids = F.pad(position_ids, (0, padding), value=0)
            elif sequence_length > self.max_length:
                if self.truncation == "error":
                    raise ValueError(f"{sequence_length=} is larger than {self.max_length=}")
                if self.truncation == "left":
                    input_ids = input_ids[-self.max_length :]
                    attention_mask = attention_mask[-self.max_length :]
                    loss_mask = loss_mask[-self.max_length :]
                    position_ids = position_ids[-self.max_length :]
                elif self.truncation == "right":
                    input_ids = input_ids[: self.max_length]
                    attention_mask = attention_mask[: self.max_length]
                    loss_mask = loss_mask[: self.max_length]
                    position_ids = position_ids[: self.max_length]
                else:
                    raise ValueError(f"Unknown truncation method {self.truncation}")
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "loss_mask": loss_mask,
            }
        if self.pad_mode != DatasetPadMode.NO_PADDING:
            raise ValueError(f"Unknown pad mode {self.pad_mode}")
        if sequence_length > self.max_length:
            if self.truncation == "error":
                raise ValueError(f"{sequence_length=} is larger than {self.max_length=}")
            if self.truncation != "right":
                raise AssistantTargetDatasetError(
                    "assistant-target SFT no-padding data only supports right truncation or error"
                )
            input_ids = input_ids[: self.max_length]
            loss_mask = loss_mask[: self.max_length]
            position_ids = position_ids[: self.max_length]
        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        if item < 0 or item >= len(self):
            raise IndexError(item)
        is_padding = item >= self._real_length
        source_item = self._padding_source_item if is_padding else item
        result = self._build_exact_training_item(source_item)
        if is_padding:
            # veRL normalizes SFT loss by the global loss-mask token count.  A
            # zero-mask row satisfies the even sampler contract without adding
            # any duplicate supervision.
            result["loss_mask"] = torch.zeros_like(result["loss_mask"])
        return result
