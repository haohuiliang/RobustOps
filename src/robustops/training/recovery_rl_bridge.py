"""Lightweight exact-token adapter, independently testable without veRL/torch."""
import math
import time
from uuid import uuid4

from robustops.evaluation.tool_protocol import TOOL_CALL_END, parse_tool_response


class TokenBridge:
    """Append observations to sampled prefixes; never re-tokenize generated actions.

    Stop at the first complete Hermes call, matching execute-first-then-reinfer.
    Assistant closing delimiters inserted by the controller have zero loss.
    """
    def __init__(self, tokenizer, generate, sampling, initial_verifier, max_tokens, seed):
        self.tokenizer, self.generate = tokenizer, generate
        self.sampling, self.verifier = sampling, initial_verifier
        self.max_tokens, self.seed = max_tokens, seed
        self.ids, self.prompt_ids, self.mask, self.logprobs = [], [], [], []
        self.turns = 0
        self.generated_seconds = 0.

    def restore_prefix(self, ids, turns):
        """A shared state is conditioning only; train exclusively new suffix tokens."""
        if self.ids or not ids or turns < 1:
            raise ValueError('invalid or already initialized boundary token state')
        self.ids = list(ids)
        self.prompt_ids = list(ids)
        self.turns = turns

    def text(self, content):
        return len(self.tokenizer.encode(content, add_special_tokens=False))

    def suffix(self, messages):
        if not messages or messages[-1]['role'] != 'tool':
            raise ValueError('continuing rollout requires a real tool observation')
        # The supported model template has a single user wrapper for tool output.
        tool = self.tokenizer.apply_chat_template([messages[-1]], tokenize=False, add_generation_prompt=True)
        end = self.tokenizer.convert_tokens_to_ids('<|im_end|>')
        close = '\n' if self.ids and self.ids[-1] == end else '<|im_end|>\n'
        return self.tokenizer.encode(close + tool, add_special_tokens=False)

    def context(self, messages, tools):
        if self.ids:
            return len(self.ids) + len(self.suffix(messages))
        return self.initial_context(messages, tools)

    def initial_context(self, messages, tools):
        """Original-task accounting, also before a saved trajectory is restored."""
        return len(self.tokenizer.apply_chat_template(messages, tools=tools or None,
                        tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False))

    async def complete(self, messages, tools):
        await self.verifier.measure()
        if not self.ids:
            self.prompt_ids = self.tokenizer.apply_chat_template(messages, tools=tools,
                                tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False)
            self.ids = list(self.prompt_ids)
        else:
            suffix = self.suffix(messages)
            self.ids.extend(suffix)
            self.mask.extend([0] * len(suffix))
            self.logprobs.extend([0.] * len(suffix))
        params = dict(self.sampling, max_tokens=self.max_tokens, seed=self.seed + self.turns,
                      stop=[TOOL_CALL_END], include_stop_str_in_output=True)
        started = time.monotonic()
        output = await self.generate(request_id=uuid4().hex, prompt_ids=list(self.ids), sampling_params=params)
        self.generated_seconds += time.monotonic() - started
        tokens = list(output.token_ids)
        if getattr(output, 'stop_reason', None) == 'aborted':
            raise RuntimeError('inference engine aborted the rollout; not a task failure')
        if not tokens or output.log_probs is None or len(tokens) != len(output.log_probs):
            raise ValueError('rollout must return real generated token IDs and log probabilities')
        if not all(math.isfinite(x) for x in output.log_probs):
            raise ValueError('non-finite rollout log probability')
        prompt_length = len(self.ids)
        self.ids.extend(tokens)
        self.mask.extend([1] * len(tokens))
        self.logprobs.extend(output.log_probs)
        self.turns += 1
        print(f'RECOVERY_TURN {self.turns} prompt={prompt_length} generated={len(tokens)}', flush=True)
        text = self.tokenizer.decode(tokens, skip_special_tokens=False)
        finish_reason = 'length' if len(tokens) >= self.max_tokens else 'stop'
        message = parse_tool_response(text, finish_reason=finish_reason, call_id=f'call_{self.turns}')
        return message, dict(
            prompt_tokens=prompt_length, completion_tokens=len(tokens),
            finish_reason=finish_reason)
