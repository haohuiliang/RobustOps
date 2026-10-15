"""Fixed teacher-forced SFT validation on the post-update RL actor."""
from pathlib import Path
from time import perf_counter

import torch
from verl.utils import tensordict_utils as tu

from robustops.training.recovery_rl_core import write_json


def prepare_validation(tokenizer, path, *, max_length=32768):
    from robustops.training.decision_sft_dataset import DecisionSFTDataset
    from omegaconf import OmegaConf
    config = OmegaConf.create(dict(messages_key='messages_json', tools_key='tools_json',
        enable_thinking_key='enable_thinking', enable_thinking_default=False,
        pad_mode='no_padding', max_length=max_length, truncation='error'))
    dataset = DecisionSFTDataset(str(path), tokenizer, config)
    if not len(dataset):
        raise ValueError('SFT validation requires at least one record')
    return [dataset[i] for i in range(len(dataset))]


def validation_batch(items, multiple=2):
    if not items or multiple < 1:
        raise ValueError('Validation requires records and a positive worker count')
    # Padding keeps worker shards equal; reduce_validation excludes all pads.
    padded = items + [items[-1]] * (-len(items) % multiple)
    data = tu.get_tensordict(dict(
        input_ids=torch.nested.as_nested_tensor([r['input_ids'] for r in padded], layout=torch.jagged),
        position_ids=torch.nested.as_nested_tensor([r['position_ids'] for r in padded], layout=torch.jagged),
        loss_mask=torch.nested.as_nested_tensor([r['loss_mask'] for r in padded], layout=torch.jagged),
    ))
    tu.assign_non_tensor(data, temperature=1.0, calculate_entropy=False,
        calculate_sum_pi_squared=False, compute_loss=False, return_model_output=True, pad_mode='no_padding')
    return data


def reduce_validation(items, log_probs):
    if not items:
        raise ValueError('SFT validation requires at least one record')
    # Engine log_probs[t] scores input_ids[t+1]. Exclude the last prediction,
    # which points across a packed record boundary, and preserve event weights.
    losses = []
    for row, probs in zip(items, log_probs.unbind()[:len(items)], strict=True):
        weights = row['loss_mask']
        if len(probs) != len(weights) or weights[0] != 0:
            raise ValueError('validation token alignment differs from SFT')
        losses.append(float(-(probs[:-1].float() * weights[1:].float()).sum()))
    return dict(loss=sum(losses)/len(losses), 
                records=len(losses), row_losses=losses)


def evaluate_validation(trainer, step):
    start = perf_counter()
    cfg = trainer.config.recovery
    if not hasattr(trainer, '_recovery_val_items'):
        trainer._recovery_val_items = prepare_validation(
            trainer.tokenizer, cfg.sft_val_file, max_length=cfg.max_context_tokens)
    items = trainer._recovery_val_items
    output = trainer.actor_rollout_wg.compute_log_prob(validation_batch(
        items, multiple=trainer.actor_rollout_wg.world_size))
    result = reduce_validation(items, tu.get(output, 'log_probs'))
    result.update(step=step, seconds=perf_counter()-start, temperature=1.0,
                  source=str(cfg.sft_val_file), scope='post-update teacher-forced fixed SFT; no rollout')
    write_json(Path(cfg.output)/'val_loss'/f'{step}.json', result)
    return {'val/sft_loss': result['loss'],
            'val/sft_records': result['records'], 'val/sft_seconds': result['seconds']}
