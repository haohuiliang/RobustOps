"""Validate, publish, rotate and export complete training checkpoints."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from robustops.training.recovery_rl_core import write_json


def write_checkpoint_status(path, value):
    path = Path(path)
    temporary = path.with_suffix('.writing')
    write_json(temporary, value)
    temporary.replace(path)


def checkpoint_world_size(path, *, expected_world_size=None):
    """Require a complete, consistent rank set without loading model or optimizer tensors."""
    import re
    path = Path(path)
    world_size = None
    for kind in ('model', 'optim', 'extra_state'):
        shards = list((path / 'actor').glob(f'{kind}_world_size_*.pt'))
        parsed = [re.fullmatch(rf'{kind}_world_size_(\d+)_rank_(\d+)\.pt', p.name) for p in shards]
        sizes = {int(m[1]) for m in parsed if m is not None}
        ranks = {int(m[2]) for m in parsed if m is not None}
        if (not shards or any(m is None for m in parsed) or len(sizes) != 1
                or next(iter(sizes)) <= 0 or ranks != set(range(next(iter(sizes))))
                or len(shards) != next(iter(sizes))
                or any(not p.is_file() or p.stat().st_size == 0 for p in shards)):
            raise RuntimeError(f'incomplete {kind} checkpoint: {path}')
        size = next(iter(sizes))
        if world_size is not None and size != world_size:
            raise RuntimeError(f'inconsistent {kind} checkpoint world size: {path}')
        world_size = size
    config = path / 'actor/fsdp_config.json'
    if config.is_file() and json.loads(config.read_text())['world_size'] != world_size:
        raise RuntimeError(f'checkpoint FSDP world size differs from rank files: {path}')
    if expected_world_size is not None and world_size != expected_world_size:
        raise RuntimeError(f'Checkpoint has {world_size} GPU ranks; selected {expected_world_size}. '
                           'Resume with the same GPU count, or start from an HF export with a new optimizer.')
    return world_size


def validate_resumable_checkpoint(path, *, expected_world_size=None):
    path = Path(path)
    world_size = checkpoint_world_size(path, expected_world_size=expected_world_size)
    required = [path / 'data.pt', path / 'recovery_state.json']
    if any(not p.is_file() or p.stat().st_size == 0 for p in required):
        raise RuntimeError(f'incomplete resumable checkpoint: {path}')
    state = json.loads((path / 'recovery_state.json').read_text())
    if state['step'] != int(path.name.removeprefix('global_step_')):
        raise RuntimeError(f'checkpoint cursor mismatch: {path}')
    return world_size


def checkpoint_generations(root):
    return sorted((p for p in Path(root).glob('global_step_*')
                   if p.is_dir() and p.name.removeprefix('global_step_').isdigit()),
                  key=lambda p: int(p.name.removeprefix('global_step_')))


def checkpoint_milestone(recovery, step):
    """Save configured milestones even between periodic checkpoints."""
    encoded = recovery.get('checkpoint_state_json')
    if not encoded:
        return False
    checkpoint = json.loads(encoded)
    return step in checkpoint.get('hf_export_steps', [])


def finish_checkpoint(directory, step, *, max_to_keep=None):
    """Publish a complete generation before rotating older checkpoints."""
    if max_to_keep is not None and (type(max_to_keep) is not int or max_to_keep <= 0):
        raise ValueError("max_to_keep must be null or a positive integer")
    directory = Path(directory)
    current = directory / f'global_step_{step}'
    validate_resumable_checkpoint(current)
    marker = directory / 'latest_checkpointed_iteration.txt'
    temporary = marker.with_suffix('.writing')
    temporary.write_text(str(step))
    temporary.replace(marker)
    if max_to_keep is None:
        return
    generations = {int(p.name.removeprefix('global_step_'))
                   for p in checkpoint_generations(directory)
                   if int(p.name.removeprefix('global_step_')) <= step}
    retained = set(sorted(generations)[-max_to_keep:])
    for old in checkpoint_generations(directory):
        old_step = int(old.name.removeprefix('global_step_'))
        if old_step < step and old_step not in retained:
            shutil.rmtree(old)


def save_after_training_failure(trainer, dataloader_state, error, *, completed_step,
                                phase='safe', skip_reason=None, reuse_only=False):
    """Best-effort save at a verified commit boundary, preserving the error."""
    failed_step = getattr(trainer, 'global_steps', 0)
    cfg = trainer.config.recovery
    record = dict(failed_step=failed_step, saved_step=completed_step,
                  original_error=str(error), status='saving', phase=phase)
    status_path = Path(cfg.output)/'emergency_checkpoints'/f'{failed_step}.json'
    try:
        if skip_reason:
            record.update(status='skipped', reason=skip_reason)
            return
        if completed_step <= 0:
            record.update(status='skipped', reason='no_completed_update')
            return
        checkpoint = json.loads(cfg.checkpoint_state_json)
        directory = Path(checkpoint['directory'])
        existing = directory / f'global_step_{completed_step}'
        marker = directory / 'latest_checkpointed_iteration.txt'
        published = marker.is_file() and int(marker.read_text()) == completed_step
        # After a write error only a published generation can be reused.
        if existing.exists() and (not reuse_only or published):
            try:
                validate_resumable_checkpoint(existing)
            except RuntimeError:
                pass
            else:
                finish_checkpoint(directory, completed_step,
                    max_to_keep=checkpoint.get('max_to_keep'))
                record.update(status='already_saved', path=str(existing))
                return
        if reuse_only:
            record.update(status='skipped', reason='checkpoint_write_failed_no_retry')
            return
        # This file is the last committed rollout's immutable planner,
        # reserve and replay state, not the manager mutated by the failed step.
        state = json.loads((Path(cfg.output)/'recovery_states'/f'{completed_step}.json').read_text())
        if state['step'] != completed_step or state['next_visit'] != completed_step * 32:
            raise RuntimeError('emergency checkpoint recovery cursor mismatch')
        if dataloader_state is None:
            raise RuntimeError('missing pre-rollout dataloader state')
        print(f'RECOVERY_EMERGENCY_SAVE_BEGIN failed_step={failed_step} saved_step={completed_step}', flush=True)
        trainer.checkpoint_manager.sleep_replicas()
        trainer.global_steps = completed_step
        trainer._save_checkpoint(dataloader_state=dataloader_state, export_model=False)
        if int(marker.read_text()) != completed_step:
            raise RuntimeError('emergency checkpoint was not published')
        validate_resumable_checkpoint(existing)
        record.update(status='saved', path=str(existing))
    except Exception as save_error:
        record.update(status='failed', save_error=repr(save_error))
    finally:
        trainer.global_steps = failed_step
        print('RECOVERY_EMERGENCY_SAVE '+json.dumps(record), flush=True)
        try:
            write_json(status_path, record)
        except Exception as log_error:
            print(f'RECOVERY_EMERGENCY_LOG_FAILED {log_error!r}', flush=True)


def export_full_model(checkpoint, output, step):
    """Synchronous CPU merge to a standalone HF model; survives checkpoint rotation."""
    checkpoint, output = Path(checkpoint), Path(output)
    validate_resumable_checkpoint(checkpoint)
    marker = output/'model_exports'/f'{step}.json'
    if marker.exists():
        model = Path(json.loads(marker.read_text())['path'])
        validate_hf_export(model)
        return model
    target_root = output / 'models'
    target_root.mkdir(parents=True, exist_ok=True)
    target = target_root/f'global_step_{step}'
    partial = target.with_name(target.name+'.partial')
    if partial.exists() or target.exists():
        raise FileExistsError(f'unpublished HF export retained for inspection: {target}')
    partial.mkdir()
    command = [sys.executable, '-m', 'verl.model_merger', 'merge', '--backend', 'fsdp',
               '--local_dir', str(checkpoint/'actor'), '--target_dir', str(partial), '--output-dtype', 'float32']
    started = time.monotonic()
    output.joinpath('model_exports').mkdir(parents=True, exist_ok=True)
    print(f'RECOVERY_HF_EXPORT_BEGIN step={step} path={target}', flush=True)
    try:
        with (output/'model_exports'/f'{step}.log').open('wb') as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT,
                           env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), timeout=7200)
        details = validate_hf_export(partial)
        write_json(partial/'MODEL_PROVENANCE.json', dict(step=step, source_checkpoint=str(checkpoint),
            format='HuggingFace safetensors', weight_dtype='float32', full_model=True,
            resumable=False, **details))
        partial.rename(target)
        write_checkpoint_status(marker, dict(step=step, path=str(target), source_checkpoint=str(checkpoint),
            format='HuggingFace safetensors', weight_dtype='float32', full_model=True,
            seconds=time.monotonic()-started, **details))
    except BaseException as exc:
        write_json(output/'model_exports'/f'{step}_failed.json', dict(error=repr(exc), partial=str(partial)))
        raise
    print(f'RECOVERY_HF_EXPORT_COMPLETE step={step} path={target}', flush=True)
    return target


def validate_hf_export(model, *, allowed_dtypes=('F32',)):
    """Read only safetensors headers, index and tokenizer metadata, never load weights."""
    import struct
    model = Path(model)
    for name in ('config.json','tokenizer.json','tokenizer_config.json'):
        if not (model/name).is_file():
            raise ValueError(f'missing HF component: {model/name}')
    tensors, files = {}, sorted(model.glob('*.safetensors'))
    if not files:
        raise ValueError('HF export has no safetensors weights')
    for file in files:
        with file.open('rb') as stream:
            n = struct.unpack('<Q',stream.read(8))[0]
            header = json.loads(stream.read(n))
        values = {k:v for k,v in header.items() if k!='__metadata__'}
        if not values or 8+n+max(v['data_offsets'][1] for v in values.values())!=file.stat().st_size:
            raise ValueError(f'incomplete safetensors: {file}')
        if any(v['dtype'] not in allowed_dtypes for v in values.values()):
            raise ValueError(f'expected preserved weights with dtype {allowed_dtypes}: {file}')
        if set(tensors)&set(values):
            raise ValueError('duplicate exported tensors')
        tensors.update({key:file.name for key in values})
    index = model/'model.safetensors.index.json'
    if index.exists() and json.loads(index.read_text())['weight_map']!=tensors:
        raise ValueError('HF index/weight shard mismatch')
    if len(files)>1 and not index.exists():
        raise ValueError('sharded HF export requires its index')
    return dict(tensor_count=len(tensors), weight_bytes=sum(p.stat().st_size for p in files))
