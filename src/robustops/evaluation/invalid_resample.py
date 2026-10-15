"""Bounded whole-attempt retries; model errors retain their original classification."""
import json
from datetime import datetime, timezone
from robustops.evaluation.tool_protocol import model_protocol_retry_reason

RETRY_SEED_STRIDE = 1000003


def resolved_path(folder):
    marker = folder / 'resolution.json'
    if not marker.exists():
        return folder / 'attempt.json'
    result = json.loads(marker.read_text())
    if result['selected']:
        return folder / result['selected']
    return None if result['skipped'] else folder / 'attempt.json'


async def resolve_attempt(folder, seed, row, sampling, generate, retries=4):
    """First valid, protocol-complete outcome wins, including business failure."""
    marker = folder / 'resolution.json'
    # A completed old run keeps its recorded selection policy on resume.
    # New runs retry model format failures as well as infrastructure invalids.
    retry_protocol = (not marker.exists()
                      or json.loads(marker.read_text()).get('retry_model_protocol_error', False))
    history = []
    for retry in range(retries + 1):
        current = folder if retry == 0 else folder / f'invalid_retry_{retry:02}'
        path = current / 'attempt.json'
        actual_seed = seed + retry * RETRY_SEED_STRIDE
        expected = dict(sampling, seed=actual_seed)
        if path.exists():
            attempt = json.loads(path.read_text())
        else:
            if current.exists():
                current.rename(current.with_name(current.name + '_incomplete_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')))
            attempt = await generate(current, actual_seed)
        if (attempt['task_id'], attempt['mode']) != (row['task_id'], row['mode']) or attempt['sampling'] != expected:
            raise ValueError(f'Resume identity or sampling mismatch: {path}')
        # Failed cleanup cannot safely be hidden by replacing a trajectory.
        if not attempt['cleanup']['succeeded']:
            raise RuntimeError(f'Evaluation cleanup failed: {path}')
        protocol_reason = model_protocol_retry_reason(attempt)
        retry_reason = ('infrastructure_invalid' if not attempt['valid'] else
                        protocol_reason if retry_protocol else None)
        history.append(dict(path=str(path.relative_to(folder)), seed=actual_seed,
                            valid=attempt['valid'], error=attempt.get('error'),
                            retry_reason=retry_reason, protocol_error=(attempt.get('trajectory') or {}).get('protocol_error')))
        selected = str(path.relative_to(folder)) if retry_reason is None else None
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'resolution.json').write_text(json.dumps(dict(
            selected=selected, skipped=not selected and retry == retries,
            invalid_attempts=sum(not x['valid'] for x in history),
            model_protocol_attempts=sum(x['retry_reason'] == 'model_protocol_error' for x in history),
            retry_model_protocol_error=retry_protocol,
            exhausted_reason=retry_reason if not selected and retry == retries else None,
            history=history), indent=2))
        if selected:
            return attempt, path
    return None, None
