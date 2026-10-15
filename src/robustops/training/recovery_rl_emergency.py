"""Safe commit boundaries and cooperative shutdown for normalization training."""
from copy import deepcopy
import json
import math
from pathlib import Path
import signal
import threading


class TrainingStopRequested(RuntimeError):
    pass


def check_stop_request(output):
    path = Path(output)/'stop_request.json'
    if path.exists():
        request = json.loads(path.read_text())
        raise TrainingStopRequested(request.get('reason', 'cooperative stop requested'))


class TrainingFailureGuard:
    def __init__(self, trainer):
        self.trainer = trainer
        self.completed_step = 0
        self.loader_state = None
        self.phase = 'initializing'
        self.stop_reason = None
        self.handlers = {}

    def __enter__(self):
        # Ray may execute the trainer outside Python's main thread. The launcher
        # also writes stop_request.json, which does not depend on signal routing.
        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
                self.handlers[sig] = signal.getsignal(sig)
                signal.signal(sig, self.request_stop)
        return self

    def request_stop(self, signum, frame):
        self.stop_reason = f'signal {signum}'
        from robustops.training.recovery_rl_storage import write_checkpoint_status
        try:
            write_checkpoint_status(Path(self.trainer.config.recovery.output)/'stop_request.json',
                                 dict(reason=self.stop_reason))
        except OSError:
            # The in-memory request still takes effect at the next boundary.
            pass

    def check_stop(self):
        if self.stop_reason:
            raise TrainingStopRequested(self.stop_reason)
        check_stop_request(self.trainer.config.recovery.output)

    def commit(self, step, metrics=None):
        # The launcher already rejects nonfinite actor metrics. Do so before
        # marking this optimizer update safe, rather than exporting bad weights.
        for key, value in (metrics or {}).items():
            if key.startswith('actor/') and key.split('/')[-1] in ('grad_norm', 'pg_loss', 'kl_loss', 'ppo_kl'):
                if not math.isfinite(float(value)):
                    self.phase = 'nonfinite_update'
                    raise RuntimeError(f'Non-finite actor training metric: {key}={value}')
        state = deepcopy(self.trainer.train_dataloader.state_dict())
        self.completed_step, self.loader_state = step, state
        self.phase = 'safe'
        self.check_stop()

    def before_update(self):
        self.check_stop()
        self.phase = 'updating'

    def __exit__(self, kind, error, traceback):
        try:
            if error is not None:
                from robustops.training.recovery_rl_storage import save_after_training_failure
                skip = None
                if self.phase in ('initializing', 'updating', 'nonfinite_update'):
                    skip = self.phase
                # These failures make new distributed checkpoint RPCs unsafe.
                chain, cursor = [], error
                while cursor is not None and id(cursor) not in {id(e) for e in chain}:
                    chain.append(cursor)
                    cursor = cursor.__cause__ or cursor.__context__
                if any(any(cls.__name__ in ('RayActorError', 'ActorDiedError', 'WorkerCrashedError', 'NodeDiedError')
                           for cls in type(e).__mro__) for e in chain):
                    skip = 'distributed_worker_unavailable'
                if any(any(token in str(e).lower() for token in
                           ('device-side assert', 'illegal memory access', 'nccl error', 'nccl communicator was aborted'))
                       for e in chain):
                    skip = 'cuda_or_collective_unusable'
                save_after_training_failure(self.trainer, self.loader_state, error,
                    completed_step=self.completed_step, phase=self.phase,
                    skip_reason=skip, reuse_only=self.phase == 'checkpoint')
        finally:
            for sig, handler in self.handlers.items():
                signal.signal(sig, handler)
        return False


async def request_training_stop(process, output, config, reason):
    """Give a live trainer time to drain episodes and save before owned cleanup."""
    import asyncio
    from robustops.training.recovery_rl_storage import write_checkpoint_status
    try:
        write_checkpoint_status(Path(output)/'stop_request.json', dict(reason=reason))
    except OSError as exc:
        print(f'RECOVERY_STOP_REQUEST_FAILED {exc!r}', flush=True)
        return
    grace = config['episode_timeout_seconds'] + 600
    print(f'RECOVERY_GRACEFUL_STOP_REQUEST grace_seconds={grace} reason={reason}', flush=True)
    try:
        await asyncio.wait_for(process.wait(), timeout=grace)
    except asyncio.TimeoutError:
        print('RECOVERY_GRACEFUL_STOP_TIMEOUT; continuing owned process cleanup', flush=True)
