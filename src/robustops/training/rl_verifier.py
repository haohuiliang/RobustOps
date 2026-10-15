"""Bind private verifier measurements to executed task environments."""

from functools import lru_cache
from robustops.training.recovery_rl_data import load_task


@lru_cache(maxsize=1027)
def cached_task(root, domain, mode, task_id):
    return load_task(root, domain, mode, task_id)


class InitialVerifier:
    """Bind after seeding; measure the initial state before the first model action."""

    def __init__(self, task, controller):
        self.task, self.controller = task, controller
        self.initial = None
        self.caller = None

    def bind(self, caller, tools):
        self.caller = caller
        self.tools = tools
        self.delegate = (
            self.controller.bind(caller, tools) if self.controller else caller
        )
        return self

    async def measure(self):
        if self.initial is None:
            self.initial = [
                await self.caller.environments[v["gym_name"]].run_verifier(v, i)
                for i, v in enumerate(self.task.verifiers, 1)
            ]

    async def call_tool(self, name, arguments):
        await self.measure()
        return await self.delegate.call_tool(name, arguments)

    async def after_tool_event(self, event):
        hook = getattr(self.controller, "after_tool_event", None)
        if hook is not None:
            await hook(event)

    def report(self):
        return (
            self.controller.report()
            if self.controller
            else dict(enabled=False, injected_calls=0, events=[])
        )

    @property
    def stop_after_capture(self):
        return bool(getattr(self.controller, "stop_after_capture", False))
