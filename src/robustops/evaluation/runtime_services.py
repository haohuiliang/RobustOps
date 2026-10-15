"""Own model/MCP processes and record model calls for rollout execution."""

import asyncio, json, os, signal, socket, subprocess, time, sys
from pathlib import Path
import httpx
from robustops.evaluation.clean_baseline import OpenAIChatClient
from robustops.launch.hardware import device_for_worker

ROOT = Path(__file__).resolve().parents[3]
SERVERS = {
    "calendar": ("gym-calendar", 18003),
    "email": ("gym-email-mcp", 18004),
    "drive": ("gym-google-drive-mcp", 18009),
    "csm": ("sn-csm-server", 18011),
    "hr": ("sn-hr-internal", 18008),
    "itsm": ("gym-itsm-mcp", 18006),
    "teams": ("gym-teams-mcp", 18002),
}


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


async def launch(required, out, *, port_offset=1000):
    processes, urls = [], {}
    try:
        for domain, (server, default_port) in SERVERS.items():
            if server not in required:
                continue
            port = default_port + port_offset
            with socket.socket() as sock:
                if sock.connect_ex(("127.0.0.1", port)) == 0:
                    raise RuntimeError(
                        f"occupied port {port}; refusing unknown runtime"
                    )
            env = dict(os.environ, ROBUSTOPS_MCP_HOST_PORT=str(port))
            with (out / f"{domain}_mcp.log").open("w") as log:
                p = subprocess.Popen(
                    [
                        "bash",
                        str(
                            ROOT
                            / "backends/enterpriseops/launch_udocker.sh"
                        ),
                        domain,
                    ],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            processes.append(p)
            url = f"http://127.0.0.1:{port}"
            urls[server] = url
            async with httpx.AsyncClient(trust_env=False) as client:
                for _ in range(120):
                    if p.poll() is not None:
                        raise RuntimeError(f"{domain} MCP exited; see log")
                    try:
                        if (
                            await client.get(url + "/openapi.json", timeout=2)
                        ).status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    await asyncio.sleep(1)
                else:
                    raise RuntimeError(f"{domain} MCP readiness timeout")
            print("MCP ready: " + domain, flush=True)
        return processes, urls
    except BaseException:
        stop_processes(processes)
        raise


def stop_processes(processes):
    for p in reversed(processes):
        if p.poll() is None:
            os.killpg(p.pid, signal.SIGTERM)
    for p in processes:
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()


class Model:
    def __init__(self, cfg, path, seed, key=None):
        self.path = path
        self.n = 0
        self.cfg = cfg
        self.seed = seed
        self.client = OpenAIChatClient(
            endpoint=cfg["endpoint"],
            model_name=cfg["model"],
            temperature=cfg["temperature"],
            max_tokens=cfg.get("max_tokens", 2048),
            seed=seed,
            enable_thinking=False,
            provider=cfg["provider"],
            api_key=key,
        )

    async def complete(self, messages, tools):
        self.n += 1
        dump(
            self.path / f"{self.n:03}.request.json",
            dict(
                messages=messages,
                tools=tools,
                model=self.cfg["model"],
                seed=self.seed,
                temperature=self.cfg["temperature"],
                max_tokens=self.cfg.get("max_tokens", 2048),
                started_at=time.time(),
            ),
        )
        m, u = await self.client.complete(messages, tools)
        dump(
            self.path / f"{self.n:03}.response.json",
            dict(message=m, usage=u, finished_at=time.time()),
        )
        print(
            "REQUEST",
            self.path.parent.parent.parent.name,
            self.path.parent.parent.name,
            self.path.parent.name,
            self.n,
            flush=True,
        )
        return m, u


class FaultReplica:
    """One owned GPU server and MCP port group; databases are per-attempt UUIDs."""

    def __init__(self, gpu_id, model_path, tokenizer, served_name, output, *, max_context_tokens=32768):
        self.gpu_id = gpu_id
        self.max_context_tokens = max_context_tokens
        self.model_path = model_path
        self.tokenizer = tokenizer
        self.served_name = served_name
        self.output = Path(output)
        self.port = 8010 + gpu_id
        self.endpoint = f"http://127.0.0.1:{self.port}/v1"
        self.gpu = None
        self.mcp = []
        self.log = None
        self.urls = {}

    async def start(self, required):
        self.output.mkdir(parents=True, exist_ok=False)
        device = device_for_worker(self.gpu_id)
        used = subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                device.uuid,
                "--query-compute-apps=pid",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
        if used:
            raise RuntimeError(f"GPU {self.gpu_id} already occupied")
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1", self.port)) == 0:
                raise RuntimeError(f"Model port {self.port} occupied")
        cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            str(self.model_path),
            "--tokenizer",
            str(self.tokenizer),
            "--served-model-name",
            self.served_name,
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--dtype",
            "bfloat16",
            "--max-model-len",
            str(self.max_context_tokens),
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            "hermes",
        ]
        dump(
            self.output / "environment.json",
            dict(
                command=cmd,
                gpu_id=self.gpu_id,
                gpu_uuid=device.uuid,
                gpu_model=device.name,
                endpoint=self.endpoint,
                mcp_port_offset=1000 + 100 * self.gpu_id,
            ),
        )
        self.log = (self.output / "gpu_server.log").open("w")
        # An absolute Python path does not activate its environment in the
        # parent shell. FlashInfer invokes ninja by name during GPU startup.
        model_env = dict(os.environ, CUDA_VISIBLE_DEVICES=device.uuid)
        model_env["PATH"] = (
            str(Path(sys.executable).parent) + os.pathsep + model_env.get("PATH", "")
        )
        self.gpu = subprocess.Popen(
            cmd,
            env=model_env,
            stdout=self.log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        async with httpx.AsyncClient(trust_env=False) as client:
            for _ in range(600):
                if self.gpu.poll() is not None:
                    raise RuntimeError(f"GPU {self.gpu_id} model startup failed")
                try:
                    if (
                        await client.get(
                            f"http://127.0.0.1:{self.port}/health", timeout=2
                        )
                    ).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(1)
            else:
                raise RuntimeError("Model readiness timeout")
        self.mcp, self.urls = await launch(
            required, self.output, port_offset=1000 + 100 * self.gpu_id
        )
        dump(
            self.output / "owned_processes.json",
            dict(gpu=self.gpu.pid, mcp=[p.pid for p in self.mcp]),
        )

    def close(self):
        stop_processes(self.mcp)
        if self.gpu is not None:
            stop_processes([self.gpu])
        if self.log:
            self.log.close()
        if self.output.exists():
            dump(
                self.output / "resource_cleanup.json",
                dict(
                    gpu_stopped=self.gpu is None or self.gpu.poll() is not None,
                    mcp_stopped=all(p.poll() is not None for p in self.mcp),
                ),
            )
