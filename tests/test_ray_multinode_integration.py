from __future__ import annotations

import errno
import json
import os
import socket
import subprocess
import sys
from contextlib import suppress
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT_PREFIX = "RAY_MULTINODE_RESULT="


def _run_multinode_probe() -> None:
    import ray
    import torch
    from ray.cluster_utils import Cluster
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

    from tinyexp.tiny_engine.accelerator import CPUAccelerator
    from tinyexp.utils.ray_utils import (
        get_num_worker_options,
        get_placement_group,
        get_placement_group_node_ids,
        start_ray_rendezvous_store,
    )

    cluster = Cluster()
    pg = None
    rendezvous_actor = None
    workers = []
    try:
        cluster.add_node(num_cpus=1, num_gpus=0, include_dashboard=False)
        gpu_node = cluster.add_node(num_cpus=2, num_gpus=2)
        ray.init(address=cluster.address)
        cluster.wait_for_nodes()

        pg = get_placement_group(
            num_worker=2,
            num_gpus_per_worker=1,
            num_cpus_per_worker=1,
            strategy="PACK",
            timeout_s=10,
        )
        node_ids = get_placement_group_node_ids(pg, 2)
        assert set(node_ids) == {gpu_node.node_id}

        rendezvous_actor, master_addr, master_port = start_ray_rendezvous_store(
            pg,
            world_size=2,
            timeout_s=10,
        )
        nodes_by_id = {str(node["NodeID"]): node for node in ray.nodes() if node["Alive"]}
        assert master_addr == nodes_by_id[node_ids[0]]["NodeManagerAddress"]

        def try_bind_rendezvous_port(address: str, port: int) -> int | None:
            with socket.socket() as sock:
                try:
                    sock.bind((address, port))
                except OSError as exc:
                    return exc.errno
            return None

        bind_probe = ray.remote(num_cpus=0)(try_bind_rendezvous_port)
        bind_errno = ray.get(
            bind_probe.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=0,
                )
            ).remote(master_addr, master_port),
            timeout=10,
        )
        assert bind_errno == errno.EADDRINUSE

        options_list = get_num_worker_options(
            pg,
            num_worker=2,
            gpu_ratio=1,
            num_cpus_per_worker=1,
            master_addr=master_addr,
            master_port=master_port,
            node_ids=node_ids,
        )

        @ray.remote
        class Worker:
            def run(self) -> dict[str, object]:
                accelerator = CPUAccelerator()
                value = torch.tensor(float(accelerator.rank + 1))
                reduced = accelerator.reduce_sum(value).item()
                result = {
                    "rank": accelerator.rank,
                    "sum": reduced,
                    "node_id": ray.get_runtime_context().get_node_id(),
                    "master_addr": accelerator.master_addr,
                    "master_port": accelerator.master_port,
                    "agent_store": os.environ.get("TORCHELASTIC_USE_AGENT_STORE"),
                }
                accelerator.destroy()
                return result

        workers = [Worker.options(**options).remote() for options in options_list]
        results = ray.get([worker.run.remote() for worker in workers], timeout=30)
        results.sort(key=lambda result: int(result["rank"]))

        assert [result["rank"] for result in results] == [0, 1]
        assert [result["sum"] for result in results] == [3.0, 3.0]
        assert {result["node_id"] for result in results} == {gpu_node.node_id}
        assert {result["master_addr"] for result in results} == {master_addr}
        assert {result["master_port"] for result in results} == {master_port}
        assert {result["agent_store"] for result in results} == {"True"}

        print(
            RESULT_PREFIX
            + json.dumps(
                {
                    "bundle_node_ids": node_ids,
                    "gpu_node_id": gpu_node.node_id,
                    "master_addr": master_addr,
                    "master_port": master_port,
                    "port_reserved": True,
                    "results": results,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    finally:
        for worker in workers:
            with suppress(Exception):
                ray.kill(worker, no_restart=True)
        if rendezvous_actor is not None:
            with suppress(Exception):
                ray.kill(rendezvous_actor, no_restart=True)
        if pg is not None and ray.is_initialized():
            with suppress(Exception):
                ray.util.remove_placement_group(pg)
        if ray.is_initialized():
            ray.shutdown()
        cluster.shutdown()


def test_cpu_only_head_with_logical_gpu_workers_uses_reserved_worker_rendezvous() -> None:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(ROOT) if not existing_pythonpath else f"{ROOT}{os.pathsep}{existing_pythonpath}"

    result = subprocess.run(  # noqa: S603
        [sys.executable, str(Path(__file__).resolve()), "--probe"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )

    assert result.returncode == 0, f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
    result_lines = [line for line in result.stdout.splitlines() if line.startswith(RESULT_PREFIX)]
    assert len(result_lines) == 1
    payload = json.loads(result_lines[0][len(RESULT_PREFIX) :])
    assert payload["port_reserved"] is True
    assert set(payload["bundle_node_ids"]) == {payload["gpu_node_id"]}


if __name__ == "__main__":
    _run_multinode_probe()
