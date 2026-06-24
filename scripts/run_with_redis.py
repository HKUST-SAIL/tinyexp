from __future__ import annotations

import shlex
import subprocess
import sys

from tinyexp import RedisCfgMixin
from tinyexp.utils.redis_utils import RedisClusterManager


def main(argv: list[str]) -> int:
    if argv[:1] == ["--"]:
        argv = argv[1:]
    if not argv:
        print(
            "Usage: python scripts/run_with_redis.py -- <command> [args...]",
            file=sys.stderr,
        )
        return 2

    redis_cache_cfg = RedisCfgMixin().redis_cache_cfg
    redis_cluster_manager = None

    try:
        if redis_cache_cfg.redis_cache_enabled:
            redis_cluster_manager = RedisClusterManager(
                ports=redis_cache_cfg.redis_cache_shard_ports,
                max_memory_per_port=redis_cache_cfg.redis_cache_max_memory
                // len(redis_cache_cfg.redis_cache_shard_ports),
            )
            redis_status = redis_cluster_manager.start_redis_cluster()
        else:
            redis_status = True

        print(f"Redis status:\033[32m{redis_status}\033[0m", flush=True)
        if not redis_status:
            return 1

        print(f"Running command: {shlex.join(argv)}", flush=True)
        return subprocess.call(argv)  # noqa: S603
    finally:
        if redis_cluster_manager is not None:
            redis_cluster_manager.stop_redis_cluster()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
