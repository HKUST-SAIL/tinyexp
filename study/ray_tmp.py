import ray

assert not ray.is_initialized(), "Ray should not already be initialized"

ray_env = {"env_vars": {"RAY_DEDUP_LOGS": "0"}}
ray.init(runtime_env=ray_env)
