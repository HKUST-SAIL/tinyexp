import subprocess
import time

import redis


class RedisClusterManager:
    def __init__(self, ports: list, max_memory_per_port: int):
        """
        Initialize RedisClusterManager with specified ports and max memory per port.
        Notice that the host is assumed to be localhost.
        """
        self.redis_processes = []
        self.redis_clients = []
        self.ports = ports
        self.max_memory = max_memory_per_port

    def start_redis_cluster(self) -> bool:
        """
        Start multiple Redis server instances

        Returns:
            bool: True if all Redis servers started successfully, False otherwise.
        """
        try:
            for i, port in enumerate(self.ports):
                # Start Redis server process
                redis_process = subprocess.Popen(
                    [
                        "redis-server",
                        "--port",
                        str(port),
                        "--daemonize",
                        "no",
                        "--save",
                        "",
                        "--appendonly",
                        "no",
                        "--maxmemory",
                        f"{self.max_memory}gb",
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                self.redis_processes.append(redis_process)

                # Wait for Redis server to start
                time.sleep(1)

                # Create Redis client connection
                redis_client = redis.StrictRedis(host="localhost", port=port, decode_responses=False)
                redis_client.ping()
                self.redis_clients.append(redis_client)

                print(f"Redis shard {i} started on port {port}")

            return True

        except Exception as e:
            print(f"Failed to start Redis cluster: {e}")
            self.stop_redis_cluster()
            print(e)
            return False

    def stop_redis_cluster(self):
        """Stop all Redis servers"""
        for process in self.redis_processes:
            if process and process.poll() is None:  # Check if process is still running
                try:
                    process.terminate()
                    process.wait(timeout=5)  # Add timeout
                except subprocess.TimeoutExpired:
                    process.kill()  # Force terminate
                    process.wait()
                except Exception as e:
                    print(f"Error stopping Redis process: {e}")
        self.redis_processes.clear()
        self.redis_clients.clear()

    def get_redis_memory_info(self):
        """Get Redis memory usage info"""
        memory_info = {}
        for i, client in enumerate(self.redis_clients):
            try:
                info = client.info("memory")
                used_memory = info["used_memory"] / 1024 / 1024  # MB
                used_memory_human = info["used_memory_human"]
                memory_info[f"redis_{self.ports[i]}"] = {
                    "used_memory_mb": used_memory,
                    "used_memory_human": used_memory_human,
                }
            except Exception as e:
                memory_info[f"redis_{self.ports[i]}"] = {"error": str(e)}
        return memory_info
