import logging
import os
import time

import ray
import ray.runtime_context  # <--- 新增导入

# 1. 初始化 Ray
if ray.is_initialized():
    ray.shutdown()
ray.init(num_cpus=2)
print("Ray initialized.")

# --- 示例 1: 在主程序中使用标准 Python logger ---
print("\n--- 示例 1: 主程序日志 ---")
main_logger = logging.getLogger(__name__)
main_logger.setLevel(logging.DEBUG)

main_logger.info("这是一个来自主脚本的 INFO 消息。")
main_logger.warning("这是一个来自主脚本的 WARNING 消息。")
main_logger.debug("这是一个来自主脚本的 DEBUG 消息。默认情况下可能不显示，除非级别设置为 DEBUG。")
main_logger.error("这是一个来自主脚本的 ERROR 消息。")

# --- 示例 2: 在 Ray 远程任务中记录日志 ---
print("\n--- 示例 2: 远程任务日志 ---")


@ray.remote
def my_remote_task(task_id):
    task_logger = logging.getLogger(f"task_logger_{task_id}")
    task_logger.setLevel(logging.INFO)

    task_logger.info(f"任务 {task_id}: 远程任务开始执行。")
    time.sleep(0.1)
    if task_id % 2 == 0:
        task_logger.warning(f"任务 {task_id}: 这是一个偶数任务，发出警告。")
    else:
        task_logger.debug(f"任务 {task_id}: 这是一个奇数任务，发出调试信息 (可能不显示)。")
    task_logger.info(f"任务 {task_id}: 远程任务执行完毕。")
    return f"Task {task_id} finished."


futures = [my_remote_task.remote(i) for i in range(3)]
results = ray.get(futures)
print(f"远程任务结果: {results}")

# --- 示例 3: 在 Ray Actor 中记录日志 ---
print("\n--- 示例 3: Actor 日志 ---")


@ray.remote
class MyActor:
    def __init__(self, actor_id):
        self.actor_id = actor_id
        self.actor_logger = logging.getLogger(f"actor_logger_{actor_id}")
        self.actor_logger.setLevel(logging.INFO)
        self.actor_logger.info(f"Actor {self.actor_id}: Actor 已初始化。")

    def perform_action(self, value):
        self.actor_logger.info(f"Actor {self.actor_id}: 正在执行操作，接收值: {value}")
        if value < 0:
            self.actor_logger.error(f"Actor {self.actor_id}: 接收到无效的负值: {value}")
            raise ValueError("Value cannot be negative")
        time.sleep(0.05)
        self.actor_logger.debug(f"Actor {self.actor_id}: 操作完成，结果为 {value * 2}")
        return value * 2


actor = MyActor.remote(101)
print(f"Actor 方法调用结果: {ray.get(actor.perform_action.remote(7))}")

try:
    print(f"Actor 方法调用结果 (错误): {ray.get(actor.perform_action.remote(-2))}")
except Exception as e:
    print(f"捕获到 Actor 错误: {e}")

# --- 示例 4: 检查 Ray 日志文件 (修正后) ---
print("\n--- 示例 4: 检查 Ray 日志文件 ---")

# 获取当前 Ray 运行时上下文
runtime_context = ray.runtime_context.get_runtime_context()
# 从运行时上下文中获取会话目录
ray_session_dir = runtime_context.get_session_dir()
ray_log_dir = os.path.join(ray_session_dir, "logs")

print(f"Ray 会话目录: {ray_session_dir}")
print(f"Ray 日志目录: {ray_log_dir}")
print("你可以在该目录下找到 worker_*.log, dashboard.log 等文件，查看详细日志。")

if os.path.exists(ray_log_dir):
    worker_log_files = [f for f in os.listdir(ray_log_dir) if f.startswith("worker") and f.endswith(".log")]
    if worker_log_files:
        print(f"示例 worker 日志文件: {os.path.join(ray_log_dir, worker_log_files[0])}")
    else:
        print("未找到 worker 日志文件，可能 Ray 尚未完全写入或没有 worker 进程。")
else:
    print(f"Ray 日志目录 '{ray_log_dir}' 不存在。")

# --- 清理 ---
ray.shutdown()
print("\nRay shut down.")
