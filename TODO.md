# TODO

> 来源：2026-08-26 代码审查。基线为当前 `main`（`2cb506d`）。
> 本文件只记录会影响训练正确性、分布式稳定性、数据一致性或 CI 的重要事项。

## P0：优先修复

### [x] P0-1 修复 CUDA DDP 下 optimizer 绑定旧参数的问题（正常路径已修复）

- **原问题**：optimizer 在 CPU 模型上创建，之后模型被移动到 CUDA。若模型迁移产生新的
  `Parameter` 对象，optimizer 可能继续持有旧参数。
- **本次处理**：
  - MNIST/ResNet 在创建 optimizer 前显式执行 `model.to(accelerator.device)`；
  - DDP accelerator 对齐正常 CUDA 路径：设置当前 device、单进程不初始化/包装 DDP，
    多进程显式传递 `device_ids` 和 `output_device`；
  - 增加单进程回归测试，并在 2×RTX 4080 上完成双进程 smoke test。
- **验证**：正常 PyTorch CUDA 转换语义下，模型与 optimizer 参数一致，梯度存在，
  `optimizer.step()` 后权重发生变化。
- **剩余边界**：未实现 `torch.__future__.set_overwrite_module_params_on_conversion(True)` 下的
  通用参数重绑定；HF Accelerate 1.10.1 的普通 CUDA 路径也不处理该特殊场景。若后续需要支持，
  应单独设计旧参数到新参数的映射和 optimizer state 迁移。

### [ ] P0-2 修复 PyTorch 2.6+ checkpoint 无法加载的问题

- **位置**：`tinyexp/exp_mixins/basic_mixins.py:393`
- **关联保存逻辑**：`tinyexp/examples/mnist_exp.py:309-311`、
  `tinyexp/examples/resnet_exp.py:459-461`
- **问题**：`torch.load()` 未显式指定 `weights_only`。当前 PyTorch 2.7.1 的默认安全加载模式
  无法反序列化 `numpy.random.get_state()` 保存的 RNG 对象。
- **影响**：单进程训练保存的 checkpoint 使用 `resume_from` 时会抛出
  `UnpicklingError`，无法恢复训练。
- **方向**：对可信 checkpoint 显式使用 `weights_only=False`，或把 RNG 状态改为仅包含基础类型/
  Tensor 的可安全序列化格式。
- **验收**：在 PyTorch 2.7.1 下覆盖“保存带 RNG 状态 -> 新进程加载 -> 恢复随机序列”的测试，
  同时明确不可信 checkpoint 的安全边界。

## P1：分布式与数据正确性

### [ ] P1-1 重新设计 InfiniteSampler 的 epoch 边界和 DataLoader 消费契约

- **位置**：`tinyexp/dataset/sampler.py:80-84,95-98`、
  `tinyexp/examples/resnet_exp.py:403-421`
- **问题**：ResNet 只创建一次无限 `train_iter`；后续 `set_epoch()` 不会重新定位已有 iterator。
  同时 sampler 的 `len()` 与 DataLoader 按 batch 向上取整后的实际消费数量不一致。
- **影响**：非整除数据集可能出现 epoch 间样本重复、遗漏或错位，shuffle/断点恢复语义也不可靠。
- **方向**：优先改为有限的 epoch sampler；或在每个 epoch 重建 iterator，并统一全局样本数、
  rank 分片数、`drop_last` 和 batch 数的定义。
- **验收**：覆盖 `size % world_size != 0`、不同 batch size、resume 后的 rank 流；验证每个逻辑
  epoch 的样本集合和长度符合明确契约。

### [ ] P1-2 保证分布式异常路径清理 process group

- **位置**：`tinyexp/examples/mnist_exp.py:160-184`、
  `tinyexp/examples/resnet_exp.py:300-324`、
  `tinyexp/examples/pi_exp.py:47-64`
- **问题**：`accelerator.destroy()` 只在正常返回时执行。
- **影响**：数据加载、训练、评估或 checkpoint 保存异常时，其他 rank 可能永久阻塞在
  collective/barrier，留下僵尸进程。
- **方向**：所有实验入口使用 `try/finally` 调用 `accelerator.destroy()`；必要时增加
  distributed error propagation。
- **验收**：模拟单个 rank 抛异常，确认其他 rank 在有界时间内退出且 process group 被销毁。

### [ ] P1-3 完善分布式 resume 的 RNG、sampler 和 DataLoader worker 状态

- **位置**：MNIST `tinyexp/examples/mnist_exp.py:251-253,309-311`；
  ResNet `tinyexp/examples/resnet_exp.py:391-393,459-461`
- **问题**：只有 `world_size == 1` 时才保存/恢复 RNG 状态。
- **影响**：DDP resume 后 dropout、数据增强、worker 随机序列与原运行不连续，结果可能明显偏离。
- **方向**：保存每个 rank 的 RNG/sampler 状态，或明确文档声明仅保证近似恢复而非 bit-exact resume。
- **验收**：多进程中断并恢复测试，至少验证 epoch、global step、采样顺序和随机增强策略符合约定。

### [ ] P1-4 修复 Ray head 节点固定导致的 GPU placement timeout

- **位置**：`tinyexp/utils/ray_utils.py:58-62`
- **问题**：bundle 0 无条件绑定 Ray head 节点，同时要求 worker GPU。
- **影响**：head 无 GPU、GPU 位于其他节点时，即使总 GPU 资源足够，placement group 也永远无法调度。
- **方向**：不要把“rank 0 必须在 head”与“GPU bundle 必须在 head”硬编码绑定；改用可达的
  rendezvous actor/address，或显式选择有 GPU 的 rank 0 节点。
- **验收**：覆盖 CPU-only head + GPU worker、多节点 GPU 不均匀拓扑。

### [ ] P1-5 消除 Ray `MASTER_PORT` 释放后的竞态

- **位置**：`tinyexp/utils/ray_utils.py:184-190`、
  `tinyexp/exp_mixins/basic_mixins.py:155-180`
- **问题**：通过临时 socket 获取端口后立即关闭，actor 启动前端口可能被其他进程占用。
- **影响**：rank 0 TCPStore 可能 `EADDRINUSE`，其他 rank 随后超时。
- **方向**：保留监听 socket 直到 rank 0 接管，或由专用 rendezvous 服务分配并验证端口。
- **验收**：模拟端口在 actor 启动前被抢占，确认失败可诊断且不会无限等待。

### [ ] P1-6 Ray 资源准入检查使用可调度资源，而不是只看总容量

- **位置**：`tinyexp/exp_mixins/basic_mixins.py:118-143`
- **问题**：自动 worker sizing 和资源校验使用 `ray.cluster_resources()`，没有考虑共享集群中
  已被其他任务占用的资源。
- **影响**：可能计算出当前无法放置的 worker 数，随后 placement group 长时间等待或超时。
- **方向**：结合 `ray.available_resources()` 和 placement group 实际调度结果；区分“总容量不足”
  与“当前资源繁忙”。
- **验收**：在有并发 Ray 作业占用资源时，worker 数量和错误信息符合预期。

### [ ] P1-7 修复 Redis Cluster 启动重试和外部命令超时

- **位置**：`tinyexp/cli/run_with_redis.py:444-467,631-637`、
  `tinyexp/utils/redis_utils.py:561-577`
- **问题**：
  1. `RedisCluster(...)` 在 `try` 外创建，初始连接失败抛出的 `RedisClusterException` 不会进入重试；
  2. `redis-cli --cluster create` 没有超时，可能无限阻塞；
  3. wrapper 版本在全局锁内执行外部命令，会阻塞其他注册请求。
- **影响**：Redis 启动失败时可能卡死，无法按配置的 timeout 清理。
- **方向**：将客户端构造纳入重试范围，捕获 cluster-specific exceptions；为 subprocess 设置硬超时；
  不要在锁内执行外部命令。
- **验收**：覆盖不可达节点、半启动集群、`redis-cli` 卡住和超时清理场景。

### [ ] P1-8 不要忽略 Redis rendezvous finish barrier 失败

- **位置**：`tinyexp/cli/run_with_redis.py:279-286`
- **问题**：`wait_for_rendezvous_finish()` 返回 `False` 后仍正常返回子进程退出码，并在
  `finally` 中销毁本节点 Redis。
- **影响**：其他 rank 仍运行时，共享 Redis 集群可能被提前拆除，导致数据访问失败。
- **方向**：barrier 失败应转为非零退出/协调错误；在所有节点确认完成前保留服务，增加心跳、
  租约或可靠的退出协议。
- **验收**：一个节点提前退出或网络中断时，其他节点不会被静默破坏，且最终状态可诊断。

### [ ] P1-9 校验 Redis Cluster bus port，避免非法端口和冲突

- **位置**：`tinyexp/utils/redis_utils.py:157-161,275-283`
- **问题**：只校验数据端口，却固定使用 `port + 10000` 作为 cluster bus port。
- **影响**：高数据端口会生成超过 `65535` 的非法 bus port；不同配置还可能发生 data/bus
  端口冲突。
- **方向**：集群模式校验 bus port 范围，并检查所有 data/bus 端口互不冲突；最好支持显式 bus port。
- **验收**：覆盖边界端口、相邻端口和多节点重复配置。

### [ ] P1-10 补齐 Redis Cluster 专属异常的容错语义

- **位置**：`tinyexp/utils/redis_utils.py:101-117`
- **问题**：`safe_get/safe_set` 只捕获 `RedisError`，未覆盖独立继承树中的
  `RedisClusterException`（例如 `SlotNotCoveredError`）。
- **影响**：集群拓扑未就绪或临时失效时，所谓 `safe_*` 方法仍可能直接使训练崩溃。
- **方向**：捕获 cluster-specific exceptions，做有限重试/拓扑刷新，并区分缓存未命中与后端故障。
- **验收**：模拟 slot 未覆盖、节点短暂不可达和恢复后的读写行为。

### [ ] P1-11 加强静态 Ray cluster CLI 参数和 readiness 超时校验

- **位置**：`tinyexp/cli/run_with_ray_cluster.py:62,116-123,173-187,208-259,282-299`
- **问题**：
  1. 未校验 `0 <= node_rank < node_count`；
  2. 允许 `--ray-port=0`，但后续仍使用 `head_addr:0` 探活和连接；
  3. `ray_alive_count()` 内部的 subprocess 没有硬超时，可能超过 `--wait-timeout` 后仍阻塞。
- **影响**：错误拓扑可能等待很久、启动多个 head，或清理不及时。
- **方向**：解析阶段拒绝非法 rank；禁止/正确处理动态端口；为 readiness 子进程设置 timeout。
- **验收**：覆盖非法 rank、port 0、网络黑洞和 head 启动失败。

### [ ] P1-12 修复 `set_cfg()` 对普通 Mapping 字段的处理

- **位置**：`tinyexp/__init__.py:118-121`
- **问题**：所有 dict 都被当作嵌套对象递归；用户自定义 `dict` 字段（如 `model_kwargs`）
  会把字典 key 当作对象属性查找并抛出 `UnknownConfigurationKeyError`。
- **方向**：仅对嵌套 dataclass/config object 递归；对 `Mapping` 字段整体赋值或按 key 更新。
- **验收**：覆盖 `dict`、`ListConfig`、`Optional` 和多层嵌套配置的 CLI override。

## P2：重要改进

### [ ] P2-1 修复 Ray launcher 的 runtime 所有权管理

- **位置**：`tinyexp/exp_mixins/basic_mixins.py:112,192-195`
- **问题**：无条件 `ray.init()`，且无条件在结束时 `ray.shutdown()`。
- **影响**：无法安全嵌入已初始化的 Ray 应用，也可能关闭调用方拥有的 runtime。
- **方向**：记录是否由当前 launcher 初始化 Ray，仅清理自己拥有的 runtime。

### [ ] P2-2 修复 `update_ema()` 的包装模型和 buffer 处理

- **位置**：`tinyexp/utils/model_utils.py:12-17`
- **问题**：只规范化当前 model 的 `module.` 前缀；两边都被 DDP 包装时可能 `KeyError`。
  同时只更新 parameters，不更新 BatchNorm 等 buffers。
- **方向**：统一 unwrap/key 规范，校验参数集合；按明确策略同步 buffers。

### [ ] P2-3 让 `persistent_workers` 随 `num_workers` 配置

- **位置**：`tinyexp/examples/resnet_exp.py:270-275,289-294`
- **问题**：`persistent_workers=True` 固定开启；当 override 将 worker 数设为 `0` 时，
  DataLoader 直接抛出 `ValueError`。
- **方向**：使用 `persistent_workers=(num_workers > 0)`，并增加 CPU/调试配置测试。

### [ ] P2-4 修复 mypy 门禁

- **位置**：`pyproject.toml:83-91`、`tox.ini:18`
- **现状**：当前 `uv run mypy` 报告 **122 个错误**；因此 tox 的类型检查无法通过。
- **方向**：补齐公共 API、accelerator、Ray/Redis 边界和 examples 的类型；或明确调整检查范围，
  不要保留“配置为强门禁但长期无法通过”的状态。

## 当前验证记录

- `pytest`：133 passed，1 skipped
- `ruff`：通过
- `mypy`：失败，122 errors
- 当前审查未修改业务代码；CUDA、真实多节点 Ray 和 Redis Cluster 场景仍需在对应环境补充回归测试。
