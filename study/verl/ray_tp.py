import ray
import torch
import torch.nn as nn
import socket
import os
import time
import logging

class ColumnTPLayer(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")

        self.layer = nn.Linear(input_size, output_size // int(os.environ['WORLD_SIZE']), bias=False).cuda()

    def forward(self, x):
        ret = self.layer(x.cuda())

        output_tensor = torch.zeros(size=(int(os.environ['WORLD_SIZE']), ret.shape[0], ret.shape[1]), dtype=ret.dtype, device=ret.device)

        torch.distributed.all_gather_into_tensor(output_tensor, ret, async_op=False)

        output_tensor = torch.cat(output_tensor.unbind(dim=0), dim=-1)

        return output_tensor

    def load_weights(self, weight):
        rank = int(os.environ['RANK'])

        world_size = int(os.environ['WORLD_SIZE'])
        dim_per_rank = weight.shape[0] // world_size
        self.layer.weight.data.copy_(weight[rank*dim_per_rank: (rank+1)*dim_per_rank, :])

    def state_dict(self):
        return self.layer.state_dict()

ray.init()

master_addr = ray._private.services.get_node_ip_address()
with socket.socket() as sock:
    sock.bind(('', 0))
    master_port = sock.getsockname()[1]

num_gpus = 2
workers = []

for i in range(num_gpus):

    options = {'runtime_env': {'env_vars': {'WORLD_SIZE': str(num_gpus), 'RANK': str(i), 'MASTER_ADDR': master_addr, 'MASTER_PORT': str(master_port)}}}

    workers.append(ray.remote(num_gpus=1)(ColumnTPLayer).options(**options).remote(2, 6))

batch_size = 10
input_data = torch.randn(batch_size, 2)

full_layer = torch.nn.Linear(2, 6, bias=False)
weight = full_layer.state_dict()['weight']

ret_list = []
for i in range(num_gpus):
    _ = ray.get(workers[i].load_weights.remote(weight))

for i in range(num_gpus):
    ret_list.append(workers[i].forward.remote(input_data))

ret = ray.get(ret_list)

ray.shutdown()

fl_ret = full_layer(input_data).cpu()

torch.testing.assert_close(ret[0], ret[1])
torch.testing.assert_close(ret[0].cpu(), fl_ret)