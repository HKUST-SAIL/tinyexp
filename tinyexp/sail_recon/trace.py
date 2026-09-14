"""Full-tensor hashes at operator boundaries, without retaining GPU activations."""

import hashlib

import torch


def tensor_record(value):
    value = value.detach().contiguous().cpu()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest(),
        "finite": bool(torch.isfinite(value).all()),
    }


def flatten_tensors(value, prefix=""):
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from flatten_tensors(child, f"{prefix}/{key}")
    elif isinstance(value, (list, tuple)):
        for key, child in enumerate(value):
            yield from flatten_tensors(child, f"{prefix}/{key}")


class Trace:
    def __init__(self, model):
        self.records = {}
        self.handles = []
        self.phase = "anchor"
        names = {"aggregator.patch_embed", "camera_head", "point_head", "depth_head"}
        for group in ("frame_blocks", "global_blocks", "global_reloc_blocks"):
            for i in range(24):
                names.add(f"aggregator.{group}.{i}")
                names.add(f"aggregator.{group}.{i}.attn")
        for name, module in model.named_modules():
            if name in names:
                self.handles.append(module.register_forward_hook(self.hook(name)))

    def hook(self, name):
        def record(module, args, output):
            for suffix, tensor in flatten_tensors(output):
                key = f"{self.phase}/{name}{suffix}"
                self.records.setdefault(key, []).append(tensor_record(tensor))

        return record

    def cache(self, model):
        self.records["anchor/camera_token"] = [tensor_record(model.cam_token_last_layer)]
        for i, block in enumerate(model.aggregator.global_reloc_blocks):
            for kind in ("k_cache", "v_cache"):
                self.records[f"anchor/cache/{i}/{kind}"] = [tensor_record(getattr(block.attn, kind))]

    def close(self):
        for handle in self.handles:
            handle.remove()
