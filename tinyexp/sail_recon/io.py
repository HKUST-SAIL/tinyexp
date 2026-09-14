"""Deterministic input selection and demo-compatible artifact serialization."""

# ruff: noqa: TRY003 -- descriptive errors at the input boundary.

import hashlib
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData, PlyElement


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def uniform_sample(total, select):
    if not 0 < select <= total:
        raise ValueError("Require 0 < select <= total")
    step = total / select
    return [int(i * step) for i in range(select)]


def image_paths(root, limit=0, stride=1):
    if stride < 1 or limit < 0:
        raise ValueError("stride must be positive and limit nonnegative")
    paths = sorted(p for p in Path(root).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
    paths = paths[::stride]
    if limit:
        paths = paths[:limit]
    if not paths:
        raise ValueError(f"No images found in {root}")
    return paths


def cpu_predictions(predictions):
    return [{key: torch.as_tensor(value).detach().cpu() for key, value in view.items()} for view in predictions]


def save_poses(predictions, path):
    poses = [np.linalg.inv(np.vstack([view["extrinsic"][0].numpy(), np.array([0, 0, 0, 1])])) for view in predictions]
    with open(path, "w") as stream:
        for pose in poses:
            stream.write(" ".join(map(str, pose[:3].reshape(-1))) + "\n")


def save_pointcloud(predictions, path, downsample_ratio=10):
    """Match upstream confidence median, random subsampling, ordering and PLY types."""
    points, colors = [], []
    for view in predictions:
        xyz = view["point_map_by_unprojection"].squeeze(0).reshape(-1, 3)
        rgb = view["rgbs"].squeeze(0).permute(1, 2, 0).reshape(-1, 3)
        valid = torch.isfinite(xyz).all(dim=1) & (xyz.norm(dim=1) > 0)
        valid &= (view["dpt_cnf"] > torch.quantile(view["dpt_cnf"], 0.5)).flatten()
        xyz, rgb = xyz[valid], rgb[valid]
        if downsample_ratio > 1 and len(xyz) >= downsample_ratio:
            indices = torch.randperm(len(xyz))[: len(xyz) // downsample_ratio]
            xyz, rgb = xyz[indices], rgb[indices]
        points.append(xyz)
        colors.append(rgb)
    xyz, rgb = torch.cat(points).numpy(), torch.cat(colors).numpy()
    if rgb.size and rgb.max() <= 1:
        rgb = rgb * 255
    rgb = rgb.astype(np.uint8)
    vertices = np.empty(
        len(xyz), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    )
    for i, key in enumerate(("x", "y", "z")):
        vertices[key] = xyz[:, i]
    for i, key in enumerate(("red", "green", "blue")):
        vertices[key] = rgb[:, i]
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(str(path))
