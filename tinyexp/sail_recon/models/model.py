"""Inference stages for SAIL-Recon; parameter names match upstream checkpoints.

Low-level transformer/head operators retain upstream arithmetic and attribution.
This module owns scene lifecycle, decoding, and per-view output assembly.
"""

# ruff: noqa: TRY003 -- no custom exception hierarchy for inference input validation.

import torch
from torch import nn

from tinyexp.sail_recon.heads.camera_head import CameraHead
from tinyexp.sail_recon.heads.dpt_head import DPTHead
from tinyexp.sail_recon.models.aggregator import Aggregator
from tinyexp.sail_recon.utils.geometry import unproject_depth_map_to_point_map
from tinyexp.sail_recon.utils.pose_enc import pose_encoding_to_extri_intri


class SailRecon(nn.Module):
    def __init__(self, kv_cache=False):
        super().__init__()
        self.aggregator = Aggregator(kv_cache=kv_cache)
        self.camera_head = CameraHead(dim_in=2048)
        self.point_head = DPTHead(dim_in=2048, output_dim=4, activation="inv_log", conf_activation="expp1")
        self.depth_head = DPTHead(dim_in=2048, output_dim=2, activation="exp", conf_activation="expp1")
        self.cam_token_last_layer = None
        self._scene_ready = False

    @staticmethod
    def validate_images(images):
        if images.ndim != 4 or images.shape[0] == 0 or images.shape[1] != 3:
            raise ValueError("Expected nonempty [views, 3, height, width] images")
        if any(size < 14 or size % 14 for size in images.shape[-2:]):
            raise ValueError("Image dimensions must be positive multiples of patch size 14")

    def clear_scene(self):
        for block in self.aggregator.global_reloc_blocks:
            block.attn.clear_kv_cache()
        self.cam_token_last_layer = None
        self._scene_ready = False

    @torch.no_grad()
    def build_scene(self, images, fix_rank=300):
        self.validate_images(images)
        if not self.aggregator.global_reloc_blocks[0].attn.kv_cache:
            raise RuntimeError("Construct SailRecon(kv_cache=True) for scene caching")
        if not hasattr(self.aggregator, "global_blocks"):
            raise RuntimeError("Global blocks were released; construct a new model for another scene")
        if fix_rank <= 0:
            raise ValueError("fix_rank must be positive")
        self.clear_scene()
        features, _, camera_token = self.aggregator(
            images.unsqueeze(0), list(range(len(images))), [], fix_rank=fix_rank
        )
        self.cam_token_last_layer = camera_token.clone()
        features.clear()
        self._scene_ready = True

    def release_global_blocks(self):
        if not self._scene_ready:
            raise RuntimeError("Build a scene before releasing global blocks")
        del self.aggregator.global_blocks

    def decode_camera(self, features, image_size):
        poses = self.camera_head(features, self.cam_token_last_layer)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(poses[-1], image_size)
        return {"extrinsic": extrinsic, "intrinsic": intrinsic}

    def decode_geometry(self, features, images, patch_start):
        points, point_confidence = self.point_head(features, images=images, patch_start_idx=patch_start)
        depth, depth_confidence = self.depth_head(features, images=images, patch_start_idx=patch_start)
        return {"point_map": points, "xyz_cnf": point_confidence, "depth_map": depth, "dpt_cnf": depth_confidence}

    @torch.no_grad()
    def localize(self, images, fix_rank=300, memory_save=False, save_depth=True, fast_reloc=False, ret_img=False):
        self.validate_images(images)
        if not self._scene_ready:
            raise RuntimeError("Call build_scene before localize")
        rgbs = images.unsqueeze(0)
        features, patch_start = self.aggregator.forward_with_cache(rgbs, fix_rank=fix_rank)
        with torch.autocast(device_type="cuda", enabled=False):
            predictions = self.decode_camera(features, rgbs.shape[-2:])
            if not fast_reloc:
                geometry = self.decode_geometry(features, rgbs, patch_start)
                if not memory_save:
                    predictions.update({key: geometry[key] for key in ("point_map", "xyz_cnf")})
                    predictions["rgbs"] = rgbs
                    predictions["point_map_by_unprojection"] = unproject_depth_map_to_point_map(
                        geometry["depth_map"].squeeze(0),
                        predictions["extrinsic"].squeeze(0),
                        predictions["intrinsic"].squeeze(0),
                    )[None]
                if save_depth:
                    predictions.update({key: geometry[key] for key in ("depth_map", "dpt_cnf")})
                predictions["cam_tokens"] = features[-1][:, :, 0]
                if ret_img:
                    predictions["images"] = rgbs
        return [{key: value[:, i] for key, value in predictions.items()} for i in range(len(images))]

    @torch.no_grad()
    def co3d_forward(self, views, no_reloc_list, reloc_list, fix_rank=300):
        """Joint anchor/query pose inference used by upstream CO3D evaluation."""
        if self.aggregator.global_reloc_blocks[0].attn.kv_cache:
            raise RuntimeError("Construct SailRecon(kv_cache=False) for joint inference")
        anchors = torch.cat([views[i]["img"] for i in no_reloc_list], dim=0)
        queries = torch.cat([views[i]["img"] for i in reloc_list], dim=0)
        images = torch.cat([anchors, queries], dim=0).unsqueeze(0)
        features, _, camera_token = self.aggregator(
            images,
            list(range(len(no_reloc_list))),
            [i + len(no_reloc_list) for i in reloc_list],
            fix_rank=fix_rank,
        )
        with torch.autocast("cuda", enabled=False):
            poses = self.camera_head(features, camera_token)
            extrinsic, intrinsic = pose_encoding_to_extri_intri(poses[-1], images.shape[-2:])
        return extrinsic

    def tmp_forward(self, views, no_reloc_list=None, reloc_list=None, fix_rank=300):
        """Compatibility for upstream dataset inference drivers (anchor-only)."""
        images = views if isinstance(views, torch.Tensor) else torch.cat([v["img"] for v in views], dim=0)
        if reloc_list or (no_reloc_list is not None and no_reloc_list != list(range(len(images)))):
            raise ValueError("tmp_forward builds an anchor-only scene; use co3d_forward for joint inference")
        return self.build_scene(images, fix_rank)

    def reloc(
        self,
        views,
        no_reloc_list=None,
        fix_rank=300,
        memory_save=True,
        save_depth=True,
        fast_reloc=False,
        ret_img=False,
    ):
        images = views if isinstance(views, torch.Tensor) else torch.cat([v["img"] for v in views], dim=0)
        return self.localize(images, fix_rank, memory_save, save_depth, fast_reloc, ret_img)
