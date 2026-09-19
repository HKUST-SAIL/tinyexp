import importlib
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from plyfile import PlyData

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "third_party" / "sail-recon"))

from sailrecon.layers.attention import Attention as ReferenceAttention
from sailrecon.models.aggregator import build_allow_block as reference_mask
from sailrecon.utils.load_fn import load_and_preprocess_images as reference_preprocess

from tinyexp.sail_recon.io import cpu_predictions, save_pointcloud, uniform_sample
from tinyexp.sail_recon.layers.attention import Attention
from tinyexp.sail_recon.models.aggregator import build_allow_block
from tinyexp.sail_recon.models.model import SailRecon
from tinyexp.sail_recon.utils.load_fn import load_and_preprocess_images


@pytest.mark.parametrize("mode", ["crop", "pad"])
def test_preprocess_mixed_shapes_rgba_grayscale(tmp_path, mode):
    paths = []
    rng = np.random.default_rng(10)
    for i, shape in enumerate(((121, 80, 4), (83, 191, 3), (77, 101))):
        path = tmp_path / f"{i}.png"
        Image.fromarray(rng.integers(0, 256, shape, dtype=np.uint8)).save(path)
        paths.append(str(path))
    actual = load_and_preprocess_images(paths, mode)
    expected = reference_preprocess(paths, mode)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.shape[0:2] == (3, 3)
    assert all(size % 14 == 0 for size in actual.shape[-2:])


def test_selection_and_mask():
    assert uniform_sample(7, 3) == [0, 2, 4]
    for total, select in ((0, 0), (3, 4), (3, 0)):
        with pytest.raises(ValueError):
            uniform_sample(total, select)
    torch.testing.assert_close(build_allow_block(4, [0, 1], [2, 3]), reference_mask(4, [0, 1], [2, 3]))


@pytest.mark.parametrize("fused", [True, False])
def test_attention_full_tensor(fused):
    torch.manual_seed(8)
    reference = ReferenceAttention(32, num_heads=4, qk_norm=True, fused_attn=fused).eval()
    candidate = Attention(32, num_heads=4, qk_norm=True, fused_attn=fused).eval()
    candidate.load_state_dict(reference.state_dict(), strict=True)
    x = torch.randn(2, 17, 32)
    torch.testing.assert_close(candidate(x), reference(x), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="KV cache upstream explicitly requires CUDA")
def test_attention_cache_full_tensor():
    torch.manual_seed(9)
    reference = ReferenceAttention(32, num_heads=4, kv_cache=True).eval().cuda()
    candidate = Attention(32, num_heads=4, kv_cache=True).eval().cuda()
    candidate.load_state_dict(reference.state_dict())
    with torch.no_grad():
        for count in (11, 5, 3):
            x = torch.randn(1, count, 32, device="cuda")
            torch.testing.assert_close(candidate(x), reference(x), rtol=0, atol=0)
        torch.testing.assert_close(candidate.k_cache, reference.k_cache, rtol=0, atol=0)
    candidate.clear_kv_cache()
    assert candidate.k_cache is None and candidate.v_cache is None


def test_geometry_pose_roundtrip():
    original = importlib.import_module("sailrecon.utils.pose_enc")
    replica = importlib.import_module("tinyexp.sail_recon.utils.pose_enc")
    torch.manual_seed(11)
    encoding = torch.randn(1, 3, 9)
    encoding[..., -2:] = 1.0
    expected = original.pose_encoding_to_extri_intri(encoding, (392, 518))
    actual = replica.pose_encoding_to_extri_intri(encoding, (392, 518))
    for x, y in zip(actual, expected):
        torch.testing.assert_close(x, y, rtol=0, atol=0)


def test_empty_cloud_writes_valid_ply(tmp_path):
    view = {
        "point_map_by_unprojection": torch.zeros(1, 14, 14, 3),
        "rgbs": torch.zeros(1, 3, 14, 14),
        "dpt_cnf": torch.ones(1, 14, 14),
    }
    path = tmp_path / "empty.ply"
    save_pointcloud([view], path)
    assert len(PlyData.read(path)["vertex"]) == 0


def test_numpy_unprojection_boundary():
    source = [{"points": np.ones((1, 3, 3), dtype=np.float64), "depth": torch.ones(1, 3, 3)}]
    result = cpu_predictions(source)
    assert result[0]["points"].dtype == torch.float64
    assert result[0]["depth"].device.type == "cpu"


@pytest.mark.parametrize("shape", [(0, 3, 14, 14), (1, 1, 14, 14), (1, 3, 15, 14)])
def test_invalid_image_shapes(shape):
    with pytest.raises(ValueError):
        SailRecon.validate_images(torch.zeros(shape))
