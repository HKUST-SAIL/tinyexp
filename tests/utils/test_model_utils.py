from __future__ import annotations

import torch

from tinyexp.utils.model_utils import update_ema


def test_update_ema_updates_parameters() -> None:
    model = torch.nn.Linear(2, 1, bias=False)
    ema_model = torch.nn.Linear(2, 1, bias=False)

    with torch.no_grad():
        model.weight.copy_(torch.tensor([[2.0, 4.0]]))
        ema_model.weight.copy_(torch.tensor([[10.0, 20.0]]))

    update_ema(ema_model, model, decay=0.5)

    expected = torch.tensor([[6.0, 12.0]])  # 10*0.5+2*0.5, 20*0.5+4*0.5
    assert torch.allclose(ema_model.weight, expected)
