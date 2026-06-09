from __future__ import annotations

import torch
from torch import nn

from lerobot.policies.pi0.modeling_pi0 import _filter_state_dict_for_compatible_shapes as pi0_filter
from lerobot.policies.pi05.modeling_pi05 import _filter_state_dict_for_compatible_shapes as pi05_filter


def test_pi_pretrained_shape_filter_keeps_compatible_tensors_and_skips_robot_heads():
    model = nn.Module()
    model.backbone = nn.Linear(3, 4)
    model.action_out_proj = nn.Linear(4, 17)
    checkpoint = {
        "backbone.weight": torch.ones(4, 3),
        "backbone.bias": torch.ones(4),
        "action_out_proj.weight": torch.ones(12, 4),
        "action_out_proj.bias": torch.ones(12),
    }

    for shape_filter in (pi0_filter, pi05_filter):
        compatible, skipped = shape_filter(model, checkpoint)

        assert set(compatible) == {"backbone.weight", "backbone.bias"}
        assert skipped == [
            ("action_out_proj.weight", (12, 4), (17, 4)),
            ("action_out_proj.bias", (12,), (17,)),
        ]
