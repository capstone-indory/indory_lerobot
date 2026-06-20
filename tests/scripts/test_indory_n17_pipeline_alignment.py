from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.gr00t_n17.indory_n17_robot_smoke import BASE_KEYS, HEAD_MOTORS, LEFT_MOTORS, RIGHT_MOTORS

ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = ROOT / "data/gr00t_n17/indory_xlerobot_pick_delivery_86ep10hz"
CHECKPOINT = (
    ROOT
    / "outputs/train/indory_xlerobot_groot_n17_86ep10hz_20k"
    / "indory_xlerobot_groot_n17_86ep10hz_20k/checkpoint-10000"
)

pytestmark = pytest.mark.skipif(
    not DATASET_ROOT.is_dir() or not CHECKPOINT.is_dir(),
    reason="Indory GR00T dataset/checkpoint artifacts are local and not tracked in the public repository.",
)


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def test_training_dataset_schema_matches_live_inference_groups():
    info = read_json(DATASET_ROOT / "meta/info.json")
    modality = read_json(DATASET_ROOT / "meta/modality.json")
    expected_names = [*LEFT_MOTORS, *RIGHT_MOTORS, *HEAD_MOTORS, *BASE_KEYS]

    assert info["fps"] == 10
    assert info["features"]["observation.state"]["names"] == expected_names
    assert info["features"]["action"]["names"] == expected_names

    assert modality["state"] == {
        "left_arm": {"start": 0, "end": 5},
        "left_gripper": {"start": 5, "end": 6},
        "right_arm": {"start": 6, "end": 11},
        "right_gripper": {"start": 11, "end": 12},
        "head": {"start": 12, "end": 14},
        "base_velocity": {"start": 14, "end": 17},
    }
    assert modality["action"] == modality["state"]
    assert modality["video"] == {
        "head": {"original_key": "observation.images.head"},
        "left_wrist": {"original_key": "observation.images.left_wrist"},
        "right_wrist": {"original_key": "observation.images.right_wrist"},
    }


def test_checkpoint_processor_matches_indory_dataset_modalities():
    processor_cfg = read_json(CHECKPOINT / "processor_config.json")["processor_kwargs"]
    embodiment = processor_cfg["modality_configs"]["new_embodiment"]

    assert processor_cfg["use_relative_action"] is True
    assert embodiment["video"]["delta_indices"] == [0]
    assert embodiment["video"]["modality_keys"] == ["head", "left_wrist", "right_wrist"]
    assert embodiment["state"]["delta_indices"] == [0]
    assert embodiment["state"]["modality_keys"] == [
        "left_arm",
        "left_gripper",
        "right_arm",
        "right_gripper",
        "head",
        "base_velocity",
    ]
    assert embodiment["action"]["delta_indices"] == list(range(16))
    assert embodiment["action"]["modality_keys"] == embodiment["state"]["modality_keys"]
    assert [cfg["rep"] for cfg in embodiment["action"]["action_configs"]] == [
        "RELATIVE",
        "ABSOLUTE",
        "RELATIVE",
        "ABSOLUTE",
        "RELATIVE",
        "ABSOLUTE",
    ]


def test_recording_wrapper_defaults_match_training_dataset_rate_and_cap():
    record_script = (ROOT / "scripts/indory_mac_record.sh").read_text(encoding="utf-8")

    assert 'fps="${INDORY_FPS:-10}"' in record_script
    assert 'max_relative_target="${INDORY_MAX_RELATIVE_TARGET:-10.0}"' in record_script
