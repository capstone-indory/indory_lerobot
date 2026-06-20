#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEROBOT_DIR="${LEROBOT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
GROOT_DIR="${GROOT_DIR:-$(cd "${LEROBOT_DIR}/.." && pwd)/Isaac-GR00T-N1.7}"
DATASET_PATH="${DATASET_PATH:-${LEROBOT_DIR}/data/gr00t_n17/indory_xlerobot_pick_delivery_86ep10hz}"
MODALITY_CONFIG_PATH="${MODALITY_CONFIG_PATH:-${SCRIPT_DIR}/indory_xlerobot_config.py}"

cd "$GROOT_DIR"

export NO_ALBUMENTATIONS_UPDATE="${NO_ALBUMENTATIONS_UPDATE:-1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export DATASET_PATH
export MODALITY_CONFIG_PATH

uv run python - <<'PY'
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

from huggingface_hub import HfApi, hf_hub_download

from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.embodiment_tags import EmbodimentTag


dataset_path = Path(os.environ["DATASET_PATH"])
modality_config_path = Path(os.environ["MODALITY_CONFIG_PATH"])
hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

errors: list[str] = []

if not dataset_path.is_dir():
    errors.append(f"Dataset path does not exist: {dataset_path}")
if not modality_config_path.is_file():
    errors.append(f"Modality config does not exist: {modality_config_path}")

if dataset_path.is_dir():
    info_path = dataset_path / "meta" / "info.json"
    modality_path = dataset_path / "meta" / "modality.json"
    relative_stats_path = dataset_path / "meta" / "relative_stats.json"
    stats_path = dataset_path / "meta" / "stats.json"
    for path in [info_path, modality_path, relative_stats_path, stats_path]:
        if not path.is_file():
            errors.append(f"Missing dataset metadata: {path}")
    if info_path.is_file():
        info = json.loads(info_path.read_text())
        expected = {
            "codebase_version": "v2.1",
            "total_episodes": 86,
            "total_frames": 58167,
            "fps": 10,
        }
        for key, value in expected.items():
            if info.get(key) != value:
                errors.append(f"Unexpected info.json {key}: got {info.get(key)!r}, expected {value!r}")
        features = info.get("features", {})
        for key in [
            "action",
            "observation.state",
            "observation.images.head",
            "observation.images.left_wrist",
            "observation.images.right_wrist",
        ]:
            if key not in features:
                errors.append(f"Missing feature in info.json: {key}")

required_hf_files = {
    "nvidia/GR00T-N1.7-3B": [
        "config.json",
        "processor_config.json",
        "statistics.json",
        "embodiment_id.json",
    ],
    "nvidia/Cosmos-Reason2-2B": [
        "config.json",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.json",
        "merges.txt",
    ],
}

try:
    who = HfApi().whoami(token=hf_token) if hf_token else HfApi().whoami()
    print(f"huggingface auth OK user={who.get('name') or who.get('fullname')}")
    access_token = who.get("auth", {}).get("accessToken", {})
    if access_token:
        print(
            "huggingface token "
            f"name={access_token.get('displayName')!r} role={access_token.get('role')!r}"
        )
        fine_grained = access_token.get("fineGrained")
        if fine_grained:
            scoped_entities = [
                scope.get("entity", {}).get("name")
                for scope in fine_grained.get("scoped", [])
                if scope.get("entity", {}).get("name")
            ]
            print(
                "huggingface token fineGrained "
                f"canReadGatedRepos={fine_grained.get('canReadGatedRepos')!r} "
                f"global={fine_grained.get('global', [])!r} "
                f"scoped_entities={scoped_entities!r}"
            )
except Exception as exc:
    errors.append(f"huggingface auth failed: {type(exc).__name__}: {str(exc).splitlines()[0]}")

for repo, filenames in required_hf_files.items():
    try:
        info = HfApi().model_info(repo, token=hf_token)
        print(f"{repo}: model_info OK gated={info.gated!r} private={info.private!r}")
    except Exception as exc:
        errors.append(f"{repo}: model_info failed: {type(exc).__name__}: {str(exc).splitlines()[0]}")
        continue
    for filename in filenames:
        try:
            path = hf_hub_download(repo, filename, token=hf_token)
            print(f"{repo}: {filename} OK {path}")
        except Exception as exc:
            errors.append(
                f"{repo}: {filename} download failed: {type(exc).__name__}: {str(exc).splitlines()[0]}"
            )

if modality_config_path.is_file() and dataset_path.is_dir():
    try:
        spec = importlib.util.spec_from_file_location("indory_xlerobot_config", modality_config_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        tag = EmbodimentTag.NEW_EMBODIMENT.value
        loader = LeRobotEpisodeLoader(str(dataset_path), MODALITY_CONFIGS[tag])
        traj = loader[0]
        expected_cols = [
            "language.annotation.human.task_description",
            "state.left_arm",
            "state.left_gripper",
            "state.right_arm",
            "state.right_gripper",
            "state.head",
            "state.base_velocity",
            "action.left_arm",
            "action.left_gripper",
            "action.right_arm",
            "action.right_gripper",
            "action.head",
            "action.base_velocity",
            "video.head",
            "video.left_wrist",
            "video.right_wrist",
        ]
        missing = [col for col in expected_cols if col not in traj.columns]
        if missing:
            errors.append(f"Loader missing columns: {missing}")
        print(f"dataset loader OK episodes={len(loader)} episode0_rows={len(traj)}")
    except Exception as exc:
        errors.append(f"Dataset loader failed: {type(exc).__name__}: {str(exc).splitlines()[0]}")

try:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    print(result.stdout.strip())
except Exception as exc:
    errors.append(f"nvidia-smi failed: {type(exc).__name__}: {exc}")

if errors:
    print("N1.7 preflight failed:", file=sys.stderr)
    for error in errors:
        print(f"- {error}", file=sys.stderr)
    sys.exit(2)

print("N1.7 preflight OK")
PY
