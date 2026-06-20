from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WRAPPER = ROOT / "scripts/gr00t_n17/run_indory_n17_guarded_one_step.sh"
SMOKE_WRAPPER = ROOT / "scripts/gr00t_n17/run_indory_n17_robot_smoke.sh"


def write_fake_stack(tmp_path: Path, *, dry_run_increments: bool = False) -> Path:
    scripts = tmp_path / "scripts/gr00t_n17"
    scripts.mkdir(parents=True)
    (scripts / "run_probe_indory_zmq.sh").write_text(
        """#!/usr/bin/env bash
set -euo pipefail
python - <<'PY'
import json
import os
from pathlib import Path

counter = Path(os.environ["COUNTER_FILE"])
print(json.dumps({"rpc": {"health": {"accepted_commands": int(counter.read_text())}}}))
PY
""",
        encoding="utf-8",
    )
    (scripts / "run_indory_n17_robot_smoke.sh").write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
python - <<'PY'
import os
from pathlib import Path

counter = Path(os.environ["COUNTER_FILE"])
value = int(counter.read_text())
if os.environ.get("SEND") == "1":
    duration_s = float(os.environ.get("DURATION_S", "0") or "0")
    fps = float(os.environ.get("FPS", "1") or "1")
    n_steps = 1 if duration_s <= 0 else max(1, int(round(duration_s * fps)))
    counter.write_text(str(value + n_steps + 1))
elif {str(dry_run_increments)}:
    counter.write_text(str(value + 1))
print("fake smoke SEND=" + os.environ.get("SEND", ""))
PY
""",
        encoding="utf-8",
    )
    for path in scripts.iterdir():
        path.chmod(0o755)
    return tmp_path


def run_wrapper(tmp_path: Path, *, send: bool, dry_run_increments: bool = False) -> subprocess.CompletedProcess[str]:
    fake_root = write_fake_stack(tmp_path, dry_run_increments=dry_run_increments)
    counter = tmp_path / "counter.txt"
    counter.write_text("7", encoding="utf-8")
    env = {
        **os.environ,
        "LEROBOT_DIR": str(fake_root),
        "COUNTER_FILE": str(counter),
        "REMOTE_IP": "127.0.0.1",
        "DURATION_S": "3",
        "FPS": "1",
    }
    if send:
        env["SEND"] = "1"
        env["CONFIRM_OPERATOR_READY"] = "1"
    return subprocess.run(
        ["bash", str(WRAPPER)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_guarded_wrapper_dry_run_forces_send_zero_and_keeps_counter(tmp_path):
    result = run_wrapper(tmp_path, send=True)

    assert result.returncode == 0, result.stderr
    assert "fake smoke SEND=0" in result.stdout
    assert "fake smoke SEND=1" in result.stdout
    assert "accepted_commands_pre=7" in result.stdout
    assert "accepted_commands_mid=7" in result.stdout
    assert "accepted_commands_post=11" in result.stdout
    assert "accepted_commands_delta=4" in result.stdout


def test_guarded_wrapper_refuses_if_dry_run_changes_counter(tmp_path):
    result = run_wrapper(tmp_path, send=True, dry_run_increments=True)

    assert result.returncode == 4
    assert "Dry-run changed accepted_commands (7 -> 8); refusing to send." in result.stderr
    assert "fake smoke SEND=1" not in result.stdout


def test_guarded_wrapper_default_cap_matches_recording_pipeline(tmp_path):
    env = {
        **os.environ,
        "LEROBOT_DIR": str(tmp_path),
        "REMOTE_IP": "127.0.0.1",
        "DURATION_S": "3",
        "FPS": "10",
        "SEND": "1",
        "CONFIRM_OPERATOR_READY": "1",
        "PRINT_COMMAND": "1",
    }

    result = subprocess.run(
        ["bash", str(WRAPPER)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("MAX_RELATIVE_TARGET=10.0") == 2


def test_robot_smoke_wrapper_send_default_cap_matches_recording_pipeline():
    env = {
        **os.environ,
        "REMOTE_IP": "127.0.0.1",
        "SEND": "1",
        "PRINT_COMMAND": "1",
    }

    result = subprocess.run(
        ["bash", str(SMOKE_WRAPPER)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--max-relative-target 10.0" in result.stdout
