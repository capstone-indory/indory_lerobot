#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

SO101_PI05 = "CoRL2026-CSI/pi05_teleop_fold_towel"
PI0_BASE = "lerobot/pi0_base"
SO101_GROOT = "CoRL2026-CSI/Gr00t_n1.5-IsaacLab-SO101-Multi_Task-30fps_8epoch"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch Indory XLerobot training from a SO101 Pi0/Pi0.5 or GR00T pretrained policy."
    )
    parser.add_argument(
        "--dataset-repo-id",
        default=os.environ.get("INDORY_DATASET_REPO_ID", "capstone-indory/indory_xlerobot_pick_delivery"),
    )
    parser.add_argument("--dataset-root", default="data/indory_xlerobot_pick_delivery_head_patched")
    parser.add_argument("--dataset-video-backend", default="pyav")
    parser.add_argument("--policy-type", choices=("pi05", "pi0", "groot"), default="pi05")
    parser.add_argument("--pretrained-path", default=None)
    parser.add_argument("--policy-optimizer-lr", type=float, default=None)
    parser.add_argument("--policy-scheduler-decay-lr", type=float, default=None)
    parser.add_argument("--policy-scheduler-warmup-steps", type=int, default=None)
    parser.add_argument("--policy-scheduler-decay-steps", type=int, default=None)
    parser.add_argument("--pi-gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pi-compile-model", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--pi-freeze-vision-encoder", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--pi-train-expert-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--job-name", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-freq", type=int, default=5_000)
    parser.add_argument("--log-freq", type=int, default=100)
    parser.add_argument("--eval-freq", type=int, default=0)
    parser.add_argument("--num-gpus", type=int, default=None)
    parser.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"), default="bf16")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wandb-project", default="indory_xlerobot")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--detach", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--smoke", action="store_true", help="Run a 1-step training smoke test.")
    parser.add_argument("--precache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-checkpoint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hf-disable-xet", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--require-wandb-login",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail early when --wandb-mode=online but no W&B API key is configured.",
    )
    return parser.parse_args()


def default_pretrained_path(policy_type: str) -> str:
    if policy_type == "pi05":
        return SO101_PI05
    if policy_type == "pi0":
        return PI0_BASE
    if policy_type == "groot":
        return SO101_GROOT
    raise ValueError(policy_type)


def detect_num_gpus() -> int:
    try:
        import torch

        return torch.cuda.device_count()
    except Exception:
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible_devices:
            return len([part for part in cuda_visible_devices.split(",") if part.strip()])
        return 0


def build_env(repo_root: Path, args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    src_path = str(repo_root / "src")
    env["PYTHONPATH"] = src_path + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    if args.hf_disable_xet:
        env.setdefault("HF_HUB_DISABLE_XET", "1")
    env["WANDB_MODE"] = args.wandb_mode
    return env


def build_precache_cmd(args: argparse.Namespace, pretrained_path: str) -> list[str]:
    return [
        sys.executable,
        "scripts/indory_precache_policy.py",
        "--policy-type",
        args.policy_type,
        "--pretrained-path",
        pretrained_path,
    ]


def build_train_cmd(args: argparse.Namespace, pretrained_path: str) -> list[str]:
    num_gpus = args.num_gpus if args.num_gpus is not None else detect_num_gpus()
    output_dir = resolve_output_dir(args)
    job_name = resolve_job_name(args)
    steps = 1 if args.smoke else args.steps
    save_freq = 1 if args.smoke else args.save_freq
    save_checkpoint = args.save_checkpoint and not args.smoke

    if args.smoke:
        output_dir = resolve_output_dir(args)
        job_name = resolve_job_name(args)

    if num_gpus > 1:
        cmd = [
            sys.executable,
            "-m",
            "accelerate.commands.launch",
            "--multi_gpu",
            f"--num_processes={num_gpus}",
            "--num_machines=1",
            f"--mixed_precision={args.mixed_precision}",
            "--dynamo_backend=no",
            "-m",
            "lerobot.scripts.lerobot_train",
        ]
    else:
        cmd = [sys.executable, "-m", "lerobot.scripts.lerobot_train"]

    train_args = [
        f"--dataset.repo_id={args.dataset_repo_id}",
        f"--dataset.root={args.dataset_root}",
        f"--dataset.video_backend={args.dataset_video_backend}",
        f"--output_dir={output_dir}",
        f"--job_name={job_name}",
        f"--batch_size={args.batch_size}",
        f"--steps={steps}",
        f"--num_workers={args.num_workers}",
        f"--resume={str(args.resume).lower()}",
        f"--save_checkpoint={str(save_checkpoint).lower()}",
        f"--save_freq={save_freq}",
        f"--log_freq={args.log_freq}",
        f"--eval_freq={args.eval_freq}",
        f"--policy.type={args.policy_type}",
        f"--policy.pretrained_path={pretrained_path}",
        "--policy.push_to_hub=false",
        "--wandb.enable=true",
        f"--wandb.project={args.wandb_project}",
        f"--wandb.mode={args.wandb_mode}",
    ]

    if args.wandb_entity:
        train_args.append(f"--wandb.entity={args.wandb_entity}")

    policy_overrides = {
        "optimizer_lr": args.policy_optimizer_lr,
        "scheduler_decay_lr": args.policy_scheduler_decay_lr,
        "scheduler_warmup_steps": args.policy_scheduler_warmup_steps,
        "scheduler_decay_steps": args.policy_scheduler_decay_steps,
    }

    if args.policy_type in {"pi0", "pi05"}:
        train_args.extend(
            [
                "--policy.dtype=bfloat16",
                f"--policy.gradient_checkpointing={str(args.pi_gradient_checkpointing).lower()}",
                f"--policy.compile_model={str(args.pi_compile_model).lower()}",
                f"--policy.freeze_vision_encoder={str(args.pi_freeze_vision_encoder).lower()}",
                f"--policy.train_expert_only={str(args.pi_train_expert_only).lower()}",
            ]
        )
    elif args.policy_type == "groot":
        train_args.extend(
            [
                "--policy.chunk_size=16",
                "--policy.n_action_steps=16",
                "--policy.tune_llm=false",
                "--policy.tune_visual=false",
                "--policy.tune_projector=true",
                "--policy.tune_diffusion_model=true",
                "--policy.use_bf16=true",
            ]
        )
    train_args.extend(
        f"--policy.{name}={value}" for name, value in policy_overrides.items() if value is not None
    )

    return cmd + train_args


def print_command(label: str, cmd: list[str]) -> None:
    print(f"{label}:")
    print(shlex.join(cmd))


def resolve_output_dir(args: argparse.Namespace) -> str:
    if args.output_dir:
        return args.output_dir
    if args.smoke:
        return f"outputs/train/_smoke_indory_xlerobot_{args.policy_type}_so101"
    return f"outputs/train/indory_xlerobot_{args.policy_type}_so101_wandb"


def resolve_job_name(args: argparse.Namespace) -> str:
    if args.job_name:
        return args.job_name
    if args.smoke:
        return f"_smoke_indory_xlerobot_{args.policy_type}_so101"
    return f"indory_xlerobot_{args.policy_type}_so101"


def resolve_log_file(args: argparse.Namespace) -> Path:
    if args.log_file:
        return Path(args.log_file)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("outputs/train/_logs") / f"{resolve_job_name(args)}_{timestamp}.log"


def detach_args(argv: list[str]) -> list[str]:
    return [arg for arg in argv if arg not in {"--detach", "--no-detach"}]


def launch_detached(repo_root: Path, env: dict[str, str], args: argparse.Namespace) -> int:
    log_file = resolve_log_file(args)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    child_cmd = [sys.executable, str(Path(__file__).resolve()), *detach_args(sys.argv[1:])]
    with log_file.open("ab", buffering=0) as log:
        process = subprocess.Popen(
            child_cmd,
            cwd=repo_root,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    print(f"detached_pid={process.pid}")
    print(f"log_file={log_file}")
    print("command:")
    print(shlex.join(child_cmd))
    return 0


def wandb_api_key_configured(env: dict[str, str]) -> bool:
    if env.get("WANDB_API_KEY"):
        return True

    try:
        result = subprocess.run(
            [sys.executable, "-m", "wandb", "status"],
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return False

    try:
        status = json.loads(result.stdout[result.stdout.index("{") :])
    except (ValueError, json.JSONDecodeError):
        return False
    return bool(status.get("api_key"))


def output_dir_has_conflict(args: argparse.Namespace) -> bool:
    return Path(resolve_output_dir(args)).exists() and not args.resume


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)
    env = build_env(repo_root, args)
    pretrained_path = args.pretrained_path or default_pretrained_path(args.policy_type)

    precache_cmd = build_precache_cmd(args, pretrained_path)
    train_cmd = build_train_cmd(args, pretrained_path)

    if args.dry_run:
        if args.precache:
            print_command("precache", precache_cmd)
        print_command("train", train_cmd)
        return 0

    if args.wandb_mode == "online" and args.require_wandb_login and not wandb_api_key_configured(env):
        print(
            "W&B online mode requested, but no API key is configured. "
            "Run `python -m wandb login` in the lerobot environment, "
            "set WANDB_API_KEY, or use `--wandb-mode offline`.",
            file=sys.stderr,
        )
        return 2

    if output_dir_has_conflict(args):
        print(
            f"Output directory already exists: {resolve_output_dir(args)}. "
            "Use --output-dir for a new run or --resume to continue an existing run.",
            file=sys.stderr,
        )
        return 3

    if args.detach:
        return launch_detached(repo_root, env, args)

    if args.precache:
        subprocess.run(precache_cmd, env=env, check=True)
    return subprocess.run(train_cmd, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
