#!/usr/bin/env python

import os

import torch
from accelerate import Accelerator


def main() -> None:
    accelerator = Accelerator()
    rank = os.environ.get("RANK", "0")
    local_rank = os.environ.get("LOCAL_RANK", "0")
    current_device = torch.cuda.current_device() if torch.cuda.is_available() else None
    device_name = torch.cuda.get_device_name(current_device) if current_device is not None else "cpu"
    print(
        f"rank={rank} local_rank={local_rank} "
        f"cuda={torch.cuda.is_available()} devices={torch.cuda.device_count()} "
        f"accelerator_device={accelerator.device} "
        f"current_device={current_device} name={device_name}",
        flush=True,
    )
    accelerator.end_training()


if __name__ == "__main__":
    main()
