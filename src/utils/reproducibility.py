"""Reproducibility utilities for consistent experiments."""

import os
import random
import numpy as np
import torch
import subprocess
from typing import Dict, Any


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)


def get_git_commit() -> str:
    try:
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("ascii")
            .strip()
        )
        return commit
    except:
        return "unknown"


def get_env_info() -> Dict[str, Any]:
    """Get environment information."""
    info = {
        "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "git_commit": get_git_commit(),
    }

    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()

    return info


def print_env_info():
    info = get_env_info()
    print("Environment Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
