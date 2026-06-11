"""Training utilities and defaults for the Indory parcel grasp detector."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DATASET = REPO_ROOT / "data" / "parcel_obb_dataset"
DEFAULT_MODEL_DIR = REPO_ROOT / "data" / "models" / "parcel_obb"
DEFAULT_MODEL_PATH = DEFAULT_MODEL_DIR / "best.pt"
