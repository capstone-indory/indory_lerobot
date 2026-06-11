# Indory Parcel Grasp Detector

This package owns the parcel grasp detector training pipeline used by the
Indory live-control runner.

- `train_yolo.py`: prepares the parcel OBB dataset, applies offline augmentation, and trains the YOLO-OBB model.
- Runtime model output: `data/models/parcel_obb/best.pt`.
- Runtime success detector alias: `--success-detector parcel-grasp-yolo`.

Example:

```bash
PYTHONPATH=src python -m lerobot.indory.parcel_grasp_detector.train_yolo --prepare-only
PYTHONPATH=src python -m lerobot.indory.parcel_grasp_detector.train_yolo --device 0
```

The model file is a local runtime artifact under `data/`, which is ignored by
git in this repo.
