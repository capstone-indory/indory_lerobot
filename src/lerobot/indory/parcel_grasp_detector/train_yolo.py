#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import shutil
import subprocess
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ZIP = REPO_ROOT / "아카이브.zip"
DEFAULT_ARCHIVES = [DEFAULT_ZIP, REPO_ROOT / "아카이브 2.zip"]
DEFAULT_DATASET = REPO_ROOT / "data" / "parcel_obb_dataset"
DEFAULT_MODEL_DIR = REPO_ROOT / "data" / "models" / "parcel_obb"


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare and train the Indoory parcel YOLO detector.")
    parser.add_argument("--archive", type=Path, action="append", default=None)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--base-model", default="yolo11n-obb.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="")
    parser.add_argument("--val-every", type=int, default=5)
    parser.add_argument("--augment-copies", type=int, default=6)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    archives = args.archive if args.archive else [path for path in DEFAULT_ARCHIVES if path.exists()]
    prepare_dataset(archives, args.dataset, args.val_every, args.augment_copies)
    if args.prepare_only:
        print(f"dataset ready: {args.dataset}")
        return 0
    return train(args)


def prepare_dataset(archives: list[Path], dataset: Path, val_every: int, augment_copies: int) -> None:
    if not archives:
        raise SystemExit("no archives found")
    for archive in archives:
        if not archive.exists():
            raise SystemExit(f"archive not found: {archive}")
    raw_dir = dataset / "raw"
    for generated in (dataset / "images", dataset / "labels"):
        if generated.exists():
            shutil.rmtree(generated)
    for path in (
        raw_dir,
        dataset / "images" / "train",
        dataset / "images" / "val",
        dataset / "labels" / "train",
        dataset / "labels" / "val",
    ):
        path.mkdir(parents=True, exist_ok=True)

    for archive_index, archive in enumerate(archives, start=1):
        with zipfile.ZipFile(archive) as zf:
            for member in zf.namelist():
                name = Path(member).name
                if not name or member.startswith("__MACOSX/") or name.startswith("._"):
                    continue
                if Path(name).suffix.lower() not in {".heic", ".heif", ".jpg", ".jpeg", ".png"}:
                    continue
                out = raw_dir / f"a{archive_index}_{name}"
                if not out.exists():
                    out.write_bytes(zf.read(member))

    raw_images = []
    for index, src in enumerate(sorted(raw_dir.iterdir())):
        if src.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            split = "val" if val_every > 0 and (index + 1) % val_every == 0 else "train"
            dst = dataset / "images" / split / src.with_suffix(".jpg").name
            if not dst.exists():
                shutil.copy2(src, dst)
            raw_images.append(dst)
        elif src.suffix.lower() in {".heic", ".heif"}:
            split = "val" if val_every > 0 and (index + 1) % val_every == 0 else "train"
            dst = dataset / "images" / split / src.with_suffix(".jpg").name
            convert_heic(src, dst)
            raw_images.append(dst)

    if not raw_images:
        raise SystemExit(f"no train images extracted from {', '.join(str(path) for path in archives)}")
    labeled = 0
    train_originals = []
    for image in raw_images:
        split = image.parent.name
        label = dataset / "labels" / split / f"{image.stem}.txt"
        polygon = auto_label_blue_parcel_obb(image)
        if polygon is None:
            label.write_text("", encoding="utf-8")
            continue
        label.write_text(format_obb_label(polygon), encoding="utf-8")
        labeled += 1
        if split == "train":
            train_originals.append((image, polygon))

    augmented = augment_train_set(dataset, train_originals, augment_copies)

    data_yaml = dataset / "parcel.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {dataset}",
                "train: images/train",
                "val: images/val",
                "task: obb",
                "names:",
                "  0: parcel",
                "",
            ]
        ),
        encoding="utf-8",
    )
    train_count = len(list((dataset / "images" / "train").glob("*.jpg")))
    val_count = len(list((dataset / "images" / "val").glob("*.jpg")))
    print(
        f"prepared {len(raw_images)} images from {len(archives)} archives at {dataset} "
        f"({train_count} train, {val_count} val)"
    )
    print(f"auto-labeled {labeled} rotated parcel boxes, augmented {augmented} train images")


def convert_heic(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        import pillow_heif
        from PIL import Image

        pillow_heif.register_heif_opener()
        with Image.open(src) as image:
            image.convert("RGB").save(dst, quality=95)
        return
    except Exception:
        pass

    magick = shutil.which("magick")
    if magick:
        subprocess.run([magick, str(src), str(dst)], check=True)
        return
    heif_convert = shutil.which("heif-convert")
    if heif_convert:
        subprocess.run([heif_convert, str(src), str(dst)], check=True)
        return
    raise SystemExit(
        "HEIC conversion requires one of: `pip install pillow-heif`, ImageMagick `magick`, or `heif-convert`."
    )


def auto_label_blue_parcel_obb(image_path: Path) -> list[tuple[float, float]] | None:
    import cv2
    import numpy as np

    image = cv2.imread(str(image_path))
    if image is None:
        return None
    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([82, 20, 50]), np.array([125, 255, 255]))
    mask = cv2.medianBlur(mask, 7)
    kernel = np.ones((11, 11), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > max(500.0, width * height * 0.01)]
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = expand_polygon(box, max(width, height) * 0.015, width, height)
    return [(float(x) / width, float(y) / height) for x, y in order_polygon(box)]


def augment_train_set(
    dataset: Path, originals: list[tuple[Path, list[tuple[float, float]]]], copies: int
) -> int:
    if copies <= 0:
        return 0
    made = 0
    for image_path, polygon in originals:
        image = read_cv_image(image_path)
        if image is None:
            continue
        height, width = image.shape[:2]
        for index in range(copies):
            rng = random.Random(f"{image_path.stem}-{index}")
            augmented, aug_polygon = augment_image_and_polygon(image, polygon, rng)
            if aug_polygon is None:
                continue
            out_image = dataset / "images" / "train" / f"{image_path.stem}_aug{index + 1:02d}.jpg"
            out_label = dataset / "labels" / "train" / f"{image_path.stem}_aug{index + 1:02d}.txt"
            cv2_imwrite(out_image, augmented)
            out_label.write_text(format_obb_label(aug_polygon), encoding="utf-8")
            made += 1
    return made


def augment_image_and_polygon(
    image: "np.ndarray", polygon: list[tuple[float, float]], rng: random.Random
) -> tuple["np.ndarray", list[tuple[float, float]] | None]:
    import cv2
    import numpy as np

    height, width = image.shape[:2]
    points = np.array([[x * width, y * height] for x, y in polygon], dtype=np.float32)

    if rng.random() < 0.5:
        image = cv2.flip(image, 1)
        points[:, 0] = width - points[:, 0]

    angle = rng.uniform(-65.0, 65.0)
    scale = rng.uniform(0.78, 1.18)
    tx = rng.uniform(-0.12, 0.12) * width
    ty = rng.uniform(-0.12, 0.12) * height
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, scale)
    matrix[0, 2] += tx
    matrix[1, 2] += ty
    warped = cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    hom = np.concatenate([points, np.ones((4, 1), dtype=np.float32)], axis=1)
    transformed = hom @ matrix.T
    transformed[:, 0] = np.clip(transformed[:, 0], 0, width)
    transformed[:, 1] = np.clip(transformed[:, 1], 0, height)
    if cv2.contourArea(transformed.astype(np.float32)) < width * height * 0.01:
        return warped, None

    alpha = rng.uniform(0.78, 1.24)
    beta = rng.uniform(-22.0, 22.0)
    warped = cv2.convertScaleAbs(warped, alpha=alpha, beta=beta)
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * rng.uniform(0.82, 1.22), 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * rng.uniform(0.85, 1.18), 0, 255)
    warped = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    if rng.random() < 0.35:
        noise = rng.uniform(2.0, 7.0)
        warped = np.clip(warped.astype(np.float32) + np.random.default_rng(rng.randrange(1_000_000)).normal(0, noise, warped.shape), 0, 255).astype(np.uint8)
    if rng.random() < 0.25:
        warped = cv2.GaussianBlur(warped, (3, 3), 0)

    ordered = order_polygon(transformed)
    return warped, [(float(x) / width, float(y) / height) for x, y in ordered]


def order_polygon(points) -> list[tuple[float, float]]:
    import numpy as np

    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    ordered = pts[np.argsort(angles)]
    start = np.argmin(ordered.sum(axis=1))
    ordered = np.roll(ordered, -start, axis=0)
    return [(float(x), float(y)) for x, y in ordered]


def expand_polygon(points, pad: float, width: int, height: int):
    import numpy as np

    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    center = pts.mean(axis=0)
    vectors = pts - center
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    lengths[lengths == 0] = 1.0
    expanded = pts + vectors / lengths * float(pad)
    expanded[:, 0] = np.clip(expanded[:, 0], 0, width)
    expanded[:, 1] = np.clip(expanded[:, 1], 0, height)
    return expanded


def format_obb_label(polygon: list[tuple[float, float]]) -> str:
    coords = []
    for x, y in polygon:
        coords.extend([max(0.0, min(1.0, x)), max(0.0, min(1.0, y))])
    return "0 " + " ".join("%.6f" % value for value in coords) + "\n"


def read_cv_image(path: Path):
    import cv2

    return cv2.imread(str(path))


def cv2_imwrite(path: Path, image) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 94])


def train(args: argparse.Namespace) -> int:
    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise SystemExit(f"ultralytics is not installed: {exc}\nInstall with: pip install ultralytics") from exc

    args.model_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.base_model)
    train_kwargs = {
        "data": str(args.dataset / "parcel.yaml"),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "project": str(args.model_dir),
        "name": "train",
        "exist_ok": True,
    }
    if args.device:
        train_kwargs["device"] = args.device
    model.train(**train_kwargs)
    best = args.model_dir / "train" / "weights" / "best.pt"
    if best.exists():
        target = args.model_dir / "best.pt"
        shutil.copy2(best, target)
        print(f"runtime model ready: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
