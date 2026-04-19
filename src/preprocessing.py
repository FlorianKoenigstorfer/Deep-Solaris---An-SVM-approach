"""
preprocessing.py — Image preprocessing utilities for binary image classification.

Pipeline per image: resize with padding → grayscale → 4× rotational augmentation.
"""

import logging
import os

import pandas as pd
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

_SUBDIRS = [
    ("Train", "Positives", "train", "positive"),
    ("Train", "Negatives", "train", "negative"),
    ("Test",  "Positives", "test",  "positive"),
    ("Test",  "Negatives", "test",  "negative"),
]


def resize_with_padding(image: Image.Image, target_size: tuple = (75, 75)) -> Image.Image:
    """Scale *image* to fit within *target_size* preserving aspect ratio, pad with black."""
    target_w, target_h = target_size
    orig_w, orig_h = image.size

    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)

    resized = image.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new(image.mode, (target_w, target_h), 0)
    offset_x = (target_w - new_w) // 2
    offset_y = (target_h - new_h) // 2
    canvas.paste(resized, (offset_x, offset_y))

    return canvas


def generate_rotations(image: Image.Image) -> dict:
    """Return four CCW rotational copies of *image* (0°, 90°, 180°, 270°).

    ``expand=False`` is safe because the image is square after preprocessing.
    """
    return {
        "0":   image,
        "90":  image.rotate(90,  expand=False),
        "180": image.rotate(180, expand=False),
        "270": image.rotate(270, expand=False),
    }


def preprocess_dataset(
    data_root: str,
    output_root: str,
    target_size: tuple = (75, 75),
) -> pd.DataFrame:
    """Preprocess every image in the dataset and save rotational augmentations.

    For each valid image: resize_with_padding → grayscale → generate_rotations.
    Each of the four variants is saved under *output_root* mirroring the
    split/label directory structure, with a rotation suffix before the extension
    (e.g. ``roof_001_rot90.png``).

    Returns
    -------
    pd.DataFrame
        One row per saved variant. Columns: ``original_filepath``,
        ``output_filepath``, ``split``, ``label``, ``rotation``,
        ``width``, ``height``.
    """
    candidates = []
    for split_dir, label_dir, split, label in _SUBDIRS:
        folder = os.path.join(data_root, split_dir, label_dir)
        if not os.path.isdir(folder):
            continue
        for fname in sorted(os.listdir(folder)):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in _IMAGE_EXTENSIONS:
                continue
            candidates.append((os.path.join(folder, fname), split, label))

    rows = []
    for fpath, split, label in tqdm(candidates, desc="Preprocessing images"):
        try:
            image = Image.open(fpath)
            image.load()  # Ensure the file is fully read before processing.
        except Exception as exc:
            logger.warning("Skipping %s — could not load: %s", fpath, exc)
            continue

        try:
            padded = resize_with_padding(image, target_size)
            gray = padded.convert("L")
            rotations = generate_rotations(gray)
        except Exception as exc:
            logger.warning("Skipping %s — preprocessing failed: %s", fpath, exc)
            continue

        # Mirror the split/label directory structure under output_root.
        rel_path = os.path.relpath(fpath, data_root)
        rel_dir = os.path.dirname(rel_path)
        base, ext = os.path.splitext(os.path.basename(rel_path))

        out_dir = os.path.join(output_root, rel_dir)
        os.makedirs(out_dir, exist_ok=True)

        for angle, rot_img in rotations.items():
            out_fname = f"{base}_rot{angle}{ext}"
            out_path = os.path.join(out_dir, out_fname)
            try:
                rot_img.save(out_path)
            except Exception as exc:
                logger.warning("Could not save %s: %s", out_path, exc)
                continue

            rows.append({
                "original_filepath": fpath,
                "output_filepath":   out_path,
                "split":             split,
                "label":             label,
                "rotation":          angle,
                "width":             rot_img.width,
                "height":            rot_img.height,
            })

    return pd.DataFrame(
        rows,
        columns=["original_filepath", "output_filepath", "split", "label", "rotation", "width", "height"],
    )
