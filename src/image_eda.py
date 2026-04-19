"""
image_eda.py — Reusable helper functions for binary image classification EDA.
"""

import hashlib
import os

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def load_image_manifest(data_root: str) -> pd.DataFrame:
    """Walk the four dataset subdirectories and build one row per image file.

    Returns
    -------
    pd.DataFrame
        Columns: ``filepath``, ``split`` (train/test), ``label``
        (positive/negative), ``is_corrupt``, ``width``, ``height``,
        ``mode``, ``file_size_bytes``.
    """
    subdirs = [
        ("Train", "Positives", "train", "positive"),
        ("Train", "Negatives", "train", "negative"),
        ("Test",  "Positives", "test",  "positive"),
        ("Test",  "Negatives", "test",  "negative"),
    ]

    rows = []
    for split_dir, label_dir, split, label in subdirs:
        folder = os.path.join(data_root, split_dir, label_dir)
        if not os.path.isdir(folder):
            continue
        for fname in os.listdir(folder):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in _IMAGE_EXTENSIONS:
                continue
            fpath = os.path.join(folder, fname)
            row = {
                "filepath": fpath,
                "split": split,
                "label": label,
                "is_corrupt": False,
                "width": np.nan,
                "height": np.nan,
                "mode": np.nan,
                "file_size_bytes": np.nan,
            }
            try:
                img = Image.open(fpath)
                img.verify()          # catches truncated / bad header files
                # Re-open after verify (verify() leaves the file-pointer in a
                # non-reusable state).
                img = Image.open(fpath)
                row["width"] = img.width
                row["height"] = img.height
                row["mode"] = img.mode
                row["file_size_bytes"] = os.path.getsize(fpath)
            except Exception:
                row["is_corrupt"] = True
            rows.append(row)

    return pd.DataFrame(rows)


def compute_pixel_stats(df: pd.DataFrame, sample_n: int = 500) -> pd.DataFrame:
    """Compute per-channel and brightness statistics for non-corrupt images.

    For each (split, label) group up to *sample_n* images are sampled
    (random seed 42) to keep runtime manageable.

    Returns a DataFrame indexed by ``filepath`` with columns
    ``mean_r/g/b``, ``std_r/g/b``, ``mean_brightness``, ``std_brightness``.
    """
    valid = df[~df["is_corrupt"]].copy()

    records = []
    rng = np.random.default_rng(42)

    for (split, label), group in valid.groupby(["split", "label"]):
        if len(group) > sample_n:
            idx = rng.choice(group.index, size=sample_n, replace=False)
            sample = group.loc[idx]
        else:
            sample = group

        filepaths = sample["filepath"].tolist()
        for fpath in tqdm(filepaths, desc=f"Pixel stats [{split}/{label}]", leave=False):
            try:
                img = Image.open(fpath).convert("RGB")
                arr = np.asarray(img, dtype=np.float32)
                gray = np.asarray(img.convert("L"), dtype=np.float32)

                records.append({
                    "filepath": fpath,
                    "mean_r": float(arr[:, :, 0].mean()),
                    "mean_g": float(arr[:, :, 1].mean()),
                    "mean_b": float(arr[:, :, 2].mean()),
                    "std_r":  float(arr[:, :, 0].std()),
                    "std_g":  float(arr[:, :, 1].std()),
                    "std_b":  float(arr[:, :, 2].std()),
                    "mean_brightness": float(gray.mean()),
                    "std_brightness":  float(gray.std()),
                })
            except Exception:
                pass

    return pd.DataFrame(records).set_index("filepath")


def compute_mean_image(filepaths: list, target_size: tuple = (128, 128)) -> np.ndarray:
    """Return the per-pixel mean image over a collection of files as uint8 (H, W, 3).

    Each image is resized to *target_size* and converted to RGB. Unreadable
    files are silently skipped. Returns zeros if no files could be loaded.
    """
    accumulator = np.zeros((target_size[1], target_size[0], 3), dtype=np.float64)
    count = 0

    for fpath in tqdm(filepaths, desc="Computing mean image"):
        try:
            img = Image.open(fpath).convert("RGB").resize(target_size, Image.LANCZOS)
            accumulator += np.asarray(img, dtype=np.float64)
            count += 1
        except Exception:
            pass

    if count == 0:
        return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

    return (accumulator / count).clip(0, 255).astype(np.uint8)


def detect_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Compute MD5 hashes and flag duplicate images.

    Only non-corrupt images are hashed. Returns a copy of *df* with two
    additional columns: ``file_hash`` (hex MD5 or NaN for corrupt files)
    and ``is_duplicate`` (True if the hash appears more than once).
    """
    result = df.copy()
    # Use object dtype from the start so string hashes can be assigned
    # without a pandas FutureWarning about incompatible dtype.
    result["file_hash"] = pd.array([None] * len(result), dtype=object)
    result["is_duplicate"] = False

    valid_idx = result.index[~result["is_corrupt"]].tolist()

    for idx in tqdm(valid_idx, desc="Hashing files"):
        fpath = result.at[idx, "filepath"]
        try:
            with open(fpath, "rb") as fh:
                digest = hashlib.md5(fh.read()).hexdigest()
            result.at[idx, "file_hash"] = digest
        except Exception:
            pass

    hash_counts = result["file_hash"].value_counts()
    duplicate_hashes = set(hash_counts[hash_counts > 1].index)
    result["is_duplicate"] = result["file_hash"].isin(duplicate_hashes)

    return result
