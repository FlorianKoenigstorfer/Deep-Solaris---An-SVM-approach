"""Image preprocessing utilities for binary image classification."""

import os

import pandas as pd
from PIL import Image
from tqdm import tqdm

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

_SUBDIRS = [
    ("Train", "Positives", "train", "positive"),
    ("Train", "Negatives", "train", "negative"),
    ("Test",  "Positives", "test",  "positive"),
    ("Test",  "Negatives", "test",  "negative"),
]


def resize_with_padding(image, target_size=(75, 75)):
    """Scale image to fit target_size preserving aspect ratio, pad with black."""
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


def generate_rotations(image):
    """Return dict of four CCW rotational copies (0°, 90°, 180°, 270°)."""
    return {
        "0":   image,
        "90":  image.rotate(90,  expand=False),
        "180": image.rotate(180, expand=False),
        "270": image.rotate(270, expand=False),
    }


def preprocess_dataset(data_root, output_root, target_size=(75, 75)):
    """Resize, grayscale, and 4× rotate every image; returns a DataFrame of saved variants."""
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
        image = Image.open(fpath)
        padded = resize_with_padding(image, target_size)
        gray = padded.convert("L")
        rotations = generate_rotations(gray)

        rel_path = os.path.relpath(fpath, data_root)
        rel_dir = os.path.dirname(rel_path)
        base, ext = os.path.splitext(os.path.basename(rel_path))

        out_dir = os.path.join(output_root, rel_dir)
        os.makedirs(out_dir, exist_ok=True)

        for angle, rot_img in rotations.items():
            out_fname = f"{base}_rot{angle}{ext}"
            out_path = os.path.join(out_dir, out_fname)
            rot_img.save(out_path)

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
