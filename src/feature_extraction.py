"""HOG feature extraction utilities for binary image classification."""

import os

import numpy as np
import pandas as pd
from PIL import Image
from skimage.feature import hog
from tqdm import tqdm


def extract_hog_features(image_array, orientations=11, pixels_per_cell=(8, 8),
                         cells_per_block=(3, 3), block_norm="L1"):
    """Extract a 1D HOG feature vector from a 2D float64 image array."""
    return hog(
        image_array,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm=block_norm,
        visualize=False,
        feature_vector=True,
    )


def build_feature_matrix(preprocessed_df, hog_params=None):
    """Extract HOG features for every image; returns (X, y, filepaths, groups)."""
    params = {
        "orientations":    11,
        "pixels_per_cell": (8, 8),
        "cells_per_block": (3, 3),
        "block_norm":      "L1",
        **(hog_params or {}),
    }

    feature_rows = []
    labels = []
    filepaths = []
    groups = []

    for _, row in tqdm(preprocessed_df.iterrows(), total=len(preprocessed_df),
                       desc="Extracting HOG features"):
        image = Image.open(row["output_filepath"])
        image.load()
        arr = np.asarray(image, dtype=np.float64) / 255.0
        features = extract_hog_features(arr, **params)

        feature_rows.append(features)
        labels.append(1 if row["label"] == "positive" else 0)
        filepaths.append(row["output_filepath"])
        # groups = original source path so the four rotations of one image share a fold
        groups.append(row["original_filepath"])

    X = np.array(feature_rows, dtype=np.float64)
    y = np.array(labels, dtype=int)
    return X, y, filepaths, np.array(groups, dtype=str)


def save_feature_matrix(X, y, filepaths, groups, output_path):
    """Save X, y, filepaths, and groups to a compressed .npz file."""
    np.savez_compressed(
        output_path,
        X=X,
        y=y,
        filepaths=np.array(filepaths, dtype=str),
        groups=np.array(groups, dtype=str),
    )


def load_feature_matrix(input_path):
    """Load a feature matrix saved by save_feature_matrix; returns (X, y, filepaths, groups)."""
    data = np.load(input_path, allow_pickle=False)
    return data["X"], data["y"], data["filepaths"], data["groups"]


def make_ablation_npz_name(output_dir, pixels_per_cell, cells_per_block, orientations):
    """Construct a canonical ablation .npz filename, e.g. features_train_ppc8_cpb3_ori11.npz."""
    ppc = pixels_per_cell[0]
    cpb = cells_per_block[0]
    fname = f"features_train_ppc{ppc}_cpb{cpb}_ori{orientations}.npz"
    return os.path.join(output_dir, fname)
