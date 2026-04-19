# Solar Panel Detection from Aerial Imagery — HOG + SVM

A reproducible image-classification pipeline that detects solar panels in aerial tiles using **Histogram of Oriented Gradients (HOG) features** and a **Support Vector Machine** classifier. This repository is an exact reproduction of the experimental results reported in my MSc thesis at Maastricht University — the joint (kernel, C, γ) grid search, both HOG ablation studies, the F1-optimal threshold tuning, and the holdout evaluation are all reproducible from the artefacts in `data/` by running the four notebooks in order.

> **Reference:** Koenigstorfer, F. (2018). *Solar panel detection from aerial imagery using HOG features and Support Vector Machines.* MSc thesis, Maastricht University.

---

## Headline Results

| Metric | Grouped 5-fold CV (training) | Holdout @ 0.5 | Holdout @ 0.36 |
|---|---|---|---|
| Accuracy  | —      | 0.7649 | 0.7357 |
| ROC AUC   | —      | 0.8275 | 0.8275 |
| F1        | 0.6801 | 0.6837 | 0.6961 |
| Precision | —      | 0.7096 | 0.6247 |
| Recall    | —      | 0.6596 | 0.7859 |

**Optimal configuration:**

| Component | Value |
|---|---|
| Classifier | SVM (RBF kernel) |
| Kernel | RBF |
| Regularisation (C) | 10.0 |
| Gamma (γ) | 0.0001 |
| Class weighting | `class_weight="balanced"` |
| Decision threshold | 0.36 (F1-optimal, tuned on out-of-fold probabilities) |
| HOG orientation bins | 11 |
| HOG pixels per cell | 8 × 8 |
| HOG cells per block | 3 × 3 |
| HOG block normalisation | L1 |
| Input image size | 75 × 75 grayscale |
| Augmentation | 4× rotational (0°, 90°, 180°, 270°) |

---

## What this project demonstrates

- **End-to-end pipeline ownership** — EDA, preprocessing, feature engineering, model selection, and evaluation are each isolated in their own notebook backed by a helper module in `src/`.
- **Disciplined experiment design** — three sub-questions are investigated in sequence, each using `StratifiedGroupKFold` with `groups=original_filepath` so the four rotations of the same source image never straddle a fold boundary. `StandardScaler` is fitted inside each fold via a `Pipeline` to prevent feature-scaling leakage.
- **Honest evaluation** — the test set is held out from every cross-validation step and only touched once for the final report, at both the default (0.5) and an F1-tuned threshold.
- **Reproducibility** — every ablation feature matrix is cached to a named `.npz` file so the model-training notebook can re-run the entire study in seconds without recomputing HOG features.
- **Classical-ML interpretability** — HOG + SVM is a deliberately interpretable baseline. The thesis discussion compares this approach against the obvious deep-learning alternatives and motivates the trade-off.

---

## Dataset

The dataset comprises aerial imagery tiles collected by the German federal statistical office (**Destatis**) and the Dutch national statistical office (**Centraal Bureau voor de Statistiek, CBS**). Labelling was performed by two MSc students at Maastricht University.

### Dataset counts

| Split | Positive | Negative | Total | Ratio (pos / neg) |
|---|---|---|---|---|
| Train   |   818 | 1,200 | 2,018 | 0.682 |
| Test    |   188 |   300 |   488 | 0.627 |
| **All** | **1,006** | **1,500** | **2,506** | 0.671 |

Zero corrupt files and zero MD5-duplicates were found across the 2,506 source images. One image is stored as RGBA rather than RGB (`Train/Positives/0917100000000790.png`); this is resolved automatically by the grayscale conversion in notebook 1.

### Dataset Structure

```
data/
├── Train/
│   ├── Positives/    ← training positive-class images
│   └── Negatives/    ← training negative-class images
└── Test/
    ├── Positives/    ← test positive-class images
    └── Negatives/    ← test negative-class images
```

Supported file extensions: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff` (case-insensitive).

---

## Project Layout

```
.
├── data/                            ← dataset (see structure above)
├── notebooks/
│   ├── 0 image_eda.ipynb            ← exploratory data analysis
│   ├── 1 preprocessing.ipynb        ← resize, grayscale, rotational augmentation
│   ├── 2 feature_extraction.ipynb   ← HOG feature extraction + ablation configs
│   └── 3 model_training.ipynb       ← grid search, ablations, final model, evaluation
├── src/
│   ├── image_eda.py                 ← EDA helper functions
│   ├── preprocessing.py             ← preprocessing helper functions
│   ├── feature_extraction.py        ← HOG feature extraction helpers
│   └── model.py                     ← SVM training, evaluation, plot utilities
├── requirements.txt
└── README.md
```

---

## Setup

**Requirements:** Python ≥ 3.11

```bash
pip install -r requirements.txt
```

Run the notebooks in order:

```bash
jupyter notebook "notebooks/0 image_eda.ipynb"
jupyter notebook "notebooks/1 preprocessing.ipynb"
jupyter notebook "notebooks/2 feature_extraction.ipynb"
jupyter notebook "notebooks/3 model_training.ipynb"
```

Each notebook is idempotent: preprocessed images and `.npz` feature matrices are skipped if they already exist on disk, so notebooks 2 and 3 can be re-run cheaply during analysis.

---

## Notebook 0 — Image EDA (`0 image_eda.ipynb`)

Exploratory analysis of the raw aerial-tile dataset before any processing.

| # | Section | Contents |
|---|---|---|
| 1 | Dataset Overview | Total/corrupt counts, per split×label counts, class-balance ratio, grouped bar chart |
| 2 | Sample Images | 2×5 grid of random training images — one row Positives, one row Negatives |
| 3 | Image Dimensions & Consistency | Unique sizes, width/height histograms, scatter plot, aspect-ratio KDE, `all_same_size` flag |
| 4 | Color Space Analysis | Mode counts (**2,505 RGB + 1 RGBA**), non-RGB flags, mean channel bar chart with std error bars, brightness KDE, mean images |
| 5 | Brightness & Contrast | Brightness KDE train vs test, contrast KDE by label, very-dark / very-bright image flagging |
| 6 | Data Quality Issues | Corrupt file listing, MD5-based duplicate detection, quality summary table |
| 7 | Train / Test Consistency Check | Overlapping KDEs for 5 metrics, mean/std summary table, interpretation of distributional gaps |

The Section 7 interpretation confirms that train and test distributions overlap to within 1 σ on every channel and brightness metric — i.e. the splits are drawn from the same distribution and a model that overfits the training set will not be rescued by a kind test set.

### Helper Functions (`src/image_eda.py`)

| Function | Description |
|---|---|
| `load_image_manifest(data_root)` | Walks all four subdirs, returns one-row-per-image DataFrame with metadata and corruption flag |
| `compute_pixel_stats(df, sample_n)` | Computes per-channel mean/std and brightness stats for up to `sample_n` images per group |
| `compute_mean_image(filepaths, target_size)` | Returns per-pixel mean of all images resized to `target_size` as a uint8 RGB array |
| `detect_duplicates(df)` | Hashes every file with MD5, adds `file_hash` and `is_duplicate` columns |

---

## Notebook 1 — Preprocessing (`1 preprocessing.ipynb`)

### Pipeline

Per source image: **resize with aspect-ratio-preserving padding (LANCZOS) → grayscale (mode `L`) → 4× rotational augmentation (0°, 90°, 180°, 270° CCW)**. All saved variants are 75 × 75 pixels, 8-bit grayscale, identical in format. The 4× multiplier is intentional: aerial tiles have no canonical "up", so rotational invariance is a free regulariser.

| # | Section | Contents |
|---|---|---|
| 1 | Run Preprocessing | Executes the full resize → grayscale → 4× rotation pipeline; saves images and `preprocessed_df.csv` |
| 2 | Preprocessing Summary | Counts of original vs. augmented images, augmentation multiplier, per-split/label breakdown |
| 3 | Rotation Visualisation | Visual grid showing the four rotational variants for a sample image |
| 4 | Sanity Checks | Verifies output dimensions (75×75), grayscale mode (`L`), and file integrity on a random sample |

### Helper Functions (`src/preprocessing.py`)

| Function | Description |
|---|---|
| `resize_with_padding(image, target_size)` | Scales with LANCZOS resampling to fit within `target_size`, adds symmetric black padding; preserves aspect ratio with no distortion |
| `generate_rotations(image)` | Returns a dict with keys `"0"`, `"90"`, `"180"`, `"270"` mapping to CCW-rotated PIL images |
| `preprocess_dataset(data_root, output_root, target_size)` | Full pipeline: resize → grayscale → 4× rotations; saves all variants and returns a manifest DataFrame |

---

## Notebook 2 — Feature Extraction (`2 feature_extraction.ipynb`)

Extracts HOG features in two modes:

1. **Optimal configuration** — train and test feature matrices saved as `features_train.npz` / `features_test.npz` for the final model.
2. **Ablation configurations** — one named `.npz` per cell/block and orientation-bin combination, so the model-training notebook can run all ablation studies by loading pre-computed features rather than recomputing them.

### HOG parameters (optimal configuration)

| Parameter | Value |
|---|---|
| Orientation bins | 11 |
| Pixels per cell | 8 × 8 |
| Cells per block | 3 × 3 |
| Block normalisation | L1 |

| # | Section | Contents |
|---|---|---|
| 1 | Optimal HOG Feature Extraction (Training Set) | Extracts HOG features for the training split using the optimal parameters; saves `features_train.npz` |
| 2 | Optimal HOG Feature Extraction (Test Set) | Same for the test split; saves `features_test.npz` |
| 3 | HOG Visualisation | Side-by-side display of a random Positive and Negative image with their HOG visualisations |
| 4 | Ablation Feature Extraction (Cell/Block Size) | Extracts HOG features for 12 cell/block combinations (ppc ∈ {4, 8, 12, 16} × cpb ∈ {2, 3, 4}); orientations fixed at 11 |
| 5 | Ablation Feature Extraction (Orientation Bins) | Extracts HOG features for 6 orientation bin counts (6, 8, 9, 11, 13, 18); ppc=8, cpb=3 fixed |
| 6 | Sanity Checks | Reloads and verifies `.npz` shapes, checks for all-zero rows, spot-checks ablation file dimensionality |

### Helper Functions (`src/feature_extraction.py`)

| Function | Description |
|---|---|
| `extract_hog_features(image_array, ...)` | Thin wrapper around `skimage.feature.hog`; returns a 1D feature vector |
| `build_feature_matrix(preprocessed_df, hog_params)` | Iterates the preprocessed manifest, extracts HOG for each image, returns `(X, y, filepaths, groups)` — `groups` holds `original_filepath` for each row, for use with `StratifiedGroupKFold` in notebook 3 |
| `save_feature_matrix(X, y, filepaths, groups, output_path)` | Saves to a compressed `.npz` with keys `X`, `y`, `filepaths`, `groups` |
| `load_feature_matrix(input_path)` | Loads an `.npz` and returns `(X, y, filepaths, groups)` |
| `make_ablation_npz_name(output_dir, pixels_per_cell, cells_per_block, orientations)` | Constructs a canonical ablation filename, e.g. `features_train_ppc8_cpb3_ori11.npz` |

### Output file locations

| Path | Description |
|---|---|
| `data/preprocessed/` | Preprocessed 75×75 grayscale images (4 rotations per source image) |
| `data/preprocessed_df.csv` | Manifest with one row per saved variant (filepath, split, label, rotation, dimensions) |
| `data/features_train.npz` | Compressed HOG feature matrix for the training set (optimal config) |
| `data/features_test.npz` | Compressed HOG feature matrix for the test set (optimal config) |
| `data/features_train_ppc{N}_cpb{M}_ori{K}.npz` | One file per ablation configuration |

---

## Notebook 3 — Model Training & Evaluation (`3 model_training.ipynb`)

This notebook identifies the optimal HOG+SVM configuration with leakage-free grouped cross-validation, tunes the F1-optimal decision threshold on out-of-fold probabilities, and evaluates on the holdout test set at both the default and the tuned threshold.

| # | Section | Contents |
|---|---|---|
| 1 | Setup & Data Loading | Loads the optimal train/test feature matrices including `groups`; prints the class balance ratio |
| 2 | SVM Assumptions Check | Documents feature-scale and class-imbalance sensitivity; shows class counts |
| 3 | Sub-question 1: Joint (kernel, C, γ) Tuning | `tune_svm` runs a joint GridSearchCV over linear and RBF kernels under `StratifiedGroupKFold` |
| 4 | Sub-question 2: Cell / Block Size Ablation | `run_hog_ablation` over 12 cell/block configs using the best params from §3 |
| 5 | Sub-question 3: Orientation Bin Ablation | `run_hog_ablation` over 6 orientation bin counts using the best params from §3 |
| 6 | Threshold Tuning | `tune_threshold` sweeps 91 thresholds on OOF probabilities and picks the F1-optimal one |
| 7 | Train Final Model & Holdout Evaluation | Evaluates `best_estimator` on the holdout set at threshold 0.5 **and** at the tuned threshold |
| 8 | Results Summary | Consolidated CV-vs-holdout comparison with honest interpretation |

### Helper Functions (`src/model.py`)

| Function | Description |
|---|---|
| `make_group_cv` | Returns a `StratifiedGroupKFold(n_splits, shuffle=True, random_state=42)` splitter |
| `tune_svm` | Joint GridSearchCV over linear and RBF kernels; returns `(results_df, best_estimator, best_params)` |
| `tune_threshold` | Sweeps decision thresholds on OOF probabilities; returns F1-optimal threshold + per-threshold DataFrame |
| `evaluate_on_holdout` | Predict on the held-out test set at a given threshold; returns a full metric dict |
| `run_hog_ablation` | `cross_validate` each ablation `.npz` under `StratifiedGroupKFold` using `best_params` |
| `plot_roc_curve` / `plot_precision_recall_curve` / `plot_confusion_matrix` / `plot_threshold_sweep` | Diagnostic plots |

---

## Notebook execution order

1. `notebooks/0 image_eda.ipynb` — exploratory data analysis (informational; produces no artefacts consumed downstream).
2. `notebooks/1 preprocessing.ipynb` — resize, grayscale, and 4× rotational augmentation; produces preprocessed images and `preprocessed_df.csv`.
3. `notebooks/2 feature_extraction.ipynb` — extract HOG feature vectors; saves the optimal `.npz` files plus all ablation configurations.
4. `notebooks/3 model_training.ipynb` — joint (kernel, C, γ) GridSearchCV under `StratifiedGroupKFold`, both ablation studies, F1-optimal threshold tuning, and dual holdout evaluation.

---

## Possible next steps

The thesis discussion proposes three natural extensions:

- **More data.** Acquiring additional labelled tiles from the DeepSolaris or DeepGeoStat datasets would likely improve generalisation further.
- **Higher resolution imagery.** Tiles are currently downsampled to 75×75; higher-resolution input could reveal finer texture and edge cues.
- **Deep learning comparison.** A CNN (e.g. ResNet-18) or a Vision Transformer fine-tuned on this dataset would provide a natural comparison point. HOG + SVM is interpretable and fast to train — a useful baseline against which any learned-feature approach should be measured.
