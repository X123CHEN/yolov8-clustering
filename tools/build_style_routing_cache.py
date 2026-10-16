#!/usr/bin/env python3
"""Build portable style-clustering parameters and routing-distance caches.

This consumes the immutable artifacts produced by the original hard-clustering
split.  It never refits the scaler or K-means model and fails unless both the
nearest-centre labels and soft-assignment labels exactly reproduce the saved
hard labels.
"""

import argparse
import csv
import hashlib
import json
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np


COUNTRY_ORDER = ("India", "Japan", "USA")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_legacy_models(source_root):
    """Load the original pickle, including its NumPy fallback class names."""
    script_dir = source_root.parent / "tools" / "style_cluster"
    sys.path.insert(0, str(script_dir))
    try:
        import split_style_domains  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(f"Cannot import original clustering classes: {exc}") from exc

    # Old pickles may record either split_style_domains.* or __main__.*.
    import __main__
    for name in ("NumpyStandardScaler", "NumpyKMeans"):
        if hasattr(split_style_domains, name):
            setattr(__main__, name, getattr(split_style_domains, name))

    with (source_root / "info" / "scaler.pkl").open("rb") as handle:
        scaler = pickle.load(handle)
    with (source_root / "info" / "kmeans.pkl").open("rb") as handle:
        kmeans = pickle.load(handle)
    return scaler, kmeans


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"image_path", "country", "pseudo_domain", "split"}
    if not rows or not required.issubset(rows[0]):
        raise RuntimeError(f"Invalid pseudo-domain CSV: {path}")
    return rows


def expected_paths(source_root, split):
    paths = []
    for country in COUNTRY_ORDER:
        path_file = source_root / "features" / f"{country}_{split}_paths.txt"
        paths.extend(path_file.read_text(encoding="utf-8").splitlines())
    return paths


def stable_softmax_from_distance(distance_sq, distance_scale, temperature=1.0):
    if temperature <= 0 or distance_scale <= 0:
        raise ValueError("temperature and distance_scale must be positive")
    logits = -distance_sq / (temperature * distance_scale)
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities


def validate_and_build(source_root, split, mean, scale, centers, distance_scale=None):
    features_path = source_root / "features" / f"all_{split}_features.npy"
    csv_path = source_root / "info" / f"{split}_pseudo_domain.csv"
    features = np.load(features_path, allow_pickle=False)
    rows = read_rows(csv_path)
    paths = [row["image_path"] for row in rows]
    labels = np.asarray([int(row["pseudo_domain"]) for row in rows], dtype=np.int64)
    countries = np.asarray([row["country"] for row in rows])

    if len(features) != len(rows):
        raise RuntimeError(f"{split}: feature/CSV count mismatch")
    if paths != expected_paths(source_root, split):
        raise RuntimeError(f"{split}: feature path order does not match pseudo-domain CSV")

    normalized = (features - mean) / scale
    distance_sq = np.sum((normalized[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    if not np.isfinite(distance_sq).all():
        raise RuntimeError(f"{split}: non-finite distances found")
    nearest = np.argmin(distance_sq, axis=1)
    mismatch = np.flatnonzero(nearest != labels)
    if mismatch.size:
        i = int(mismatch[0])
        raise RuntimeError(
            f"{split}: nearest-centre mismatch at {paths[i]}: "
            f"saved={labels[i]}, nearest={nearest[i]}, d2={distance_sq[i].tolist()}"
        )

    if distance_scale is None:
        distance_scale = float(np.median(np.min(distance_sq, axis=1)))
    probabilities = stable_softmax_from_distance(distance_sq, distance_scale)
    soft_labels = np.argmax(probabilities, axis=1)
    mismatch = np.flatnonzero(soft_labels != labels)
    if mismatch.size:
        i = int(mismatch[0])
        raise RuntimeError(
            f"{split}: soft-label mismatch at {paths[i]}: "
            f"saved={labels[i]}, soft={soft_labels[i]}, d2={distance_sq[i].tolist()}"
        )

    return {
        "image_paths": np.asarray(paths),
        "hard_labels": labels,
        "distance_sq": distance_sq.astype(np.float32),
        "countries": countries,
        "splits": np.full(len(rows), split),
    }, distance_scale


def save_cache(path, arrays):
    np.savez_compressed(path, **arrays)


def parse_args():
    project = Path(__file__).resolve().parents[1]
    default_source = project.parent / "yolov8-pretrain" / "pseudo_style_txt"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=default_source)
    parser.add_argument("--output-dir", type=Path, default=project / "style_cache")
    return parser.parse_args()


def main():
    args = parse_args()
    source_root = args.source_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    scaler, kmeans = load_legacy_models(source_root)
    mean = np.asarray(scaler.mean_, dtype=np.float64)
    scale = np.asarray(scaler.scale_, dtype=np.float64)
    centers = np.asarray(kmeans.cluster_centers_, dtype=np.float64)
    if mean.shape != scale.shape or centers.shape != (3, mean.size):
        raise RuntimeError(
            f"Unexpected clustering shapes: mean={mean.shape}, scale={scale.shape}, "
            f"centers={centers.shape}"
        )
    if np.any(scale <= 0) or not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise RuntimeError("Invalid scaler parameters")

    train, distance_scale = validate_and_build(
        source_root, "train", mean, scale, centers
    )
    val, _ = validate_and_build(
        source_root, "val", mean, scale, centers, distance_scale=distance_scale
    )

    params_path = output_dir / "cluster_params.npz"
    np.savez_compressed(
        params_path,
        scaler_mean=mean,
        scaler_scale=scale,
        cluster_centers=centers,
    )
    save_cache(output_dir / "train_style_distances.npz", train)
    save_cache(output_dir / "val_style_distances.npz", val)

    for filename in ("scaler.pkl", "kmeans.pkl"):
        shutil.copy2(source_root / "info" / filename, output_dir / filename)

    expert_paths = [
        Path(__file__).resolve().parents[1]
        / "results" / "Pre_train" / f"Style_{index}" / "best_epoch_weights.pth"
        for index in range(3)
    ]
    missing_experts = [str(path) for path in expert_paths if not path.is_file()]
    if missing_experts:
        raise RuntimeError(f"Missing style expert weights: {missing_experts}")

    metadata = {
        "schema_version": 1,
        "num_clusters": 3,
        "feature_layer": "feat1",
        "style_vector": "channel_mean_std",
        "feature_dimension": int(mean.size),
        "input_shape": [640, 640],
        "distance_type": "squared_euclidean",
        "distance_scale": distance_scale,
        "distance_scale_mode": "train_median_min_d2",
        "kmeans_random_state": 0,
        "kmeans_n_init": 10,
        "cluster_implementation": (source_root / "info" / "cluster_impl.txt").read_text().strip(),
        "portable_parameters": params_path.name,
        "source_train_files": [
            "India_train.txt", "Japan_train.txt", "United_States_train.txt"
        ],
        "source_val_files": ["India_val.txt", "Japan_val.txt", "United_States_val.txt"],
        "style_expert_paths": [str(path) for path in expert_paths],
        "style_expert_sha256": [sha256(path) for path in expert_paths],
        "source_artifact_sha256": {
            name: sha256(source_root / relative)
            for name, relative in {
                "scaler.pkl": Path("info/scaler.pkl"),
                "kmeans.pkl": Path("info/kmeans.pkl"),
                "all_train_features.npy": Path("features/all_train_features.npy"),
                "all_val_features.npy": Path("features/all_val_features.npy"),
                "train_pseudo_domain.csv": Path("info/train_pseudo_domain.csv"),
                "val_pseudo_domain.csv": Path("info/val_pseudo_domain.csv"),
            }.items()
        },
        "sample_counts": {"train": len(train["hard_labels"]), "val": len(val["hard_labels"])},
        "hard_label_match_rate": {"train": 1.0, "val": 1.0},
        "soft_label_match_rate": {"train": 1.0, "val": 1.0},
        "style_index_mapping": {
            "0": "Style_0", "1": "Style_1", "2": "Style_2"
        },
    }
    metadata_path = output_dir / "cluster_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"train samples: {len(train['hard_labels'])}, match: 100.000000%")
    print(f"val samples:   {len(val['hard_labels'])}, match: 100.000000%")
    print(f"distance_scale: {distance_scale:.15g}")
    print(f"cache directory: {output_dir}")


if __name__ == "__main__":
    main()
