"""Source-train-only Student-t memberships with immutable, content-addressed caches."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assignment(z, centers, alpha):
    distance = ((z[:, None, :] - centers[None, :, :]) ** 2).sum(2)
    logits = -0.5 * (alpha + 1) * np.log1p(distance / alpha)
    weights = np.exp(logits - logits.max(1, keepdims=True))
    return weights / weights.sum(1, keepdims=True)


def source_inputs(config):
    root = Path(config["analysis_root"])
    return {"features": root / "features/style_features.npy",
            "metadata": root / "features/metadata.csv",
            "extraction": root / "features/extraction_metadata.json",
            "mean": root / "results/scaler_mean.npy", "scale": root / "results/scaler_scale.npy",
            "centers": root / "results/cluster_centers.npy",
            "labels": root / "results/final_cluster_labels.csv"}


def ensure_cache(config):
    sources = source_inputs(config)
    protocol = dict(method=config["method"], alpha=config["alpha"],
                    sources={key: dict(path=str(path), sha256=sha256(path)) for key, path in sources.items()})
    key = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()[:16]
    output = Path(config["cache_root"]) / f"student_t_alpha{config['alpha']:g}_{key}"
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["protocol"] != protocol:
            raise ValueError("Membership cache protocol mismatch")
        for name, digest in manifest["artifacts"].items():
            if sha256(output / name) != digest:
                raise ValueError(f"Membership cache corrupted: {name}")
        return output, manifest
    if output.exists():
        raise FileExistsError(f"Incomplete membership cache; inspect before retrying: {output}")
    with sources["metadata"].open() as f:
        metadata = list(csv.DictReader(f))
    with sources["labels"].open() as f:
        label_rows = list(csv.DictReader(f))
    extraction = json.loads(sources["extraction"].read_text())
    if extraction["features_sha256"] != protocol["sources"]["features"]["sha256"] or extraction["metadata_sha256"] != protocol["sources"]["metadata"]["sha256"]:
        raise ValueError("Feature extraction hashes mismatch")
    paths = [str(Path(r["image_path"]).resolve(strict=True)) for r in metadata]
    labels_by_path = {str(Path(r["image_path"]).resolve(strict=True)): int(r["latent_domain"]) for r in label_rows}
    if len(paths) != len(set(paths)) or len(label_rows) != len(labels_by_path) or set(paths) != set(labels_by_path):
        raise ValueError("Duplicate or mismatched membership image paths")
    if any(r["split"] != "train" or r["country"] not in {"India", "Japan", "United_States"} for r in metadata):
        raise ValueError("Only source training images may generate memberships")
    labels = np.asarray([labels_by_path[path] for path in paths], dtype=np.int64)
    raw = np.load(sources["features"], allow_pickle=False).astype(np.float64)
    mean, scale, centers = [np.load(sources[k], allow_pickle=False) for k in ("mean", "scale", "centers")]
    if raw.shape != (len(paths), 256) or mean.shape != (256,) or scale.shape != (256,) or centers.shape != (2, 256) or (scale <= 0).any():
        raise ValueError("Unexpected fixed P3 mean/std feature or center dimensions")
    z = (raw - mean) / scale
    q = assignment(z, centers, config["alpha"])
    if not np.isfinite(q).all() or not np.allclose(q.sum(1), 1) or not np.array_equal(q.argmax(1), labels):
        raise ValueError("Invalid probabilities or hard-label mismatch")
    entropy = -(q * np.log(np.maximum(q, np.finfo(q.dtype).tiny))).sum(1)
    output.mkdir(parents=True)
    np.savez_compressed(output / "membership.npz", paths=np.asarray(paths), q=q, hard_labels=labels, entropy=entropy)
    with (output / "membership.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "country", "hard_cluster", "q0", "q1", "entropy"])
        for i, row in enumerate(metadata):
            writer.writerow([paths[i], row["country"], labels[i], *q[i], entropy[i]])
    representatives = {"boundary": np.argsort(q.max(1))[:12].tolist()}
    for k in range(2):
        candidates = np.flatnonzero(labels == k)
        representatives[f"cluster_{k}"] = candidates[np.argsort(-q[candidates, k])[:12]].tolist()
    (output / "representatives.json").write_text(json.dumps({group: [dict(image_path=paths[i], q=q[i].tolist()) for i in indices]
                                for group, indices in representatives.items()}, ensure_ascii=False, indent=2))
    make_figures(output, z, q, entropy, paths, representatives)
    manifest = dict(protocol=protocol, images=len(paths), agreement=1.0, prior=q.mean(0).tolist(),
                    hard_counts=np.bincount(labels, minlength=2).tolist(),
                    max_q_percentiles=dict(zip(["min", "p25", "p50", "p75", "max"], np.percentile(q.max(1), [0,25,50,75,100]).tolist())),
                    entropy_mean=float(entropy.mean()), entropy_normalized_mean=float(entropy.mean()/np.log(2)),
                    fraction_max_q_below_06=float((q.max(1)<.6).mean()),
                    fraction_max_q_above_08=float((q.max(1)>.8).mean()))
    manifest["artifacts"] = {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return output, manifest


def make_figures(output, z, q, entropy, paths, representatives):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(q[:, 0], bins=40, alpha=.6, label="q0")
    axes[0].hist(q[:, 1], bins=40, alpha=.6, label="q1")
    axes[0].legend(); axes[0].set_title("Fixed Student-t membership")
    axes[1].hist(entropy / np.log(2), bins=40); axes[1].set_title("Entropy / log(2)")
    fig.tight_layout(); fig.savefig(output / "membership_distribution.png", dpi=160); plt.close(fig)
    centered = z - z.mean(0)
    _, vectors = np.linalg.eigh(centered.T @ centered)
    xy = centered @ vectors[:, -2:][:, ::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(xy[:, 0], xy[:, 1], c=q[:, 1], s=3, cmap="coolwarm", vmin=0, vmax=1)
    fig.colorbar(scatter, ax=ax, label="q1"); ax.set(xlabel="PC1", ylabel="PC2", title="PCA for visualization only")
    fig.tight_layout(); fig.savefig(output / "pca_q1.png", dpi=160); plt.close(fig)
    for group, indices in representatives.items():
        fig, axes = plt.subplots(3, 4, figsize=(12, 8))
        for ax, i in zip(axes.flat, indices):
            with Image.open(paths[i]) as im:
                im.thumbnail((320, 220)); ax.imshow(im.convert("RGB"))
            ax.set_title(f"q=[{q[i,0]:.3f}, {q[i,1]:.3f}]", fontsize=9); ax.axis("off")
        fig.suptitle(group); fig.tight_layout(); fig.savefig(output / f"{group}.jpg", dpi=120); plt.close(fig)


def load_aligned(output, records, owners, assignment_kind):
    with np.load(Path(output) / "membership.npz", allow_pickle=False) as data:
        paths, soft, labels = data["paths"].tolist(), data["q"], data["hard_labels"]
    if set(paths) != set(records) or len(paths) != len(records):
        raise ValueError("Membership/train path set mismatch")
    mapping = {path: i for i, path in enumerate(paths)}
    indices = [mapping[path] for path in sorted(records)]
    hard = labels[indices]
    if not np.array_equal(hard, [owners[path] for path in sorted(records)]):
        raise ValueError("Expert cluster IDs differ from cached membership IDs")
    q = soft[indices] if assignment_kind == "soft" else np.eye(soft.shape[1], dtype=np.float64)[hard]
    return q.astype(np.float32), q.mean(0).astype(np.float32)
