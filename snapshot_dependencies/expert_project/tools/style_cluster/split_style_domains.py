#!/usr/bin/env python
import argparse
import csv
import pickle
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.utils import cvtColor, preprocess_input, resize_image  # noqa: E402
from nets import backbone as backbone_module  # noqa: E402


class NumpyStandardScaler:
    def fit(self, x):
        self.mean_ = x.mean(axis=0)
        self.scale_ = x.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, x):
        return (x - self.mean_) / self.scale_

    def fit_transform(self, x):
        return self.fit(x).transform(x)


class NumpyKMeans:
    def __init__(self, n_clusters=3, init="k-means++", n_init=10, max_iter=300, tol=1e-4, random_state=0):
        if init != "k-means++":
            raise ValueError("NumpyKMeans only supports init='k-means++'.")
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.init = init

    def fit(self, x):
        best_inertia = None
        best_centers = None
        best_labels = None
        base_rng = np.random.RandomState(self.random_state)
        for _ in range(self.n_init):
            rng = np.random.RandomState(base_rng.randint(0, 2**31 - 1))
            centers = self._init_kmeans_plus_plus(x, rng)
            labels = None
            for _iter in range(self.max_iter):
                distances = self._squared_distances(x, centers)
                new_labels = distances.argmin(axis=1)
                new_centers = centers.copy()
                for k in range(self.n_clusters):
                    members = x[new_labels == k]
                    if len(members) > 0:
                        new_centers[k] = members.mean(axis=0)
                    else:
                        farthest = distances.min(axis=1).argmax()
                        new_centers[k] = x[farthest]
                shift = np.linalg.norm(new_centers - centers)
                centers = new_centers
                labels = new_labels
                if shift <= self.tol:
                    break
            inertia = self._squared_distances(x, centers).min(axis=1).sum()
            if best_inertia is None or inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers.copy()
                best_labels = labels.copy()
        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = float(best_inertia)
        return self

    def fit_predict(self, x):
        return self.fit(x).labels_

    def predict(self, x):
        return self._squared_distances(x, self.cluster_centers_).argmin(axis=1)

    def _init_kmeans_plus_plus(self, x, rng):
        n_samples = x.shape[0]
        centers = np.empty((self.n_clusters, x.shape[1]), dtype=x.dtype)
        first = rng.randint(n_samples)
        centers[0] = x[first]
        closest_dist_sq = self._squared_distances(x, centers[:1]).reshape(-1)
        for c in range(1, self.n_clusters):
            total = closest_dist_sq.sum()
            if total <= 0:
                candidate = rng.randint(n_samples)
            else:
                probs = closest_dist_sq / total
                candidate = rng.choice(n_samples, p=probs)
            centers[c] = x[candidate]
            new_dist_sq = self._squared_distances(x, centers[c:c + 1]).reshape(-1)
            closest_dist_sq = np.minimum(closest_dist_sq, new_dist_sq)
        return centers

    @staticmethod
    def _squared_distances(x, centers):
        return ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def patch_backbone_forward():
    def device_safe_forward(self, x):
        return self.act(self.bn(self.conv(x)))

    backbone_module.Conv.forward = device_safe_forward

    def backbone_forward_without_device_move(self, x):
        x = self.stem(x)
        x = self.dark2(x)
        x = self.dark3(x)
        feat1 = x
        x = self.dark4(x)
        feat2 = x
        x = self.dark5(x)
        feat3 = x
        return feat1, feat2, feat3

    backbone_module.Backbone.forward = backbone_forward_without_device_move


def parse_txt(txt_path, country_name, max_samples=None):
    samples = []
    skipped = 0
    txt_path = Path(txt_path)
    with txt_path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                skipped += 1
                continue
            image_path = line.split()[0]
            if not Path(image_path).exists():
                skipped += 1
                print(f"WARNING: missing image, skip line {line_no}: {image_path}")
                continue
            samples.append({
                "country": country_name,
                "image_path": image_path,
                "line": line,
                "source_txt": str(txt_path),
            })
            if max_samples is not None and len(samples) >= max_samples:
                break
    if skipped:
        print(f"WARNING: {txt_path} skipped {skipped} empty/missing-image lines.")
    return samples


def load_backbone(weights, phi, device):
    patch_backbone_forward()
    model = backbone_module.Backbone(phi=phi, pretrained=False)
    checkpoint = torch.load(weights, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]

    model_dict = model.state_dict()
    load_dict = {}
    extra_keys = []
    mismatch_keys = []
    for key, value in checkpoint.items():
        clean_key = key[7:] if key.startswith("module.") else key
        if clean_key.startswith("backbone."):
            clean_key = clean_key[len("backbone."):]
        else:
            continue
        if clean_key in model_dict and tuple(model_dict[clean_key].shape) == tuple(value.shape):
            load_dict[clean_key] = value
        elif clean_key in model_dict:
            mismatch_keys.append(clean_key)
        else:
            extra_keys.append(clean_key)

    missing_keys = [k for k in model_dict.keys() if k not in load_dict]
    model_dict.update(load_dict)
    model.load_state_dict(model_dict)
    model.to(device)
    model.eval()
    print(
        "Backbone weights loaded: "
        f"loaded={len(load_dict)}, missing={len(missing_keys)}, "
        f"mismatch={len(mismatch_keys)}, extra={len(extra_keys)}"
    )
    if missing_keys:
        print("WARNING: first missing backbone keys:", missing_keys[:10])
    if mismatch_keys:
        print("WARNING: first mismatched backbone keys:", mismatch_keys[:10])
    return model


def preprocess_image(image_path, input_shape, letterbox_image=False):
    image = Image.open(image_path)
    image = cvtColor(image)
    image = resize_image(image, (input_shape[1], input_shape[0]), letterbox_image)
    image = np.array(image, dtype="float32")
    image = preprocess_input(image)
    image = np.transpose(image, (2, 0, 1))
    return torch.from_numpy(image)


def compute_style_vector(feat1):
    mean = feat1.mean(dim=[2, 3])
    std = feat1.std(dim=[2, 3])
    return torch.cat([mean, std], dim=1)


def extract_features_for_samples(samples, model, batch_size, device, input_shape, split, country, letterbox_image):
    features = []
    valid_samples = []
    first_shape_printed = False
    for start in range(0, len(samples), batch_size):
        batch_samples = samples[start:start + batch_size]
        tensors = []
        tensor_samples = []
        for sample in batch_samples:
            try:
                tensors.append(preprocess_image(sample["image_path"], input_shape, letterbox_image))
                tensor_samples.append(sample)
            except Exception as exc:
                print(f"WARNING: failed to read image, skip {sample['image_path']}: {exc}")
        if not tensors:
            continue
        images = torch.stack(tensors, dim=0).to(device)
        with torch.no_grad():
            feat1, _feat2, _feat3 = model(images)
            if not first_shape_printed:
                print(f"Using {country} {split} feature map shape: {list(feat1.shape)}")
                first_shape_printed = True
            style = compute_style_vector(feat1).cpu().numpy().astype("float32")
        features.append(style)
        valid_samples.extend(tensor_samples)
    if features:
        return np.concatenate(features, axis=0), valid_samples
    return np.empty((0, 0), dtype="float32"), valid_samples


def get_cluster_impl():
    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        print("Using sklearn StandardScaler + KMeans.")
        return StandardScaler, KMeans, "sklearn"
    except Exception as exc:
        print(f"WARNING: sklearn unavailable, using numpy fallback: {exc}")
        return NumpyStandardScaler, NumpyKMeans, "numpy_fallback"


def fit_cluster(all_train_features, k, seed):
    StandardScaler, KMeans, impl = get_cluster_impl()
    scaler = StandardScaler()
    features_norm = scaler.fit_transform(all_train_features)
    kmeans = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=seed)
    labels = kmeans.fit_predict(features_norm)
    return scaler, kmeans, labels, impl


def ensure_dirs(output_dir):
    paths = {
        "country_split": Path(output_dir) / "country_split",
        "combined_style": Path(output_dir) / "combined_style",
        "info": Path(output_dir) / "info",
        "features": Path(output_dir) / "features",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def save_features(features_dir, country, split, features, samples):
    np.save(features_dir / f"{country}_{split}_features.npy", features)
    with (features_dir / f"{country}_{split}_paths.txt").open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(sample["image_path"] + "\n")


def write_country_split_files(samples_by_country, labels_by_country, output_dir, split, k):
    country_split = output_dir / "country_split"
    for country, samples in samples_by_country.items():
        labels = labels_by_country[country]
        handles = []
        try:
            for style_id in range(k):
                path = country_split / f"{country}_style{style_id}_{split}.txt"
                handles.append(path.open("w", encoding="utf-8"))
            for sample, label in zip(samples, labels):
                handles[int(label)].write(sample["line"] + "\n")
        finally:
            for handle in handles:
                handle.close()


def combine_style_files(country_names, output_dir, split, k):
    country_split = output_dir / "country_split"
    combined_style = output_dir / "combined_style"
    combined_paths = []
    for style_id in range(k):
        out_path = combined_style / f"Style_{style_id}_{split}.txt"
        with out_path.open("w", encoding="utf-8") as out:
            for country in country_names:
                in_path = country_split / f"{country}_style{style_id}_{split}.txt"
                if not in_path.exists():
                    continue
                with in_path.open("r", encoding="utf-8") as inp:
                    for line in inp:
                        out.write(line)
        combined_paths.append(out_path)
    return combined_paths


def save_pseudo_domain_csv(samples_by_country, labels_by_country, output_path, split):
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "country", "pseudo_domain", "split", "source_txt"])
        for country, samples in samples_by_country.items():
            for sample, label in zip(samples, labels_by_country[country]):
                writer.writerow([sample["image_path"], country, int(label), split, sample["source_txt"]])


def save_cluster_summary(samples_by_country, labels_by_country, output_path, k):
    country_names = list(samples_by_country.keys())
    rows = []
    for style_id in range(k):
        counts = {}
        total = 0
        for country in country_names:
            labels = labels_by_country[country]
            count = int(np.sum(labels == style_id))
            counts[country] = count
            total += count
        rows.append((style_id, total, counts))

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pseudo_domain", "total"] + country_names)
        for style_id, total, counts in rows:
            writer.writerow([f"Style_{style_id}", total] + [counts[c] for c in country_names])
    return rows


def save_class_distribution(samples_by_country, labels_by_country, output_path, k):
    class_counts = [defaultdict(int) for _ in range(k)]
    class_ids = set()
    for country, samples in samples_by_country.items():
        for sample, label in zip(samples, labels_by_country[country]):
            for item in sample["line"].split()[1:]:
                parts = item.split(",")
                if len(parts) >= 5:
                    cls = parts[-1]
                    class_counts[int(label)][cls] += 1
                    class_ids.add(cls)
    class_ids = sorted(class_ids, key=lambda x: int(x) if x.isdigit() else x)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pseudo_domain"] + [f"class_{cls}" for cls in class_ids])
        for style_id in range(k):
            writer.writerow([f"Style_{style_id}"] + [class_counts[style_id][cls] for cls in class_ids])


def print_summary(rows, samples_by_country, title, k):
    country_names = list(samples_by_country.keys())
    total_samples = sum(len(samples_by_country[c]) for c in country_names)
    print(f"\n===== {title} Cluster Summary =====")
    for style_id, total, counts in rows:
        parts = [f"total={total}"] + [f"{country}={counts[country]}" for country in country_names]
        print(f"Style_{style_id}: " + ", ".join(parts))
        if total_samples and total / total_samples < 0.05:
            print(
                f"WARNING: Style_{style_id} contains only "
                f"{100.0 * total / total_samples:.2f}% of {title.lower()} samples."
            )
    for country in country_names:
        labels = samples_by_country[country]["_labels_for_warning"]
        if len(labels) == 0:
            continue
        for style_id in range(k):
            ratio = float(np.sum(labels == style_id)) / len(labels)
            if ratio > 0.90:
                print(
                    f"WARNING: {100.0 * ratio:.2f}% of {country} samples are "
                    f"assigned to Style_{style_id}."
                )


def split_labels_by_lengths(labels, country_names, lengths):
    labels_by_country = {}
    offset = 0
    for country, length in zip(country_names, lengths):
        labels_by_country[country] = labels[offset:offset + length]
        offset += length
    return labels_by_country


def process_split(args, model, output_paths, scaler=None, kmeans=None, split="train"):
    txts = args.train_txts if split == "train" else args.val_txts
    names = args.train_names if split == "train" else args.val_names
    if not txts:
        return None
    samples_by_country = {}
    features_by_country = {}
    for txt_path, country in zip(txts, names):
        samples = parse_txt(txt_path, country, args.max_samples_per_txt)
        features, valid_samples = extract_features_for_samples(
            samples=samples,
            model=model,
            batch_size=args.batch_size,
            device=args.runtime_device,
            input_shape=args.input_shape,
            split=split,
            country=country,
            letterbox_image=args.letterbox_image,
        )
        samples_by_country[country] = valid_samples
        features_by_country[country] = features
        if args.save_features:
            save_features(output_paths["features"], country, split, features, valid_samples)
        print(f"{country} {split}: samples={len(samples)}, valid={len(valid_samples)}, features={features.shape}")

    country_names = list(samples_by_country.keys())
    features_list = [features_by_country[country] for country in country_names]
    if not features_list or any(feature.size == 0 for feature in features_list):
        raise RuntimeError(f"No valid {split} features were extracted.")
    all_features = np.concatenate(features_list, axis=0)
    if args.save_features:
        np.save(output_paths["features"] / f"all_{split}_features.npy", all_features)

    if split == "train":
        scaler, kmeans, all_labels, impl = fit_cluster(all_features, args.k, args.seed)
        with (output_paths["info"] / "scaler.pkl").open("wb") as f:
            pickle.dump(scaler, f)
        with (output_paths["info"] / "kmeans.pkl").open("wb") as f:
            pickle.dump(kmeans, f)
        with (output_paths["info"] / "cluster_impl.txt").open("w", encoding="utf-8") as f:
            f.write(impl + "\n")
    else:
        all_features_norm = scaler.transform(all_features)
        all_labels = kmeans.predict(all_features_norm)

    lengths = [len(samples_by_country[country]) for country in country_names]
    labels_by_country = split_labels_by_lengths(all_labels, country_names, lengths)
    write_country_split_files(samples_by_country, labels_by_country, output_paths["root"], split, args.k)
    combined_paths = combine_style_files(country_names, output_paths["root"], split, args.k)
    save_pseudo_domain_csv(
        samples_by_country,
        labels_by_country,
        output_paths["info"] / f"{split}_pseudo_domain.csv",
        split,
    )
    rows = save_cluster_summary(
        samples_by_country,
        labels_by_country,
        output_paths["info"] / f"cluster_summary_{split}.csv",
        args.k,
    )
    save_class_distribution(
        samples_by_country,
        labels_by_country,
        output_paths["info"] / f"class_distribution_{split}.csv",
        args.k,
    )

    warning_samples = {
        country: {"_labels_for_warning": labels_by_country[country]}
        for country in country_names
    }
    print_summary(rows, warning_samples, split.capitalize(), args.k)
    print("\nOutput files:")
    for path in combined_paths:
        print(path)
    return scaler, kmeans


def validate_args(args):
    if len(args.train_txts) != len(args.train_names):
        raise ValueError("--train_txts and --train_names must have the same length.")
    if args.val_txts or args.val_names:
        if not args.val_txts or not args.val_names:
            raise ValueError("--val_txts and --val_names must be provided together.")
        if len(args.val_txts) != len(args.val_names):
            raise ValueError("--val_txts and --val_names must have the same length.")
    if args.k <= 0:
        raise ValueError("--k must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")


def parse_args():
    parser = argparse.ArgumentParser(description="Split source images into pseudo style domains.")
    parser.add_argument("--train_txts", nargs="+", required=True)
    parser.add_argument("--train_names", nargs="+", required=True)
    parser.add_argument("--val_txts", nargs="+", default=None)
    parser.add_argument("--val_names", nargs="+", default=None)
    parser.add_argument("--weights", default="model_data/yolov8_s.pth")
    parser.add_argument("--classes_path", default="model_data/rdd_classes.txt")
    parser.add_argument("--phi", default="s", choices=["n", "s", "m", "l", "x"])
    parser.add_argument("--input_shape", nargs=2, type=int, default=[640, 640])
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--output_dir", default="pseudo_style_txt")
    parser.add_argument("--save_features", action="store_true")
    parser.add_argument("--letterbox_image", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_samples_per_txt", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA requested but unavailable. Falling back to CPU.")
        args.runtime_device = torch.device("cpu")
    else:
        args.runtime_device = torch.device(args.device)

    output_paths = ensure_dirs(args.output_dir)
    output_paths["root"] = Path(args.output_dir)

    model = load_backbone(args.weights, args.phi, args.runtime_device)
    scaler, kmeans = process_split(args, model, output_paths, split="train")
    if args.val_txts:
        process_split(args, model, output_paths, scaler=scaler, kmeans=kmeans, split="val")


if __name__ == "__main__":
    main()
