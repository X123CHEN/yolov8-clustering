import json
import os
from pathlib import Path

import numpy as np


def canonicalize_path(path):
    return os.path.normcase(os.path.abspath(os.path.normpath(str(path).strip())))


class StyleDistanceCache:
    """Strict path-to-(hard label, squared distances) lookup."""

    def __init__(self, cache_path, metadata_path):
        cache_path = Path(cache_path)
        with np.load(cache_path, allow_pickle=False) as data:
            paths = data["image_paths"]
            labels = data["hard_labels"].astype(np.int64)
            distances = data["distance_sq"].astype(np.float32)
        if distances.shape != (len(paths), 3) or labels.shape != (len(paths),):
            raise RuntimeError(f"Invalid style cache shapes in {cache_path}")
        self._lookup = {}
        for path, label, distance in zip(paths, labels, distances):
            key = canonicalize_path(path)
            if key in self._lookup:
                raise RuntimeError(f"Duplicate path in style cache: {path}")
            self._lookup[key] = (int(label), distance)
        self.metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        self.distance_scale = float(self.metadata["distance_scale"])
        if self.distance_scale <= 0:
            raise RuntimeError("distance_scale must be positive")
        nearest = distances.argmin(axis=1)
        soft_main = (-distances / self.distance_scale).argmax(axis=1)
        mismatch = np.flatnonzero((nearest != labels) | (soft_main != labels))
        if mismatch.size:
            i = int(mismatch[0])
            raise RuntimeError(
                f"Style cache label mismatch at {paths[i]}: saved={labels[i]}, "
                f"nearest={nearest[i]}, soft={soft_main[i]}, d2={distances[i].tolist()}"
            )
        self.match_rate = 1.0

    def lookup(self, image_path):
        key = canonicalize_path(image_path)
        if key not in self._lookup:
            raise KeyError(f"Image path is absent from style cache: {image_path}")
        label, distance = self._lookup[key]
        return label, distance.copy()
