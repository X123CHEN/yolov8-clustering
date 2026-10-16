"""Frozen source descriptors, main-process assignments, online soft moments."""
import csv
from pathlib import Path
import numpy as np
from utils.latent_membership import source_inputs, assignment


class OnlineCenters:
    def __init__(self, config, records, initial_q):
        self.config = config
        paths = source_inputs(config['membership'])
        with paths['metadata'].open() as f: metadata = list(csv.DictReader(f))
        raw = np.load(paths['features']).astype(np.float64)
        raw = (raw - np.load(paths['mean'])) / np.load(paths['scale'])
        index = {str(Path(r['image_path']).resolve()): i for i, r in enumerate(metadata)}
        self.features = raw[[index[key] for key in sorted(records)]]
        self.centers = np.load(paths['centers']).copy()
        self.initial_centers = self.centers.copy()
        self.q = np.asarray(initial_q).copy()
        self.prior = self.q.astype(np.float64).mean(0).astype(np.float32)
        self.N = self.prior.astype(np.float64).copy()
        self.M = self.N[:, None] * self.centers
        self.updates = self.window_batches = self.window_images = self.skipped = 0
        self.mass = np.zeros(2); self.moment = np.zeros_like(self.centers)
        # Verify one coordinate/assignment convention without model inference.
        assert np.allclose(assignment(self.features, self.centers, 2), self.q, atol=2e-7)

    def accumulate(self, indices, q):
        s = self.features[np.asarray(indices)]
        q = np.asarray(q, dtype=np.float64)
        self.mass += q.sum(0); self.moment += q.T @ s
        self.window_images += len(s); self.window_batches += 1
        if self.window_batches >= self.config['center_update_window']:
            return self.commit()
        return None

    def commit(self):
        if not self.window_images: return None
        self.updates += 1
        rho = (self.updates + self.config['center_update_offset']) ** -self.config['center_update_power']
        valid = self.mass >= 1.
        self.skipped += int((~valid).sum())
        self.N[valid] = (1-rho)*self.N[valid] + rho*self.mass[valid]/self.window_images
        self.M[valid] = (1-rho)*self.M[valid] + rho*self.moment[valid]/self.window_images
        self.centers = self.M / np.maximum(self.N[:, None], 1e-12)
        if not np.isfinite(self.centers).all(): raise RuntimeError('Nonfinite online centers')
        self.q = assignment(self.features, self.centers, self.config['membership']['alpha']).astype(np.float32)
        self.prior = self.q.astype(np.float64).mean(0).astype(np.float32)
        entropy = -(self.q * np.log(np.maximum(self.q, 1e-12))).sum(1)
        row = dict(update=self.updates, rho=rho, images=self.window_images, batches=self.window_batches,
            prior=self.prior.tolist(), smoothed_mass=(self.N/self.N.sum()).tolist(),
            mean_max_q=float(self.q.max(1).mean()), median_max_q=float(np.median(self.q.max(1))),
            entropy=float(entropy.mean()), separation=float(np.linalg.norm(self.centers[0]-self.centers[1])),
            relative_displacement=float(np.linalg.norm(self.centers-self.initial_centers)/np.linalg.norm(self.initial_centers)),
            skipped=self.skipped)
        self.mass.fill(0); self.moment.fill(0)
        self.window_batches = self.window_images = 0
        return row

    def state(self):
        return {k: getattr(self, k) for k in ['centers','N','M','updates','window_batches','window_images','skipped','mass','moment']}

    def restore(self, state):
        for k, v in state.items(): setattr(self, k, v)
        self.q = assignment(self.features, self.centers, self.config['membership']['alpha']).astype(np.float32)
        self.prior = self.q.astype(np.float64).mean(0).astype(np.float32)
