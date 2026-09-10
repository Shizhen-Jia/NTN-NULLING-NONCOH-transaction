"""Rectangular BS cells, horizontal sector sampling, and channel dominance."""
import numpy as np


def channel_dominance(h, target_tx, margin_db, min_channel_norm, eps=1e-12):
    """Compare Frobenius channel powers against every non-serving sector."""
    h = np.asarray(h, dtype=np.complex128)
    target = np.asarray(target_tx, dtype=int)
    if h.ndim != 4 or h.shape[2] < 2 or target.shape != (h.shape[0],):
        raise ValueError("Expected h[UE, RX antenna, TX, TX antenna] and one target per UE.")
    if np.any((target < 0) | (target >= h.shape[2])):
        raise ValueError("Target TX index is out of range.")
    if not np.isfinite(margin_db) or margin_db < 0:
        raise ValueError("tn_association_margin_db must be finite and nonnegative.")
    if not np.isfinite(min_channel_norm) or min_channel_norm < 0:
        raise ValueError("tn_min_channel_norm must be finite and nonnegative.")
    power = np.sum(np.abs(h) ** 2, axis=(1, 3))
    rows = np.arange(len(target))
    serving = power[rows, target]
    others = power.copy()
    others[rows, target] = -np.inf
    competitor = np.max(others, axis=1)
    finite = np.all(np.isfinite(power), axis=1)
    margin = np.full(len(target), -np.inf)
    positive = finite & (serving > 0) & (competitor > 0)
    margin[positive] = 10 * (np.log10(serving[positive]) - np.log10(competitor[positive]))
    margin[finite & (serving > 0) & (competitor == 0)] = np.inf
    valid = (finite & (np.sqrt(serving) > max(min_channel_norm, eps))
             & (serving > competitor) & (margin >= margin_db))
    return valid, margin, serving, competitor


class SectorTNSampler:
    """Keep one Bernoulli indoor/outdoor choice per region throughout retries."""

    def __init__(self, x, y, building, zmin, zmax, bs_pos, *,
                 outdoor_probability, yaw_offset_rad, rng, max_attempts=256,
                 outdoor_height=1.8, indoor_roof_offset=-1.5):
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        building = np.asarray(building, dtype=bool)
        zmin, zmax = np.asarray(zmin, float), np.asarray(zmax, float)
        bs = np.asarray(bs_pos, float)
        shape = (len(self.y), len(self.x))
        if len(self.x) < 2 or len(self.y) < 2:
            raise ValueError("The scene grid must have at least two coordinates per axis.")
        if not (np.all(np.diff(self.x) > 0) and np.all(np.diff(self.y) > 0)):
            raise ValueError("Grid coordinates must be strictly increasing.")
        if any(a.shape != shape for a in (building, zmin, zmax)):
            raise ValueError("Grid attribute shapes must match (len(y), len(x)).")
        if bs.shape != (4, 3) or not np.all(np.isfinite(bs)):
            raise ValueError("Sector TN drop requires exactly four finite BS positions.")
        if not np.isfinite(outdoor_probability) or not 0 <= outdoor_probability <= 1:
            raise ValueError("tn_outdoor_probability must be between 0 and 1.")
        if (isinstance(max_attempts, bool) or int(max_attempts) != max_attempts
                or max_attempts < 1):
            raise ValueError("tn_max_attempts must be a positive integer.")
        if not all(np.isfinite(v) for v in (yaw_offset_rad, outdoor_height, indoor_roof_offset)):
            raise ValueError("Orientation and UE heights must be finite.")
        self.rng = rng
        self.max_attempts = int(max_attempts)
        self.is_outdoor = rng.random(12) < outdoor_probability
        self.draw_counts = np.zeros(12, dtype=int)
        self.region_grid = np.full(shape, -1, dtype=np.int16)
        self.rectangles = np.empty((4, 4), dtype=float)
        xmid = (self.x[0] + self.x[-1]) / 2
        ymid = (self.y[0] + self.y[-1]) / 2
        inside = ((bs[:, 0] >= self.x[0]) & (bs[:, 0] <= self.x[-1])
                  & (bs[:, 1] >= self.y[0]) & (bs[:, 1] <= self.y[-1]))
        cells = (bs[:, 0] >= xmid).astype(int) + 2 * (bs[:, 1] >= ymid)
        if not np.all(inside) or len(np.unique(cells)) != 4:
            raise ValueError("Each quadrant must contain exactly one BS.")
        xx, yy = np.meshgrid(self.x, self.y)
        cell_grid = (xx >= xmid).astype(int) + 2 * (yy >= ymid)
        centers = yaw_offset_rad + np.arange(3) * 2 * np.pi / 3
        for b, cell in enumerate(cells):
            dx, dy = xx - bs[b, 0], yy - bs[b, 1]
            angle = np.arctan2(dy, dx)
            delta = np.angle(np.exp(1j * (angle[..., None] - centers)))
            sector = np.argmin(np.abs(delta), axis=-1)
            # Half-open quadrant boundaries and argmin make boundary ownership unique.
            use = (cell_grid == cell) & ((dx * dx + dy * dy) > 0)
            self.region_grid[use] = 3 * b + sector[use]
            self.rectangles[b] = [
                self.x[0] if cell % 2 == 0 else xmid,
                xmid if cell % 2 == 0 else self.x[-1],
                self.y[0] if cell < 2 else ymid,
                ymid if cell < 2 else self.y[-1],
            ]
        self.z = np.where(building, zmax + indoor_roof_offset, zmin + outdoor_height)
        self.pools = []
        for target in range(12):
            eligible = ((self.region_grid == target) & np.isfinite(self.z)
                        & (building == (not self.is_outdoor[target])))
            pool = np.flatnonzero(eligible)
            if pool.size == 0:
                kind = "outdoor" if self.is_outdoor[target] else "indoor"
                raise ValueError(f"No {kind} grid points in BS {target // 3}, sector {target % 3}. "
                                 "The sampled UE type will not be changed.")
            self.pools.append(rng.permutation(pool))

    def draw(self, targets, count=1):
        """Draw up to count unused grid points per target, respecting its budget."""
        if isinstance(count, bool) or int(count) != count or count < 1:
            raise ValueError("Candidate batch size must be a positive integer.")
        selected, labels = [], []
        for target in np.asarray(targets, dtype=int).reshape(-1):
            if target < 0 or target >= 12:
                raise ValueError("Region index must be between 0 and 11.")
            start = self.draw_counts[target]
            stop = min(start + int(count), len(self.pools[target]), self.max_attempts)
            flat = self.pools[target][start:stop]
            selected.extend(flat.tolist())
            labels.extend([int(target)] * len(flat))
            self.draw_counts[target] = stop
        flat = np.asarray(selected, dtype=int)
        row, col = np.unravel_index(flat, self.region_grid.shape)
        pos = np.column_stack((self.x[col], self.y[row], self.z[row, col]))
        return pos, np.asarray(labels, dtype=int)

    def exhausted(self, targets):
        return [int(t) for t in targets
                if self.draw_counts[t] >= min(len(self.pools[t]), self.max_attempts)]

