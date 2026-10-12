import time

import numpy as np


class TraversabilityMap:
    """2D traversability grid indexed by [row, column].

    Dimensions, offsets, and resolution are in metres. Offsets locate the
    lower-left grid boundary; columns increase along x and rows along y.
    Partial boundary cells are included by rounding dimensions up.

    Values are -1 (unknown) or 0-100 (increasing traversability). These scores
    have the opposite meaning to ROS occupancy probabilities. Floating-point
    storage preserves fractional scores during decay.
    """

    def __init__(self, h: float, w: float, offset_x: float, offset_y: float,
                 resolution: float, decay_time: float | None = None):
        for name, value in (("h", h), ("w", w), ("resolution", resolution)):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not np.isfinite([offset_x, offset_y]).all():
            raise ValueError("Offsets must be finite")

        self.width = w
        self.height = h
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.resolution = resolution
        self.decay_time = decay_time

        rows = int(np.ceil(h / resolution))
        columns = int(np.ceil(w / resolution))
        self.data = np.full((rows, columns), -1.0, dtype=np.float64)
        self._last_decay = time.monotonic()

    def _trav_decay(self):
        """Decay known scores with decay_time as the half-life in seconds.

        Call periodically to refresh the map. Unknown cells remain unknown;
        zero remains zero. Scores approach zero rather than expiring to unknown.
        """
        if self.decay_time is None: return
        now = time.monotonic()
        elapsed = now - self._last_decay
        self._last_decay = now
        known = self.data >= 0
        self.data[known] *= np.exp2(-elapsed / self.decay_time)

    def __getitem__(self, index):
        return self.data[index]

    def __setitem__(self, index, value):
        """Add a score or slice, keeping updated cells in the range 0-100.

        Decay existing observations before adding fresh ones, treating unknown
        cells as zero and clipping totals to 0-100. Negative scores subtract
        from existing values without resetting cells to unknown. New scores
        are not aged for time that elapsed before they were observed. Write via this method
        rather than directly to data to preserve that behavior.
        """
        values = np.asarray(value, dtype=np.float64)
        if not np.isfinite(values).all() or np.any(values > 100):
            raise ValueError("Traversability must be finite and at most 100")
        # Validate indexing/broadcasting before changing any map state.
        target = np.empty_like(self.data[index])
        target[...] = values
        self._trav_decay()
        self.data[index] = np.clip(
            np.maximum(self.data[index], 0) + target, 0, 100,
        )
