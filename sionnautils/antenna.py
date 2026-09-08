"""Discrete antenna-pattern interpolation for Sionna RT 2.x."""

import drjit as dr
import mitsuba as mi
import numpy as np

# Importing Sionna RT selects its Mitsuba variant before the grid is created.
import sionna.rt  # noqa: F401


class PatternInterpGrid:
    """Sample a uniformly spaced antenna grid with nearest-neighbor lookup.

    ``Ev`` and ``Eh`` contain the complex field components on a
    ``(num_theta, num_phi)`` grid. The callable returned through ``pattern``
    accepts Mitsuba/Dr.Jit angles and returns Sionna RT-compatible complex
    arrays. This preserves the previous nearest-neighbor lookup behavior while
    using Sionna RT 2.x's native backend.
    """

    def __init__(
        self,
        Ev,
        Eh,
        phi_range=None,
        theta_range=None,
        dtype_real=np.float32,
    ):
        dtype_real = np.dtype(dtype_real)
        if dtype_real not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise ValueError("dtype_real must be numpy.float32 or numpy.float64")

        dtype_complex = np.complex64 if dtype_real == np.dtype(np.float32) else np.complex128
        ev_grid = np.asarray(Ev, dtype=dtype_complex)
        eh_grid = np.asarray(Eh, dtype=dtype_complex)
        if ev_grid.ndim != 2 or eh_grid.ndim != 2:
            raise ValueError("Ev and Eh must be two-dimensional arrays")
        if ev_grid.shape != eh_grid.shape:
            raise ValueError("Ev and Eh must have the same shape")
        if min(ev_grid.shape) < 1:
            raise ValueError("Ev and Eh must not be empty")

        self.dtype_real = dtype_real
        self.dtype_complex = np.dtype(dtype_complex)
        self.theta_range = self._validate_range(theta_range, (0.0, np.pi), "theta_range")
        self.phi_range = self._validate_range(phi_range, (0.0, 2.0 * np.pi), "phi_range")
        self.ntheta, self.nphi = ev_grid.shape
        self.stheta = (self.ntheta - 1) / (self.theta_range[1] - self.theta_range[0])
        self.sphi = (self.nphi - 1) / (self.phi_range[1] - self.phi_range[0])

        ev_flat = ev_grid.reshape(-1)
        eh_flat = eh_grid.reshape(-1)
        self._ev_grid = mi.Complex2f(mi.Float(ev_flat.real), mi.Float(ev_flat.imag))
        self._eh_grid = mi.Complex2f(mi.Float(eh_flat.real), mi.Float(eh_flat.imag))

    @staticmethod
    def _validate_range(value, default, name):
        result = np.asarray(default if value is None else value, dtype=np.float64)
        if result.shape != (2,) or not np.all(np.isfinite(result)):
            raise ValueError(f"{name} must contain two finite values")
        if result[1] <= result[0]:
            raise ValueError(f"{name} must be strictly increasing")
        return result

    def pattern(self, theta, phi):
        """Return the nearest sampled vertical and horizontal field values."""
        theta = mi.Float(theta)
        phi = mi.Float(phi)

        fold = theta > dr.pi
        theta = dr.select(fold, 2.0 * dr.pi - theta, theta)
        phi = dr.select(fold, phi + dr.pi, phi)
        phi = dr.select(phi < 0.0, phi + 2.0 * dr.pi, phi)

        theta_index = mi.Int32(self.stheta * (theta - self.theta_range[0]))
        phi_index = mi.Int32(self.sphi * (phi - self.phi_range[0]))
        theta_index = dr.clip(theta_index, 0, self.ntheta - 1)
        phi_index = dr.clip(phi_index, 0, self.nphi - 1)
        flat_index = mi.UInt32(theta_index * self.nphi + phi_index)

        ev = dr.gather(mi.Complex2f, self._ev_grid, flat_index)
        eh = dr.gather(mi.Complex2f, self._eh_grid, flat_index)
        return ev, eh
