"""Physics baseline for coasting apogee prediction (NumPy/pandas only).

After burnout the vertical motion of a statically stable rocket is close to

    dv/dt = -g - k * v**2          (v > 0, quadratic drag, k = rho*Cd*A / 2m)

whose apex is available in closed form:

    H = h + ln(1 + k*v**2/g) / (2k) = h + (v**2 / 2g) * ln(1+x)/x,  x = k*v**2/g

The drag constant ``k`` is *measured*, not learned: with gravity-removed
vertical acceleration ``a`` the coast residual ``-a - g`` equals ``k*v**2``.
A causal weighted least-squares fit over the samples seen since coast onset
gives ``k`` on board, so the baseline needs no training data and transfers to
a vehicle whose drag lies outside the simulated range.

Limits (stated, not hidden):

* Only coasting is modelled.  During thrust the remaining impulse is unknown,
  so the baseline returns NaN until coast is confirmed.
* Vertical-only: with a tilted trajectory the fitted ``k`` absorbs
  ``k_true*|V|/v_z``.  Density change with altitude is ignored.
* A constant accelerometer bias ``b`` biases ``k`` by about ``b/v**2``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

SIM_G = 9.81


class PhysicsBaselineError(ValueError):
    """Raised when inputs violate the baseline contract."""


def ballistic_apogee(h, v, k, *, gravity: float = SIM_G) -> np.ndarray:
    """Closed-form apex of vertical coast with quadratic drag.

    ``k = 0`` reduces exactly to the drag-free energy height ``h + v**2/2g``.
    Non-positive ``v`` (at or past the apex) returns ``h``.  Negative ``k`` is
    rejected because it would mean drag adds energy.
    """
    h = np.asarray(h, dtype=float)
    v = np.asarray(v, dtype=float)
    k = np.asarray(k, dtype=float)
    if gravity <= 0:
        raise PhysicsBaselineError("gravity must be positive")
    if np.any(k[np.isfinite(k)] < 0):
        raise PhysicsBaselineError("drag constant k must be non-negative")
    v_up = np.maximum(v, 0.0)
    x = k * v_up**2 / gravity
    # ln(1+x)/x -> 1 as x -> 0; evaluate the limit explicitly for stability.
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = np.where(x > 1e-12, np.log1p(x) / np.where(x > 1e-12, x, 1.0), 1.0 - 0.5 * x)
    return h + v_up**2 / (2.0 * gravity) * factor


def fit_drag_constant(
    accel_inertial: Sequence[float],
    velocity: Sequence[float],
    *,
    gravity: float = SIM_G,
    v_min: float = 5.0,
) -> float:
    """Weighted least squares for ``-a - g = k v**2`` (weights grow with v**2).

    Returns NaN when no sample is usable and clips a negative fit to zero.
    """
    a = np.asarray(accel_inertial, dtype=float)
    v = np.asarray(velocity, dtype=float)
    if a.shape != v.shape:
        raise PhysicsBaselineError("acceleration and velocity must have the same shape")
    use = np.isfinite(a) & np.isfinite(v) & (v > v_min)
    if not use.any():
        return float("nan")
    v2 = v[use] ** 2
    k = float(np.sum((-a[use] - gravity) * v2) / np.sum(v2 * v2))
    return max(k, 0.0)


@dataclass(frozen=True)
class CoastConfig:
    """Causal coast detection and drag-fit settings.

    ``accel_threshold`` is in gravity-removed m/s**2.  Coast is latched once
    ``confirm_samples`` consecutive samples fall below it; thrust tail-off
    above the threshold is never treated as coast.  ``settle_s`` then skips
    that long after onset before the first fit and output, because thrust
    tail-off still adds impulse there and would bias ``k`` low (apex high).
    ``fit_window_s=None`` fits ``k`` on every settled sample (expanding
    window).

    Defaults were tuned on the train+validation flights of the 200-flight
    simulation only (test held out): -0.95 g, 5 samples, 0.5 s settle,
    expanding window.  ``settle_s`` is tied to the simulated motor's tail-off;
    re-check it for another motor.  A longer settle lowers the error mostly by
    dropping early (harder) endpoints, so compare models on matched endpoints.
    """

    accel_threshold: float = -0.95 * SIM_G
    confirm_samples: int = 5
    settle_s: float = 0.5
    fit_window_s: float | None = None
    v_min: float = 5.0
    gravity: float = SIM_G

    def validate(self) -> None:
        if self.confirm_samples < 1:
            raise PhysicsBaselineError("confirm_samples must be >= 1")
        if self.settle_s < 0:
            raise PhysicsBaselineError("settle_s must be non-negative")
        if self.fit_window_s is not None and self.fit_window_s <= 0:
            raise PhysicsBaselineError("fit_window_s must be positive or None")
        if self.gravity <= 0 or self.v_min < 0:
            raise PhysicsBaselineError("gravity must be positive and v_min non-negative")


def _flight_baseline(
    t: np.ndarray, h: np.ndarray, v: np.ndarray, a: np.ndarray, cfg: CoastConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(t)
    coast = np.zeros(n, dtype=bool)
    k_est = np.full(n, np.nan)
    below = np.isfinite(a) & (a < cfg.accel_threshold)
    run = 0
    onset = None
    for i in range(n):
        run = run + 1 if below[i] else 0
        if onset is None and run >= cfg.confirm_samples:
            onset = i - cfg.confirm_samples + 1
    if onset is None:
        return coast, k_est, np.full(n, np.nan)
    # Latch: coast is declared once confirmed and settled, never released.
    confirmed = onset + cfg.confirm_samples - 1
    settled = int(np.searchsorted(t, t[onset] + cfg.settle_s - 1e-9, side="left"))
    onset = max(settled, onset)
    coast[max(confirmed, onset):] = True
    if not coast.any():
        return coast, k_est, np.full(n, np.nan)

    use = np.isfinite(a) & np.isfinite(v) & (v > cfg.v_min)
    use[:onset] = False
    v2 = np.where(use, v, 0.0) ** 2
    num = np.cumsum(np.where(use, (-a - cfg.gravity) * v2, 0.0))
    den = np.cumsum(v2 * v2)
    if cfg.fit_window_s is None:
        num_w, den_w = num, den
    else:
        start = np.searchsorted(t, t - cfg.fit_window_s, side="left")
        start = np.maximum(start, onset)
        prev = start - 1
        num_w = num - np.where(prev >= 0, num[np.maximum(prev, 0)], 0.0)
        den_w = den - np.where(prev >= 0, den[np.maximum(prev, 0)], 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.where(den_w > 0, num_w / den_w, np.nan)
    # Carry the last valid fit forward once v drops below v_min near apex.
    k = pd.Series(np.where(coast, k, np.nan)).ffill().to_numpy()
    k = np.where(coast, np.maximum(k, 0.0), np.nan)
    k_est[:] = k
    apex = np.full(n, np.nan)
    ok = coast & np.isfinite(k) & np.isfinite(h) & np.isfinite(v)
    apex[ok] = ballistic_apogee(h[ok], v[ok], k[ok], gravity=cfg.gravity)
    return coast, k_est, apex


def online_ballistic_apogee(
    frame: pd.DataFrame,
    *,
    config: CoastConfig | None = None,
    flight_col: str = "flight_id",
    time_col: str = "time",
    h_col: str = "altitude_rel",
    v_col: str = "velocity_z",
    accel_col: str = "acceleration_inertial",
) -> pd.DataFrame:
    """Causal per-row physics baseline.

    Returns ``flight_id, time, coast, k_est, apogee_ballistic`` aligned with
    ``frame`` order.  Every output row depends only on that flight's rows at
    or before it.  ``apogee_ballistic`` is NaN until coast is confirmed.
    """
    cfg = config or CoastConfig()
    cfg.validate()
    missing = [c for c in (flight_col, time_col, h_col, v_col, accel_col) if c not in frame.columns]
    if missing:
        raise PhysicsBaselineError(f"missing columns: {missing}")
    parts = []
    for flight_id, group in frame.groupby(flight_col, sort=False):
        t = group[time_col].to_numpy(dtype=float)
        if len(t) > 1 and np.any(np.diff(t) <= 0):
            raise PhysicsBaselineError(f"time must be strictly increasing in flight {flight_id}")
        coast, k, apex = _flight_baseline(
            t,
            group[h_col].to_numpy(dtype=float),
            group[v_col].to_numpy(dtype=float),
            group[accel_col].to_numpy(dtype=float),
            cfg,
        )
        parts.append(pd.DataFrame({
            flight_col: flight_id,
            time_col: t,
            "coast": coast,
            "k_est": k,
            "apogee_ballistic": apex,
        }, index=group.index))
    if not parts:
        return pd.DataFrame(columns=[flight_col, time_col, "coast", "k_est", "apogee_ballistic"])
    return pd.concat(parts).loc[frame.index]


def kinematic_smoother(
    frame: pd.DataFrame,
    *,
    window_s: float = 1.5,
    flight_col: str = "flight_id",
    time_col: str = "time",
    baro_col: str = "altitude_noisy_rel",
    accel_col: str = "acceleration_inertial",
) -> pd.DataFrame:
    """Causal least-squares altitude/velocity from barometer + accelerometer.

    Over the trailing window ending at sample ``e`` the measured acceleration
    is integrated twice (``V``, ``P`` = cumulative trapezoids), so the only
    unknowns are the endpoint state:

        baro_j ~= h_e + v_e*tau_j + (P_j - P_e - V_e*tau_j),   tau_j = t_j - t_e <= 0

    A two-parameter linear fit gives ``h_e`` and ``v_e``.  No dynamics are
    learned; the window trades barometer noise against accelerometer bias
    (a bias ``b`` adds about ``b*window_s/2`` to ``v``).  Rows with fewer than
    three finite samples in the window return NaN.

    The 1.5 s default was chosen on train+validation flights with the flight
    firmware UKF replay grid (0.3-4 s scanned).  There the smoothed velocity
    error was 0.44 m/s against 5.0 m/s for the UKF velocity.
    """
    if window_s <= 0:
        raise PhysicsBaselineError("window_s must be positive")
    parts = []
    for flight_id, group in frame.groupby(flight_col, sort=False):
        t = group[time_col].to_numpy(dtype=float)
        z = group[baro_col].to_numpy(dtype=float)
        a = group[accel_col].to_numpy(dtype=float)
        a_filled = np.where(np.isfinite(a), a, 0.0)
        dt = np.diff(t, prepend=t[0])
        vel = np.cumsum(0.5 * (a_filled + np.roll(a_filled, 1)) * np.where(np.arange(len(t)) > 0, dt, 0.0))
        pos = np.cumsum(0.5 * (vel + np.roll(vel, 1)) * np.where(np.arange(len(t)) > 0, dt, 0.0))
        start = np.searchsorted(t, t - window_s - 1e-9, side="left")
        h_fit = np.full(len(t), np.nan)
        v_fit = np.full(len(t), np.nan)
        for e in range(len(t)):
            s = start[e]
            tau = t[s:e + 1] - t[e]
            y = z[s:e + 1] - (pos[s:e + 1] - pos[e] - vel[e] * tau)
            ok = np.isfinite(y) & np.isfinite(a[s:e + 1])
            if ok.sum() < 3:
                continue
            tau, y = tau[ok], y[ok]
            tm, ym = tau.mean(), y.mean()
            var = np.sum((tau - tm) ** 2)
            if var <= 0:
                continue
            slope = np.sum((tau - tm) * (y - ym)) / var
            v_fit[e] = slope
            h_fit[e] = ym - slope * tm
        parts.append(pd.DataFrame({flight_col: flight_id, time_col: t, "h_smooth": h_fit,
                                   "v_smooth": v_fit}, index=group.index))
    if not parts:
        return pd.DataFrame(columns=[flight_col, time_col, "h_smooth", "v_smooth"])
    return pd.concat(parts).loc[frame.index]


@dataclass(frozen=True)
class ApogeeReference:
    """Reference apex for a log that may stop before apogee.

    ``observed_max`` is a lower bound whenever ``apogee_observed`` is False;
    it must not be scored as the true apogee.  ``extrapolated`` adds the
    ballistic remaining climb from the final state with the drag constant
    fitted over the whole logged coast.
    """

    apogee_observed: bool
    observed_max: float
    extrapolated: float
    final_velocity: float
    k_fit: float
    coast_samples: int

    @property
    def best_estimate(self) -> float:
        return self.observed_max if self.apogee_observed else self.extrapolated


def apogee_reference(
    time: Sequence[float],
    h: Sequence[float],
    v: Sequence[float],
    accel_inertial: Sequence[float],
    *,
    config: CoastConfig | None = None,
    apex_velocity: float = 0.5,
) -> ApogeeReference:
    """Build an honest apogee reference for one logged flight.

    The log counts as reaching apogee only when the final velocity is at or
    below ``apex_velocity`` (m/s) or the altitude maximum lies before the end.
    """
    cfg = config or CoastConfig()
    cfg.validate()
    t = np.asarray(time, dtype=float)
    h = np.asarray(h, dtype=float)
    v = np.asarray(v, dtype=float)
    a = np.asarray(accel_inertial, dtype=float)
    if not (len(t) == len(h) == len(v) == len(a)) or len(t) == 0:
        raise PhysicsBaselineError("time/h/v/accel must be non-empty and equally long")
    observed_max = float(np.nanmax(h))
    i_max = int(np.nanargmax(h))
    final_v = float(v[-1])
    apogee_observed = bool(i_max < len(h) - 1 and h[-1] < observed_max) or final_v <= apex_velocity
    coast, _, _ = _flight_baseline(t, h, v, a, cfg)
    if coast.any():
        onset = int(np.argmax(coast))
        k = fit_drag_constant(a[onset:], v[onset:], gravity=cfg.gravity, v_min=cfg.v_min)
    else:
        k = float("nan")
    if apogee_observed:
        extrapolated = observed_max
    elif np.isfinite(k):
        extrapolated = float(max(observed_max, ballistic_apogee(h[-1], final_v, k, gravity=cfg.gravity)))
    else:
        extrapolated = float("nan")
    return ApogeeReference(
        apogee_observed=apogee_observed,
        observed_max=observed_max,
        extrapolated=extrapolated,
        final_velocity=final_v,
        k_fit=float(k),
        coast_samples=int(coast.sum()),
    )
