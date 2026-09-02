"""Monte Carlo audit for the persistent two-state Gaussian HMM.

The script validates the oracle-scale long-lag estimator used in
Theorem ``persistence-local-upper`` and compares it with the lag-one plug-in.
It also checks the two-point confidence lower-bound construction across the
regular/saturated transition.

Only NumPy and Pillow are required.  The simulation is streaming: no full
T-by-replications array is stored.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class SimulationResult:
    u_hat_long: np.ndarray
    u_hat_lag1: np.ndarray
    r_hat_long: np.ndarray
    r_hat_lag1: np.ndarray
    k: int
    m_long: int


def simulate_long_subsampled(
    *,
    u: float,
    t_length: int,
    replications: int,
    a: float,
    sigma: float,
    u_star: float,
    alpha: float,
    beta: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Simulate the exact marginal law of the long-lag statistic.

    Subsampling the original chain every k observations gives another
    symmetric two-state chain with switching probability
    {1-(1-2u)^k}/2.  This avoids generating observations that the statistic
    never uses and is distributionally identical for this experiment.
    """
    k = math.ceil(1.0 / u_star)
    m_long = (t_length - 1) // k
    if m_long == 0:
        midpoint = 0.5 * (alpha + beta) * u_star
        return (
            np.full(replications, midpoint),
            np.full(replications, np.nan),
            k,
            m_long,
        )

    rho = (1.0 - 2.0 * u) ** k
    sampled_switch_probability = (1.0 - rho) / 2.0
    y = rng.choice(np.array([-1.0, 1.0]), size=replications)
    x_previous = a * y + sigma * rng.standard_normal(replications)
    product_sum = np.zeros(replications)
    for _ in range(m_long):
        flips = rng.random(replications) < sampled_switch_probability
        y[flips] *= -1.0
        x_current = a * y + sigma * rng.standard_normal(replications)
        product_sum += x_previous * x_current
        x_previous = x_current

    r_hat = product_sum / (a * a * m_long)
    lower = (1.0 - 2.0 * beta * u_star) ** k
    upper = (1.0 - 2.0 * alpha * u_star) ** k
    projected = _project(r_hat, lower, upper)
    u_hat = (1.0 - np.power(projected, 1.0 / k)) / 2.0
    return u_hat, r_hat, k, m_long


def v_infinity(u: float, a: float, sigma: float) -> float:
    return sigma * sigma + a * a * (1.0 - u) / u


def _project(x: np.ndarray, lower: float, upper: float) -> np.ndarray:
    return np.minimum(np.maximum(x, lower), upper)


def simulate_estimators(
    *,
    u: float,
    t_length: int,
    replications: int,
    a: float,
    sigma: float,
    u_star: float,
    alpha: float,
    beta: float,
    rng: np.random.Generator,
) -> SimulationResult:
    """Simulate both product-moment estimators from stationary trajectories."""
    if not (0.0 < alpha <= 1.0 <= beta):
        raise ValueError("Require 0 < alpha <= 1 <= beta.")
    if not (alpha * u_star <= u <= beta * u_star):
        raise ValueError("The true u must belong to the prescribed local cell.")
    if t_length < 2:
        raise ValueError("t_length must be at least 2.")

    k = math.ceil(1.0 / u_star)
    m_long = (t_length - 1) // k
    a2 = a * a

    y = rng.choice(np.array([-1.0, 1.0]), size=replications)
    x_previous = a * y + sigma * rng.standard_normal(replications)
    x_sampled_previous = x_previous.copy()
    sum_lag1 = np.zeros(replications)
    sum_long = np.zeros(replications)

    # Vectorize short time blocks while retaining the exact same trajectory
    # for both estimators.  The cell cap keeps peak memory bounded when the
    # number of replications is large.
    block_size = max(16, min(256, 1_500_000 // replications))
    t_start = 1
    while t_start < t_length:
        width = min(block_size, t_length - t_start)
        flips = rng.random((replications, width)) < u
        parity = np.bitwise_and(
            np.cumsum(flips, axis=1, dtype=np.uint16), 1
        ).astype(np.int8)
        signs = 1 - 2 * parity
        y_block = y[:, None] * signs
        x_block = a * y_block + sigma * rng.standard_normal((replications, width))

        sum_lag1 += x_previous * x_block[:, 0]
        if width > 1:
            sum_lag1 += np.sum(x_block[:, :-1] * x_block[:, 1:], axis=1)

        t_end = t_start + width - 1
        first_sample = ((t_start + k - 1) // k) * k
        if first_sample <= t_end:
            sample_columns = np.arange(first_sample, t_end + 1, k) - t_start
            for column in sample_columns:
                x_sampled_current = x_block[:, column]
                sum_long += x_sampled_previous * x_sampled_current
                x_sampled_previous = x_sampled_current.copy()

        y = y_block[:, -1].copy()
        x_previous = x_block[:, -1].copy()
        t_start += width

    r_hat_lag1 = sum_lag1 / (a2 * (t_length - 1))
    lag1_lower = 1.0 - 2.0 * beta * u_star
    lag1_upper = 1.0 - 2.0 * alpha * u_star
    r_lag1_projected = _project(r_hat_lag1, lag1_lower, lag1_upper)
    u_hat_lag1 = (1.0 - r_lag1_projected) / 2.0

    if m_long == 0:
        # The theorem prescribes an arbitrary point in the cell when no block
        # is observed.  The midpoint avoids favouring either endpoint.
        u_hat_long = np.full(replications, 0.5 * (alpha + beta) * u_star)
        r_hat_long = np.full(replications, np.nan)
    else:
        r_hat_long = sum_long / (a2 * m_long)
        long_lower = (1.0 - 2.0 * beta * u_star) ** k
        long_upper = (1.0 - 2.0 * alpha * u_star) ** k
        r_long_projected = _project(r_hat_long, long_lower, long_upper)
        u_hat_long = (1.0 - np.power(r_long_projected, 1.0 / k)) / 2.0

    return SimulationResult(
        u_hat_long=u_hat_long,
        u_hat_lag1=u_hat_lag1,
        r_hat_long=r_hat_long,
        r_hat_lag1=r_hat_lag1,
        k=k,
        m_long=m_long,
    )


def exact_moment_sd(
    *, u: float, k: int, m: int, a: float, sigma: float
) -> float:
    """Exact SD of M^{-1} sum X_{rk}X_{(r+1)k}/a^2.

    The normalized products are one-dependent.  Their variance is
    (1+s^2)^2-rho^2 and their adjacent covariance is s^2 rho^2, where
    s^2=sigma^2/a^2 and rho=(1-2u)^k.
    """
    if m < 1:
        return math.nan
    rho = (1.0 - 2.0 * u) ** k
    s2 = (sigma * sigma) / (a * a)
    variance = (1.0 + s2) ** 2 - rho * rho
    adjacent_covariance = s2 * rho * rho
    variance_mean = (
        m * variance + 2.0 * (m - 1) * adjacent_covariance
    ) / (m * m)
    return math.sqrt(variance_mean)


def delta_sd_v(
    *, u: float, k: int, m: int, a: float, sigma: float
) -> float:
    """First-order SD of v-hat induced by the unprojected moment."""
    rho = (1.0 - 2.0 * u) ** k
    inverse_derivative = abs(1.0 - 2.0 * u) / (2.0 * k * rho)
    return (
        (a * a) / (u * u)
        * inverse_derivative
        * exact_moment_sd(u=u, k=k, m=m, a=a, sigma=sigma)
    )


def _error_summaries(u_hat: np.ndarray, u: float, a: float, sigma: float) -> dict:
    target = v_infinity(u, a, sigma)
    estimates = v_infinity(u_hat, a, sigma)
    errors = estimates - target
    absolute = np.abs(errors)
    return {
        "rmse": float(np.sqrt(np.mean(errors * errors))),
        "median_abs": float(np.median(absolute)),
        "q90_abs": float(np.quantile(absolute, 0.90)),
        "q95_abs": float(np.quantile(absolute, 0.95)),
        "errors": errors,
    }


def run_power_experiment(
    *, output_dir: Path, quick: bool, seed: int
) -> tuple[list[dict], dict]:
    a = 1.0
    sigma = 1.0
    alpha = 0.25
    beta = 4.0
    t_length = 8193 if quick else 65537
    replications = 500 if quick else 2500
    u_values = np.array([0.0400, 0.0300, 0.0225, 0.0169, 0.0127, 0.0095])
    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    for u in u_values:
        u_star = u
        result = simulate_estimators(
            u=float(u),
            t_length=t_length,
            replications=replications,
            a=a,
            sigma=sigma,
            u_star=float(u_star),
            alpha=alpha,
            beta=beta,
            rng=rng,
        )
        long_summary = _error_summaries(result.u_hat_long, float(u), a, sigma)
        lag1_summary = _error_summaries(result.u_hat_lag1, float(u), a, sigma)
        long_sd_r = exact_moment_sd(
            u=float(u), k=result.k, m=result.m_long, a=a, sigma=sigma
        )
        lag1_sd_r = exact_moment_sd(
            u=float(u), k=1, m=t_length - 1, a=a, sigma=sigma
        )
        long_delta = delta_sd_v(
            u=float(u), k=result.k, m=result.m_long, a=a, sigma=sigma
        )
        lag1_delta = delta_sd_v(
            u=float(u), k=1, m=t_length - 1, a=a, sigma=sigma
        )
        long_lower = (1.0 - 2.0 * beta * u_star) ** result.k
        long_upper = (1.0 - 2.0 * alpha * u_star) ** result.k
        lag1_lower = 1.0 - 2.0 * beta * u_star
        lag1_upper = 1.0 - 2.0 * alpha * u_star
        lambda_value = 1.0 - 2.0 * float(u)
        rho_long = lambda_value ** result.k
        long_inverse_derivative = (
            (a * a)
            / (float(u) ** 2)
            * abs(lambda_value)
            / (2.0 * result.k * rho_long)
        )
        lag1_inverse_derivative = (a * a) / (2.0 * float(u) ** 2)
        long_r_error = result.r_hat_long - rho_long
        lag1_r_error = result.r_hat_lag1 - lambda_value
        long_linearized_rmse = long_inverse_derivative * float(
            np.sqrt(np.mean(long_r_error * long_r_error))
        )
        lag1_linearized_rmse = lag1_inverse_derivative * float(
            np.sqrt(np.mean(lag1_r_error * lag1_r_error))
        )
        rows.append(
            {
                "u": float(u),
                "u_star": float(u_star),
                "T": t_length,
                "replications": replications,
                "k": result.k,
                "M": result.m_long,
                "long_rmse": long_summary["rmse"],
                "lag1_rmse": lag1_summary["rmse"],
                "long_q90": long_summary["q90_abs"],
                "lag1_q90": lag1_summary["q90_abs"],
                "long_delta_sd": long_delta,
                "lag1_delta_sd": lag1_delta,
                "long_linearized_rmse": long_linearized_rmse,
                "lag1_linearized_rmse": lag1_linearized_rmse,
                "long_rmse_over_delta": long_summary["rmse"] / long_delta,
                "lag1_rmse_over_delta": lag1_summary["rmse"] / lag1_delta,
                "long_r_sd_over_exact": float(np.std(result.r_hat_long, ddof=1))
                / long_sd_r,
                "lag1_r_sd_over_exact": float(np.std(result.r_hat_lag1, ddof=1))
                / lag1_sd_r,
                "long_r_bias_z": float(np.mean(long_r_error))
                / (long_sd_r / math.sqrt(replications)),
                "lag1_r_bias_z": float(np.mean(lag1_r_error))
                / (lag1_sd_r / math.sqrt(replications)),
                "long_projection_rate": float(
                    np.mean(
                        (result.r_hat_long <= long_lower)
                        | (result.r_hat_long >= long_upper)
                    )
                ),
                "lag1_projection_rate": float(
                    np.mean(
                        (result.r_hat_lag1 <= lag1_lower)
                        | (result.r_hat_lag1 >= lag1_upper)
                    )
                ),
            }
        )

    log_u = np.log([row["u"] for row in rows])
    slopes = {
        "long_empirical": float(
            np.polyfit(log_u, np.log([row["long_rmse"] for row in rows]), 1)[0]
        ),
        "lag1_empirical": float(
            np.polyfit(log_u, np.log([row["lag1_rmse"] for row in rows]), 1)[0]
        ),
        "long_delta": float(
            np.polyfit(
                log_u, np.log([row["long_delta_sd"] for row in rows]), 1
            )[0]
        ),
        "lag1_delta": float(
            np.polyfit(
                log_u, np.log([row["lag1_delta_sd"] for row in rows]), 1
            )[0]
        ),
        "long_linearized_empirical": float(
            np.polyfit(
                log_u,
                np.log([row["long_linearized_rmse"] for row in rows]),
                1,
            )[0]
        ),
        "lag1_linearized_empirical": float(
            np.polyfit(
                log_u,
                np.log([row["lag1_linearized_rmse"] for row in rows]),
                1,
            )[0]
        ),
    }
    _write_csv(output_dir / "power_experiment.csv", rows)
    return rows, slopes


def run_boundary_experiment(
    *, output_dir: Path, quick: bool, seed: int
) -> list[dict]:
    a = 1.0
    sigma = 1.0
    alpha = 1.0
    beta = 1.5
    u0 = 0.01
    delta = 0.05
    ell = math.log(1.0 / (4.0 * delta))
    replications = 1500 if quick else 10000
    tu_values = [
        1.00,
        2.00,
        3.00,
        4.00,
        5.00,
        6.00,
        8.00,
        12.00,
        20.00,
        40.00,
        80.00,
        160.00,
        320.00,
        640.00,
        1280.00,
        2560.00,
        5120.00,
    ]
    if quick:
        tu_values = tu_values[:11]
    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    for tu_value in tu_values:
        t_length = 1 + round(tu_value / u0)
        c_value = (t_length - 1) * u0 / ell
        d_value = min(
            u0 / 2.0,
            math.sqrt(5.0 * u0 * ell / (8.0 * (t_length - 1))),
        )
        u1 = u0 + d_value
        threshold = (a * a) * d_value / (2.0 * u0 * u1)
        results = []
        summaries = []
        for u in (u0, u1):
            u_hat, r_hat, k, m_long = simulate_long_subsampled(
                u=u,
                t_length=t_length,
                replications=replications,
                a=a,
                sigma=sigma,
                u_star=u0,
                alpha=alpha,
                beta=beta,
                rng=rng,
            )
            results.append((u_hat, r_hat, k, m_long))
            summaries.append(_error_summaries(u_hat, u, a, sigma))
        fail0 = float(np.mean(np.abs(summaries[0]["errors"]) >= threshold))
        fail1 = float(np.mean(np.abs(summaries[1]["errors"]) >= threshold))
        empirical_radius_90 = max(summaries[0]["q90_abs"], summaries[1]["q90_abs"])
        diameter = (a * a) * (beta - alpha) / (alpha * beta * u0)
        natural_regular = (
            (abs(a) + sigma) ** 2
            * math.sqrt(math.log(6.0 / delta))
            / (u0 ** 1.5 * math.sqrt(t_length - 1))
        )
        kl_complete = (t_length - 1) * (
            u0 * math.log(u0 / u1)
            + (1.0 - u0) * math.log((1.0 - u0) / (1.0 - u1))
        )
        bh_probability = 0.25 * math.exp(-kl_complete)
        rows.append(
            {
                "c_Tu_over_ell": c_value,
                "T": t_length,
                "Tu": (t_length - 1) * u0,
                "ell_lower": ell,
                "d": d_value,
                "branch": "saturated" if math.isclose(d_value, u0 / 2.0) else "regular",
                "k": results[0][2],
                "M": results[0][3],
                "threshold_half_separation": threshold,
                "failure_u0": fail0,
                "failure_u1": fail1,
                "max_failure": max(fail0, fail1),
                "complete_KL": kl_complete,
                "complete_KL_over_ell": kl_complete / ell,
                "BH_probability": bh_probability,
                "empirical_radius_90": empirical_radius_90,
                "radius90_over_a2_over_u": empirical_radius_90 / ((a * a) / u0),
                "radius90_over_min_natural_scale": empirical_radius_90
                / min(diameter, natural_regular),
                "diameter": diameter,
                "natural_regular_scale": natural_regular,
                "replications_per_alternative": replications,
            }
        )

    _write_csv(output_dir / "boundary_experiment.csv", rows)
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _log_map(value: float, low: float, high: float, start: int, end: int) -> float:
    return start + (math.log(value) - math.log(low)) / (math.log(high) - math.log(low)) * (
        end - start
    )


def make_figure(
    output_path: Path, power_rows: list[dict], slopes: dict, boundary_rows: list[dict]
) -> None:
    width, height = 1800, 760
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = _font(30, bold=True)
    label_font = _font(22)
    small_font = _font(18)
    draw.text((55, 25), "Persistent-path Monte Carlo audit", fill="black", font=title_font)

    panels = [(80, 105, 850, 665), (950, 105, 1720, 665)]
    for left, top, right, bottom in panels:
        draw.rectangle((left, top, right, bottom), outline="#444444", width=2)

    # Left panel: log-log RMSE slopes.
    left, top, right, bottom = panels[0]
    u_values = [row["u"] for row in power_rows]
    all_errors = [row[key] for row in power_rows for key in ("long_rmse", "lag1_rmse")]
    u_low, u_high = min(u_values) * 0.90, max(u_values) * 1.10
    e_low, e_high = min(all_errors) * 0.75, max(all_errors) * 1.35
    colors = {"long_rmse": "#0068b5", "lag1_rmse": "#c43c39"}
    labels = {"long_rmse": "long lag", "lag1_rmse": "lag 1"}
    for key in ("long_rmse", "lag1_rmse"):
        points = []
        for row in power_rows:
            x = _log_map(row["u"], u_low, u_high, left + 65, right - 30)
            y = _log_map(row[key], e_low, e_high, bottom - 55, top + 45)
            points.append((x, y))
        draw.line(points, fill=colors[key], width=5)
        for x, y in points:
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=colors[key])
    draw.text((left + 20, top + 12), "RMSE versus persistence hazard u", font=label_font, fill="black")
    draw.text((left + 20, bottom + 18), "log-log axes; T fixed", font=small_font, fill="#444444")
    draw.text(
        (left + 390, top + 62),
        f"long slope {slopes['long_empirical']:.3f} (target -1.5)",
        font=small_font,
        fill=colors["long_rmse"],
    )
    draw.text(
        (left + 390, top + 92),
        f"lag-1 plug-in slope {slopes['lag1_empirical']:.3f}",
        font=small_font,
        fill=colors["lag1_rmse"],
    )
    draw.text(
        (left + 390, top + 122),
        f"lag-1 linearized {slopes['lag1_linearized_empirical']:.3f} (target -2)",
        font=small_font,
        fill=colors["lag1_rmse"],
    )

    # Right panel: lower-bound pair failure and branch switch.
    left, top, right, bottom = panels[1]
    c_values = [row["c_Tu_over_ell"] for row in boundary_rows]
    failures = [row["max_failure"] for row in boundary_rows]
    points = []
    for c_value, failure in zip(c_values, failures):
        x = _log_map(c_value, min(c_values) * 0.85, max(c_values) * 1.15, left + 65, right - 30)
        y = bottom - 55 - failure * (bottom - top - 100)
        points.append((x, y))
    draw.line(points, fill="#5b2c83", width=5)
    for x, y in points:
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill="#5b2c83")
    y_delta = bottom - 55 - 0.10 * (bottom - top - 100)
    draw.line((left + 65, y_delta, right - 30, y_delta), fill="#777777", width=2)
    x_switch = _log_map(2.5, min(c_values) * 0.85, max(c_values) * 1.15, left + 65, right - 30)
    draw.line((x_switch, top + 45, x_switch, bottom - 55), fill="#d18b00", width=3)
    draw.text((left + 20, top + 12), "Two-point lower-bound diagnostic", font=label_font, fill="black")
    draw.text((left + 20, top + 62), "max failure probability", font=small_font, fill="#5b2c83")
    draw.text((left + 20, top + 92), "delta = 0.05", font=small_font, fill="#777777")
    draw.text((x_switch + 8, top + 135), "d branch changes", font=small_font, fill="#a46900")
    draw.text((x_switch + 8, top + 162), "at Tu/ell = 2.5", font=small_font, fill="#a46900")
    draw.text((left + 20, bottom + 18), "horizontal axis: (T-1)u / log(1/(4 delta))", font=small_font, fill="#444444")

    image.save(output_path, dpi=(180, 180))


def write_report(
    output_path: Path, power_rows: list[dict], slopes: dict, boundary_rows: list[dict]
) -> None:
    long_r_ratios = [row["long_r_sd_over_exact"] for row in power_rows]
    lag1_r_ratios = [row["lag1_r_sd_over_exact"] for row in power_rows]
    long_delta_ratios = [row["long_rmse_over_delta"] for row in power_rows]
    lag1_delta_ratios = [row["lag1_rmse_over_delta"] for row in power_rows]
    minimum_failure = min(row["max_failure"] for row in boundary_rows)
    maximum_kl_ratio = max(row["complete_KL_over_ell"] for row in boundary_rows)
    minimum_bh_probability = min(row["BH_probability"] for row in boundary_rows)
    with output_path.open("w", encoding="utf-8") as stream:
        stream.write("Monte Carlo audit of the persistent-path result\n")
        stream.write("================================================\n")
        stream.write(f"Empirical long-lag RMSE slope: {slopes['long_empirical']:.6f}\n")
        stream.write(f"Delta-predicted long-lag slope: {slopes['long_delta']:.6f}\n")
        stream.write(f"Empirical lag-1 RMSE slope: {slopes['lag1_empirical']:.6f}\n")
        stream.write(
            "Empirical linearized lag-1 slope: "
            f"{slopes['lag1_linearized_empirical']:.6f}\n"
        )
        stream.write(f"Delta-predicted lag-1 slope: {slopes['lag1_delta']:.6f}\n")
        stream.write(
            "Long-lag empirical/exact moment-SD range: "
            f"[{min(long_r_ratios):.4f}, {max(long_r_ratios):.4f}]\n"
        )
        stream.write(
            "Lag-1 empirical/exact moment-SD range: "
            f"[{min(lag1_r_ratios):.4f}, {max(lag1_r_ratios):.4f}]\n"
        )
        stream.write(
            "Long-lag RMSE/delta-SD range: "
            f"[{min(long_delta_ratios):.4f}, {max(long_delta_ratios):.4f}]\n"
        )
        stream.write(
            "Lag-1 RMSE/delta-SD range: "
            f"[{min(lag1_delta_ratios):.4f}, {max(lag1_delta_ratios):.4f}]\n"
        )
        stream.write(
            "Minimum max two-point failure probability at the theoretical "
            f"half-separation: {minimum_failure:.4f} (required >= 0.05).\n"
        )
        stream.write(f"Maximum exact complete-KL/ell ratio: {maximum_kl_ratio:.6f}\n")
        stream.write(
            f"Minimum exact Bretagnolle-Huber probability: {minimum_bh_probability:.6f}\n"
        )
        stream.write("The confidence perturbation switches branch at Tu/ell = 2.5 exactly.\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="small deterministic smoke run")
    parser.add_argument("--output-dir", default="numerical_validation")
    parser.add_argument("--seed", type=int, default=20260827)
    arguments = parser.parse_args()

    output_dir = Path(arguments.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    power_rows, slopes = run_power_experiment(
        output_dir=output_dir, quick=arguments.quick, seed=arguments.seed
    )
    boundary_rows = run_boundary_experiment(
        output_dir=output_dir, quick=arguments.quick, seed=arguments.seed + 1
    )
    make_figure(output_dir / "persistent_path_validation.png", power_rows, slopes, boundary_rows)
    write_report(output_dir / "monte_carlo_report.txt", power_rows, slopes, boundary_rows)

    print((output_dir / "monte_carlo_report.txt").read_text(encoding="utf-8"))
    print("Power experiment")
    for row in power_rows:
        print(
            f"u={row['u']:.3f} k={row['k']:3d} M={row['M']:4d} "
            f"RMSE(long)={row['long_rmse']:.5g} "
            f"RMSE(lag1)={row['lag1_rmse']:.5g} "
            f"SD-ratios=({row['long_r_sd_over_exact']:.3f},"
            f" {row['lag1_r_sd_over_exact']:.3f})"
        )
    print("Boundary experiment")
    for row in boundary_rows:
        print(
            f"c={row['c_Tu_over_ell']:5.2f} T={row['T']:5d} "
            f"M={row['M']:3d} branch={row['branch']:9s} "
            f"max-failure={row['max_failure']:.3f} "
            f"q90={row['empirical_radius_90']:.4g}"
        )


if __name__ == "__main__":
    main()
